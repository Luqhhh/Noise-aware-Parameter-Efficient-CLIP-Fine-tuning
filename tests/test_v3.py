"""CPU checks of weighted supervision, replay isolation, and full pair delivery."""
from __future__ import annotations

import copy
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms as T
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducibility/aegis_f1"))
from v2 import training_utils
from v2.core import logical_backward
from v2.plan import dump, sha
from v3.core import (PairedImages, isolated_cpu_rng, mix_supervision, paired_draws,
                     weighted_backward)
from v3.plan import ARMS, check_recipe, frozen_groups, normalize_targets, verify, write_rows
from v3.report import compare, paired_metrics
from v3.runtime import fit_arm, run_pair


def augmentation(**updates):
    cfg = dict(mixup=.2, cutmix=1., mix_prob=.8)
    cfg.update(updates)
    return cfg


def test_weighted_microbatch_gradient_matches_full_logical_loss():
    torch.manual_seed(13)
    full = torch.nn.Linear(7, 5)
    micro = copy.deepcopy(full)
    images = torch.randn(11, 7)
    p = training_utils.one_hot(torch.arange(11) % 5, 5, .15)
    w = torch.tensor([0., .2, .8, .5, 1., .1, .3, 0., .7, .8, .4])
    masses = p * w[:, None]
    expected = -(masses * full(images).log_softmax(1)).sum() / w.sum()
    expected.backward()
    actual = weighted_backward(micro, images, masses, w.sum(), 3, nullcontext)
    assert actual == pytest.approx(float(expected.detach()), rel=1e-6)
    for a, b in zip(full.parameters(), micro.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-6, atol=1e-7)


def test_original_arm_is_the_v2_unweighted_logical_loss():
    torch.manual_seed(13)
    a = torch.nn.Linear(7, 5)
    b = copy.deepcopy(a)
    images = torch.randn(11, 7)
    p = training_utils.one_hot(torch.arange(11) % 5, 5, .15)
    expected = logical_backward(a, images, p, 3, nullcontext)
    actual = weighted_backward(b, images, p, torch.tensor(11.), 3, nullcontext)
    assert actual == pytest.approx(expected, rel=1e-6)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=2e-6, atol=1e-7)


def test_zero_supervision_still_has_a_finite_zero_gradient_and_an_update():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
    before = model.weight.detach().clone()
    loss = weighted_backward(model, torch.ones(4, 3), torch.zeros(4, 2), torch.tensor(1e-8), 3, nullcontext)
    assert loss == 0
    assert all(torch.count_nonzero(p.grad) == 0 for p in model.parameters())
    optimizer.step()
    assert all(float(optimizer.state[p]["step"]) == 1 for p in model.parameters())
    assert not torch.equal(before, model.weight)  # common AdamW update, not a skipped batch


@pytest.mark.parametrize("mode", ["mixup", "cutmix", "none"])
def test_mixing_has_identical_pixels_and_no_zero_weight_label_leakage(mode, monkeypatch):
    monkeypatch.setattr(training_utils, "rand_bbox", lambda *args: (1, 3, 0, 2))
    monkeypatch.setattr(torch, "randperm", lambda *args, **kwargs: torch.tensor([1, 0]))
    x = torch.stack([torch.zeros(3, 4, 4), torch.ones(3, 4, 4)])
    p = torch.eye(2)
    cfg = augmentation(mixup=.2 if mode == "mixup" else 0, cutmix=1. if mode == "cutmix" else 0,
                       mix_prob=0 if mode == "none" else 1)
    raw_images, raw_mass, _, raw_trace = mix_supervision(x, p, torch.ones(2), cfg, 13)
    images, masses, denominator, trace = mix_supervision(x, p, torch.tensor([0., .8]), cfg, 13)
    torch.testing.assert_close(images, raw_images, rtol=0, atol=0)
    assert trace == raw_trace and torch.count_nonzero(masses[:, 0]) == 0
    assert denominator.item() == pytest.approx(.8)
    assert masses.sum().item() == pytest.approx(.8)
    torch.testing.assert_close(raw_mass.sum(1), torch.ones(2))
    if mode == "cutmix":
        torch.testing.assert_close(masses[:, 1], torch.tensor([.2, .6]))


def test_original_label_sampler_keeps_sqrt_class_exposure_and_frozen_replay():
    labels = [0] * 4000 + [1] * 16000
    draws = paired_draws(labels, 2, 13, 2, 96)
    assert draws.shape == (2, 19968)
    assert (np.array(labels)[draws] == 1).mean() == pytest.approx(2 / 3, abs=.02)
    np.testing.assert_array_equal(draws, paired_draws(labels, 2, 13, 2, 96))


def test_image_loader_keeps_pixels_when_exif_metadata_is_malformed(tmp_path, monkeypatch):
    path = tmp_path / "valid_pixels.jpg"
    Image.new("RGB", (8, 8), (120, 30, 10)).save(path)

    def reject_exif(image):
        raise SyntaxError("not a TIFF file")

    monkeypatch.setattr(training_utils.ImageOps, "exif_transpose", reject_exif)
    loaded = training_utils.load_image(path)
    assert loaded.mode == "RGB" and loaded.size == (8, 8)
    with Image.open(path) as source:
        assert loaded.getpixel((0, 0)) == source.convert("RGB").getpixel((0, 0))


def test_image_replay_is_independent_of_worker_rng_and_arm(tmp_path):
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    Image.fromarray(image).save(tmp_path / "a.png")
    transform = T.Compose([T.RandomCrop(16), T.RandomHorizontalFlip(), T.ToTensor()])
    rows = [dict(relative_path="a.png")]
    data = PairedImages(tmp_path, rows, transform, np.array([0, 0, 0, 0]), 13, 1)
    serial = torch.stack([data[i][0] for i in range(len(data))])
    parallel = torch.cat([x for x, _ in torch.utils.data.DataLoader(
        data, batch_size=2, num_workers=2, multiprocessing_context="spawn", timeout=30)])
    torch.testing.assert_close(serial, parallel, rtol=0, atol=0)
    with isolated_cpu_rng(1234):
        torch.testing.assert_close(torch.stack([data[i][0] for i in range(len(data))]), serial, rtol=0, atol=0)


def target_fixture():
    rows = [dict(image_path="train/0000/a.png", label=0, content_group="a"),
            dict(image_path="train/0001/b.png", label=1, content_group="b"),
            dict(image_path="train/0000/c.png", label=0, content_group="c")]
    payload = dict(image_paths=[r["image_path"] for r in rows], class_names=["0000", "0001"],
                   labels=torch.tensor([0, 1, 0]), targets=torch.tensor([0, 1, 1]),
                   weights=torch.tensor([.2, .8, .5]), original_alpha=torch.zeros(3),
                   kept=torch.tensor([True, True, False]), pseudo=torch.tensor([False, False, True]),
                   agreement=torch.tensor([.3, .8, .1]))
    return payload, rows


def test_supervision_is_keyed_by_path_and_keeps_original_labels():
    payload, rows = target_fixture()
    result = normalize_targets(payload, rows[::-1], ["0000", "0001"])
    np.testing.assert_array_equal(result["labels"], [0, 1, 0])
    np.testing.assert_array_equal(result["targets"], [1, 1, 0])
    np.testing.assert_allclose(result["weights"], [.5, .8, .2])


@pytest.mark.parametrize("corruption", ["holdout", "label", "mapping", "nan", "coverage", "mask"])
def test_supervision_rejects_wrong_scope_and_invalid_targets(corruption):
    payload, rows = target_fixture()
    if corruption == "holdout":
        payload["image_paths"][0] = "train/0000/holdout.png"
    elif corruption == "label":
        payload["labels"][0] = 1
    elif corruption == "mapping":
        payload["class_names"].reverse()
    elif corruption == "nan":
        payload["weights"][0] = float("nan")
    elif corruption == "coverage":
        payload["weights"][:1] = 0
        payload["kept"][:1] = False
    else:
        payload["pseudo"][0] = True
    with pytest.raises(ValueError):
        normalize_targets(payload, rows, ["0000", "0001"])


def test_few_support_and_tail_groups_use_original_support_not_relabels():
    payload, rows = target_fixture()
    values = normalize_targets(payload, rows, payload["class_names"])
    groups = frozen_groups(rows, values, 2, .1, 1)
    assert groups["tail_classes"] == [1] and groups["few_support_classes"] == [1]
    assert groups["train_rows"] == [2, 1] and groups["active_target_rows"] == [1, 2]


@pytest.mark.parametrize("field,value", [("full_train", True), ("resolution_ladder", True),
                                        ("execution_authorized", True), ("sampler_labels", "targets"),
                                        ("epochs", 10), ("image_size", 576)])
def test_recipe_cannot_silently_expand_the_fixed_pair(field, value):
    recipe = json.loads((ROOT / "configs/v3/recipe.json").read_text())
    check_recipe(recipe)
    recipe[field] = value
    with pytest.raises(ValueError):
        check_recipe(recipe)


def test_false_authorization_never_queries_cuda(tmp_path, monkeypatch):
    auth = tmp_path / "authorization.json"
    dump(auth, dict(authorized=False))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("Unauthorized CUDA query"))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: pytest.fail("Unauthorized CUDA query"))
    with pytest.raises(ValueError, match="not authorized"):
        run_pair(tmp_path / "missing_plan.json", auth)
    assert not torch.cuda.is_initialized()


def test_pair_authorization_binds_operation_and_plan_without_cost_gate(tmp_path, monkeypatch):
    from v3.plan import authorize
    plan_path, auth_path = [tmp_path / name for name in ("plan.json", "auth.json")]
    dump(plan_path, {})
    plan = dict(total_updates=20, recipe=dict(epochs=2))
    monkeypatch.setattr("v3.plan.verify", lambda path: plan)
    auth = dict(authorized=True, operation="train", plan_sha256=sha(plan_path))
    dump(auth_path, auth)
    assert authorize(plan_path, auth_path, "train")[1] == plan
    auth["plan_sha256"] = "other"
    dump(auth_path, auth)
    with pytest.raises(ValueError, match="bind"):
        authorize(plan_path, auth_path, "train")


def test_pair_metrics_report_all_corrections_regressions_and_overlapping_slices():
    labels = [0, 0, 1, 1, 2, 2]
    a, b = [0, 1, 1, 0, 1, 2], [0, 0, 0, 1, 2, 1]
    groups = dict(tail_classes=[2], other_classes=[0, 1], few_support_classes=[1, 2])
    report = paired_metrics(labels, a, b, 3, groups)
    assert report["slices"]["all"]["corrections"] == 3
    assert report["slices"]["all"]["regressions"] == 2
    assert report["slices"]["all"]["net"] == 1
    assert report["slices"]["tail"]["net"] == 0
    assert report["decision"] == "no_support_for_ladder"
    assert not report["automatic_ladder"] and not report["clean_labels_known"]


def test_verified_inputs_reject_mutation(tmp_path):
    from v3 import plan as module
    recipe = json.loads((ROOT / "configs/v3/recipe.json").read_text())
    for name in ("train_manifest.csv", "val_manifest.csv", "shared_config.json", "supervision.npz", "draws.npy", "groups.json"):
        (tmp_path / name).write_text("frozen")
    payload = dict(recipe=recipe, arms=list(ARMS), execution_authorized=False,
                   v1_binding=dict(stage="repechage", data_version=recipe["data_version"]),
                   inputs={str(p): sha(p) for p in tmp_path.iterdir()})
    path = tmp_path / "plan.json"
    dump(path, payload)
    module.verify(path)
    (tmp_path / "draws.npy").write_text("changed")
    with pytest.raises(ValueError, match="Frozen input changed"):
        module.verify(path)


def test_preparation_rejects_previous_competition_stage_before_loading_targets(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from v3.plan import prepare
    source = dict(source=dict(partition="train_dev"), project=dict(data_version="20260921", stage="repechage"))
    monkeypatch.setattr("aegis_clip.v1_pipeline.load_recipe", lambda path: source)
    monkeypatch.setattr("aegis_clip.v1_pipeline.StageContext", lambda cfg: SimpleNamespace(manifest=dict(stage="preliminary")))
    monkeypatch.setattr("aegis_clip.v1_pipeline.load_artifact", lambda *args: pytest.fail("Previous-stage targets were read"))
    with pytest.raises(ValueError, match="different competition stage"):
        prepare(ROOT / "configs/v3/recipe.json", tmp_path / "prepared")


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = torch.nn.Linear(3, 5)
        self.head = torch.nn.Linear(5, 3)

    def forward(self, images):
        return self.head(self.visual(images.mean((2, 3))))


@pytest.fixture
def cpu_pair(tmp_path, monkeypatch):
    # Synthetic image/gradient integration only, never an official-data score.
    image_root, workspace = tmp_path / "images", tmp_path / "prepared"
    image_root.mkdir()
    workspace.mkdir()
    rows = []
    for i in range(9):
        Image.new("RGB", (40, 40), (20 * i, 255 - 20 * i, 50)).save(image_root / f"{i}.png")
        rows.append(dict(index=i, image_path=f"train/{i}.png", relative_path=f"{i}.png",
                         label=i % 3, content_group=str(i), source_sha256=sha(image_root / f"{i}.png")))
    val = rows[:3]
    cfg = yaml.safe_load((ROOT / "configs/v2/stages/s1_384.yaml").read_text())
    cfg["data"].update(image_size=32, batch_size=4, num_workers=0)
    cfg["train"].update(epochs=2, seed=13)
    cfg["augment"].update(color_jitter=0, rand_augment=False, random_erase=0)
    labels = np.array([r["label"] for r in rows])
    supervision = dict(labels=labels, targets=labels.copy(), original_alpha=np.zeros(9, np.float32),
                       weights=np.array([0, .2, .8, .5, .4, .9, .7, .3, .8], np.float32))
    groups = frozen_groups(rows, supervision, 3, .1, 1)
    dump(workspace / "groups.json", groups)
    write_rows(workspace / "val_manifest.csv", val)
    plan = dict(classes=["0000", "0001", "0002"], train_root=str(image_root), steps_per_epoch=2,
                updates_per_arm=4, experiment_id="synthetic_cpu_pair", recipe=dict(micro_batch_size=3))
    dump(workspace / "plan.json", plan)
    run_root = workspace / "runs/train"
    results = {}
    draws = paired_draws(labels, 3, 13, 2, 4)
    for arm in ARMS:
        with isolated_cpu_rng(13):
            model = TinyClassifier()
        results[arm] = fit_arm(model, plan, workspace, cfg, rows, val, draws, supervision,
                               arm, run_root / arm, torch.device("cpu"))
    dump(run_root / "status.json", dict(status="complete", plan_sha256=sha(workspace / "plan.json")))
    monkeypatch.setattr("v3.report.verify", lambda path: plan)
    return workspace, run_root, results


def test_full_cpu_pair_has_same_init_updates_trace_and_bound_report(cpu_pair):
    workspace, run_root, results = cpu_pair
    assert results[ARMS[0]]["initial_model_sha256"] == results[ARMS[1]]["initial_model_sha256"]
    assert all(r["updates"] == 4 for r in results.values())
    assert (run_root / ARMS[0] / "steps.jsonl").read_bytes() == (run_root / ARMS[1] / "steps.jsonl").read_bytes()
    report = compare(workspace / "plan.json", run_root, workspace / "comparison")
    assert report["paired_trace_verified"] and report["slices"]["all"]["rows"] == 3
    assert (workspace / "comparison/paired.csv").is_file()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("corruption", ["checkpoint", "predictions", "trace", "init", "endpoint"])
def test_pair_comparison_rejects_unpaired_or_changed_results(cpu_pair, corruption):
    workspace, run_root, _ = cpu_pair
    arm_root = run_root / ARMS[1]
    status = json.loads((arm_root / "status.json").read_text())
    if corruption in ("checkpoint", "predictions", "trace"):
        name = dict(checkpoint="selected.pt", predictions="predictions.npz", trace="steps.jsonl")[corruption]
        with (arm_root / name).open("ab") as handle:
            handle.write(b"changed")
    elif corruption == "init":
        status["initial_model_sha256"] = "different"
        dump(arm_root / "status.json", status)
    else:
        status["updates"] = 3
        dump(arm_root / "status.json", status)
    with pytest.raises(ValueError):
        compare(workspace / "plan.json", run_root, workspace / "comparison")
