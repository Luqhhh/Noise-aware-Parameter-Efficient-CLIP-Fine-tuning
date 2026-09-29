"""CPU strategy tests: effective adapters, leakage boundaries and resumability."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from clip.model import VisionTransformer
from torch.utils.data import DataLoader, TensorDataset

from aegis_clip.b448_pipeline import (
    Images, image_transform, load_artifact, load_recipe, save_artifact, seed_training,
    train_loop, validate_calibration, validate_rows,
)
from aegis_clip.b448_strategy import (
    B448Classifier, WeightAverage, aligned_positions, confident_keep, denoised_targets,
    fit_training_bias, knn_signals, target_probabilities, trainable_state,
    weighted_mixup_loss,
)
from aegis_clip.submission import create_submission

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_model(seed=42):
    seed_training(seed)
    visual = VisionTransformer(input_resolution=32, patch_size=8, width=32,
                               layers=2, heads=4, output_dim=16)
    return B448Classifier(visual, 3, image_size=32, rank=2, alpha=4, blocks=2)


def test_aligned_positions_preserve_cls_and_odd_grid_anchors():
    source = torch.randn(50, 8)
    result = aligned_positions(source, 13)
    assert torch.equal(result[0], source[0])
    anchors = result[1:].reshape(13, 13, 8)[::2, ::2]
    torch.testing.assert_close(anchors, source[1:].reshape(7, 7, 8), rtol=0, atol=0)
    with pytest.raises(ValueError, match="square"):
        aligned_positions(torch.zeros(49, 2), 14)


def test_native_qkv_output_and_mlp_adapters_are_effective_and_frozen_base_stays_fixed():
    seed_training(42)
    visual = VisionTransformer(32, 8, 32, 2, 4, 16).float().eval()
    original = copy.deepcopy(visual)
    model = B448Classifier(visual, 3, image_size=32, rank=2, alpha=4, blocks=2)
    images = torch.randn(4, 3, 32, 32)
    model.eval()
    torch.testing.assert_close(model.visual(images), original(images))
    assert len(model.adapted_modules) == 8
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert all("lora_" in name or name.startswith("head.") for name in trainable)
    before = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
    model.train()
    model(images).square().mean().backward()
    for block in model.visual.transformer.resblocks:
        assert block.attn.parametrizations.in_proj_weight[0].lora_B.grad.abs().sum() > 0
        assert block.attn.out_proj.parametrizations.weight[0].lora_B.grad.abs().sum() > 0
        assert block.mlp.c_fc.parametrizations.weight[0].lora_B.grad.abs().sum() > 0
        assert block.mlp.c_proj.parametrizations.weight[0].lora_B.grad.abs().sum() > 0
    opt.step()
    assert not torch.allclose(model.visual(images), original(images))
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(p, before[name])


def test_chunked_knn_excludes_same_content_and_handles_no_neighbours():
    features = torch.tensor([[1., 0.], [1., 0.], [.8, .2], [0., 1.]])
    labels = torch.tensor([0, 1, 0, 1])
    groups = torch.tensor([0, 0, 1, 2])
    agreement, prediction = knn_signals(features, labels, features, labels, 2, k=1,
        query_groups=groups, gallery_groups=groups, query_chunk=1, gallery_chunk=1)
    assert agreement[0] == 1 and agreement[1] == 0
    assert prediction[:2].tolist() == [0, 0]
    _, empty = knn_signals(features[:1], labels[:1], features[:1], labels[:1], 2,
        query_groups=groups[:1], gallery_groups=groups[:1])
    assert empty.tolist() == [-1]
    with pytest.raises(ValueError, match="Both"):
        knn_signals(features, labels, features, labels, 2, query_groups=groups)


def test_confident_learning_keeps_uncertain_and_never_selects_absent_classes():
    labels = torch.tensor([0, 0, 1, 1])
    probabilities = torch.tensor([[.8, .15, .05], [.25, .65, .1], [.1, .8, .1], [.45, .4, .15]])
    keep, thresholds = confident_keep(probabilities, labels)
    assert keep.tolist() == [True, False, True, True]
    assert torch.isinf(thresholds[2])


def test_pseudolabel_requires_confidence_margin_and_independent_knn_consensus():
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    p = torch.tensor([[.95, .05], [.1, .9], [.2, .8], [.1, .9], [.9, .1], [.9, .1]])
    targets = denoised_targets(p, labels, torch.ones(6), torch.tensor([0, 1, 0, 1, 0, -1]))
    assert targets["pseudo"].tolist() == [False, True, False, False, True, False]
    assert targets["targets"][1] == 1
    assert targets["weights"][2] == 0 and targets["weights"][5] == 0


def test_mixup_uses_weighted_targets_and_soft_original_labels_even_when_mixed():
    logits = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    probabilities = target_probabilities(torch.tensor([1, 0]), torch.tensor([0, 1]),
                                          torch.tensor([.25, 0.]), 2, .1)
    weights = torch.tensor([1., 0.])
    loss = weighted_mixup_loss(logits, probabilities, weights, torch.tensor([1, 0]), .7)
    expected = -(probabilities[0] * (.7 * logits[0].log_softmax(0)
                + .3 * logits[1].log_softmax(0))).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_ema_and_swa_update_weights_without_output_fusion():
    ema = WeightAverage({"w": torch.tensor([0.])}, .9)
    ema.update({"w": torch.tensor([10.])})
    torch.testing.assert_close(ema.state["w"], torch.tensor([1.]))
    swa = WeightAverage({"w": torch.tensor([-99.])})
    for v in (2., 4., 6.):
        swa.update({"w": torch.tensor([v])})
    assert swa.count == 3 and swa.state["w"].item() == 4
    restored = WeightAverage.restore(swa.payload())
    restored.update({"w": torch.tensor([8.])})
    assert restored.state["w"].item() == 5


def test_training_bias_has_uniform_mass_and_requires_class_coverage():
    logits = torch.tensor([[3., 0., 0.], [3., 0., 0.], [3., 0., 0.], [3., 0., 0.]])
    labels = torch.tensor([0, 0, 1, 2])
    bias = fit_training_bias(logits, labels, iterations=20)
    torch.testing.assert_close((logits + bias).softmax(1), torch.full((4, 3), 1/3))
    with pytest.raises(ValueError, match="every class"):
        fit_training_bias(logits, torch.zeros(4, dtype=torch.long))


@pytest.mark.parametrize("bad", ["content", "path", "outside", "mapping"])
def test_split_scope_rejects_leakage_and_invalid_roots(bad):
    train = [dict(image_path="train/0000/a.jpg", label="0", content_group="a")]
    val = [dict(image_path="train/0001/b.jpg", label="1", content_group="b")]
    if bad == "content":
        val[0]["content_group"] = "a"
    elif bad == "path":
        val = copy.deepcopy(train)
    elif bad == "outside":
        train[0]["image_path"] = "test/a.jpg"
    else:
        train[0]["label"] = "1"
    with pytest.raises(ValueError):
        validate_rows(train, val, ["0000", "0001"])


def test_artifacts_fail_closed_on_stage_or_checksum_change(tmp_path):
    path = tmp_path / "targets.pt"
    binding = {"stage": "repechage", "data_version": "20260921"}
    save_artifact(path, dict(targets=torch.tensor([1])), binding)
    assert load_artifact(path, binding)["targets"].item() == 1
    with pytest.raises(ValueError, match="binding"):
        load_artifact(path, {"stage": "initial"})
    with pytest.raises(FileExistsError):
        save_artifact(path, {}, binding)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_artifact(path, binding)


def test_calibration_rejects_test_fitting_or_another_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"toy")
    from aegis_clip.runtime import sha256_file
    decoder = dict(scales=[448, 512, 576])
    context = SimpleNamespace(classes=["0000", "0001"], config=dict(decode=decoder))
    payload = dict(source="official_training_val_dev", test_data_used=False,
        checkpoint_sha256=sha256_file(checkpoint), decoder=decoder, bias=torch.zeros(2))
    validate_calibration(payload, context, checkpoint)
    payload["test_data_used"] = True
    with pytest.raises(ValueError, match="Calibration"):
        validate_calibration(payload, context, checkpoint)


def test_real_training_loop_resume_matches_uninterrupted_ema_swa_and_valid_package(tmp_path):
    rng = torch.Generator().manual_seed(5)
    images = torch.randn(8, 3, 32, 32, generator=rng)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    loader = DataLoader(TensorDataset(images, torch.arange(8)), batch_size=2, shuffle=True)
    targets = dict(targets=labels, labels=labels, weights=torch.ones(8), original_alpha=torch.zeros(8),
                   class_names=["0000", "0001", "0002"], stats={"used": 8})
    cfg = dict(seed=42, epochs=3, lora_lr=.001, head_lr=.005, amp=False,
               ema_decay=.9, swa_enabled=True, swa_start=2, label_smoothing=.1,
               grad_clip=1., mixup_alpha=.2, model_config={"test": "tiny"})
    binding = dict(stage="synthetic_cpu", recipe_sha256="toy")
    uninterrupted = tiny_model()
    full = train_loop(uninterrupted, loader, targets, cfg, binding, tmp_path / "full")
    split = tiny_model()
    first = train_loop(split, loader, targets, cfg, binding, tmp_path / "split", stop_after=1)
    resume_model = tiny_model()  # Same frozen initialization, as with official CLIP.
    seed_training(99)  # Resumption must restore RNG independently of ambient state.
    last_epoch = train_loop(resume_model, loader, targets, cfg, binding, tmp_path / "split", resume=first,
                            stop_after=cfg["epochs"])
    # Also recover a stop after the last epoch was sealed, before selected.pt.
    resumed = train_loop(tiny_model(), loader, targets, cfg, binding, tmp_path / "split", resume=last_epoch)
    a, b = load_artifact(full, binding), load_artifact(resumed, binding)
    assert a["swa_epochs"] == b["swa_epochs"] == [2, 3]
    for name in a["selected_state"]:
        torch.testing.assert_close(a["selected_state"][name], b["selected_state"][name], rtol=0, atol=0)
    # Reconstruct the frozen initialization, then load selected trainable tensors.
    from aegis_clip.b448_strategy import load_trainable_state
    inference = tiny_model()
    load_trainable_state(inference, b["selected_state"])
    inference.eval()
    preds = inference(images[:3]).argmax(1).tolist()
    names = [f"synthetic_{i}.jpg" for i in range(3)]
    create_submission([(name, targets["class_names"][pred]) for name, pred in zip(names, preds)],
        names, tmp_path / "submission", resumed, inference_mode="synthetic_cpu",
        tta_risk_acknowledged=True, valid_labels=set(targets["class_names"]), space_after_comma=True)
    csv_path = tmp_path / "submission/pred_results.csv"
    with zipfile.ZipFile(tmp_path / "submission/submission.zip") as archive:
        assert archive.namelist() == ["pred_results.csv"]
        assert archive.read("pred_results.csv") == csv_path.read_bytes()
    assert b", 000" in csv_path.read_bytes()


def test_fixed_tta_crops_to_448_at_all_resize_scales_and_does_not_stretch():
    from PIL import Image
    image = Image.new("RGB", (800, 600), color="green")
    for scale in (448, 512, 576):
        assert image_transform(448, scale=scale)(image).shape == (3, 448, 448)


def test_recipe_swa_is_default_off_and_compute_requires_execute():
    config = load_recipe(ROOT / "configs/rematch750_b448_strategy.yaml")
    assert config["train"]["swa_enabled"] is False
    assert config["model"]["rank"] == 32 and config["model"]["blocks"] == 12
    run = subprocess.run([sys.executable, "-m", "aegis_clip.cli.b448_strategy", "train",
                          "--config", "/nonexistent", "--device", "cuda"], capture_output=True, text=True)
    assert run.returncode == 2 and "require --execute" in run.stderr


def test_image_symlink_escape_fails_without_reading_external_file(tmp_path):
    root = tmp_path / "train"
    (root / "0000").mkdir(parents=True)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"not an image")
    (root / "0000/a.jpg").symlink_to(outside)
    ds = Images(root, [{"image_path": "train/0000/a.jpg"}], image_transform(32))
    with pytest.raises(ValueError, match="symlink"):
        ds[0]


def test_calibrate_and_infer_complete_cpu_workflow_uses_bound_training_bias(tmp_path, monkeypatch):
    from PIL import Image
    from aegis_clip import b448_pipeline as pipeline
    train_root, test_root = tmp_path / "train", tmp_path / "test"
    rows = []
    for c in range(3):
        directory = train_root / f"{c:04d}"
        directory.mkdir(parents=True)
        for i in range(2):
            name = f"{i}.jpg"
            Image.new("RGB", (48, 40), color=(c * 80, i * 100, 40)).save(directory / name)
            rows.append(dict(image_path=f"train/{c:04d}/{name}", label=str(c), content_group=f"{c}-{i}"))
    test_root.mkdir()
    test_rows = []
    for i in range(3):
        name = f"synthetic_{i}.jpg"
        Image.new("RGB", (48, 40), color=(i * 80, 100, 40)).save(test_root / name)
        test_rows.append(dict(image_path=f"test/{name}", label="-1"))
    mapping = tmp_path / "class_to_idx.json"
    mapping.write_text(json.dumps({f"{i:04d}": i for i in range(3)}))
    with (tmp_path / "test_manifest.csv").open("w") as handle:
        import csv
        writer = csv.DictWriter(handle, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerows(test_rows)
    checkpoint = tmp_path / "toy_checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    model = tiny_model()
    context = SimpleNamespace(config=dict(model=dict(image_size=32),
        train=dict(batch_size=2, num_workers=0),
        decode=dict(scales=[32, 40, 48], flip=True, bias_iterations=20, bias_strength=1.),
        output=dict(root=str(tmp_path / "output"))), calibration=rows,
        train_root=train_root, test_root=test_root, classes=["0000", "0001", "0002"],
        binding=dict(stage="synthetic_cpu"), manifest=dict(test_samples=3),
        reference=dict(data=dict(dataset_manifest=str(tmp_path / "dataset_manifest.json"), class_mapping=str(mapping))))
    monkeypatch.setattr(pipeline, "load_selected", lambda *args: (model, dict(selected_policy="ema")))
    path = pipeline.calibrate(context, checkpoint, "cpu")
    bias = load_artifact(path, context.binding)
    assert bias["test_data_used"] is False and bias["samples"] == 6
    out = pipeline.infer(context, checkpoint, "cpu")
    assert "All checks passed!" in (out / "submission_check.log").read_text()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["prediction_count"] == 3
    assert manifest["platform_score"] is None
    assert manifest["status"] == "unpromoted_research_candidate"
    # This is a frozen input to inference: test prediction collection cannot
    # silently refit or mutate the saved calibration artifact.
    torch.testing.assert_close(load_artifact(path, context.binding)["bias"], bias["bias"])


def test_target_pipeline_trains_teacher_on_training_partition_only(tmp_path):
    from aegis_clip.b448_pipeline import prepare_targets
    rng = torch.Generator().manual_seed(3)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    features = torch.eye(3)[labels] + .02 * torch.randn(9, 3, generator=rng)
    rows = [dict(image_path=f"train/{int(label):04d}/{i}.jpg", label=str(int(label)),
                 content_group=f"g{i}") for i, label in enumerate(labels)]
    context = SimpleNamespace(config=dict(project=dict(seed=42),
        denoise=dict(knn_k=2, query_chunk=2, gallery_chunk=3, keep_ratio=.9,
            high_agreement_floor=.7, teacher_epochs=2, teacher_batch_size=3,
            teacher_lr=.01, minimum_weight=.2, pseudo_threshold=.7,
            pseudo_margin=.2, pseudo_soft_alpha=0.), output=dict(root=str(tmp_path))),
        train=rows, classes=["0000", "0001", "0002"], features=lambda: features,
        binding=dict(stage="synthetic_cpu", partition="training_only"))
    result = load_artifact(prepare_targets(context, "cpu"), context.binding)
    assert result["image_paths"] == [r["image_path"] for r in rows]
    assert result["labels"].tolist() == labels.tolist()
    assert result["stats"]["original_samples"] == 9
    assert result["stats"]["used"] > 0
    assert torch.isfinite(result["weights"]).all()


def test_resume_rejects_modified_target_artifact(tmp_path):
    model = tiny_model()
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    loader = DataLoader(TensorDataset(torch.zeros(8, 3, 32, 32), torch.arange(8)), batch_size=2)
    targets = dict(targets=labels, labels=labels, weights=torch.ones(8), original_alpha=torch.zeros(8),
                   class_names=["0000", "0001", "0002"], stats={"used": 8})
    cfg = dict(seed=42, epochs=3, lora_lr=.001, head_lr=.005, amp=False,
               ema_decay=.9, swa_enabled=False, swa_start=2, label_smoothing=.1,
               grad_clip=1., mixup_alpha=.2, model_config={"test": "tiny"}, targets_sha256="old")
    binding = dict(stage="synthetic_cpu")
    first = train_loop(model, loader, targets, cfg, binding, tmp_path, stop_after=1)
    cfg["targets_sha256"] = "changed"
    with pytest.raises(ValueError, match="targets have changed"):
        train_loop(tiny_model(), loader, targets, cfg, binding, tmp_path, resume=first)
