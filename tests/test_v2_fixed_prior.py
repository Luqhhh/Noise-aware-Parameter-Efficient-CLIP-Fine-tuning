"""Decoder invariance, independent prior fit and fail-closed delivered provenance."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_v2_fixed_prior as run
from verify_v1_768_full_delivery import numpy_uniform_bias
from aegis_clip.v1_test_bias import fit_test_uniform_bias


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def delivered(tmp_path):
    cfg = run.load_config(ROOT / "configs/v2_fixed_prior_20261002.json")
    source, stage, test = tmp_path / "source", tmp_path / "stage", tmp_path / "test"
    source.mkdir(); stage.mkdir(); test.mkdir()
    cfg.update(delivery_root=str(source), stage_root=str(stage))
    names = [f"{i}.jpg" for i in range(9)]
    for name in names:
        (test / name).write_bytes(b"unit-test placeholder, never decoded")
    (stage / "test_manifest.csv").write_text("image_path,file_sha256\n" + "".join(
        f"test/{n},{run.sha256_file(test / n)}\n" for n in names))
    for name in ("val_dev.csv", "train_dev.csv", "full_train.csv"):
        (stage / name).write_text("image_path,label\ntrain/a.jpg,0\n")
    dump(stage / "class_to_idx.json", {f"{i:04d}": i for i in range(3)})
    hashes = {p.name: run.sha256_file(p) for p in stage.iterdir()}
    manifest = dict(data_version="20260921", external_data=False, num_classes=3, test_samples=9,
                    train_samples=1, test_root=str(test), files=hashes)
    dump(stage / "dataset_manifest.json", manifest)
    spec = dict(epochs=5, image_size=576, steps_per_epoch=1858, total_updates=9290,
                config="configs/full_576.json")
    final_config = {"synthetic": True}
    config_bytes = json.dumps(final_config).encode()
    spec["config_sha256"] = run.digest_bytes(config_bytes)
    source_inputs = {"/source/" + name: h for name, h in hashes.items()}
    source_inputs["/source/train_manifest.csv"] = "original-current-stage-manifest"
    for name in ("reproducibility/aegis_f1/v2/model.py", "reproducibility/aegis_f1/v2/training_utils.py",
                 "reproducibility/aegis_f1/aegis_clip/model.py"):
        source_inputs["/source/" + name] = run.sha256_file(ROOT / name)
    plan = dict(experiment_id="V2_20260929", data_version="20260921", strategy_revision="v2_recipe_1",
                recipe=dict(views=run.VIEWS, official_sha256=cfg["official_sha256"],
                            num_classes=3, test_rows=9, official_train_rows=1, ema_enabled=False),
                inputs=source_inputs, stages={s: spec for s in ("s2_448", "s3_576", "full_576")})
    plan_bytes = json.dumps(plan).encode()
    cfg["expected_plan_sha256"] = run.digest_bytes(plan_bytes)
    binding = dict(experiment_id=plan["experiment_id"], data_version="20260921", stage="full_576",
                   plan_sha256=cfg["expected_plan_sha256"],
                   strategy_revision=plan["strategy_revision"], complete=True, completed_epochs=5,
                   official_sha256=cfg["official_sha256"], stage_config_sha256=spec["config_sha256"],
                   manifest_sha256="original-current-stage-manifest")
    payload = dict(binding=binding, epoch=4, global_step=9290, config=final_config, image_size=576,
                   num_classes=3, ema_enabled=False, ema_updates=0, metrics={"chosen": "raw"},
                   model={"toy.weight": torch.ones(3, 2)})
    checkpoint = source / run.FINAL
    checkpoint.parent.mkdir(parents=True)
    torch.save(payload, checkpoint)
    raw = np.arange(9) % 3
    run.write_submission(source / "submission", names, raw, 3)
    report = dict(status="package_ready", weights="raw", views=run.VIEWS, rows=9,
                  checkpoint_sha256=run.sha256_file(checkpoint),
                  csv_sha256=run.sha256_file(source / "submission/pred_results.csv"),
                  zip_sha256=run.sha256_file(source / "submission/submission.zip"))
    dump(source / "submission/report.json", report)
    members = {"plan.json": plan_bytes, "configs/full_576.json": config_bytes,
               "controller.json": json.dumps(dict(status="completed_delivered", training_completed=True,
                                 plan_sha256=cfg["expected_plan_sha256"])).encode(),
               "runs/full_576/last.binding.json": json.dumps(dict(binding=binding,
                                sha256=run.sha256_file(checkpoint))).encode()}
    for s in ("s2_448", "s3_576", "full_576"):
        members[f"runs/{s}/status.json"] = json.dumps(dict(status="complete", updates=9290)).encode()
    with tarfile.open(source / "delivery_metadata.tar.gz", "w:gz") as archive:
        for name, value in members.items():
            info = tarfile.TarInfo(name); info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
    files = [dict(relative=str(p.relative_to(source)), bytes=p.stat().st_size, sha256=run.sha256_file(p))
             for p in source.rglob("*") if p.is_file()]
    receipt = dict(status="verified_before_shutdown", files=files, submission_check_passed=True,
                   plan_sha256=cfg["expected_plan_sha256"])
    dump(source / "delivery_receipt.json", receipt)
    return cfg, source, stage


def test_original_probability_decoder_is_preserved_and_not_logit_sum():
    # A logit sum and a probability average can select different classes.
    views = np.array([[[7, -3, -2]], [[-1, 2, 1]], [[-1, 2, 1]], [[-1, 2, 1]]], np.float32)
    soft = np.exp(views - views.max(2, keepdims=True))
    soft /= soft.sum(2, keepdims=True)
    mean = soft.mean(0)
    assert views.sum(0).argmax(1).item() != mean.argmax(1).item()
    np.testing.assert_array_equal(run.probability_scores(mean).argmax(1), mean.argmax(1))


def test_one_fixed_bias_matches_float64_without_forcing_equal_argmax_counts():
    torch.set_num_threads(2)
    rng = np.random.default_rng(33)
    scores = rng.normal(size=(79, 5)).astype(np.float32) + [3, -1, 0, 0, 1]
    p = np.exp(scores-scores.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    logp = run.probability_scores(p)
    original = logp.copy()
    bias = fit_test_uniform_bias(torch.from_numpy(logp), iterations=200).numpy()
    expected = numpy_uniform_bias(logp, 200)
    np.testing.assert_allclose(bias, expected, atol=2e-6, rtol=0)
    np.testing.assert_array_equal(logp, original)
    z = logp + bias; aligned = np.exp(z-z.max(1, keepdims=True)); aligned /= aligned.sum(1, keepdims=True)
    np.testing.assert_allclose(aligned.mean(0), .2, atol=1e-6, rtol=0)
    assert np.unique(np.bincount(aligned.argmax(1), minlength=5)).size > 1


@pytest.mark.parametrize("bad", [np.zeros((2, 3)), np.full((2, 3), np.nan),
                                 np.array([[1.2, -.2]]), np.ones((2, 1))])
def test_invalid_probability_matrix_is_rejected(bad):
    with pytest.raises(ValueError, match="Invalid"):
        run.probability_scores(bad)


def test_missing_or_tampered_delivery_is_rejected_before_cuda(delivered, monkeypatch):
    cfg, source, _ = delivered
    monkeypatch.setattr(run, "ensure_idle_cuda", lambda: pytest.fail("CUDA reached before provenance passed"))
    (source / "submission/pred_results.csv").write_text("bad")
    with pytest.raises(ValueError, match="Incomplete delivery"):
        run.run(cfg, ROOT / "configs/v2_fixed_prior_20261002.json", source / "new")
    (source / "delivery_receipt.json").unlink()
    with pytest.raises(ValueError, match="Waiting"):
        run.verified_delivery(cfg)
    assert not (source / "new").exists()


def test_verified_delivery_strictly_checks_completed_checkpoint(delivered):
    cfg, source, _ = delivered
    result = run.verified_delivery(cfg)
    assert result["payload"]["global_step"] == 9290
    assert len(result["files"]) == 9
    cfg["expected_plan_sha256"] = "wrong-plan"
    with pytest.raises(ValueError, match="wrong local"):
        run.verified_delivery(cfg)


def test_mapping_corruption_rejected_even_if_local_manifest_is_rewritten(delivered):
    cfg, _, stage = delivered
    dump(stage / "class_to_idx.json", {"0000": 1, "0001": 0, "0002": 2})
    manifest = run.read_json(stage / "dataset_manifest.json")
    manifest["files"]["class_to_idx.json"] = run.sha256_file(stage / "class_to_idx.json")
    dump(stage / "dataset_manifest.json", manifest)
    with pytest.raises(ValueError, match="mapping"):
        run.verified_delivery(cfg)


def test_test_image_changes_rejected_before_any_cuda_operation(delivered, monkeypatch):
    cfg, source, stage = delivered
    manifest = run.read_json(stage / "dataset_manifest.json")
    (Path(manifest["test_root"]) / "0.jpg").write_bytes(b"corruption")
    monkeypatch.setattr(run, "ensure_idle_cuda", lambda: pytest.fail("CUDA reached for changed images"))
    with pytest.raises(ValueError, match="pixels changed"):
        run.run(cfg, ROOT / "configs/v2_fixed_prior_20261002.json", source / "new")
    assert not (source / "new").exists()


def test_archive_duplicate_or_symlink_metadata_is_rejected(tmp_path):
    path = tmp_path / "metadata.tar"
    with tarfile.open(path, "w") as archive:
        for _ in range(2):
            info = tarfile.TarInfo("plan.json"); info.size = 2
            archive.addfile(info, io.BytesIO(b"{}"))
        link = tarfile.TarInfo("linked.json"); link.type = tarfile.SYMTYPE; link.linkname = "/etc/passwd"
        archive.addfile(link)
    with tarfile.open(path) as archive:
        for name in ("plan.json", "linked.json"):
            with pytest.raises(ValueError, match="ambiguous"):
                run.metadata_member(archive, name)


def test_run_two_packages_full_cpu_replay(delivered, monkeypatch):
    cfg, source, _ = delivered
    config = source / "protocol.json"; dump(config, cfg)
    rng = np.random.default_rng(9)
    p = .1*rng.random(size=(9, 3)) + np.eye(3)[np.arange(9) % 3]
    p /= p.sum(1, keepdims=True)
    p = p.astype(np.float32)
    monkeypatch.setattr(run, "ensure_idle_cuda", lambda: None)
    monkeypatch.setattr(run, "load_model", lambda *a: object())
    monkeypatch.setattr(run, "collect_probabilities", lambda *a: p)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "synthetic_cpu")
    output = source / "candidate"
    run.run(cfg, config, output)
    report = run.read_json(output / "report.json")
    independent = run.read_json(output / "independent_verification.json")
    assert report["status"] == "completed_verified_delivery"
    assert independent["full_float64_decision_agreement"]
    assert report["platform_score"] is None and report["model_parameter_updates"] is False
    for name in ("submission_bias", "submission_raw"):
        assert independent["packages"][name]["rows"] == 9
    with pytest.raises(ValueError, match="Output exists"):
        run.run(cfg, config, output)


def test_changing_temperature_or_bias_strength_is_not_supported(tmp_path):
    cfg = run.load_config(ROOT / "configs/v2_fixed_prior_20261002.json")
    cfg["bias_strength"] = .8
    path = tmp_path / "config.json"; dump(path, cfg)
    with pytest.raises(ValueError, match="protocol changed"):
        run.load_config(path)


def test_only_small_low_margin_raw_portability_differences_can_pass():
    cfg = run.load_config(ROOT / "configs/v2_fixed_prior_20261002.json")
    p = np.tile(np.array([.501, .499], np.float32), (1000, 1))
    original = np.zeros(1000, dtype=np.int64); original[0] = 1
    assert run.raw_replay(p, original, cfg)["raw_replay_passed"]
    original[1] = 1
    assert not run.raw_replay(p, original, cfg)["raw_replay_passed"]
    original[1] = 0; p[0] = [.55, .45]
    assert not run.raw_replay(p, original, cfg)["raw_replay_passed"]
