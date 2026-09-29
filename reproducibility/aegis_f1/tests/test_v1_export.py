"""Post-training export matches online SWA and rejects incompatible epochs."""
from __future__ import annotations

import json
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from clip.model import VisionTransformer
from torch.utils.data import DataLoader, TensorDataset

from aegis_clip.runtime import sha256_file
from aegis_clip.v1_export import export_swa
from aegis_clip.v1_pipeline import load_artifact, load_recipe, save_artifact, seed_training, train_loop
from aegis_clip.v1_strategy import V1Classifier, WeightAverage

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def tiny_model():
    seed_training(42)
    return V1Classifier(VisionTransformer(32, 8, 32, 2, 4, 16), 3,
                        image_size=32, blocks=2, rank=2, alpha=4)


@pytest.fixture
def trained(tmp_path):
    recipe = load_recipe(ROOT / "configs/v1_train_20260929.yaml")
    recipe["model"].update(image_size=32, blocks=2, rank=2, alpha=4.)
    recipe["train"].update(epochs=3, batch_size=2, num_workers=0, amp=False,
                           swa_start=2, ema_decay=.9, lora_lr=.001, head_lr=.005)
    recipe["decode"].update(scales=[32, 40, 48], bias_iterations=20)
    recipe["output"]["root"] = str(tmp_path / "run")
    context = SimpleNamespace(config=recipe, classes=["0000", "0001", "0002"],
                              binding=dict(stage="synthetic_cpu", recipe_sha256="fixture"))
    generator = torch.Generator().manual_seed(5)
    images = torch.randn(8, 3, 32, 32, generator=generator)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    loader = DataLoader(TensorDataset(images, torch.arange(8)), batch_size=2, shuffle=True)
    targets = dict(targets=labels, labels=labels, weights=torch.ones(8), original_alpha=torch.zeros(8),
                   class_names=context.classes, stats=dict(used=8))
    target_path = Path(recipe["output"]["root"]) / "targets.pt"
    save_artifact(target_path, targets, context.binding)
    cfg = dict(recipe["train"], seed=42, model_config=recipe["model"], targets_sha256=sha256_file(target_path))
    source = target_path.parent / "training"
    train_loop(tiny_model(), loader, targets, cfg, context.binding, source)
    return SimpleNamespace(context=context, source=source, cfg=cfg, loader=loader,
                           targets=targets, images=images, output=tmp_path / "export")


def rewrite_checkpoint(trained, change):
    path = trained.source / "epoch_03.pt"
    payload = load_artifact(path, trained.context.binding)
    change(payload)
    path.unlink()
    path.with_suffix(".sha256.json").unlink()
    save_artifact(path, payload, trained.context.binding)


def test_posthoc_ema_matches_online_swa_and_loads_as_one_selected_checkpoint(trained, monkeypatch):
    before = {path.name: sha256_file(path) for path in trained.source.iterdir()}
    exported = export_swa(trained.context, trained.output)
    online = train_loop(tiny_model(), trained.loader, trained.targets, dict(trained.cfg, swa_enabled=True),
                        trained.context.binding, trained.output.parent / "online")
    actual = load_artifact(exported, trained.context.binding)
    expected = load_artifact(online, trained.context.binding)
    assert actual["swa_epochs"] == [2, 3] and actual["selected_policy"] == "swa_ema"
    assert actual["optimizer"] is None and actual["rng"] is None
    for name, value in actual["selected_state"].items():
        torch.testing.assert_close(value, expected["selected_state"][name], rtol=0, atol=0)
    assert before == {path.name: sha256_file(path) for path in trained.source.iterdir()}
    assert not (trained.output / "calibration.pt").exists()
    recipe = load_recipe(trained.output / "config.yaml")
    assert recipe["train"] == trained.context.config["train"]
    assert recipe["output"]["root"] == str(trained.output)
    from aegis_clip import v1_pipeline as pipeline
    monkeypatch.setattr(pipeline, "build_classifier", lambda *args: tiny_model())
    context = SimpleNamespace(config=recipe, binding=trained.context.binding,
                              official="synthetic", classes=trained.context.classes)
    model, payload = pipeline.load_selected(context, exported, "cpu")
    assert model(trained.images).shape == (8, 3)
    assert payload["selected_policy"] == "swa_ema"
    from PIL import Image
    train_root, test_root = trained.output.parent / "images", trained.output.parent / "test"
    val_rows = []
    for label, name in enumerate(context.classes):
        directory = train_root / name
        directory.mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (48, 40), color=(label * 80, index * 100, 40)).save(directory / f"{index}.jpg")
            val_rows.append(dict(image_path=f"train/{name}/{index}.jpg", label=str(label)))
    test_root.mkdir()
    test_rows = []
    for index in range(3):
        name = f"synthetic_{index}.jpg"
        Image.new("RGB", (48, 40), color=(index * 80, 100, 40)).save(test_root / name)
        test_rows.append(dict(image_path=f"test/{name}", label="-1"))
    mapping = trained.output.parent / "class_to_idx.json"
    mapping.write_text(json.dumps({name: index for index, name in enumerate(context.classes)}))
    with (mapping.parent / "test_manifest.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerows(test_rows)
    context.train, context.calibration = val_rows, val_rows
    context.train_root, context.test_root = train_root, test_root
    context.manifest = dict(test_samples=3)
    context.reference = dict(data=dict(dataset_manifest=str(mapping.parent / "dataset_manifest.json"),
                                       class_mapping=str(mapping)))
    pipeline.calibrate(context, exported, "cpu")
    submission = pipeline.infer(context, exported, "cpu")
    assert "All checks passed!" in (submission / "submission_check.log").read_text()
    manifest = json.loads((submission / "manifest.json").read_text())
    assert manifest["selected_policy"] == "swa_ema" and manifest["prediction_count"] == 3


def test_raw_export_averages_raw_weights_instead_of_ema(trained):
    expected = WeightAverage(load_artifact(trained.source / "epoch_02.pt", trained.context.binding)["raw_state"])
    for epoch in (2, 3):
        expected.update(load_artifact(trained.source / f"epoch_{epoch:02d}.pt", trained.context.binding)["raw_state"])
    exported = export_swa(trained.context, trained.output, weight_source="raw")
    payload = load_artifact(exported, trained.context.binding)
    assert payload["selected_policy"] == "swa_raw"
    for name, value in payload["selected_state"].items():
        torch.testing.assert_close(value, expected.state[name], rtol=0, atol=0)


@pytest.mark.parametrize("change,message", [
    (lambda p: p.update(epoch=2), "Epoch, trajectory"),
    (lambda p: p.update(trajectory="other_seed"), "Epoch, trajectory"),
    (lambda p: p.update(classes=["0001", "0000", "0002"]), "Epoch, trajectory"),
    (lambda p: p.update(model_config={"different": "architecture"}), "Epoch, trajectory"),
    (lambda p: p.update(targets_sha256="another_target_set"), "different training targets"),
    (lambda p: p["effective_training_config"].update(seed=9), "Effective training recipe"),
    (lambda p: p.update(optimizer=None), "complete training epoch"),
    (lambda p: p.update(ema=None), "matching updated EMA"),
    (lambda p: p["ema"].update(decay=.99), "matching updated EMA"),
    (lambda p: p["ema"]["state"].pop("head.weight"), "parameter keys"),
    (lambda p: p["ema"]["state"].update({"head.weight": torch.zeros(2, 16)}), "shapes or dtypes"),
    (lambda p: p["ema"]["state"]["head.weight"].fill_(float("nan")), "finite floating-point"),
])
def test_rejects_mixed_or_incomplete_trajectory_without_partial_export(trained, change, message):
    rewrite_checkpoint(trained, change)
    with pytest.raises(ValueError, match=message):
        export_swa(trained.context, trained.output)
    assert not trained.output.exists()


def test_missing_epoch_does_not_silently_average_available_subset(trained):
    (trained.source / "epoch_03.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Complete SWA epoch window"):
        export_swa(trained.context, trained.output)
    assert not trained.output.exists()


def test_rejects_checksum_drift_and_foreign_stage(trained):
    path = trained.source / "epoch_03.pt"
    meta_path = path.with_suffix(".sha256.json")
    original = meta_path.read_text()
    meta = json.loads(original)
    meta["sha256"] = "changed_bytes"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="checksum or current-stage"):
        export_swa(trained.context, trained.output)
    meta = json.loads(original)
    meta["binding"]["stage"] = "foreign_stage"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="checksum or current-stage"):
        export_swa(trained.context, trained.output)
    assert not trained.output.exists()


def test_refuses_existing_output_and_original_run_directory(trained):
    trained.output.mkdir()
    sentinel = trained.output / "keep.txt"
    sentinel.write_text("existing artifact")
    with pytest.raises(FileExistsError, match="not empty"):
        export_swa(trained.context, trained.output)
    assert sentinel.read_text() == "existing artifact"
    with pytest.raises(ValueError, match="outside the original run"):
        export_swa(trained.context, trained.source.parent / "swa")


@pytest.mark.parametrize("start,end", [(0, 3), (2, 2), (3, 2), (2, 4)])
def test_rejects_windows_outside_fixed_training_schedule(trained, start, end):
    with pytest.raises(ValueError, match="at least two epochs"):
        export_swa(trained.context, trained.output, start_epoch=start, end_epoch=end)


def test_cli_dispatches_cpu_export_and_guard_rejects_cuda_before_loading_assets(trained, monkeypatch, capsys):
    from aegis_clip import v1_pipeline as pipeline
    from aegis_clip.cli import v1
    monkeypatch.setattr(pipeline, "StageContext", lambda config: trained.context)
    command = ["v1", "export-swa", "--config", str(ROOT / "configs/v1_train_20260929.yaml"),
               "--output-dir", str(trained.output), "--execute"]
    monkeypatch.setattr(sys, "argv", command)
    v1.main()
    assert str(trained.output / "selected.pt") in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", command + ["--device", "cuda"])
    monkeypatch.setattr(pipeline, "StageContext", lambda config: pytest.fail("must reject before reading assets"))
    with pytest.raises(SystemExit) as error:
        v1.main()
    assert error.value.code == 2


def test_cli_requires_explicit_execution_and_independent_output(monkeypatch):
    from aegis_clip.cli import v1
    monkeypatch.setattr(sys, "argv", ["v1", "export-swa", "--config", "/nonexistent"])
    with pytest.raises(SystemExit) as error:
        v1.main()
    assert error.value.code == 2
    monkeypatch.setattr(sys, "argv", ["v1", "export-swa", "--config", "/nonexistent", "--execute"])
    with pytest.raises(SystemExit) as error:
        v1.main()
    assert error.value.code == 2


def test_cli_reports_incomplete_window_as_usage_error_without_creating_output(trained, monkeypatch, capsys):
    from aegis_clip import v1_pipeline as pipeline
    from aegis_clip.cli import v1
    (trained.source / "epoch_03.pt").unlink()
    monkeypatch.setattr(pipeline, "StageContext", lambda config: trained.context)
    monkeypatch.setattr(sys, "argv", ["v1", "export-swa", "--config", str(ROOT / "configs/v1_train_20260929.yaml"),
                                     "--output-dir", str(trained.output), "--execute"])
    with pytest.raises(SystemExit) as error:
        v1.main()
    assert error.value.code == 2 and "Complete SWA epoch window" in capsys.readouterr().err
    assert not trained.output.exists()
