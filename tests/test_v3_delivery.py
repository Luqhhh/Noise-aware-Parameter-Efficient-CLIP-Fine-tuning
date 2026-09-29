"""CPU delivery check for the fixed v3 supervision arm."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducibility/aegis_f1"))
from v2.plan import dump, sha
from v3 import delivery


class TinyClassifier(torch.nn.Module):
    def __init__(self, recipe):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(2))

    def forward(self, images):
        return torch.stack([images[:, 0].mean((1, 2)), images[:, 2].mean((1, 2))], dim=1) + self.bias


def fixture(tmp_path, monkeypatch):
    workspace = tmp_path / "prepared"
    workspace.mkdir()
    plan_path = workspace / "plan.json"
    dump(plan_path, {"fixture": True})
    plan_hash = sha(plan_path)
    plan = dict(recipe=dict(epochs=2, micro_batch_size=2, source_v1_config="unused"),
                updates_per_arm=4, model_recipe={}, classes=["0000", "0001"],
                v1_binding={"scope": "test"})
    monkeypatch.setattr(delivery, "verify", lambda _: plan)
    run = workspace / "runs/train"
    arm = run / "v1_supervision"
    arm.mkdir(parents=True)
    selected = dict(model=TinyClassifier({}).state_dict(),
                    binding=dict(plan_sha256=plan_hash, arm="v1_supervision", weights="last_ema",
                                 epoch=2, updates=4, complete=True))
    torch.save(selected, arm / "selected.pt")
    cp_hash = sha(arm / "selected.pt")
    dump(run / "status.json", dict(status="complete", plan_sha256=plan_hash))
    dump(arm / "status.json", dict(status="complete", plan_sha256=plan_hash,
                                   weights="last_ema", updates=4, checkpoint_sha256=cp_hash))
    (workspace / "comparison").mkdir()
    dump(workspace / "comparison/report.json",
         dict(status="local_result", plan_sha256=plan_hash, paired_trace_verified=True,
              checkpoint_sha256={"v1_supervision": cp_hash}))
    dump(workspace / "shared_config.json",
         dict(data=dict(image_size=32, eval_resize_ratio=1., num_workers=0, prefetch_factor=2),
              train=dict(seed=13)))
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "dataset_manifest.json").write_text("{}")
    (assets / "class_to_idx.json").write_text(json.dumps({"0000": 0, "0001": 1}))
    with (assets / "test_manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "label"])
        writer.writerows([["test/a.png", -1], ["test/b.png", -1]])
    test_root = tmp_path / "test"
    test_root.mkdir()
    Image.new("RGB", (32, 32), (255, 0, 0)).save(test_root / "a.png")
    Image.new("RGB", (32, 32), (0, 0, 255)).save(test_root / "b.png")
    context = type("Context", (), dict(binding=plan["v1_binding"], classes=plan["classes"],
                                        manifest={"test_samples": 2}, test_root=test_root,
                                        reference={"data": {"dataset_manifest": str(assets / "dataset_manifest.json"),
                                                            "class_mapping": str(assets / "class_to_idx.json")}}))()
    monkeypatch.setattr(delivery, "load_recipe", lambda _: {})
    monkeypatch.setattr(delivery, "StageContext", lambda _: context)
    monkeypatch.setattr(delivery, "V2Classifier", TinyClassifier)
    return plan_path, arm


def test_v3_supervision_selected_exports_one_checked_package(tmp_path, monkeypatch):
    plan_path, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path / "submission"
    report = delivery.deliver(plan_path, output, 60, "cpu")
    assert report["status"] == "package_ready" and report["arm"] == "v1_supervision"
    assert report["rows"] == 2 and report["weights"] == "last_ema"
    assert (output / "pred_results.csv").read_text().splitlines() == ["a.png, 0000", "b.png, 0001"]
    assert "All checks passed!" in (output / "submission_check.log").read_text()


def test_v3_delivery_rejects_changed_checkpoint(tmp_path, monkeypatch):
    plan_path, arm = fixture(tmp_path, monkeypatch)
    (arm / "selected.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        delivery.checked_source(plan_path)
