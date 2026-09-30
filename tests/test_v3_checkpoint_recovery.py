"""Checkpoint recovery preserves the completed arm and selects only its last EMA."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducibility/aegis_f1"))
sys.path.insert(0, str(ROOT / "scripts"))
import recover_v3_checkpoint as recovery
from v2.plan import dump, sha
from v2.runtime import save_checkpoint


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(2))

    def forward(self, images):
        return self.bias.expand(len(images), -1)


@pytest.fixture
def endpoint(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan = dict(recipe=dict(epochs=2), updates_per_arm=4, steps_per_epoch=2, classes=["0", "1"])
    dump(plan_path, plan)
    original = tmp_path / "runs/train/original"
    original.mkdir(parents=True)
    metrics = dict(macro=1., micro=1., rows=2, classes_present=1)
    history = [dict(epoch=e, updates=e * 2, ema=metrics) for e in (1, 2)]
    payload = dict(model={"bias": torch.tensor([10., 0.])}, ema={"bias": torch.tensor([0., 10.])},
                   history=history, epoch=2, updates=4, initial_model_sha256="frozen_initial",
                   binding=dict(plan_sha256=sha(plan_path), arm="original", probe=False, complete=True))
    save_checkpoint(original / "epoch02.pt", payload)
    dump(original / "history.json", history)
    import json
    (original / "steps.jsonl").write_text("".join(json.dumps(dict(epoch=i // 2 + 1, batch=i % 2,
        optimizer_updates=i + 1, ema_updates=i + 1)) + "\n" for i in range(4)))
    return plan_path, plan, original, sha(original / "epoch02.pt")


def export(endpoint):
    plan_path, plan, original, digest = endpoint
    _, payload = recovery.checked_endpoint(plan_path, plan, digest)
    data = TensorDataset(torch.zeros(2, 1), torch.ones(2, dtype=torch.long), torch.arange(2))
    rows = [dict(label=1, image_path=f"val/{i}.png") for i in range(2)]
    return recovery.export_original(Tiny(), plan_path, plan, original, payload, DataLoader(data, batch_size=2),
                                    rows, torch.device("cpu"))


def test_completed_original_export_preserves_sources_and_uses_ema(endpoint):
    _, _, original, _ = endpoint
    before = {name: sha(original / name) for name in ("epoch02.pt", "epoch02.binding.json", "history.json", "steps.jsonl")}
    result = export(endpoint)
    assert result["status"] == "complete" and result["updates"] == 4
    assert before == {name: sha(original / name) for name in before}
    selected = torch.load(original / "selected.pt", weights_only=False)
    assert torch.equal(selected["model"]["bias"], torch.tensor([0., 10.]))
    assert selected["binding"]["weights"] == "last_ema"
    with np.load(original / "predictions.npz") as predictions:
        assert predictions["predictions"].tolist() == [1, 1]
        assert predictions["image_paths"].tolist() == ["val/0.png", "val/1.png"]


def test_recovery_rejects_changed_checkpoint(endpoint):
    plan_path, plan, _, _ = endpoint
    with pytest.raises(ValueError, match="checkpoint changed"):
        recovery.checked_endpoint(plan_path, plan, "wrong")


@pytest.mark.parametrize("change", ["truncate", "reorder"])
def test_recovery_requires_complete_ordered_trace(endpoint, change):
    plan_path, plan, original, digest = endpoint
    lines = (original / "steps.jsonl").read_text().splitlines()
    if change == "truncate":
        lines.pop()
    else:
        lines.reverse()
    (original / "steps.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="trace"):
        recovery.checked_endpoint(plan_path, plan, digest)


def test_recovery_refuses_export_overwrite(endpoint):
    export(endpoint)
    with pytest.raises(ValueError, match="overwrite"):
        export(endpoint)


def test_changed_validation_cannot_produce_selected_checkpoint(endpoint):
    plan_path, plan, original, digest = endpoint
    _, payload = recovery.checked_endpoint(plan_path, plan, digest)
    payload["history"][-1]["ema"]["micro"] = 0.5
    rows = [dict(label=1, image_path=f"val/{i}.png") for i in range(2)]
    data = TensorDataset(torch.zeros(2, 1), torch.ones(2, dtype=torch.long), torch.arange(2))
    with pytest.raises(ValueError, match="validation differs"):
        recovery.export_original(Tiny(), plan_path, plan, original, payload, DataLoader(data, batch_size=2),
                                 rows, torch.device("cpu"))
    assert not (original / "selected.pt").exists()
