import pytest
import pandas as pd
import torch

from aegis_clip.oof_rebuild import OOFInputs, rebuild_oof_logits
from aegis_clip.trajectory import (
    CrossFittedTrajectory,
    audit_final_against_reference,
    first_label_wave_reversal,
    rebuild_oof_trajectory,
    trajectory_reference_gate,
)


def test_first_label_wave_reversal_selects_epoch_before_rebound() -> None:
    assert first_label_wave_reversal(torch.tensor([0, 10, 7, 8, 6])) == 3
    assert first_label_wave_reversal(torch.tensor([0, 10, 7, 4])) == 4
    with pytest.raises(ValueError, match="Epoch-one"):
        first_label_wave_reversal(torch.tensor([1, 0]))


def test_cross_fitted_trajectory_accumulates_wrong_and_change_events() -> None:
    trajectory = CrossFittedTrajectory(
        num_samples=3,
        epochs=2,
        num_classes=3,
        top_k=2,
    )
    rows = torch.arange(3)
    labels = torch.tensor([0, 0, 1])
    trajectory.update(
        epoch_index=0,
        row_indices=rows,
        labels=labels,
        logits=torch.tensor(
            [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 4.0, 0.0]]
        ),
    )
    trajectory.update(
        epoch_index=1,
        row_indices=rows,
        labels=labels,
        logits=torch.tensor(
            [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]
        ),
    )
    result = trajectory.finalize()
    assert result["wrong_event_count"].tolist() == [0, 1, 0]
    assert result["prediction_change_count"].tolist() == [0, 1, 0]
    assert result["epoch_prediction_change"].tolist() == [0, 1]
    assert torch.allclose(
        result["epoch_oof_accuracy"], torch.tensor([2 / 3, 1.0])
    )
    assert result["topk_indices"].shape == (3, 2, 2)
    assert result["selected_base_epoch"] == 2


def test_cross_fitted_trajectory_rejects_duplicate_and_incomplete_rows() -> None:
    trajectory = CrossFittedTrajectory(
        num_samples=2,
        epochs=2,
        num_classes=2,
        top_k=1,
    )
    kwargs = {
        "epoch_index": 0,
        "row_indices": torch.tensor([0]),
        "labels": torch.tensor([0]),
        "logits": torch.tensor([[2.0, 0.0]]),
    }
    trajectory.update(**kwargs)
    with pytest.raises(ValueError, match="twice"):
        trajectory.update(**kwargs)
    with pytest.raises(RuntimeError, match="incomplete"):
        trajectory.finalize()


def test_reference_audit_and_gate_pass_for_matching_artifact(tmp_path) -> None:
    sample_ids = ["a", "b", "c"]
    labels = torch.tensor([0, 1, 2])
    folds = torch.tensor([0, 1, 0])
    logits = torch.tensor(
        [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    ).half()
    reference = tmp_path / "oof.pt"
    torch.save(
        {
            "sample_ids": sample_ids,
            "labels": labels,
            "folds": folds,
            "logits": logits,
        },
        reference,
    )
    audit = audit_final_against_reference(
        sample_ids=sample_ids,
        labels=labels,
        folds=folds,
        final_logits=logits,
        reference_path=reference,
    )
    assert audit["top1_agreement"] == 1.0
    assert audit["p_original_max_absolute_error"] == 0.0
    assert trajectory_reference_gate(audit)["passed"] is True


def test_reference_audit_fails_closed_on_lineage_mismatch(tmp_path) -> None:
    reference = tmp_path / "oof.pt"
    torch.save(
        {
            "sample_ids": ["a"],
            "labels": torch.tensor([0]),
            "folds": torch.tensor([0]),
            "logits": torch.tensor([[1.0, 0.0]]),
        },
        reference,
    )
    with pytest.raises(ValueError, match="sample IDs"):
        audit_final_against_reference(
            sample_ids=["b"],
            labels=torch.tensor([0]),
            folds=torch.tensor([0]),
            final_logits=torch.tensor([[1.0, 0.0]]),
            reference_path=reference,
        )


def test_rebuild_oof_trajectory_matches_rebuilt_reference_end_to_end(tmp_path) -> None:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(12, 4, generator=generator)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    folds = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
    assignments = pd.DataFrame(
        {
            "sample_id": [f"s{index:02d}" for index in range(len(labels))],
            "image_path": [f"train/{index:02d}.jpg" for index in range(len(labels))],
            "label": labels.tolist(),
            "fold": folds.tolist(),
        }
    )
    inputs = OOFInputs(assignments=assignments, features=features, labels=labels)
    common = {
        "num_classes": 3,
        "epochs": 3,
        "batch_size": 4,
        "infer_batch_size": 8,
        "lr": 0.01,
        "weight_decay": 0.0,
        "warmup_epochs": 1,
        "q": 0.5,
        "seed": 11,
        "device": torch.device("cpu"),
        "input_hashes": {"synthetic": "test-only"},
    }
    reference_dir = tmp_path / "reference"
    rebuild_oof_logits(inputs, reference_dir, **common)
    output_dir = tmp_path / "trajectory"
    result = rebuild_oof_trajectory(
        inputs,
        output_dir,
        reference_oof_path=reference_dir / "oof_logits.pt",
        top_k=2,
        **common,
    )
    assert result["audit"]["all_samples_seen_once_per_epoch"] is True
    assert result["audit"]["reference_gate"]["passed"] is True
    payload = torch.load(
        output_dir / "trajectory.pt", map_location="cpu", weights_only=False
    )
    assert payload["original_label_probability"].shape == (12, 3)
    assert payload["topk_indices"].shape == (12, 3, 2)
    selected_epoch = int(payload["selected_base_epoch"])
    for fold in (0, 1):
        snapshots = torch.load(
            output_dir / f"fold_{fold}" / "epoch_heads.pt",
            map_location="cpu",
            weights_only=True,
        )
        selected = torch.load(
            output_dir / f"fold_{fold}" / "selected_base_head.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert selected["selected_base_epoch"] == selected_epoch
        assert torch.equal(
            selected["state_dict"]["weight"],
            snapshots["weight"][selected_epoch - 1],
        )
        assert torch.equal(
            selected["state_dict"]["bias"],
            snapshots["bias"][selected_epoch - 1],
        )


def test_rebuild_oof_trajectory_rejects_nonfinite_features(tmp_path) -> None:
    assignments = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "image_path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "label": [0, 1, 0, 1],
            "fold": [0, 0, 1, 1],
        }
    )
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [float("nan"), 0.0], [0.0, 1.0]]
    )
    inputs = OOFInputs(
        assignments=assignments,
        features=features,
        labels=torch.tensor([0, 1, 0, 1]),
    )
    with pytest.raises(ValueError, match="finite rank-two"):
        rebuild_oof_trajectory(
            inputs,
            tmp_path / "trajectory",
            reference_oof_path=tmp_path / "unused.pt",
            num_classes=2,
            epochs=2,
            batch_size=2,
            infer_batch_size=2,
            lr=0.01,
            weight_decay=0.0,
            warmup_epochs=0,
            q=0.5,
            seed=1,
            top_k=1,
            device=torch.device("cpu"),
            input_hashes={},
        )
