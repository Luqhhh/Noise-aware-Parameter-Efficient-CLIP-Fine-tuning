from copy import deepcopy

import pytest
import torch

from aegis_clip.trust_subspace_gate import evaluate_trust_subspace_gate


def _cache(prediction: torch.Tensor, view: str, checkpoint: str) -> dict:
    labels = torch.arange(30) % 3
    logits = torch.full((30, 3), -4.0)
    logits[torch.arange(30), prediction] = 4.0
    return {
        "paths": [f"train/{index}.jpg" for index in range(30)],
        "labels": labels,
        "clean_probability": torch.ones(30),
        "pseudo_labels": labels.clone(),
        "correction_alpha": torch.zeros(30),
        "logits": logits,
        "view_mode": view,
        "checkpoint_sha256": checkpoint,
    }


def _training_row(mode: str) -> dict[str, str]:
    return {
        "epoch": "2",
        "train_trust_subspace_steps": "1452",
        "train_trust_subspace_skipped_steps": "0",
        "train_trust_subspace_basis_rank": "8",
        "train_trust_subspace_projection_steps": (
            "0" if mode == "control" else "1452"
        ),
        "train_trust_subspace_trusted_examples": "65473",
        "train_trust_subspace_uncertain_examples": "27429",
        "train_trust_subspace_retained_norm_ratio": (
            "0" if mode == "control" else "0.15"
        ),
    }


def _arguments() -> dict:
    labels = torch.arange(30) % 3
    baseline = labels.clone()
    baseline[0] = 1
    return {
        "original_m1": _cache(baseline, "attention_local_global", "f1"),
        "t0_center": _cache(baseline, "center", "t0"),
        "t0_m1": _cache(baseline, "attention_local_global", "t0"),
        "t1_center": _cache(baseline, "center", "t1"),
        "t1_m1": _cache(labels, "attention_local_global", "t1"),
        "t0_initial": {"clean_core_micro": 0.8},
        "t1_initial": {"clean_core_micro": 0.8},
        "t0_evaluation": {"mean_feature_drift": 0.005},
        "t1_evaluation": {"mean_feature_drift": 0.006},
        "t0_last_metrics": _training_row("control"),
        "t1_last_metrics": _training_row("treatment"),
    }


def test_gate_passes_only_when_mechanics_and_performance_both_pass() -> None:
    report = evaluate_trust_subspace_gate(**_arguments())
    assert report["passed"]
    assert report["status"] == "passed"
    assert report["delta_pp"]["t1_m1_minus_t0_m1_pp"]["clean_core_micro"] > 3.0


def test_gate_fails_when_projection_never_engages() -> None:
    arguments = _arguments()
    arguments["t1_last_metrics"]["train_trust_subspace_projection_steps"] = "0"
    report = evaluate_trust_subspace_gate(**arguments)
    assert not report["passed"]
    assert not report["checks"]["treatment_projection_coverage_95pct"]


def test_gate_fails_closed_on_validation_path_mismatch() -> None:
    arguments = _arguments()
    arguments["t1_m1"] = deepcopy(arguments["t1_m1"])
    arguments["t1_m1"]["paths"][0] = "test/leak.jpg"
    with pytest.raises(ValueError, match="Validation field paths differs"):
        evaluate_trust_subspace_gate(**arguments)
