import torch

import pytest

from aegis_clip.cli.compare_logit_caches import (
    _require_same_validation,
    _transition_counts,
)


def test_transition_counts_separates_corrections_and_harm() -> None:
    labels = torch.tensor([0, 1, 2, 3, 4])
    source = torch.tensor([0, 0, 2, 3, 0])
    target = torch.tensor([0, 1, 1, 0, 4])

    assert _transition_counts(source, target, labels) == {
        "samples": 5,
        "changed": 4,
        "corrected": 2,
        "harmed": 2,
        "net_correct": 0,
    }


def test_transition_counts_respects_mask() -> None:
    labels = torch.tensor([0, 1, 2, 3])
    source = torch.tensor([0, 0, 2, 3])
    target = torch.tensor([1, 1, 1, 3])
    mask = torch.tensor([False, True, True, True])

    assert _transition_counts(source, target, labels, mask) == {
        "samples": 3,
        "changed": 2,
        "corrected": 1,
        "harmed": 1,
        "net_correct": 0,
    }


def test_validation_alignment_rejects_different_paths() -> None:
    reference = {
        "paths": ["a.jpg"],
        "labels": torch.tensor([0]),
        "clean_probability": torch.tensor([1.0]),
        "pseudo_labels": torch.tensor([0]),
        "correction_alpha": torch.tensor([0.0]),
        "checkpoint_sha256": "same",
    }
    candidate = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in reference.items()
    }
    candidate["paths"] = ["b.jpg"]

    with pytest.raises(ValueError, match="paths"):
        _require_same_validation(reference, candidate)
