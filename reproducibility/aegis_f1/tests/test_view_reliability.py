from __future__ import annotations

import pytest
import torch

from aegis_clip.view_reliability import (
    BASE_VIEW_WEIGHTS,
    CVRGProtocol,
    VIEW_ORDER,
    validate_cvrg_cache,
)


def make_valid_cache(
    *, samples: int = 3, classes: int = 500, split: str = "validation"
) -> dict[str, object]:
    payload: dict[str, object] = {
        "format_version": 1,
        "split": split,
        "view_order": list(VIEW_ORDER),
        "checkpoint_sha256": "checkpoint-hash",
        "split_sha256": "split-hash",
        "protocol": CVRGProtocol().__dict__,
        "view_logits": torch.zeros(samples, 4, classes),
        "view_features": torch.nn.functional.normalize(
            torch.ones(samples, 4, 8), dim=-1
        ),
        "orientation_attention": torch.ones(samples, 2, 12, 49),
        "crop_boxes": torch.zeros(samples, 2, 4, dtype=torch.int64),
        "paths": [f"image-{index}.jpg" for index in range(samples)],
    }
    if split == "validation":
        payload.update(
            {
                "labels": torch.arange(samples) % classes,
                "clean_probability": torch.ones(samples),
                "pseudo_label": torch.arange(samples) % classes,
                "correction_alpha": torch.zeros(samples),
            }
        )
    return payload


def test_protocol_constants_match_preregistered_baseline() -> None:
    assert VIEW_ORDER == (
        "original_global",
        "original_local",
        "flipped_global",
        "flipped_local",
    )
    assert torch.equal(
        BASE_VIEW_WEIGHTS, torch.tensor([0.30, 0.20, 0.30, 0.20])
    )
    protocol = CVRGProtocol()
    assert protocol.crop_size == 160
    assert protocol.top_k == 5
    assert protocol.temperature == 1.0
    assert protocol.local_weight == 0.4
    assert protocol.flip_weight == 0.5
    assert protocol.prior_alignment_strength == 1.0


def test_validation_cache_returns_aligned_sample_count() -> None:
    assert validate_cvrg_cache(make_valid_cache(samples=3), require_labels=True) == 3


def test_validation_cache_requires_labels() -> None:
    payload = make_valid_cache(samples=3)
    del payload["labels"]
    with pytest.raises(ValueError, match="labels"):
        validate_cvrg_cache(payload, require_labels=True)


def test_test_cache_rejects_label_or_trust_fields() -> None:
    payload = make_valid_cache(split="test")
    payload["labels"] = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError, match="label-free"):
        validate_cvrg_cache(payload, require_labels=False)


def test_cache_rejects_nonfinite_logits_and_wrong_class_count() -> None:
    payload = make_valid_cache()
    payload["view_logits"][0, 0, 0] = float("nan")  # type: ignore[index]
    with pytest.raises(ValueError, match="finite"):
        validate_cvrg_cache(payload, require_labels=True)

    payload = make_valid_cache(classes=499)
    with pytest.raises(ValueError, match="500"):
        validate_cvrg_cache(payload, require_labels=True)


def test_cache_rejects_duplicate_paths_and_checkpoint_mismatch() -> None:
    payload = make_valid_cache()
    payload["paths"] = ["same.jpg", "same.jpg", "other.jpg"]
    with pytest.raises(ValueError, match="unique"):
        validate_cvrg_cache(payload, require_labels=True)

    payload = make_valid_cache()
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        validate_cvrg_cache(
            payload,
            require_labels=True,
            expected_checkpoint_sha256="different-hash",
        )
