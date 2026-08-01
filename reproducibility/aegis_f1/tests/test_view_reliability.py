from __future__ import annotations

import pytest
import torch

from aegis_clip.view_reliability import (
    BASE_VIEW_WEIGHTS,
    CVRGProtocol,
    FrozenReliabilityGate,
    VIEW_ORDER,
    compute_dynamic_view_weights,
    extract_reliability_features,
    fuse_dynamic_view_probabilities,
    predict_view_reliability,
    validate_cvrg_cache,
)
from aegis_clip.localization import fuse_global_local_flip_probabilities


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

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        validate_cvrg_cache(
            payload,
            require_labels=True,
            expected_checkpoint_sha256="different-hash",
        )

def make_view_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.zeros(1, 4, 500)
    logits[0, 0, 0] = 10.0
    logits[0, 1, 0] = 10.0
    logits[0, 2, 1] = 10.0
    logits[0, 3, 1] = 10.0
    visual = torch.eye(4).reshape(1, 4, 4)
    attention = torch.ones(1, 2, 1, 49)
    attention[0, 0, 0, :] = 0.0
    attention[0, 0, 0, 0] = 1.0
    attention[0, 1, 0, :] = 0.0
    attention[0, 1, 0, 48] = 1.0
    boxes = torch.tensor([[[0, 0, 160, 160], [64, 0, 224, 160]]])
    return logits, visual, attention, boxes


def test_feature_schema_is_39_columns_and_has_no_forbidden_fields() -> None:
    features, names = extract_reliability_features(*make_view_inputs())
    assert features.shape == (1, 4, 39)
    assert len(names) == 39
    forbidden = ("label", "class_id", "path", "neighbor", "pseudo", "raw_logit")
    assert not any(any(token in name for token in forbidden) for name in names)


def test_uniform_logits_have_unit_entropy_and_zero_margin() -> None:
    logits, visual, attention, boxes = make_view_inputs()
    logits.zero_()
    features, names = extract_reliability_features(logits, visual, attention, boxes)
    index = {name: position for position, name in enumerate(names)}
    assert torch.allclose(features[:, :, index["single.normalized_entropy"]], torch.ones(1, 4))
    assert torch.allclose(features[:, :, index["single.top1_top2_margin"]], torch.zeros(1, 4))
    assert torch.allclose(
        features[:, :, index["single.top5_probability_mass"]],
        torch.full((1, 4), 0.01),
        atol=1.0e-5,
    )


def test_pairwise_js_and_top5_jaccard_match_hand_computed_values() -> None:
    logits, visual, attention, boxes = make_view_inputs()
    p = torch.zeros(500)
    q = torch.zeros(500)
    p[0] = 0.50
    p[1] = 0.20
    p[2] = 0.15
    p[3] = 0.10
    p[4] = 0.05
    q[0] = 0.50
    q[2] = 0.20
    q[5] = 0.15
    q[6] = 0.10
    q[7] = 0.05
    logits[0, 0] = p.clamp_min(1.0e-8).log()
    logits[0, 1] = q.clamp_min(1.0e-8).log()
    features, names = extract_reliability_features(logits, visual, attention, boxes)
    index = {name: position for position, name in enumerate(names)}
    midpoint = (p + q) / 2.0
    expected_js = 0.5 * (
        (p[p > 0] * (p[p > 0] / midpoint[p > 0]).log()).sum()
        + (q[q > 0] * (q[q > 0] / midpoint[q > 0]).log()).sum()
    )
    assert torch.allclose(features[0, 0, index["pair.01.js_divergence"]], expected_js)
    assert features[0, 0, index["pair.01.top5_jaccard"]] == pytest.approx(1.0 / 4.0)


def test_flip_mapped_center_distance_uses_horizontal_equivariance() -> None:
    _, visual, attention, boxes = make_view_inputs()
    features, names = extract_reliability_features(
        torch.zeros(1, 4, 500), visual, attention, boxes
    )
    index = {name: position for position, name in enumerate(names)}
    assert features[0, 0, index["attention.flip_mapped_center_distance"]] == pytest.approx(
        0.0, abs=1.0e-6
    )


def make_zero_gate(feature_count: int = 39) -> FrozenReliabilityGate:
    return FrozenReliabilityGate(
        feature_names=tuple(f"feature.{index}" for index in range(feature_count)),
        feature_mean=torch.zeros(feature_count),
        feature_scale=torch.ones(feature_count),
        coefficient=torch.zeros(feature_count),
        intercept=0.0,
        regularization_c=1.0,
        checkpoint_sha256="checkpoint-hash",
        validation_cache_sha256="cache-hash",
        feature_schema_sha256="schema-hash",
        protocol=CVRGProtocol(),
    )


def test_dynamic_weights_are_a_simplex_and_follow_reliability() -> None:
    weights = compute_dynamic_view_weights(
        torch.tensor([[0.9, 0.5, 0.5, 0.5]])
    )
    assert torch.allclose(weights.sum(dim=1), torch.ones(1))
    assert torch.all(weights >= 0.0)
    assert weights[0, 0] > BASE_VIEW_WEIGHTS[0]


def test_zero_residual_gate_is_exact_baseline_fusion() -> None:
    logits = torch.randn(2, 4, 500)
    features = torch.zeros(2, 4, 39)
    fused, weights, reliability = fuse_dynamic_view_probabilities(
        logits, make_zero_gate(), features
    )
    baseline = fuse_global_local_flip_probabilities(
        logits[:, 0], logits[:, 1], logits[:, 2], logits[:, 3],
        local_weight=0.4, flip_weight=0.5, temperature=1.0,
    )
    assert torch.equal(fused, baseline)
    assert torch.allclose(weights, BASE_VIEW_WEIGHTS.expand(2, -1))
    assert torch.allclose(reliability, torch.full((2, 4), 0.5))


def test_gate_prediction_is_clipped() -> None:
    probabilities = predict_view_reliability(torch.zeros(2, 4, 39), make_zero_gate())
    assert probabilities.shape == (2, 4)
    assert torch.all((probabilities >= 1.0e-4) & (probabilities <= 1.0 - 1.0e-4))
