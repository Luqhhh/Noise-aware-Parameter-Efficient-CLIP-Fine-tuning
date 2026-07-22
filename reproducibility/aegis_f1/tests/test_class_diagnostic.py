import pytest
import torch

from aegis_clip.class_diagnostic import (
    diagnose_class_errors,
    spearman_correlation,
    validate_aligned_logit_caches,
)


def _cache(predictions: list[int]) -> dict[str, object]:
    logits = torch.full((6, 3), -5.0)
    logits[torch.arange(6), torch.tensor(predictions)] = 5.0
    return {
        "paths": [f"image_{index}.jpg" for index in range(6)],
        "labels": torch.tensor([0, 0, 1, 1, 2, 2]),
        "clean_probability": torch.ones(6),
        "logits": logits,
    }


def test_spearman_is_tie_aware() -> None:
    assert spearman_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman_correlation([1, 1, 1], [10, 20, 30]) is None


def test_diagnostic_tracks_corrections_and_oracle() -> None:
    center = _cache([0, 1, 1, 0, 2, 1])
    m1 = _cache([0, 0, 1, 0, 1, 1])
    m3 = _cache([0, 1, 1, 1, 2, 1])
    train = {
        "paths": [f"train_{index}.jpg" for index in range(6)],
        "labels": torch.tensor([0, 0, 1, 1, 2, 2]),
        "clean_probability": torch.tensor([1.0, 0.9, 0.8, 0.7, 0.9, 1.0]),
    }

    report, rows, confusions = diagnose_class_errors(
        center, m1, m3, train, num_classes=3
    )

    assert report["models"]["a2_center"]["correct"] == 3
    assert report["comparisons"]["m1_attention_vs_a2_center"]["net_correct"] == 0
    assert report["comparisons"]["m3_complementary_vs_a2_center"]["net_correct"] == 1
    assert report["complementarity_ceiling"]["oracle_micro_accuracy"] == pytest.approx(
        5 / 6
    )
    assert len(rows) == 3
    assert confusions[0]["errors"] >= 1


def test_alignment_rejects_different_path_order() -> None:
    first = _cache([0, 0, 1, 1, 2, 2])
    second = _cache([0, 0, 1, 1, 2, 2])
    second["paths"] = list(reversed(second["paths"]))
    with pytest.raises(ValueError, match="path order"):
        validate_aligned_logit_caches([first, second], num_classes=3)
