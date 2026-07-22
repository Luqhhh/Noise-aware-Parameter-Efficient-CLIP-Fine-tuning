import json

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from aegis_clip.oof_rebuild import (
    audit_against_historical_quality,
    generalized_cross_entropy,
    learning_rate_factor,
    load_oof_inputs,
)


def _write_cache(tmp_path):
    assignments = pd.DataFrame(
        {
            "sample_id": ["d", "a", "c", "b"],
            "image_path": [
                "train_dedup/0001/d.jpg",
                "train_dedup/0000/a.jpg",
                "train_dedup/0001/c.jpg",
                "train_dedup/0000/b.jpg",
            ],
            "label": [1, 0, 1, 0],
            "fold": [1, 0, 0, 1],
        }
    )
    assignments_path = tmp_path / "assignments.csv"
    assignments.to_csv(assignments_path, index=False)
    features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    )
    feature_path = tmp_path / "features.pt"
    torch.save(features, feature_path)
    paths = [
        "train/0000/a.jpg",
        "train/0000/b.jpg",
        "train/0001/c.jpg",
        "train/0001/d.jpg",
    ]
    paths_path = tmp_path / "paths.json"
    labels_path = tmp_path / "labels.json"
    paths_path.write_text(json.dumps(paths))
    labels_path.write_text(json.dumps([0, 0, 1, 1]))
    return assignments_path, feature_path, paths_path, labels_path


def test_generalized_cross_entropy_matches_definition() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    selected = F.softmax(logits, dim=1)[torch.arange(2), labels]
    expected = ((1.0 - selected.sqrt()) / 0.5).mean()
    assert torch.allclose(
        generalized_cross_entropy(logits, labels, q=0.5), expected
    )


def test_learning_rate_factor_matches_warmup_and_cosine_protocol() -> None:
    assert learning_rate_factor(0, total_steps=10, warmup_steps=2) == pytest.approx(
        0.5
    )
    assert learning_rate_factor(1, total_steps=10, warmup_steps=2) == pytest.approx(
        1.0
    )
    assert learning_rate_factor(2, total_steps=10, warmup_steps=2) == pytest.approx(
        1.0
    )
    assert learning_rate_factor(9, total_steps=10, warmup_steps=2) > 0.01


def test_load_oof_inputs_sorts_and_aligns_by_canonical_path(tmp_path) -> None:
    paths = _write_cache(tmp_path)
    inputs = load_oof_inputs(*paths)
    assert inputs.assignments["sample_id"].tolist() == ["a", "b", "c", "d"]
    assert inputs.labels.tolist() == [0, 0, 1, 1]
    assert torch.allclose(inputs.features.norm(dim=1), torch.ones(4))


def test_load_oof_inputs_rejects_feature_label_mismatch(tmp_path) -> None:
    assignments, features, paths, labels = _write_cache(tmp_path)
    labels.write_text(json.dumps([1, 0, 1, 1]))
    with pytest.raises(ValueError, match="label mismatch"):
        load_oof_inputs(assignments, features, paths, labels)


def test_historical_audit_is_exact_for_matching_logits(tmp_path) -> None:
    assignments_path, feature_path, paths_path, labels_path = _write_cache(tmp_path)
    inputs = load_oof_inputs(
        assignments_path, feature_path, paths_path, labels_path
    )
    logits = torch.tensor(
        [[4.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 4.0]]
    )
    probabilities = logits.softmax(dim=1)
    top2 = probabilities.topk(2, dim=1)
    quality = pd.DataFrame(
        {
            "sample_id": inputs.assignments["sample_id"],
            "original_label": inputs.labels.numpy(),
            "oof_top1": top2.indices[:, 0].numpy(),
            "p_original_label": probabilities[
                torch.arange(4), inputs.labels
            ].numpy(),
            "p_top1": top2.values[:, 0].numpy(),
            "top1_margin": (top2.values[:, 0] - top2.values[:, 1]).numpy(),
        }
    )
    quality_path = tmp_path / "quality.csv"
    quality.sample(frac=1.0, random_state=42).to_csv(quality_path, index=False)
    audit = audit_against_historical_quality(
        inputs.assignments, logits, quality_path
    )
    assert audit["top1_agreement"] == 1.0
    assert audit["p_original_max_absolute_error"] < 1.0e-7
