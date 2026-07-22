import torch

from aegis_clip.cache_diagnostic import (
    complementarity_metrics,
    prediction_metrics,
    topk_cache_predictions,
)


def test_topk_cache_predictions_uses_similarity_weighted_vote() -> None:
    bank = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1])
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prediction, margin = topk_cache_predictions(
        query, bank, labels, num_classes=2, k=2, beta=20.0
    )
    assert prediction.tolist() == [0, 1]
    assert torch.all(margin > 0)


def test_prediction_and_complementarity_metrics() -> None:
    target = torch.tensor([0, 1, 0, 1])
    baseline = torch.tensor([0, 0, 0, 1])
    candidate = torch.tensor([1, 1, 0, 0])
    clean = torch.tensor([0.9, 0.9, 0.1, 0.9])
    metrics = prediction_metrics(
        baseline, target, clean, num_classes=2, clean_threshold=0.7
    )
    complement = complementarity_metrics(
        baseline, candidate, target, clean, clean_threshold=0.7
    )
    assert metrics["clean_core_samples"] == 3
    assert abs(metrics["clean_core_micro"] - 2 / 3) < 1e-6
    assert abs(complement["oracle_clean_core_micro"] - 1.0) < 1e-6
    assert complement["candidate_rescues"] == 1
    assert complement["candidate_damages"] == 2
