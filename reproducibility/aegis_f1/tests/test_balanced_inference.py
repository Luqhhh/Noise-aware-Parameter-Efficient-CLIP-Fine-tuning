import torch

from aegis_clip.balanced_inference import (
    effective_model_prior,
    prediction_metrics,
    prior_corrected_logits,
    prior_diagnostics,
)


def test_effective_prior_is_normalized_and_reflects_model_bias() -> None:
    logits = torch.tensor([[5.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    prior = effective_model_prior(logits)
    assert torch.allclose(prior.sum(), torch.tensor(1.0))
    assert prior[0] > prior[1]


def test_prior_correction_removes_a_shared_class_bias() -> None:
    unbiased = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    source_prior = torch.tensor([0.8, 0.2])
    biased = unbiased + source_prior.log()
    corrected = prior_corrected_logits(biased, source_prior)
    assert torch.equal(corrected.argmax(dim=1), unbiased.argmax(dim=1))


def test_uniform_source_prior_changes_no_argmax() -> None:
    logits = torch.randn(7, 5)
    corrected = prior_corrected_logits(logits, torch.full((5,), 0.2))
    assert torch.equal(corrected.argmax(dim=1), logits.argmax(dim=1))


def test_prediction_metrics_report_balance_and_clean_core() -> None:
    metrics = prediction_metrics(
        torch.tensor([0, 0, 1, 1]),
        labels=torch.tensor([0, 1, 1, 1]),
        clean_probability=torch.tensor([1.0, 0.1, 1.0, 1.0]),
        pseudo_labels=torch.tensor([0, 1, 1, 1]),
        correction_alpha=torch.zeros(4),
        num_classes=2,
        clean_core_threshold=0.7,
    )
    assert metrics["raw_micro"] == 0.75
    assert metrics["clean_core_micro"] == 1.0
    assert metrics["prediction_count_min"] == 2
    assert metrics["prediction_count_max"] == 2


def test_prior_diagnostics_detect_imbalance() -> None:
    uniform = prior_diagnostics(torch.tensor([0.5, 0.5]))
    skewed = prior_diagnostics(torch.tensor([0.9, 0.1]))
    assert uniform["normalized_entropy"] > skewed["normalized_entropy"]
    assert skewed["max_min_ratio"] == 9.0
