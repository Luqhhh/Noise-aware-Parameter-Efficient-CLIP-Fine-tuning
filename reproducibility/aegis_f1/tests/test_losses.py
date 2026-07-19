import torch

from aegis_clip.losses import (
    AdaptiveLossCap,
    EarlyLearningRegularizer,
    classwise_suspicion_mask,
    class_prior_adjusted_logits,
    corrected_targets,
    mixup,
    project_conflicting_gradients,
    soft_generalized_cross_entropy,
)


def test_class_prior_adjustment_is_training_only_and_tail_aware() -> None:
    logits = torch.zeros(2, 3)
    counts = torch.tensor([100.0, 10.0, 1.0])
    unchanged = class_prior_adjusted_logits(logits, counts, tau=0.0)
    adjusted = class_prior_adjusted_logits(logits, counts, tau=1.0)
    assert torch.equal(unchanged, logits)
    assert adjusted[0, 0] > adjusted[0, 1] > adjusted[0, 2]
    assert torch.equal(logits, torch.zeros_like(logits))


def test_invalid_pseudo_label_falls_back_to_noisy_label() -> None:
    result = corrected_targets(
        torch.tensor([0, 1]),
        torch.tensor([-1, 2]),
        torch.tensor([0.5, 0.25]),
        num_classes=3,
    )
    assert torch.allclose(result.sum(dim=1), torch.ones(2))
    assert torch.allclose(result[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(result[1], torch.tensor([0.0, 0.75, 0.25]))


def test_soft_gce_reduces_to_hard_target_formula() -> None:
    logits = torch.tensor([[2.0, 0.0]])
    target = torch.tensor([[1.0, 0.0]])
    probability = logits.softmax(dim=1)[0, 0]
    expected = (1.0 - probability.pow(0.5)) / 0.5
    actual = soft_generalized_cross_entropy(logits, target, q=0.5)
    assert torch.allclose(actual, expected.unsqueeze(0))


def test_mixup_is_reproducible_and_returns_alignment() -> None:
    inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    targets = torch.eye(4)
    weights = torch.arange(1, 5, dtype=torch.float32)
    first = torch.Generator().manual_seed(7)
    second = torch.Generator().manual_seed(7)
    result_a = mixup(inputs, targets, weights, 0.2, 1.0, first)
    result_b = mixup(inputs, targets, weights, 0.2, 1.0, second)
    for left, right in zip(result_a[:3], result_b[:3]):
        assert torch.allclose(left, right)
    assert result_a[3] == result_b[3]
    assert torch.equal(result_a[4], result_b[4])


def test_adaptive_cap_is_smooth_and_finite() -> None:
    losses = torch.tensor([0.1, 0.2, 0.3, 100.0], requires_grad=True)
    cap = AdaptiveLossCap(quantile=0.75, momentum=0.0, maximum=2.0)
    transformed = cap(losses, torch.tensor([True, True, True, False]))
    assert torch.isfinite(transformed).all()
    assert transformed[-1] < losses[-1]
    transformed.sum().backward()
    assert 0.0 < losses.grad[-1] < 1.0


def test_conflicting_gradient_is_projected_to_anchor_halfspace() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([-2.0, 1.0])
    result = project_conflicting_gradients([parameter], [torch.tensor([1.0, 0.0])])
    assert result["projected"] is True
    assert float(parameter.grad @ torch.tensor([1.0, 0.0])) >= -1.0e-7


def test_elr_uses_original_negative_log_direction_and_stable_indices() -> None:
    regularizer = EarlyLearningRegularizer(
        num_examples=2,
        num_classes=2,
        momentum=0.5,
        target_weight=3.0,
        warmup_epochs=1,
        ramp_epochs=2,
    )
    indices = torch.tensor([0, 1])
    first_logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
    first = regularizer.update_and_loss(indices, first_logits)
    aligned = regularizer.update_and_loss(indices, first_logits)
    reversed_logits = first_logits.flip(dims=(1,))
    reversed_value = regularizer.update_and_loss(indices, reversed_logits)

    assert aligned < first < 0.0
    assert aligned < reversed_value
    assert torch.equal(regularizer.updates, torch.tensor([3, 3]))
    assert regularizer.rampup_weight(1) == 0.0
    assert regularizer.rampup_weight(2) == 1.5
    assert regularizer.rampup_weight(3) == 3.0


def test_classwise_suspicion_uses_equal_within_class_quota() -> None:
    labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    scores = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.4, 0.8, 0.1, 0.9, 0.7])
    mask = classwise_suspicion_mask(labels, scores, fraction=0.2)
    assert torch.equal(torch.nonzero(mask).flatten(), torch.tensor([1, 7]))
    assert torch.bincount(labels[mask], minlength=2).tolist() == [1, 1]
