import torch

from aegis_clip.losses import (
    AdaptiveLossCap,
    EarlyLearningRegularizer,
    active_forgetting_noise_suppression_losses,
    classwise_suspicion_mask,
    classwise_high_loss_filter,
    class_prior_adjusted_logits,
    consensus_conflict_mask,
    corrected_targets,
    deterministic_complementary_labels,
    double_softmax_cross_entropy,
    mixup,
    noise_tolerant_supervised_contrastive_loss,
    project_conflicting_gradients,
    soft_cross_entropy,
    soft_generalized_cross_entropy,
    smoothstep_damped_loss,
)


def test_deterministic_complementary_labels_never_repeat_given_label() -> None:
    labels = torch.tensor([0, 1, 2, 3])
    indices = torch.tensor([0, 7, 11, 99])
    first = deterministic_complementary_labels(
        labels, indices, num_classes=5, epoch=2
    )
    repeated = deterministic_complementary_labels(
        labels, indices, num_classes=5, epoch=2
    )
    assert torch.equal(first, repeated)
    assert torch.all(first != labels)


def test_active_forgetting_reverses_noisy_label_gradient() -> None:
    logits = torch.tensor([[3.0, 0.0, -1.0]], requires_grad=True)
    active, negative, count = active_forgetting_noise_suppression_losses(
        logits,
        torch.tensor([0]),
        torch.tensor([True]),
        torch.tensor([4]),
        epoch=1,
    )
    active.backward(retain_graph=True)
    assert count == 1
    assert logits.grad is not None
    assert logits.grad[0, 0] > 0.0
    assert torch.isfinite(active) and torch.isfinite(negative)


def test_active_forgetting_empty_cohort_is_differentiable_zero() -> None:
    logits = torch.randn(3, 5, requires_grad=True)
    active, negative, count = active_forgetting_noise_suppression_losses(
        logits,
        torch.tensor([0, 1, 2]),
        torch.zeros(3, dtype=torch.bool),
        torch.arange(3),
        epoch=1,
    )
    (active + negative).backward()
    assert count == 0
    assert active.item() == 0.0 and negative.item() == 0.0
    assert logits.grad is not None and torch.equal(logits.grad, torch.zeros_like(logits))


def test_noise_tolerant_contrastive_uses_class_labels_only_when_trusted() -> None:
    first = torch.tensor([[1.0, 0.0], [0.8, 0.2]])
    second = first.clone()
    labels = torch.tensor([0, 0])
    trusted_loss = noise_tolerant_supervised_contrastive_loss(
        first, second, labels, torch.tensor([True, True]), temperature=0.1
    )
    untrusted_loss = noise_tolerant_supervised_contrastive_loss(
        first, second, labels, torch.tensor([False, False]), temperature=0.1
    )
    relabeled_untrusted_loss = noise_tolerant_supervised_contrastive_loss(
        first,
        second,
        torch.tensor([0, 1]),
        torch.tensor([False, False]),
        temperature=0.1,
    )
    assert not torch.allclose(trusted_loss, untrusted_loss)
    assert torch.allclose(untrusted_loss, relabeled_untrusted_loss)


def test_noise_tolerant_contrastive_has_finite_gradients() -> None:
    first = torch.randn(4, 8, requires_grad=True)
    second = torch.randn(4, 8, requires_grad=True)
    loss = noise_tolerant_supervised_contrastive_loss(
        first,
        second,
        torch.tensor([0, 0, 1, 2]),
        torch.tensor([True, True, False, False]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert first.grad is not None and torch.isfinite(first.grad).all()


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


def test_consensus_conflict_requires_disagreement_confidence_and_evidence() -> None:
    mask = consensus_conflict_mask(
        noisy_labels=torch.tensor([0, 0, 1, 1, 2]),
        pseudo_labels=torch.tensor([1, 1, 0, 2, 2]),
        pseudo_confidence=torch.tensor([0.90, 0.80, 0.95, 0.99, 1.00]),
        correction_evidence=torch.tensor([0.1, 0.1, 0.0, 0.2, 0.3]),
        minimum_confidence=0.85,
    )
    assert torch.equal(mask, torch.tensor([True, False, False, True, False]))


def test_consensus_conflict_rejects_invalid_threshold() -> None:
    try:
        consensus_conflict_mask(
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([0.9]),
            torch.tensor([0.1]),
            minimum_confidence=1.1,
        )
    except ValueError as exc:
        assert "minimum_confidence" in str(exc)
    else:
        raise AssertionError("Invalid confidence threshold must fail closed")


def test_soft_gce_reduces_to_hard_target_formula() -> None:
    logits = torch.tensor([[2.0, 0.0]])
    target = torch.tensor([[1.0, 0.0]])
    probability = logits.softmax(dim=1)[0, 0]
    expected = (1.0 - probability.pow(0.5)) / 0.5
    actual = soft_generalized_cross_entropy(logits, target, q=0.5)
    assert torch.allclose(actual, expected.unsqueeze(0))


def test_double_softmax_matches_two_stage_definition() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    expected = -(targets * logits.softmax(dim=1).log_softmax(dim=1)).sum(dim=1)
    actual = double_softmax_cross_entropy(logits, targets)
    assert torch.allclose(actual, expected)
    actual.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_double_softmax_suppresses_confident_mismatch_gradient() -> None:
    logits_ce = torch.tensor([[-12.0, 12.0, -12.0]], requires_grad=True)
    logits_ds = logits_ce.detach().clone().requires_grad_(True)
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    soft_cross_entropy(logits_ce, targets).sum().backward()
    double_softmax_cross_entropy(logits_ds, targets).sum().backward()
    assert logits_ce.grad is not None and logits_ds.grad is not None
    assert logits_ds.grad.norm() < logits_ce.grad.norm() * 1.0e-6


def test_double_softmax_computes_fp32_from_half_logits() -> None:
    logits = torch.tensor([[2.0, 0.0]], dtype=torch.float16, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]], dtype=torch.float16)
    loss = double_softmax_cross_entropy(logits, targets)
    assert loss.dtype == torch.float32
    loss.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


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


def test_smoothstep_damping_suppresses_high_loss_and_reintroduces_final_epoch() -> None:
    losses = torch.tensor([0.1, 5.0], requires_grad=True)
    damped, delta = smoothstep_damped_loss(
        losses, maximum_delta=0.25, epoch_in_cycle=5, cycle_epochs=10
    )
    assert delta == 0.25
    assert torch.allclose(damped[0], losses[0])
    assert damped[1] < losses[1]
    damped.sum().backward()
    assert torch.isfinite(losses.grad).all()
    reintroduced, delta = smoothstep_damped_loss(
        losses.detach(), maximum_delta=0.25, epoch_in_cycle=10, cycle_epochs=10
    )
    assert delta == 0.0
    assert torch.equal(reintroduced, losses.detach())


def test_classwise_high_loss_filter_obeys_global_target_and_caps() -> None:
    losses = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    selected = classwise_high_loss_filter(
        losses,
        labels,
        torch.ones(8, dtype=torch.bool),
        remove_fraction=0.25,
        maximum_class_fraction=0.25,
        minimum_kept_per_class=2,
    )
    assert int(selected.sum()) == 2
    assert torch.bincount(labels[selected], minlength=2).tolist() == [1, 1]
