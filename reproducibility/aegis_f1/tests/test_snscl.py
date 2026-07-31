import torch

from aegis_clip.snscl import (
    ClasswiseFeatureQueue,
    StatefulSNSCL,
    StochasticFeatureEmbedding,
    classwise_queue_contrastive_loss,
)


def test_stochastic_embedding_is_finite_and_differentiable() -> None:
    module = StochasticFeatureEmbedding(8, 16, 4, initial_std=0.05)
    features = torch.randn(3, 8, requires_grad=True)
    projected, mean, log_variance, kl = module(features)
    loss = projected.square().sum() + kl
    loss.backward()
    assert projected.shape == (3, 4)
    assert mean.shape == log_variance.shape == (3, 8)
    assert torch.isfinite(loss)
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_classwise_queue_obeys_admission_and_fifo() -> None:
    queue = ClasswiseFeatureQueue(num_classes=3, queue_size=2, feature_dim=2)
    admitted = queue.enqueue(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([0, 1]),
        torch.tensor([1.0, 0.0]),
    )
    assert admitted == 1
    assert queue.valid_count == 1
    queue.enqueue(
        torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
        torch.tensor([0, 0]),
        torch.ones(2),
    )
    features, labels = queue.snapshot()
    assert features.shape == (2, 2)
    assert torch.equal(labels, torch.tensor([0, 0]))
    assert torch.allclose(features, torch.tensor([[-1.0, 0.0], [0.0, 1.0]]))


def test_queue_contrastive_loss_uses_only_matching_classes() -> None:
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    keys = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    loss, usable = classwise_queue_contrastive_loss(
        queries,
        torch.tensor([0, 2]),
        keys,
        torch.tensor([0, 1]),
        temperature=0.1,
    )
    loss.backward()
    assert usable == 1
    assert torch.isfinite(loss)
    assert queries.grad is not None and torch.isfinite(queries.grad).all()


def test_corrected_anchor_labels_keep_reliable_and_refurbish_unreliable() -> None:
    state = StatefulSNSCL(
        num_samples=2,
        num_classes=3,
        feature_dim=4,
        hidden_dim=8,
        projection_dim=2,
        queue_size=2,
        initial_std=0.05,
        mean_residual_scale=0.1,
    )
    first, weights = state.corrected_labels(
        indices=torch.tensor([0, 1]),
        noisy_labels=torch.tensor([0, 0]),
        reliability=torch.tensor([1.0, 0.0]),
        logits=torch.tensor([[0.0, 5.0, 0.0], [0.0, 5.0, 0.0]]),
        reliability_threshold=0.5,
        moving_average=0.9,
    )
    assert torch.equal(first.argmax(dim=1), torch.tensor([0, 1]))
    assert torch.equal(weights, torch.tensor([1.0, 0.0]))
    second, _ = state.corrected_labels(
        indices=torch.tensor([1]),
        noisy_labels=torch.tensor([0]),
        reliability=torch.tensor([0.0]),
        logits=torch.tensor([[0.0, 0.0, 5.0]]),
        reliability_threshold=0.5,
        moving_average=0.9,
    )
    assert second[0, 1] > second[0, 2]


def test_snscl_state_round_trip_restores_queue_and_label_memory() -> None:
    source = StatefulSNSCL(
        num_samples=2,
        num_classes=3,
        feature_dim=4,
        hidden_dim=8,
        projection_dim=2,
        queue_size=2,
        initial_std=0.05,
        mean_residual_scale=0.1,
    )
    source.queue.enqueue(
        torch.tensor([[1.0, 0.0]]), torch.tensor([2]), torch.tensor([1.0])
    )
    source.corrected_labels(
        indices=torch.tensor([0]),
        noisy_labels=torch.tensor([1]),
        reliability=torch.tensor([1.0]),
        logits=torch.zeros(1, 3),
        reliability_threshold=0.5,
        moving_average=0.9,
    )
    restored = StatefulSNSCL(
        num_samples=2,
        num_classes=3,
        feature_dim=4,
        hidden_dim=8,
        projection_dim=2,
        queue_size=2,
        initial_std=0.05,
        mean_residual_scale=0.1,
    )
    restored.load_state_dict(source.state_dict())
    assert torch.equal(restored.queue.valid, source.queue.valid)
    assert torch.equal(restored.queue.features, source.queue.features)
    assert torch.equal(restored.label_memory, source.label_memory)
    assert torch.equal(restored.label_initialized, source.label_initialized)
