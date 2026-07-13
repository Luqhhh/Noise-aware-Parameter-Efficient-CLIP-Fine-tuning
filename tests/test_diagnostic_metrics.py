"""Tests for common.diagnostic_metrics module."""

import torch
import torch.nn.functional as F
import pytest
from common.diagnostic_metrics import (
    per_sample_cross_entropy,
    per_sample_gce,
    softmax_confidence_margin_entropy,
    jensen_shannon_divergence,
    build_trimmed_class_prototypes,
    prototype_metrics,
    knn_label_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def logits_labels():
    """Return (logits, labels) with shape (N=8, C=5)."""
    rng = torch.Generator().manual_seed(42)
    logits = torch.randn(8, 5, generator=rng)
    labels = torch.randint(0, 5, (8,), generator=rng)
    return logits, labels


@pytest.fixture
def features_labels_50c():
    """Return (features, labels) for 100 samples across 5 classes."""
    rng = torch.Generator().manual_seed(123)
    features = torch.randn(100, 32, generator=rng)
    features = F.normalize(features, dim=-1)
    labels = torch.randint(0, 5, (100,), generator=rng)
    return features, labels


# ---------------------------------------------------------------------------
# per_sample_cross_entropy
# ---------------------------------------------------------------------------

class TestPerSampleCrossEntropy:
    def test_matches_pytorch_reduction_none(self, logits_labels):
        logits, labels = logits_labels
        ours = per_sample_cross_entropy(logits, labels)
        ref = F.cross_entropy(logits, labels, reduction="none")
        assert torch.allclose(ours, ref, atol=1e-6), f"max diff: {(ours - ref).abs().max()}"

    def test_shape(self, logits_labels):
        logits, labels = logits_labels
        result = per_sample_cross_entropy(logits, labels)
        assert result.shape == (logits.size(0),)

    def test_perfect_prediction_zero_loss(self):
        logits = torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
        labels = torch.tensor([0, 1])
        loss = per_sample_cross_entropy(logits, labels)
        assert (loss < 1e-4).all()


# ---------------------------------------------------------------------------
# per_sample_gce
# ---------------------------------------------------------------------------

class TestPerSampleGCE:
    def test_manual_formula(self):
        """GCE = (1 - p_y^q) / q, verify against manual computation."""
        logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        labels = torch.tensor([0, 1])
        q = 0.7
        probs = F.softmax(logits, dim=1)
        py = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        expected = (1.0 - py.pow(q)) / q
        ours = per_sample_gce(logits, labels, q=q)
        assert torch.allclose(ours, expected, atol=1e-6)

    def test_q_near_zero_stable(self, logits_labels):
        logits, labels = logits_labels
        # q=0.01 should not produce NaN
        result = per_sample_gce(logits, labels, q=0.01)
        assert torch.isfinite(result).all()

    def test_shape(self, logits_labels):
        logits, labels = logits_labels
        result = per_sample_gce(logits, labels)
        assert result.shape == (logits.size(0),)


# ---------------------------------------------------------------------------
# softmax_confidence_margin_entropy
# ---------------------------------------------------------------------------

class TestConfidenceMarginEntropy:
    def test_confidence_range(self, logits_labels):
        logits, _ = logits_labels
        conf, margin, entropy, pred = softmax_confidence_margin_entropy(logits)
        assert (conf >= 0).all() and (conf <= 1).all()
        assert (margin >= 0).all()
        assert (entropy >= 0).all()

    def test_pred_is_argmax(self, logits_labels):
        logits, _ = logits_labels
        conf, margin, entropy, pred = softmax_confidence_margin_entropy(logits)
        assert (pred == logits.argmax(dim=1)).all()

    def test_uniform_logits(self):
        logits = torch.zeros(4, 10)
        conf, margin, entropy, pred = softmax_confidence_margin_entropy(logits)
        # All classes equal → margin ≈ 0, confidence ≈ 0.1
        assert (margin < 1e-4).all()
        assert torch.allclose(conf, torch.full_like(conf, 0.1), atol=1e-4)

    def test_peaked_logits(self):
        logits = torch.tensor([[100.0, 0.0, 0.0]])
        conf, margin, entropy, pred = softmax_confidence_margin_entropy(logits)
        assert conf.item() > 0.99
        assert margin.item() > 0.99
        assert entropy.item() < 0.01


# ---------------------------------------------------------------------------
# jensen_shannon_divergence
# ---------------------------------------------------------------------------

class TestJSD:
    def test_non_negative(self, logits_labels):
        logits, _ = logits_labels
        jsd = jensen_shannon_divergence(logits, logits + 0.1)
        assert (jsd >= 0).all()

    def test_identical_logits_zero(self, logits_labels):
        logits, _ = logits_labels
        jsd = jensen_shannon_divergence(logits, logits)
        assert (jsd < 1e-6).all()

    def test_symmetric(self, logits_labels):
        logits, _ = logits_labels
        logits_b = logits + torch.randn_like(logits) * 0.5
        jsd_ab = jensen_shannon_divergence(logits, logits_b)
        jsd_ba = jensen_shannon_divergence(logits_b, logits)
        assert torch.allclose(jsd_ab, jsd_ba, atol=1e-6)


# ---------------------------------------------------------------------------
# build_trimmed_class_prototypes
# ---------------------------------------------------------------------------

class TestBuildTrimmedPrototypes:
    def test_shape(self, features_labels_50c):
        features, labels = features_labels_50c
        prototypes = build_trimmed_class_prototypes(features, labels, num_classes=5, trim_fraction=0.10)
        assert prototypes.shape == (5, 32)

    def test_normalized(self, features_labels_50c):
        features, labels = features_labels_50c
        prototypes = build_trimmed_class_prototypes(features, labels, num_classes=5)
        norms = prototypes.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-5)

    def test_empty_class_raises(self):
        features = torch.randn(10, 32)
        labels = torch.zeros(10, dtype=torch.long)  # all class 0
        with pytest.raises(ValueError, match="empty|no samples"):
            build_trimmed_class_prototypes(features, labels, num_classes=5)


# ---------------------------------------------------------------------------
# prototype_metrics
# ---------------------------------------------------------------------------

class TestPrototypeMetrics:
    def test_keys_present(self, features_labels_50c):
        features, labels = features_labels_50c
        prototypes = build_trimmed_class_prototypes(features, labels, num_classes=5)
        metrics = prototype_metrics(features, labels, prototypes)
        expected_keys = {
            "prototype_label_similarity", "prototype_top1_label",
            "prototype_top1_similarity", "prototype_second_similarity",
            "prototype_margin", "prototype_supports_noisy_label",
        }
        assert set(metrics.keys()) == expected_keys

    def test_top1_label_shape(self, features_labels_50c):
        features, labels = features_labels_50c
        prototypes = build_trimmed_class_prototypes(features, labels, num_classes=5)
        metrics = prototype_metrics(features, labels, prototypes)
        assert metrics["prototype_top1_label"].shape == (features.size(0),)
        assert metrics["prototype_margin"].shape == (features.size(0),)

    def test_supports_label_is_boolean(self, features_labels_50c):
        features, labels = features_labels_50c
        prototypes = build_trimmed_class_prototypes(features, labels, num_classes=5)
        metrics = prototype_metrics(features, labels, prototypes)
        supports = metrics["prototype_supports_noisy_label"]
        assert supports.dtype == torch.bool


# ---------------------------------------------------------------------------
# knn_label_metrics
# ---------------------------------------------------------------------------

class TestKNNLabelMetrics:
    def test_keys_present(self):
        neighbor_indices = torch.tensor([[0, 1, 2], [1, 2, 3]])
        bank_labels = torch.tensor([0, 0, 1, 1, 2, 2])
        query_labels = torch.tensor([0, 1])
        metrics = knn_label_metrics(neighbor_indices, bank_labels, query_labels, num_classes=3)
        expected_keys = {
            "knn_label_agreement", "knn_majority_label",
            "knn_majority_fraction",
        }
        assert set(metrics.keys()) == expected_keys

    def test_all_neighbors_same_label(self):
        # All 3 neighbors are label 0, query label is 0
        neighbor_indices = torch.tensor([[0, 1, 2]])
        bank_labels = torch.tensor([0, 0, 0, 1, 1, 1])
        query_labels = torch.tensor([0])
        metrics = knn_label_metrics(neighbor_indices, bank_labels, query_labels, num_classes=2)
        assert metrics["knn_label_agreement"].item() == 1.0
        assert metrics["knn_majority_label"].item() == 0
        assert metrics["knn_majority_fraction"].item() == 1.0

    def test_shape(self):
        neighbor_indices = torch.randint(0, 100, (50, 20))
        bank_labels = torch.randint(0, 10, (100,))
        query_labels = torch.randint(0, 10, (50,))
        metrics = knn_label_metrics(neighbor_indices, bank_labels, query_labels, num_classes=10)
        for k in metrics:
            assert metrics[k].shape == (50,), f"{k} has wrong shape {metrics[k].shape}"
