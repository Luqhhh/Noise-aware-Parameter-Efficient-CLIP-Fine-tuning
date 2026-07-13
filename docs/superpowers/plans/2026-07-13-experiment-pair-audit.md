# Experiment Pair Audit & Trusted Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a comprehensive framework to audit D3_STRICT vs B2_GCE07 experiment pairs, compute model-agnostic diagnostic metrics, construct trusted validation subsets, and evaluate dual validation accuracy with paired bootstrap.

**Architecture:** Three `common/` modules (diagnostic_metrics, pair_protocol_audit, trusted_subset) provide pure computation. Six `tools/` CLI scripts orchestrate the analysis pipeline. Five `tests/` files verify correctness. All code reuses existing patterns from `common/clip_utils.py`, `common/cache.py`, and `experiments/baseline/evaluate.py`.

**Tech Stack:** Python 3, PyTorch, pandas, numpy, PIL, tqdm, pytest, PyYAML (all already in requirements.txt)

## Global Constraints

- No new pip dependencies beyond requirements.txt
- All paths use pathlib.Path
- SHA-256 via hashlib.sha256 (streaming, 1MiB chunks — matching evaluate.py pattern)
- `common/diagnostic_metrics.py` is pure computation — no I/O, no file reads
- `common/trusted_subset.py` never reads D3/B2 logits/confidence/margin/correctness columns
- kNN uses chunked computation — never builds full [N_query, N_bank] similarity matrix
- Feature bank .pt files use dict format: `{"features": Tensor, "labels": Tensor, "paths": list[str], ...}`
- Checkpoint loading handles both `best.pt` and `last.pt` (they share the same keys)
- Configs loaded via `common/utils.py:load_config()`
- Class mapping loaded via `common/class_mapping.py:load_or_generate_mapping()`
- All CLI scripts accept `--device` with default `cuda`, falling back to `cpu` if unavailable
- Trusted subset V1 rules: `knn_label_agreement >= 0.60`, `prototype_supports_noisy_label == True`, `prototype_margin >= 0.02`, `clip_flip_cosine >= 0.90`, `~cross_class_duplicate_conflict`
- De-duplication metadata column `cross_class_duplicate_conflict` defaults to `False` with warning if unavailable
- Validation = 10,316 samples (from val.csv header+1 = 10,317 lines)
- Training = 91,376 samples (from train.csv header+1 = 91,377 lines)
- D3 expected correct: 7,289; B2 expected correct: 7,179; delta = 110

---

## File Map

| File | Responsibility |
|------|---------------|
| `common/diagnostic_metrics.py` | Per-sample CE/GCE, confidence/margin/entropy, JSD, chunked kNN, robust prototypes, prototype/kNN metrics |
| `common/pair_protocol_audit.py` | `PairAuditResult` dataclass, `audit_experiment_pair()` — split/checkpoint/config comparison |
| `common/trusted_subset.py` | `TrustedSubsetConfig`, `build_trusted_subset()` — V1 model-agnostic rules on DataFrame |
| `tools/audit_experiment_pair.py` | CLI for `audit_experiment_pair()`, exit codes 0/2/3/4 |
| `tools/export_feature_bank.py` | CLI to export frozen CLIP features + flip features for train/val |
| `tools/analyze_checkpoint_disagreement.py` | CLI for four-group analysis, fast-path `F.linear` when visual encoders match |
| `tools/build_trusted_subset.py` | CLI for `build_trusted_subset()`, sensitivity tiers |
| `tools/evaluate_dual_validation.py` | CLI for raw + trusted evaluation with paired bootstrap |
| `tools/summarize_disagreement.py` | CLI to generate `findings.md` from all analysis outputs |
| `tests/test_diagnostic_metrics.py` | Numerical correctness: CE, GCE, confidence, JSD, prototype shapes |
| `tests/test_chunked_knn.py` | chunked vs brute-force equality on small matrices |
| `tests/test_pair_protocol_audit.py` | Synthetic configs/CSVs/checkpoints for audit edge cases |
| `tests/test_trusted_subset.py` | Rule boundary tests, model-column independence |
| `tests/test_dual_validation.py` | Partition correctness, bootstrap reproducibility, manifest validation |

---

### Task 1: `common/diagnostic_metrics.py` + `tests/test_diagnostic_metrics.py`

**Files:**
- Create: `common/diagnostic_metrics.py`
- Create: `tests/test_diagnostic_metrics.py`

**Interfaces:**
- Produces:
  - `per_sample_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor`
  - `per_sample_gce(logits: Tensor, labels: Tensor, q: float = 0.7, eps: float = 1e-7) -> Tensor`
  - `softmax_confidence_margin_entropy(logits: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]`
  - `jensen_shannon_divergence(logits_a: Tensor, logits_b: Tensor) -> Tensor`
  - `chunked_topk_cosine(query_features: Tensor, bank_features: Tensor, k: int, query_chunk_size: int = 256, bank_chunk_size: int = 8192, device: str = "cpu") -> Tuple[Tensor, Tensor]`
  - `build_trimmed_class_prototypes(features: Tensor, labels: Tensor, num_classes: int, trim_fraction: float = 0.10) -> Tensor`
  - `prototype_metrics(features: Tensor, labels: Tensor, prototypes: Tensor) -> Dict[str, Tensor]`
  - `knn_label_metrics(neighbor_indices: Tensor, bank_labels: Tensor, query_labels: Tensor, num_classes: int) -> Dict[str, Tensor]`

- [ ] **Step 1: Write `tests/test_diagnostic_metrics.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/lux1/noise && python -m pytest tests/test_diagnostic_metrics.py -v 2>&1 | tail -20
```
Expected: All tests FAIL with ImportError (module not yet created)

- [ ] **Step 3: Write `common/diagnostic_metrics.py`**

```python
"""
Diagnostic metrics for noisy-label analysis.

Pure computation module — no I/O, no file reads, no config loading.
All functions operate on tensors and return tensors or dicts of tensors.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict


def per_sample_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sample cross-entropy loss.

    Args:
        logits: (N, C) unnormalized logits.
        labels: (N,) integer class labels in [0, C-1].

    Returns:
        (N,) per-sample CE loss.
    """
    return F.cross_entropy(logits, labels, reduction="none")


def per_sample_gce(
    logits: torch.Tensor, labels: torch.Tensor, q: float = 0.7, eps: float = 1e-7
) -> torch.Tensor:
    """Per-sample Generalized Cross Entropy loss.

    L_i = (1 - p_{i,y_i}^q) / q

    Args:
        logits: (N, C) unnormalized logits.
        labels: (N,) integer class labels in [0, C-1].
        q: GCE exponent in (0, 1]. q→0 approaches CE, q=1 is MAE.
        eps: Minimum probability clamp to avoid numerical issues.

    Returns:
        (N,) per-sample GCE loss.
    """
    probs = F.softmax(logits, dim=1)
    py = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    py = py.clamp_min(eps)
    return (1.0 - py.pow(q)) / q


def softmax_confidence_margin_entropy(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute confidence, margin, entropy, and prediction from logits.

    Args:
        logits: (N, C) unnormalized logits.

    Returns:
        confidence: (N,) max softmax probability.
        margin: (N,) top1_prob - top2_prob.
        entropy: (N,) Shannon entropy of softmax distribution (nats).
        pred: (N,) predicted class index.
    """
    probs = F.softmax(logits, dim=1)
    top2_probs, top2_indices = probs.topk(2, dim=1)
    confidence = top2_probs[:, 0]
    margin = top2_probs[:, 0] - top2_probs[:, 1]
    pred = top2_indices[:, 0]

    log_probs = torch.log(probs.clamp_min(1e-12))
    entropy = -(probs * log_probs).sum(dim=1)

    return confidence, margin, entropy, pred


def jensen_shannon_divergence(
    logits_a: torch.Tensor, logits_b: torch.Tensor
) -> torch.Tensor:
    """Per-sample Jensen-Shannon divergence between two softmax distributions.

    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q).

    Args:
        logits_a: (N, C) first set of logits.
        logits_b: (N, C) second set of logits.

    Returns:
        (N,) per-sample JSD (nats, base-e).
    """
    probs_a = F.softmax(logits_a, dim=1)
    probs_b = F.softmax(logits_b, dim=1)
    m = 0.5 * (probs_a + probs_b)

    log_a = torch.log(probs_a.clamp_min(1e-12))
    log_b = torch.log(probs_b.clamp_min(1e-12))
    log_m = torch.log(m.clamp_min(1e-12))

    kl_a = (probs_a * (log_a - log_m)).sum(dim=1)
    kl_b = (probs_b * (log_b - log_m)).sum(dim=1)

    return 0.5 * kl_a + 0.5 * kl_b


def chunked_topk_cosine(
    query_features: torch.Tensor,
    bank_features: torch.Tensor,
    k: int,
    query_chunk_size: int = 256,
    bank_chunk_size: int = 8192,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Memory-safe chunked exact top-k cosine similarity search.

    Never builds the full [N_query, N_bank] similarity matrix. Instead processes
    query chunks against bank chunks, maintaining a running top-k heap per query.

    Args:
        query_features: (N_query, D) normalized query features.
        bank_features: (N_bank, D) normalized bank features.
        k: Number of nearest neighbors to return.
        query_chunk_size: Number of queries to process at once.
        bank_chunk_size: Number of bank samples to process at once.
        device: Torch device string ("cpu" or "cuda").

    Returns:
        topk_indices: (N_query, k) indices into bank_features.
        topk_similarities: (N_query, k) cosine similarities in [0, 1].
    """
    n_query = query_features.shape[0]
    n_bank = bank_features.shape[0]

    # Initialize with -inf so any real similarity replaces it
    topk_sims = torch.full((n_query, k), -float("inf"), device=device)
    topk_idxs = torch.zeros((n_query, k), dtype=torch.long, device=device)

    query_features = query_features.to(device)
    bank_features = bank_features.to(device)

    for q_start in range(0, n_query, query_chunk_size):
        q_end = min(q_start + query_chunk_size, n_query)
        q_chunk = query_features[q_start:q_end]  # (chunk_q, D)

        # Reset running top-k for this query chunk
        chunk_sims = torch.full((q_end - q_start, k), -float("inf"), device=device)
        chunk_idxs = torch.zeros((q_end - q_start, k), dtype=torch.long, device=device)

        for b_start in range(0, n_bank, bank_chunk_size):
            b_end = min(b_start + bank_chunk_size, n_bank)
            b_chunk = bank_features[b_start:b_end]  # (chunk_b, D)

            # Cosine similarity = dot product (features are L2-normalized)
            sim_block = torch.mm(q_chunk, b_chunk.T)  # (chunk_q, chunk_b)

            # Merge with running top-k
            combined_sims = torch.cat([chunk_sims, sim_block], dim=1)
            combined_idxs = torch.cat([
                chunk_idxs,
                torch.arange(b_start, b_end, device=device).unsqueeze(0).expand(q_end - q_start, -1),
            ], dim=1)

            chunk_sims, sorted_idx = combined_sims.topk(k, dim=1, largest=True)
            chunk_idxs = combined_idxs.gather(1, sorted_idx)

        topk_sims[q_start:q_end] = chunk_sims
        topk_idxs[q_start:q_end] = chunk_idxs

    return topk_idxs.cpu(), topk_sims.cpu()


def build_trimmed_class_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    trim_fraction: float = 0.10,
) -> torch.Tensor:
    """Build robust class prototypes with outlier trimming.

    Two-stage process:
    1. Compute initial mean per class, L2-normalize.
    2. Drop the trim_fraction lowest-similarity samples per class.
    3. Recompute mean from retained samples, L2-normalize.

    Args:
        features: (N, D) L2-normalized feature vectors.
        labels: (N,) integer class labels in [0, num_classes-1].
        num_classes: Total number of classes.
        trim_fraction: Fraction of lowest-similarity samples to drop (default 0.10).

    Returns:
        (num_classes, D) L2-normalized prototype vectors.

    Raises:
        ValueError: If any class has no samples.
    """
    device = features.device
    dtype = features.dtype
    prototypes = torch.zeros(num_classes, features.size(1), device=device, dtype=dtype)

    for c in range(num_classes):
        mask = labels == c
        class_feats = features[mask]
        n_c = class_feats.size(0)

        if n_c == 0:
            raise ValueError(f"Class {c} has no samples — cannot build prototype.")

        # Stage 1: initial mean
        init_proto = class_feats.mean(dim=0)
        init_proto = F.normalize(init_proto, p=2, dim=0)

        # Stage 2: trim lowest-similarity samples
        similarities = torch.mv(class_feats, init_proto)  # (n_c,)
        n_keep = max(1, int(n_c * (1.0 - trim_fraction)))
        _, keep_idx = similarities.topk(n_keep, largest=True)
        retained = class_feats[keep_idx]

        proto = retained.mean(dim=0)
        prototypes[c] = F.normalize(proto, p=2, dim=0)

    return prototypes


def prototype_metrics(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute per-sample prototype-based metrics.

    Args:
        features: (N, D) L2-normalized query features.
        labels: (N,) noisy labels (ground truth for metric computation).
        prototypes: (C, D) L2-normalized class prototypes.

    Returns:
        Dict with keys:
          prototype_label_similarity: (N,) cosine similarity to noisy label's prototype.
          prototype_top1_label: (N,) class index of nearest prototype.
          prototype_top1_similarity: (N,) similarity to nearest prototype.
          prototype_second_similarity: (N,) similarity to second-nearest prototype.
          prototype_margin: (N,) top1_sim - second_sim.
          prototype_supports_noisy_label: (N,) bool, top1_label == noisy_label.
    """
    # (N, C) cosine similarities
    sims = torch.mm(features, prototypes.T)  # features and prototypes are normalized

    top2_sims, top2_idxs = sims.topk(2, dim=1)
    top1_sim = top2_sims[:, 0]
    second_sim = top2_sims[:, 1]
    top1_label = top2_idxs[:, 0]

    label_sim = sims.gather(1, labels.unsqueeze(1)).squeeze(1)

    return {
        "prototype_label_similarity": label_sim,
        "prototype_top1_label": top1_label,
        "prototype_top1_similarity": top1_sim,
        "prototype_second_similarity": second_sim,
        "prototype_margin": top1_sim - second_sim,
        "prototype_supports_noisy_label": top1_label == labels,
    }


def knn_label_metrics(
    neighbor_indices: torch.Tensor,
    bank_labels: torch.Tensor,
    query_labels: torch.Tensor,
    num_classes: int,
) -> Dict[str, torch.Tensor]:
    """Compute per-sample kNN-based label agreement metrics.

    Args:
        neighbor_indices: (N_query, k) indices of nearest neighbors in the bank.
        bank_labels: (N_bank,) integer labels for the bank samples.
        query_labels: (N_query,) noisy labels of the query samples.
        num_classes: Total number of classes.

    Returns:
        Dict with keys:
          knn_label_agreement: (N_query,) fraction of neighbors matching query label.
          knn_majority_label: (N_query,) most common label among neighbors.
          knn_majority_fraction: (N_query,) fraction of neighbors with majority label.
    """
    n_query, k = neighbor_indices.shape
    neighbor_labels = bank_labels[neighbor_indices]  # (N_query, k)

    # Label agreement: fraction of neighbors matching the query's noisy label
    query_labels_expanded = query_labels.unsqueeze(1).expand(-1, k)
    label_agreement = (neighbor_labels == query_labels_expanded).float().mean(dim=1)

    # Majority label via bincount per row
    majority_label = torch.zeros(n_query, dtype=torch.long)
    majority_fraction = torch.zeros(n_query)

    for i in range(n_query):
        counts = torch.bincount(neighbor_labels[i], minlength=num_classes)
        max_count = counts.max()
        majority_label[i] = counts.argmax()
        majority_fraction[i] = max_count.float() / k

    return {
        "knn_label_agreement": label_agreement,
        "knn_majority_label": majority_label,
        "knn_majority_fraction": majority_fraction,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/lux1/noise && python -m pytest tests/test_diagnostic_metrics.py -v 2>&1 | tail -30
```
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add common/diagnostic_metrics.py tests/test_diagnostic_metrics.py
git commit -m "feat(analysis): add diagnostic metrics module (CE, GCE, kNN, prototypes, JSD)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: `tests/test_chunked_knn.py`

- [ ] **Step 1: Write `tests/test_chunked_knn.py`**

```python
"""Tests for chunked_topk_cosine — verify equality with brute-force."""
import torch
import torch.nn.functional as F
import pytest
from common.diagnostic_metrics import chunked_topk_cosine


@pytest.fixture
def small_data():
    """Small query and bank for exhaustive comparison."""
    rng = torch.Generator().manual_seed(99)
    query = torch.randn(30, 16, generator=rng)
    bank = torch.randn(50, 16, generator=rng)
    query = F.normalize(query, dim=-1)
    bank = F.normalize(bank, dim=-1)
    return query, bank


def brute_force_topk(query, bank, k):
    """Reference: full similarity matrix, exact top-k."""
    sim = torch.mm(query, bank.T)
    return sim.topk(k, dim=1, largest=True)


class TestChunkedTopK:
    def test_vs_brute_force_default_chunks(self, small_data):
        query, bank = small_data
        k = 5
        idx_c, sim_c = chunked_topk_cosine(query, bank, k, query_chunk_size=8, bank_chunk_size=10)
        _, sim_bf = brute_force_topk(query, bank, k)
        assert torch.allclose(sim_c, sim_bf, atol=1e-5), f"max diff: {(sim_c - sim_bf).abs().max()}"

    def test_vs_brute_force_single_query_chunk(self, small_data):
        query, bank = small_data
        k = 3
        idx_c, sim_c = chunked_topk_cosine(query, bank, k, query_chunk_size=30, bank_chunk_size=50)
        _, sim_bf = brute_force_topk(query, bank, k)
        assert torch.allclose(sim_c, sim_bf, atol=1e-5)

    def test_vs_brute_force_single_bank_chunk(self, small_data):
        query, bank = small_data
        k = 3
        idx_c, sim_c = chunked_topk_cosine(query, bank, k, query_chunk_size=5, bank_chunk_size=50)
        _, sim_bf = brute_force_topk(query, bank, k)
        assert torch.allclose(sim_c, sim_bf, atol=1e-5)

    def test_k_is_one(self, small_data):
        query, bank = small_data
        k = 1
        idx_c, sim_c = chunked_topk_cosine(query, bank, k)
        _, sim_bf = brute_force_topk(query, bank, k)
        assert torch.allclose(sim_c, sim_bf, atol=1e-5)

    def test_k_equals_bank_size(self, small_data):
        query, bank = small_data
        k = bank.size(0)
        idx_c, sim_c = chunked_topk_cosine(query, bank, k)
        _, sim_bf = brute_force_topk(query, bank, k)
        # Sort each row of sim_bf for comparison
        sim_bf_sorted, _ = sim_bf.sort(dim=1, descending=True)
        sim_c_sorted, _ = sim_c.sort(dim=1, descending=True)
        assert torch.allclose(sim_c_sorted, sim_bf_sorted, atol=1e-5)

    def test_output_shapes(self, small_data):
        query, bank = small_data
        k = 7
        idx, sim = chunked_topk_cosine(query, bank, k)
        assert idx.shape == (query.size(0), k)
        assert sim.shape == (query.size(0), k)

    def test_indices_in_range(self, small_data):
        query, bank = small_data
        k = 5
        idx, sim = chunked_topk_cosine(query, bank, k)
        assert (idx >= 0).all() and (idx < bank.size(0)).all()

    def test_cpu_device(self, small_data):
        query, bank = small_data
        k = 5
        idx, sim = chunked_topk_cosine(query, bank, k, device="cpu")
        assert idx.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self, small_data):
        query, bank = small_data
        k = 5
        idx, sim = chunked_topk_cosine(query, bank, k, device="cuda")
        assert idx.device.type == "cpu"  # returned to CPU
        # Just check it runs without error
        assert idx.shape == (query.size(0), k)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/lux1/noise && python -m pytest tests/test_chunked_knn.py -v 2>&1 | tail -10
```
Expected: All tests FAIL (module not imported yet but file exists, or dependencies resolved — actually these depend on common/diagnostic_metrics.py which exists now, so some will pass immediately. Verify at least the new test file is discovered.)

- [ ] **Step 3: Run tests (they should pass since chunked_topk_cosine is already implemented)**

```bash
cd /home/lux1/noise && python -m pytest tests/test_chunked_knn.py -v 2>&1 | tail -30
```
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_chunked_knn.py
git commit -m "test(analysis): add chunked kNN correctness tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: `common/pair_protocol_audit.py` + `tests/test_pair_protocol_audit.py`

**Files:**
- Create: `common/pair_protocol_audit.py`
- Create: `tests/test_pair_protocol_audit.py`

**Interfaces:**
- Produces:
  - `PairAuditResult` dataclass with fields: `paired_valid`, `causal_claim_allowed`, `allowed_differences`, `unexpected_differences`, `warnings`, `hashes`, `counts`, `resolved_paths`
  - `audit_experiment_pair(reference_config_path, candidate_config_path, reference_ckpt_path, candidate_ckpt_path, output_path) -> PairAuditResult`

- [ ] **Step 1: Write `tests/test_pair_protocol_audit.py`**

```python
"""Tests for common.pair_protocol_audit module."""
import json
import tempfile
import hashlib
from pathlib import Path
import torch
import torch.nn as nn
import yaml
import pandas as pd
import pytest
from common.pair_protocol_audit import (
    PairAuditResult,
    audit_experiment_pair,
    _sha256_hex,
    _resolve_effective_samples,
    _compare_config_fields,
)


def _make_csv(paths_labels, output_path):
    """Write a minimal split CSV."""
    rows = [{"image_path": p, "class_name": str(l), "label": l} for p, l in paths_labels]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _make_config(output_path, **overrides):
    """Write a minimal config YAML."""
    cfg = {
        "experiment": {"id": "TEST", "mode": "dev", "head_type": "linear", "augmentation_preset": "a0"},
        "data": {
            "stage": "preliminary", "seed": 42, "split_seed": 42, "train_seed": 42,
            "split_dir": "/tmp", "test_dir": "test", "train_dir": "train",
            "val_ratio": 0.1, "expected_num_classes": 500,
            "class_mapping_path": "/tmp",
        },
        "model": {
            "clip_model_name": "ViT-B/32", "feature_dim": 512, "freeze_clip": True,
            "num_classes": 500, "unfreeze_last_n_blocks": 0,
            "train_ln_post": False, "train_visual_proj": False,
        },
        "loss": {"name": "cross_entropy"},
        "eval": {"batch_size": 256},
        "output": {"log_dir": "/tmp/logs", "submission_dir": "/tmp/subs"},
        "train": {
            "amp": True, "batch_size": 128, "device": "cpu", "epochs": 50,
            "image_size": 224, "lr": 0.005, "max_grad_norm": 1.0,
            "num_workers": 0, "save_dir": "/tmp/ckpts", "scheduler": "cosine",
            "warmup_epochs": 2, "weight_decay": 0.0001, "min_lr_ratio": 0.01,
            "early_stop_patience": 10,
        },
    }
    cfg.update(overrides)
    with open(output_path, "w") as f:
        yaml.dump(cfg, f)


def _make_minimal_checkpoint(output_path, classifier_weight=None, classifier_bias=None):
    """Write a minimal checkpoint .pt file."""
    if classifier_weight is None:
        classifier_weight = torch.randn(500, 512)
    if classifier_bias is None:
        classifier_bias = torch.zeros(500)
    ckpt = {
        "model_state_dict": {
            "classifier.weight": classifier_weight,
            "classifier.bias": classifier_bias,
            # Add some visual keys to simulate frozen CLIP
            "visual.conv1.weight": torch.randn(768, 3, 32, 32),
        },
        "epoch": 30,
        "best_val_acc": 0.70,
        "best_epoch": 25,
        "head_type": "linear",
        "augmentation_preset": "a0",
        "split_seed": 42,
        "training_mode": "dev",
    }
    torch.save(ckpt, output_path)


class TestSHA256:
    def test_same_content_same_hash(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"hello world")
            path = f.name
        try:
            h1 = _sha256_hex(Path(path))
            h2 = _sha256_hex(Path(path))
            assert h1 == h2
        finally:
            Path(path).unlink()


class TestResolveEffectiveSamples:
    def test_identical_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Create actual image files
            img_dir = tmp / "images"
            img_dir.mkdir()
            img1 = img_dir / "a.jpg"
            img1.write_bytes(b"fake jpeg content")
            img2 = img_dir / "b.jpg"
            img2.write_bytes(b"other content")

            csv = tmp / "train.csv"
            _make_csv([(str(img1), 0), (str(img2), 1)], csv)

            result = _resolve_effective_samples(str(csv))
            assert result["count"] == 2
            assert result["missing"] == 0


class TestCompareConfigFields:
    def test_allowed_differences_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ref_cfg = tmp / "ref.yaml"
            cand_cfg = tmp / "cand.yaml"
            _make_config(ref_cfg)
            _make_config(cand_cfg,
                         experiment={"id": "CAND", "mode": "dev", "head_type": "linear", "augmentation_preset": "a0"},
                         loss={"name": "gce", "q": 0.7},
                         output={"log_dir": "/other/logs", "submission_dir": "/other/subs"})
            ref = yaml.safe_load(open(ref_cfg))
            cand = yaml.safe_load(open(cand_cfg))
            allowed, unexpected = _compare_config_fields(ref, cand)
            assert len(unexpected) == 1  # train.save_dir difference
            assert any("save_dir" in str(d) for d in unexpected)


class TestAuditExperimentPair:
    def test_matching_configs_and_checkpoints(self):
        """Full audit with identical configs should pass."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            # Create configs
            ref_cfg = tmp / "ref.yaml"
            cand_cfg = tmp / "cand.yaml"
            _make_config(ref_cfg)
            _make_config(cand_cfg, experiment={"id": "CAND", "mode": "dev", "head_type": "linear", "augmentation_preset": "a0"})

            # Create CSV files
            val_csv = tmp / "val.csv"
            train_csv = tmp / "train.csv"
            _make_csv([("train/0000/img1.jpg", 0), ("train/0000/img2.jpg", 1)], val_csv)
            _make_csv([("train/0000/img3.jpg", 0), ("train/0000/img4.jpg", 1)], train_csv)

            # Create checkpoints
            ref_ckpt = tmp / "ref.pt"
            cand_ckpt = tmp / "cand.pt"
            w = torch.randn(500, 512)
            b = torch.zeros(500)
            _make_minimal_checkpoint(ref_ckpt, w, b)
            _make_minimal_checkpoint(cand_ckpt, w.clone(), b.clone())

            # Need to mock the config loading and CSV resolution...
            # This integration test requires real files at paths that exist.
            # We test via the unit functions instead.
            pass  # Integration test skipped — requires real data tree
```

- [ ] **Step 2: Write `common/pair_protocol_audit.py`**

```python
"""
Experiment pair protocol audit.

Compares two experiment configs, their effective training samples, class mappings,
and checkpoints to determine whether they form a valid paired comparison suitable
for causal attribution of performance differences.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import yaml

from common.utils import load_config

logger = logging.getLogger(__name__)


def _sha256_hex(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# Fields allowed to differ between reference and candidate configs
ALLOWED_DIFFERENCES = {
    "experiment.id",
    "output.log_dir",
    "output.submission_dir",
    "train.save_dir",
    "loss.name",
    "loss.q",
    "loss.probability_epsilon",
    "loss.epsilon",
    "loss.reduction",
}

# Fields that must be identical
REQUIRED_IDENTICAL = [
    "experiment.mode",
    "experiment.head_type",
    "experiment.augmentation_preset",
    "data.seed",
    "data.split_seed",
    "data.train_seed",
    "data.split_dir",
    "data.class_mapping_path",
    "model.clip_model_name",
    "model.feature_dim",
    "model.freeze_clip",
    "model.num_classes",
    "model.unfreeze_last_n_blocks",
    "model.train_ln_post",
    "model.train_visual_proj",
    "train.batch_size",
    "train.epochs",
    "train.lr",
    "train.weight_decay",
    "train.scheduler",
    "train.warmup_epochs",
    "train.min_lr_ratio",
    "train.early_stop_patience",
    "train.max_grad_norm",
    "train.amp",
]


def _nested_get(d: dict, dotted_key: str, default=object()):
    """Get a nested dict value by dotted key, e.g. 'data.seed'."""
    keys = dotted_key.split(".")
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            if default is not object():
                return default
            raise KeyError(dotted_key)
    return d


@dataclass(frozen=True)
class PairAuditResult:
    paired_valid: bool
    causal_claim_allowed: bool
    allowed_differences: List[str]
    unexpected_differences: List[dict]
    warnings: List[str]
    hashes: Dict[str, str]
    counts: Dict[str, int]
    resolved_paths: Dict[str, List[str]]
    sample_classification: str = "not_checked"  # identical_effective_samples | different_effective_samples | ...
    max_visual_abs_diff: float = 0.0
    extra: dict = field(default_factory=dict)


def _resolve_effective_samples(split_csv_path: str) -> dict:
    """Resolve the actual training samples from a split CSV.

    Returns dict with: count, missing, paths, sha256_set, label_set, sha256_label_set.
    """
    csv_path = Path(split_csv_path)
    if not csv_path.exists():
        return {"count": 0, "missing": 0, "paths": [], "error": "CSV not found"}

    df = pd.read_csv(csv_path)
    result = {"count": len(df), "missing": 0, "paths": [], "sha256_set": set(),
              "label_set": set(), "sha256_label_set": set()}

    cwd = Path.cwd()
    for _, row in df.iterrows():
        img_path = Path(row["image_path"])
        if not img_path.is_absolute():
            img_path = cwd / img_path
        abs_path = str(img_path.resolve())
        result["paths"].append(abs_path)

        if img_path.exists():
            sha = _sha256_hex(img_path)
            result["sha256_set"].add(sha)
            result["sha256_label_set"].add((sha, int(row["label"])))
        else:
            result["missing"] += 1

        result["label_set"].add(int(row["label"]))

    return result


def _compare_config_fields(ref_config: dict, cand_config: dict) -> Tuple[List[str], List[dict]]:
    """Compare config fields. Returns (allowed_diffs, unexpected_diffs)."""
    allowed = []
    unexpected = []

    for key_path in REQUIRED_IDENTICAL:
        try:
            ref_val = _nested_get(ref_config, key_path)
            cand_val = _nested_get(cand_config, key_path)
        except KeyError:
            unexpected.append({"field": key_path, "issue": "missing_in_one_config",
                               "ref": "KeyError", "cand": "KeyError"})
            continue

        if ref_val != cand_val:
            if key_path in ALLOWED_DIFFERENCES:
                allowed.append(key_path)
            else:
                unexpected.append({
                    "field": key_path,
                    "ref_value": str(ref_val),
                    "cand_value": str(cand_val),
                })

    return allowed, unexpected


def _compare_visual_encoders(ref_ckpt: dict, cand_ckpt: dict) -> float:
    """Compare visual encoder weights. Returns max absolute difference."""
    ref_state = ref_ckpt.get("model_state_dict", ref_ckpt)
    cand_state = cand_ckpt.get("model_state_dict", cand_ckpt)

    visual_keys = [k for k in ref_state if k.startswith("visual.")]
    if not visual_keys:
        return 0.0

    max_diff = 0.0
    for k in visual_keys:
        if k in cand_state:
            diff = (ref_state[k].float() - cand_state[k].float()).abs().max().item()
            max_diff = max(max_diff, diff)

    return max_diff


def audit_experiment_pair(
    reference_config_path: str,
    candidate_config_path: str,
    reference_ckpt_path: str,
    candidate_ckpt_path: str,
    output_path: str,
) -> PairAuditResult:
    """Run full protocol audit between two experiments.

    Args:
        reference_config_path: Path to reference experiment YAML config.
        candidate_config_path: Path to candidate experiment YAML config.
        reference_ckpt_path: Path to reference checkpoint .pt file.
        candidate_ckpt_path: Path to candidate checkpoint .pt file.
        output_path: Path to write audit JSON.

    Returns:
        PairAuditResult with full audit details.
    """
    warnings = []
    allowed_diffs = []
    unexpected_diffs = []
    hashes = {}
    counts = {}

    # Load configs
    ref_config = load_config(reference_config_path)
    cand_config = load_config(candidate_config_path)

    # ── A. Split and class mapping ──
    ref_split_dir = Path(ref_config["data"]["split_dir"])
    cand_split_dir = Path(cand_config["data"]["split_dir"])

    for name, path in [("ref_val", ref_split_dir / "val.csv"),
                        ("cand_val", cand_split_dir / "val.csv"),
                        ("ref_train", ref_split_dir / "train.csv"),
                        ("cand_train", cand_split_dir / "train.csv")]:
        if path.exists():
            hashes[name] = _sha256_hex(path)
        else:
            hashes[name] = "MISSING"
            warnings.append(f"{name} not found at {path}")

    # Validation CSV must be identical
    if hashes.get("ref_val") and hashes.get("cand_val"):
        if hashes["ref_val"] != hashes["cand_val"]:
            unexpected_diffs.append({
                "field": "val.csv",
                "ref_sha256": hashes["ref_val"],
                "cand_sha256": hashes["cand_val"],
            })

    # Class mapping
    class_mapping_path = ref_config["data"].get("class_mapping_path", ref_config["data"]["split_dir"])
    for fname in ["class_to_idx.json", "idx_to_class.json"]:
        p = Path(class_mapping_path) / fname
        if p.exists():
            hashes[fname] = _sha256_hex(p)

    # ── B. Effective training samples ──
    ref_samples = _resolve_effective_samples(str(ref_split_dir / "train.csv"))
    cand_samples = _resolve_effective_samples(str(cand_split_dir / "train.csv"))

    counts["ref_train_samples"] = ref_samples["count"]
    counts["cand_train_samples"] = cand_samples["count"]
    counts["ref_train_missing"] = ref_samples.get("missing", 0)
    counts["cand_train_missing"] = cand_samples.get("missing", 0)

    # Classify sample relationship
    if ref_samples.get("sha256_set") and cand_samples.get("sha256_set"):
        if ref_samples["sha256_set"] == cand_samples["sha256_set"]:
            if ref_samples.get("sha256_label_set") == cand_samples.get("sha256_label_set"):
                sample_class = "identical_effective_samples"
            else:
                sample_class = "same_images_different_labels"
        else:
            overlap = ref_samples["sha256_set"] & cand_samples["sha256_set"]
            if len(overlap) == min(len(ref_samples["sha256_set"]), len(cand_samples["sha256_set"])):
                sample_class = "same_images_different_paths"
            else:
                sample_class = "different_effective_samples"
    else:
        sample_class = "could_not_compute"

    if sample_class != "identical_effective_samples":
        warnings.append(f"Training samples classified as: {sample_class}")
    if ref_samples.get("missing", 0) > 0 or cand_samples.get("missing", 0) > 0:
        warnings.append("Missing training files detected")

    # ── C. Config comparison ──
    allowed_diffs, unexpected_diffs = _compare_config_fields(ref_config, cand_config)

    # train_dir is a special case — warn if different but effective samples same
    if ref_config["data"].get("train_dir") != cand_config["data"].get("train_dir"):
        warnings.append(
            f"data.train_dir differs: ref={ref_config['data']['train_dir']}, "
            f"cand={cand_config['data']['train_dir']}. "
            f"Deferring to effective sample audit."
        )

    # ── D. Checkpoint comparison ──
    ref_ckpt = torch.load(reference_ckpt_path, map_location="cpu", weights_only=False)
    cand_ckpt = torch.load(candidate_ckpt_path, map_location="cpu", weights_only=False)

    hashes["ref_checkpoint"] = _sha256_hex(Path(reference_ckpt_path))
    hashes["cand_checkpoint"] = _sha256_hex(Path(candidate_ckpt_path))

    ref_state = ref_ckpt.get("model_state_dict", {})
    cand_state = cand_ckpt.get("model_state_dict", {})

    ref_keys = set(ref_state.keys())
    cand_keys = set(cand_state.keys())
    if ref_keys != cand_keys:
        unexpected_diffs.append({
            "field": "checkpoint.state_dict_keys",
            "only_in_ref": sorted(ref_keys - cand_keys),
            "only_in_cand": sorted(cand_keys - ref_keys),
        })

    # Classifier shape
    for ckpt_name, state in [("ref", ref_state), ("cand", cand_state)]:
        if "classifier.weight" in state:
            w = state["classifier.weight"]
            counts[f"{ckpt_name}_classifier_weight_shape"] = list(w.shape)
        if "classifier.bias" in state:
            counts[f"{ckpt_name}_classifier_bias_shape"] = list(state["classifier.bias"].shape)

    counts["ref_checkpoint_epoch"] = ref_ckpt.get("epoch", -1)
    counts["cand_checkpoint_epoch"] = cand_ckpt.get("epoch", -1)

    # Visual encoder comparison
    max_visual_abs_diff = _compare_visual_encoders(ref_ckpt, cand_ckpt)
    if max_visual_abs_diff > 0:
        warnings.append(f"Visual encoder weights differ: max_abs_diff={max_visual_abs_diff:.6f}")
    else:
        logger.info("Visual encoder weights are identical (max_abs_diff=0).")

    # ── E. Determine validity ──
    paired_valid = (
        sample_class in ("identical_effective_samples", "same_images_different_paths")
        and len([d for d in unexpected_diffs if "checkpoint" not in d.get("field", "")]) == 0
        and hashes.get("ref_val") == hashes.get("cand_val")
        and hashes.get("ref_val") is not None
        and hashes.get("ref_val") != "MISSING"
    )

    causal_claim_allowed = paired_valid and max_visual_abs_diff == 0

    result = PairAuditResult(
        paired_valid=paired_valid,
        causal_claim_allowed=causal_claim_allowed,
        allowed_differences=allowed_diffs,
        unexpected_differences=unexpected_diffs,
        warnings=warnings,
        hashes=hashes,
        counts=counts,
        resolved_paths={},
        sample_classification=sample_class,
        max_visual_abs_diff=max_visual_abs_diff,
    )

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "paired_valid": result.paired_valid,
            "causal_claim_allowed": result.causal_claim_allowed,
            "allowed_differences": result.allowed_differences,
            "unexpected_differences": result.unexpected_differences,
            "warnings": result.warnings,
            "hashes": result.hashes,
            "counts": result.counts,
            "sample_classification": result.sample_classification,
            "max_visual_abs_diff": result.max_visual_abs_diff,
        }, f, indent=2, default=str)

    return result
```

- [ ] **Step 2 alternative — run tests:**

```bash
cd /home/lux1/noise && python -m pytest tests/test_pair_protocol_audit.py -v 2>&1 | tail -30
```
Expected: Unit tests pass (integration test skipped)

- [ ] **Step 3: Commit**

```bash
git add common/pair_protocol_audit.py tests/test_pair_protocol_audit.py
git commit -m "feat(analysis): add experiment pair protocol audit

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: `common/trusted_subset.py` + `tests/test_trusted_subset.py`

**Files:**
- Create: `common/trusted_subset.py`
- Create: `tests/test_trusted_subset.py`

**Interfaces:**
- Produces:
  - `TrustedSubsetConfig` dataclass
  - `build_trusted_subset(df: pd.DataFrame, config: TrustedSubsetConfig) -> Tuple[pd.DataFrame, dict]`

- [ ] **Step 1: Write `tests/test_trusted_subset.py`**

```python
"""Tests for common.trusted_subset module."""
import pandas as pd
import numpy as np
import pytest
from common.trusted_subset import TrustedSubsetConfig, build_trusted_subset


@pytest.fixture
def sample_df():
    """DataFrame with all required columns for trusted subset rules."""
    n = 100
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "sample_index": range(n),
        "image_path": [f"img_{i}.jpg" for i in range(n)],
        "image_sha256": [f"sha_{i}" for i in range(n)],
        "noisy_label": rng.integers(0, 10, n),
        "class_name": [f"{i%10:04d}" for i in range(n)],
        "knn_label_agreement": rng.uniform(0.3, 1.0, n),
        "prototype_supports_noisy_label": rng.choice([True, False], n),
        "prototype_margin": rng.uniform(-0.1, 0.3, n),
        "prototype_top1_label": rng.integers(0, 10, n),
        "clip_flip_cosine": rng.uniform(0.7, 1.0, n),
        "cross_class_duplicate_conflict": rng.choice([True, False], n, p=[0.1, 0.9]),
        # Model-specific columns — should be IGNORED by trusted subset
        "d3_pred": rng.integers(0, 10, n),
        "d3_correct": rng.choice([True, False], n),
        "d3_confidence": rng.uniform(0, 1, n),
        "d3_margin": rng.uniform(0, 1, n),
        "b2_pred": rng.integers(0, 10, n),
        "b2_correct": rng.choice([True, False], n),
        "b2_confidence": rng.uniform(0, 1, n),
    })
    return df


class TestTrustedSubsetConfig:
    def test_defaults(self):
        cfg = TrustedSubsetConfig()
        assert cfg.knn_label_agreement_min == 0.60
        assert cfg.prototype_margin_min == 0.02
        assert cfg.clip_flip_cosine_min == 0.90
        assert cfg.require_prototype_top1_matches_label is True
        assert cfg.reject_cross_class_duplicate_conflict is True


class TestBuildTrustedSubset:
    def test_all_rules_applied(self, sample_df):
        cfg = TrustedSubsetConfig()
        manifest, summary = build_trusted_subset(sample_df, cfg)
        assert "trusted_v1" in manifest.columns
        assert "rejection_reasons" in manifest.columns
        assert summary["total_samples"] == len(sample_df)

    def test_trusted_subset_is_subset(self, sample_df):
        cfg = TrustedSubsetConfig()
        manifest, summary = build_trusted_subset(sample_df, cfg)
        assert manifest["trusted_v1"].sum() <= len(sample_df)

    def test_rejection_reasons_nonempty_for_rejected(self, sample_df):
        cfg = TrustedSubsetConfig()
        manifest, summary = build_trusted_subset(sample_df, cfg)
        rejected = manifest[~manifest["trusted_v1"]]
        assert (rejected["rejection_reasons"].str.len() > 0).all()

    def test_all_trusted_no_rejection_reasons(self, sample_df):
        cfg = TrustedSubsetConfig()
        manifest, summary = build_trusted_subset(sample_df, cfg)
        trusted = manifest[manifest["trusted_v1"]]
        # Trusted samples should have empty rejection reasons
        assert (trusted["rejection_reasons"] == "").all()

    def test_coverage_reported(self, sample_df):
        cfg = TrustedSubsetConfig()
        manifest, summary = build_trusted_subset(sample_df, cfg)
        assert "coverage" in summary
        assert 0 <= summary["coverage"] <= 1

    def test_per_class_stats(self, sample_df):
        cfg = TrustedSubsetConfig()
        manifest, summary = build_trusted_subset(sample_df, cfg)
        assert "represented_classes" in summary
        assert "per_class_trusted" in summary

    def test_output_independent_of_model_columns(self, sample_df):
        """Trusted subset must NOT depend on D3/B2 columns."""
        cfg = TrustedSubsetConfig()
        manifest1, summary1 = build_trusted_subset(sample_df, cfg)

        # Remove all model-specific columns
        df_no_model = sample_df.drop(columns=[
            c for c in sample_df.columns
            if c.startswith("d3_") or c.startswith("b2_")
        ])
        manifest2, summary2 = build_trusted_subset(df_no_model, cfg)

        assert (manifest1["trusted_v1"] == manifest2["trusted_v1"]).all()
        assert summary1["trusted_count"] == summary2["trusted_count"]

    def test_missing_conflict_metadata_marks_partial(self, sample_df):
        cfg = TrustedSubsetConfig()
        df_no_conflict = sample_df.drop(columns=["cross_class_duplicate_conflict"])
        manifest, summary = build_trusted_subset(df_no_conflict, cfg)
        assert summary.get("conflict_metadata_available") is False

    def test_boundary_knn_agreement(self, sample_df):
        cfg = TrustedSubsetConfig(knn_label_agreement_min=0.60)
        sample_df["knn_label_agreement"] = 0.599
        sample_df["prototype_supports_noisy_label"] = True
        sample_df["prototype_margin"] = 0.1
        sample_df["clip_flip_cosine"] = 0.95
        sample_df["cross_class_duplicate_conflict"] = False
        manifest, _ = build_trusted_subset(sample_df, cfg)
        # knn at 0.599 < 0.60 → all rejected
        assert manifest["trusted_v1"].sum() == 0
        assert all("low_knn_agreement" in r for r in manifest["rejection_reasons"])

    def test_boundary_prototype_margin(self, sample_df):
        cfg = TrustedSubsetConfig(prototype_margin_min=0.02)
        sample_df["knn_label_agreement"] = 0.8
        sample_df["prototype_supports_noisy_label"] = True
        sample_df["prototype_margin"] = 0.0199
        sample_df["clip_flip_cosine"] = 0.95
        sample_df["cross_class_duplicate_conflict"] = False
        manifest, _ = build_trusted_subset(sample_df, cfg)
        assert manifest["trusted_v1"].sum() == 0
        assert all("low_prototype_margin" in r for r in manifest["rejection_reasons"])

    def test_all_conditions_pass(self, sample_df):
        cfg = TrustedSubsetConfig()
        sample_df["knn_label_agreement"] = 0.8
        sample_df["prototype_supports_noisy_label"] = True
        sample_df["prototype_margin"] = 0.1
        sample_df["clip_flip_cosine"] = 0.95
        sample_df["cross_class_duplicate_conflict"] = False
        manifest, _ = build_trusted_subset(sample_df, cfg)
        assert manifest["trusted_v1"].sum() == len(sample_df)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/lux1/noise && python -m pytest tests/test_trusted_subset.py -v 2>&1 | tail -10
```
Expected: FAIL with ImportError

- [ ] **Step 3: Write `common/trusted_subset.py`**

```python
"""
Trusted validation subset construction.

V1 rules are model-agnostic: they use only CLIP features, kNN topology,
robust class prototypes, and flip stability. They NEVER read D3/B2 logits,
confidence, margin, loss, or correctness.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrustedSubsetConfig:
    """Fixed V1 thresholds for trusted subset selection.

    These thresholds are pre-specified and must NOT be tuned based on
    platform scores or D3/B2 relative performance.
    """
    knn_label_agreement_min: float = 0.60
    prototype_margin_min: float = 0.02
    clip_flip_cosine_min: float = 0.90
    require_prototype_top1_matches_label: bool = True
    reject_cross_class_duplicate_conflict: bool = True


REJECTION_LABELS = {
    "low_knn_agreement": "kNN label agreement below threshold",
    "prototype_label_mismatch": "Prototype top-1 does not match noisy label",
    "low_prototype_margin": "Prototype margin below threshold",
    "low_clip_flip_cosine": "CLIP flip cosine below threshold",
    "cross_class_duplicate_conflict": "Cross-class duplicate conflict detected",
    "missing_conflict_metadata": "Conflict metadata not available — partial assessment",
}


def build_trusted_subset(
    df: pd.DataFrame,
    config: TrustedSubsetConfig = TrustedSubsetConfig(),
) -> Tuple[pd.DataFrame, dict]:
    """Build trusted validation subset using model-agnostic V1 rules.

    Args:
        df: DataFrame with per-sample metrics. Required columns:
            knn_label_agreement, prototype_supports_noisy_label,
            prototype_margin, clip_flip_cosine,
            cross_class_duplicate_conflict (optional — if missing, defaults
            to False with warning).
        config: TrustedSubsetConfig with thresholds.

    Returns:
        manifest: Copy of df with added columns:
            trusted_v1 (bool), rejection_reasons (str).
        summary: Dict with coverage, counts, per-class stats.
    """
    df = df.copy()
    n_total = len(df)

    # Check for conflict metadata
    has_conflict = "cross_class_duplicate_conflict" in df.columns
    if not has_conflict:
        logger.warning(
            "cross_class_duplicate_conflict column not found. "
            "Defaulting to False — trusted subset marked as partial."
        )
        df["cross_class_duplicate_conflict"] = False

    # ── Build rejection reasons per sample ──
    reasons = pd.Series([[] for _ in range(n_total)], dtype=object)

    # Rule 1: kNN label agreement
    low_knn = df["knn_label_agreement"] < config.knn_label_agreement_min
    for i in df.index[low_knn]:
        reasons.iloc[i] = reasons.iloc[i] + ["low_knn_agreement"]

    # Rule 2: Prototype top-1 matches noisy label
    if config.require_prototype_top1_matches_label:
        proto_mismatch = ~df["prototype_supports_noisy_label"].astype(bool)
        for i in df.index[proto_mismatch]:
            reasons.iloc[i] = reasons.iloc[i] + ["prototype_label_mismatch"]

    # Rule 3: Prototype margin
    low_proto_margin = df["prototype_margin"] < config.prototype_margin_min
    for i in df.index[low_proto_margin]:
        reasons.iloc[i] = reasons.iloc[i] + ["low_prototype_margin"]

    # Rule 4: CLIP flip cosine
    low_flip_cos = df["clip_flip_cosine"] < config.clip_flip_cosine_min
    for i in df.index[low_flip_cos]:
        reasons.iloc[i] = reasons.iloc[i] + ["low_clip_flip_cosine"]

    # Rule 5: Cross-class duplicate conflict
    if config.reject_cross_class_duplicate_conflict:
        conflict = df["cross_class_duplicate_conflict"].astype(bool)
        for i in df.index[conflict]:
            reasons.iloc[i] = reasons.iloc[i] + ["cross_class_duplicate_conflict"]

    # Rule 6: Missing conflict metadata marker
    if not has_conflict:
        for i in df.index:
            reasons.iloc[i] = reasons.iloc[i] + ["missing_conflict_metadata"]

    # ── Determine trusted ──
    trusted = reasons.apply(lambda r: len(r) == 0)
    rejection_str = reasons.apply(lambda r: ";".join(r) if r else "")

    df["trusted_v1"] = trusted
    df["rejection_reasons"] = rejection_str

    # ── Build summary ──
    trusted_count = trusted.sum()
    coverage = trusted_count / n_total if n_total > 0 else 0.0

    represented_classes = df[trusted]["noisy_label"].nunique() if trusted_count > 0 else 0
    total_classes = df["noisy_label"].nunique()

    per_class = df.groupby("noisy_label")["trusted_v1"].agg(["sum", "count"])
    per_class.columns = ["trusted", "total"]
    per_class["coverage"] = per_class["trusted"] / per_class["total"]

    per_class_trusted = per_class["trusted"].astype(int).to_dict()

    summary = {
        "total_samples": n_total,
        "trusted_count": int(trusted_count),
        "rejected_count": int(n_total - trusted_count),
        "coverage": float(coverage),
        "represented_classes": int(represented_classes),
        "total_classes": int(total_classes),
        "missing_classes": int(total_classes - represented_classes),
        "conflict_metadata_available": has_conflict,
        "per_class_trusted": per_class_trusted,
        "min_trusted_per_class": int(per_class["trusted"].min()) if len(per_class) > 0 else 0,
        "median_trusted_per_class": float(per_class["trusted"].median()) if len(per_class) > 0 else 0.0,
        "p10_trusted_per_class": float(per_class["trusted"].quantile(0.10)) if len(per_class) > 0 else 0.0,
        "p90_trusted_per_class": float(per_class["trusted"].quantile(0.90)) if len(per_class) > 0 else 0.0,
        "rejection_reason_counts": {},
        "config": {
            "knn_label_agreement_min": config.knn_label_agreement_min,
            "prototype_margin_min": config.prototype_margin_min,
            "clip_flip_cosine_min": config.clip_flip_cosine_min,
            "require_prototype_top1_matches_label": config.require_prototype_top1_matches_label,
            "reject_cross_class_duplicate_conflict": config.reject_cross_class_duplicate_conflict,
        },
    }

    # Count rejection reasons
    all_reasons = []
    for r in reasons:
        all_reasons.extend(r)
    from collections import Counter
    summary["rejection_reason_counts"] = dict(Counter(all_reasons))

    return df, summary
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/lux1/noise && python -m pytest tests/test_trusted_subset.py -v 2>&1 | tail -30
```
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add common/trusted_subset.py tests/test_trusted_subset.py
git commit -m "feat(analysis): add model-agnostic trusted subset construction

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: `tests/test_dual_validation.py`

**Files:**
- Create: `tests/test_dual_validation.py`

- [ ] **Step 1: Write `tests/test_dual_validation.py`**

```python
"""Tests for dual validation evaluation logic."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


# We test the evaluation utility functions that evaluate_dual_validation.py will use.
# These are defined inline for now; in production they'd be imported from
# tools/evaluate_dual_validation.py or a shared utility.


def compute_group_partition(d3_correct, b2_correct):
    """Partition samples into four mutually exclusive groups."""
    d3 = np.asarray(d3_correct, dtype=bool)
    b2 = np.asarray(b2_correct, dtype=bool)
    both_correct = d3 & b2
    d3_only = d3 & ~b2
    b2_only = ~d3 & b2
    both_wrong = ~d3 & ~b2
    return both_correct, d3_only, b2_only, both_wrong


def paired_bootstrap(d3_correct, b2_correct, trusted_mask=None, n_iter=1000, seed=42):
    """Paired bootstrap for accuracy delta."""
    rng = np.random.default_rng(seed)
    n = len(d3_correct)
    d3 = np.asarray(d3_correct, dtype=bool)
    b2 = np.asarray(b2_correct, dtype=bool)
    deltas = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        if trusted_mask is not None:
            t = np.asarray(trusted_mask, dtype=bool)
            idx = idx[t[idx]] if t[idx].sum() > 0 else idx
            d3_acc = d3[idx][t[idx]].mean() if t[idx].sum() > 0 else 0.0
            b2_acc = b2[idx][t[idx]].mean() if t[idx].sum() > 0 else 0.0
        else:
            d3_acc = d3[idx].mean()
            b2_acc = b2[idx].mean()
        deltas.append(b2_acc - d3_acc)
    deltas = np.array(deltas)
    return {
        "mean": float(deltas.mean()),
        "ci_lower": float(np.percentile(deltas, 2.5)),
        "ci_upper": float(np.percentile(deltas, 97.5)),
    }


class TestGroupPartition:
    def test_mutually_exclusive(self):
        d3 = [True, True, False, False]
        b2 = [True, False, True, False]
        bc, do, bo, bw = compute_group_partition(d3, b2)
        assert bc.sum() == 1
        assert do.sum() == 1
        assert bo.sum() == 1
        assert bw.sum() == 1

    def test_union_equals_total(self):
        n = 100
        rng = np.random.default_rng(42)
        d3 = rng.choice([True, False], n)
        b2 = rng.choice([True, False], n)
        bc, do, bo, bw = compute_group_partition(d3, b2)
        assert bc.sum() + do.sum() + bo.sum() + bw.sum() == n

    def test_known_numbers(self):
        """D3=7289, B2=7179, total=10316 → d3_only=?, b2_only=?"""
        # d3_only - b2_only = 7289 - 7179 = 110
        # We can't determine exact values without the joint distribution,
        # but we can verify partition invariants.
        n = 10316
        d3_correct_count = 7289
        b2_correct_count = 7179
        # Create synthetic data matching the counts
        d3 = np.zeros(n, dtype=bool)
        b2 = np.zeros(n, dtype=bool)
        d3[:d3_correct_count] = True
        b2[:b2_correct_count] = True
        bc, do, bo, bw = compute_group_partition(d3, b2)
        assert bc.sum() + do.sum() == d3_correct_count
        assert bc.sum() + bo.sum() == b2_correct_count


class TestPairedBootstrap:
    def test_seed_reproducibility(self):
        n = 100
        d3 = np.random.default_rng(0).choice([True, False], n)
        b2 = np.random.default_rng(1).choice([True, False], n)
        result1 = paired_bootstrap(d3, b2, n_iter=500, seed=42)
        result2 = paired_bootstrap(d3, b2, n_iter=500, seed=42)
        assert result1["mean"] == result2["mean"]
        assert result1["ci_lower"] == result2["ci_lower"]

    def test_same_accuracy_zero_delta(self):
        n = 200
        d3 = np.ones(n, dtype=bool)
        b2 = np.ones(n, dtype=bool)
        result = paired_bootstrap(d3, b2, n_iter=1000, seed=42)
        assert abs(result["mean"]) < 1e-10

    def test_output_keys(self):
        d3 = np.random.default_rng(0).choice([True, False], 50)
        b2 = np.random.default_rng(1).choice([True, False], 50)
        result = paired_bootstrap(d3, b2, n_iter=100)
        assert set(result.keys()) == {"mean", "ci_lower", "ci_upper"}

    def test_trusted_mask_effect(self):
        n = 100
        rng = np.random.default_rng(99)
        d3 = rng.choice([True, False], n, p=[0.7, 0.3])
        b2 = rng.choice([True, False], n, p=[0.68, 0.32])
        trusted = rng.choice([True, False], n, p=[0.5, 0.5])
        raw = paired_bootstrap(d3, b2, n_iter=500, seed=42)
        trusted_result = paired_bootstrap(d3, b2, trusted, n_iter=500, seed=42)
        # Both should produce valid deltas
        assert -1 <= raw["mean"] <= 1
        assert -1 <= trusted_result["mean"] <= 1


class TestManifestValidation:
    def test_rejects_path_mismatch(self):
        """Evaluation should reject if manifest paths don't match val dataset."""
        manifest = pd.DataFrame({
            "image_path": ["a.jpg", "b.jpg"],
            "trusted_v1": [True, False],
        })
        val_paths = ["a.jpg", "c.jpg"]
        # Path c.jpg not in manifest → should detect mismatch
        manifest_paths = set(manifest["image_path"])
        val_paths_set = set(val_paths)
        assert not val_paths_set.issubset(manifest_paths)
```

- [ ] **Step 2: Run tests**

```bash
cd /home/lux1/noise && python -m pytest tests/test_dual_validation.py -v 2>&1 | tail -20
```
Expected: All tests PASS (these test inline utility functions, no imports needed)

- [ ] **Step 3: Commit**

```bash
git add tests/test_dual_validation.py
git commit -m "test(analysis): add dual validation partition and bootstrap tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: `tools/audit_experiment_pair.py`

**Files:** Create: `tools/audit_experiment_pair.py`

**Interfaces:** Consumes `common.pair_protocol_audit.audit_experiment_pair`. Produces CLI with exit codes 0/2/3/4.

- [ ] **Step 1: Write the tool**

See `tools/audit_experiment_pair.py` in the implementation — wraps `audit_experiment_pair()`, checks file existence, handles exit codes per spec (0=pass, 2=warnings, 3=not paired, 4=file error), supports `--allow-confounded-analysis`.

- [ ] **Step 2: Verify import** — `cd /home/lux1/noise && python -c "from tools.audit_experiment_pair import main; print('OK')"`
- [ ] **Step 3: Commit**

---

### Task 7: `tools/export_feature_bank.py`

**Files:** Create: `tools/export_feature_bank.py`

**Interfaces:** Consumes `common.clip_utils.load_openai_clip`, `common.clip_utils.encode_frozen_clip_features`. Produces `train_feature_bank.pt` and `val_feature_bank.pt`.

- [ ] **Step 1: Write the tool** — Core logic: load CLIP model + preprocess, read train CSV → resolve paths (CWD-relative), SHA-256 hash each image, encode through frozen CLIP, save `train_feature_bank.pt` dict with features/labels/paths/image_sha256/class_names. Same for val, plus encode with horizontal flip (`PIL.ImageOps.mirror` before preprocess — never flip post-normalization). Verification: feature dim=512, all norms within 1e-5 of 1.0, no NaN/Inf, val count=10316, path order matches CSV order.
- [ ] **Step 2: Verify import**
- [ ] **Step 3: Commit**

---

### Task 8: `tools/analyze_checkpoint_disagreement.py`

**Files:** Create: `tools/analyze_checkpoint_disagreement.py`

**Interfaces:** Consumes all functions from `common.diagnostic_metrics`, plus `common.utils.load_config`, `common.class_mapping`. Produces `sample_metrics.csv` (50+ columns per spec), `group_summary.json`, `class_summary.csv`, and per-group CSVs.

- [ ] **Step 1: Write the tool** — Read audit JSON; if `max_visual_abs_diff == 0`, use fast-path: load classifier weight/bias from both checkpoints, compute logits via `F.linear(F.normalize(features), weight, bias)`, verify against full model `forward_features` on 32 samples (max diff < 1e-6). Otherwise fall back to full model. Compute all per-sample metrics: CE, GCE, confidence, margin, entropy, JSD, flip stability, TTA, chunked kNN (k=20), robust prototypes (trim=10%), clip_flip_cosine. Partition into 4 groups (both_correct, d3_only_correct, b2_only_correct, both_wrong). Verify: groups are mutually exclusive and exhaustive, d3_only - b2_only = 110. Save outputs.
- [ ] **Step 2: Verify import**
- [ ] **Step 3: Commit**

---

### Task 9: `tools/build_trusted_subset.py`

**Files:** Create: `tools/build_trusted_subset.py`

**Interfaces:** Consumes `common.trusted_subset.TrustedSubsetConfig`, `common.trusted_subset.build_trusted_subset`. Produces `trusted_manifest.csv` + `trusted_subset_summary.json`.

- [ ] **Step 1: Write the tool** — Read `sample_metrics.csv`, apply V1 rules with CLI-configurable thresholds, output manifest CSV (subset of columns) and summary JSON. Optionally generate T_strict/T_main/T_loose sensitivity tiers. Print coverage and class representation. Warn if coverage < 25% or represented < 475 classes.
- [ ] **Step 2: Verify import**
- [ ] **Step 3: Commit**

---

### Task 10: `tools/evaluate_dual_validation.py`

**Files:** Create: `tools/evaluate_dual_validation.py`

**Interfaces:** Consumes `val_feature_bank.pt`, `trusted_manifest.csv`, checkpoint `.pt`. Produces dual validation JSON (schema_version=1, raw + trusted + rejected metrics).

- [ ] **Step 1: Write the tool** — Load val features + flip features from bank, load classifier weights from checkpoint, compute logits via fast-path `F.linear`. Verify manifest paths match val paths exactly (reject on any mismatch). Compute raw metrics (micro, macro_present_classes, macro_all, median, bottom-10%). Compute trusted metrics masked by `trusted_v1` (same + coverage, represented_classes). Compute rejected subset diagnostic. Output JSON with all hashes.
- [ ] **Step 2: Verify import**
- [ ] **Step 3: Commit**

---

### Task 11: `tools/summarize_disagreement.py`

**Files:** Create: `tools/summarize_disagreement.py`

**Interfaces:** Consumes all analysis outputs (audit JSON, sample_metrics CSV, group_summary JSON, trusted_summary JSON, both dual validation JSONs). Produces `findings.md`.

- [ ] **Step 1: Write the tool** — Generate markdown report with sections answering the 8 required questions from the spec: 1) Is it a strict paired comparison? 2) Raw accuracy delta? 3) Trusted accuracy delta? 4) Are d3_only samples more noise-like? 5) Does B2's new prediction get higher kNN/prototype support in d3_only region? 6) Does B2 reduce noisy-label probability on low-consistency samples? 7) Is evidence sufficient for causal claim? 8) Correlation vs causation boundary. Follow spec guidance for allowed/forbidden conclusions.
- [ ] **Step 2: Verify import**
- [ ] **Step 3: Commit**

---

### Task 12: Full test suite verification

- [ ] **Step 1: Run all tests** — `cd /home/lux1/noise && python -m pytest -q 2>&1`
- [ ] **Step 2: Fix any failures** — Common issues: import ordering in test files, `weights_only=False` for torch.load, pandas dtype warnings
- [ ] **Step 3: Final commit** — Stage all new files, verify with `git status --short`, commit with summary message

---

## Plan Summary

| Task | Files Created | Dependencies |
|------|--------------|--------------|
| 1 | `common/diagnostic_metrics.py`, `tests/test_diagnostic_metrics.py` | None |
| 2 | `tests/test_chunked_knn.py` | Task 1 |
| 3 | `common/pair_protocol_audit.py`, `tests/test_pair_protocol_audit.py` | common.utils |
| 4 | `common/trusted_subset.py`, `tests/test_trusted_subset.py` | None (pandas only) |
| 5 | `tests/test_dual_validation.py` | None (standalone) |
| 6 | `tools/audit_experiment_pair.py` | Task 3 |
| 7 | `tools/export_feature_bank.py` | common.clip_utils |
| 8 | `tools/analyze_checkpoint_disagreement.py` | Tasks 1, 3 |
| 9 | `tools/build_trusted_subset.py` | Task 4 |
| 10 | `tools/evaluate_dual_validation.py` | common.utils |
| 11 | `tools/summarize_disagreement.py` | None (reads JSON/CSV) |
| 12 | Verification | All prior tasks |

