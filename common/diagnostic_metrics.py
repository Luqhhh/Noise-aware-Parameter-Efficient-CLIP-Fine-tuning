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
        q: GCE exponent in (0, 1]. q->0 approaches CE, q=1 is MAE.
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
