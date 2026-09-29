"""v1 primitives for the team-designed noise-robust strategy.

No timm dependency or historical fitted assets: the visual tower comes directly
from the SHA-verified official OpenAI checkpoint. All functions also run on CPU.
"""
from __future__ import annotations

import contextlib
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from aegis_clip.model import AdditiveLowRankParametrization


def aligned_positions(positions: torch.Tensor, grid: int) -> torch.Tensor:
    if positions.ndim != 2 or grid < 1:
        raise ValueError("Expected 2-D positions and a positive patch grid")
    old = math.isqrt(positions.shape[0] - 1)
    if old * old != positions.shape[0] - 1:
        raise ValueError("Source positions must have a square patch grid")
    if old == grid:
        return positions.clone()
    patches = positions[1:].reshape(old, old, -1).permute(2, 0, 1)[None]
    patches = F.interpolate(patches.float(), (grid, grid), mode="bilinear",
                            align_corners=True, antialias=False)
    patches = patches[0].permute(1, 2, 0).reshape(grid * grid, -1)
    return torch.cat([positions[:1], patches.to(positions.dtype)])


class CosineHead(nn.Module):
    def __init__(self, dimension: int, classes: int, dropout: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, dimension))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, features):
        return self.logit_scale.exp().clamp(1, 100) * (
            F.normalize(self.dropout(features.float()), dim=-1)
            @ F.normalize(self.weight, dim=-1).t())


class V1Classifier(nn.Module):
    def __init__(self, visual: nn.Module, classes: int, *, image_size=448,
                 rank=32, alpha=64.0, blocks=12):
        super().__init__()
        self.visual = visual.float()
        for p in visual.parameters():
            p.requires_grad_(False)
        patch_size = int(visual.conv1.kernel_size[0])
        if image_size < patch_size or image_size % patch_size:
            raise ValueError("Image size must be divisible by the patch size")
        visual.positional_embedding = nn.Parameter(aligned_positions(
            visual.positional_embedding.detach(), image_size // patch_size), requires_grad=False)
        visual.input_resolution = int(image_size)
        residual_blocks = visual.transformer.resblocks
        if not 1 <= blocks <= len(residual_blocks):
            raise ValueError("Invalid number of LoRA blocks")
        self.adapted_modules = []
        for index in range(len(residual_blocks) - blocks, len(residual_blocks)):
            block = residual_blocks[index]
            # Apply the rank-r adapter to the complete combined QKV projection, including K.
            # Native MHA reads weights directly; wrappers around out_proj alone
            # would silently bypass the adapter. Parametrize the actual weights.
            modules = [(block.attn, "in_proj_weight", "attn.qkv"),
                       (block.attn.out_proj, "weight", "attn.out_proj"),
                       (block.mlp.c_fc, "weight", "mlp.c_fc"),
                       (block.mlp.c_proj, "weight", "mlp.c_proj")]
            for module, key, name in modules:
                weight = getattr(module, key)
                adapter = AdditiveLowRankParametrization(*weight.shape, rank, alpha)
                parametrize.register_parametrization(module, key, adapter)
                self.adapted_modules.append(f"blocks.{index}.{name}")
        dimension = int(visual.proj.shape[1])
        self.head = CosineHead(dimension, classes)

    def train(self, mode=True):
        super().train(mode)
        # CLIP has no dropout in the tower. Frozen weights stay frozen while
        # autograd follows the inserted LoRA parameters in the native forward.
        self.visual.eval()
        return self

    def forward(self, images):
        return self.head(self.visual(images.float()))


def build_classifier(official_checkpoint, classes, model_config, device="cpu"):
    import clip
    from clip.clip import _MODELS
    from aegis_clip.runtime import sha256_file
    if sha256_file(official_checkpoint) != _MODELS["ViT-B/32"].split("/")[-2]:
        raise ValueError("Official OpenAI ViT-B/32 checkpoint SHA256 mismatch")
    clip_model, _ = clip.load(str(official_checkpoint), device="cpu", jit=False)
    visual = clip_model.visual
    if (visual.conv1.kernel_size != (32, 32)
            or len(visual.transformer.resblocks) != 12
            or visual.class_embedding.numel() != 768
            or tuple(visual.proj.shape) != (768, 512)):
        raise ValueError("Unexpected official ViT-B/32 architecture")
    model = V1Classifier(visual, classes, image_size=model_config["image_size"],
                           rank=model_config["rank"], alpha=model_config["alpha"],
                           blocks=model_config["blocks"])
    del clip_model
    return model.to(device)


@torch.no_grad()
def knn_signals(query, query_labels, gallery, gallery_labels, classes, *,
                k=16, query_groups=None, gallery_groups=None, query_chunk=128,
                gallery_chunk=8192):
    """Similarity-weighted agreement and majority, with bounded memory.

    Matching content groups (including self) are excluded when group IDs are
    supplied. Unknown/unsupported neighbourhoods return agreement=0, class=-1.
    Only training samples may enter the gallery; the pipeline validates split isolation.
    """
    if k < 1 or len(gallery) < 1 or query_chunk < 1 or gallery_chunk < 1:
        raise ValueError("Invalid kNN dimensions")
    if (query_groups is None) != (gallery_groups is None):
        raise ValueError("Both sets of content group IDs are required")
    query, gallery = F.normalize(query.float(), dim=1), F.normalize(gallery.float(), dim=1)
    k = min(k, len(gallery))
    agreements, predictions = [], []
    for start in range(0, len(query), query_chunk):
        q = query[start:start + query_chunk]
        values = q.new_full((len(q), k), -torch.inf)
        indices = torch.zeros((len(q), k), device=q.device, dtype=torch.long)
        for gs in range(0, len(gallery), gallery_chunk):
            similarities = q @ gallery[gs:gs + gallery_chunk].t()
            if query_groups is not None:
                same = (query_groups[start:start + len(q), None]
                        == gallery_groups[None, gs:gs + similarities.shape[1]])
                similarities.masked_fill_(same, -torch.inf)
            local_values, local_indices = similarities.topk(min(k, similarities.shape[1]), dim=1)
            merged = torch.cat([values, local_values], dim=1)
            all_indices = torch.cat([indices, local_indices + gs], dim=1)
            values, chosen = merged.topk(k, dim=1)
            indices = all_indices.gather(1, chosen)
        weights = values.clamp_min(0)
        labels = gallery_labels[indices]
        votes = q.new_zeros((len(q), classes))
        votes.scatter_add_(1, labels, weights)
        mass = weights.sum(1)
        agreement = (weights * (labels == query_labels[start:start + len(q), None])).sum(1)
        agreements.append(agreement / mass.clamp_min(1e-8))
        predictions.append(torch.where(mass > 0, votes.argmax(1), -1))
    return torch.cat(agreements), torch.cat(predictions)


def class_top_keep(scores, labels, classes, ratio=0.90):
    if not 0 < ratio <= 1:
        raise ValueError("keep_ratio must be in (0,1]")
    keep = torch.zeros_like(labels, dtype=torch.bool)
    for c in range(classes):
        idx = torch.where(labels == c)[0]
        if len(idx):
            count = max(1, int(len(idx) * ratio))
            keep[idx[scores[idx].topk(count).indices]] = True
    return keep


@torch.no_grad()
def confident_keep(probs, labels):
    """Upstream's conservative CL-style threshold rule, not cleanlab package.

    Classes absent from training cannot acquire a zero threshold and claim
    unrelated samples. Samples clearing no threshold retain their given label.
    """
    classes = probs.shape[1]
    thresholds = probs.new_full((classes,), torch.inf)
    for c in range(classes):
        idx = labels == c
        if idx.any():
            thresholds[c] = probs[idx, c].mean()
    above = probs >= thresholds[None]
    confident = probs.masked_fill(~above, -1).argmax(1)
    keep = ~above.any(1) | (confident == labels)
    return keep, thresholds


@torch.no_grad()
def denoised_targets(probs, labels, agreement, knn_predictions, *, minimum_weight=.2,
                     pseudo_threshold=.7, pseudo_margin=.2, pseudo_soft_alpha=0.0):
    keep, thresholds = confident_keep(probs, labels)
    top = probs.topk(2, dim=1)
    prediction, confidence = top.indices[:, 0], top.values[:, 0]
    margin = (top.values[:, 0] - top.values[:, 1]).clamp_min(0)
    p_label = probs.gather(1, labels[:, None])[:, 0]
    reliability = agreement.clamp(0, 1) * p_label.sqrt() * margin.sqrt()
    weights = torch.where(keep, minimum_weight + (1 - minimum_weight) * reliability, 0)
    pseudo = (~keep & (prediction == knn_predictions)
              & (confidence >= pseudo_threshold) & (margin >= pseudo_margin))
    targets = torch.where(pseudo, prediction, labels)
    weights[pseudo] = confidence[pseudo] * margin[pseudo].sqrt()
    alpha = torch.where(pseudo, float(pseudo_soft_alpha), 0.0)
    return dict(labels=labels, targets=targets, weights=weights, original_alpha=alpha,
                kept=keep, pseudo=pseudo, agreement=agreement, thresholds=thresholds)


def target_probabilities(targets, original_labels, original_alpha, classes, smoothing):
    base = F.one_hot(targets, classes).float()
    original = F.one_hot(original_labels, classes).float()
    distribution = (1 - original_alpha[:, None]) * base + original_alpha[:, None] * original
    return (1 - smoothing) * distribution + smoothing / classes


def weighted_mixup_loss(logits, probabilities, weights, permutation, lam):
    """Weight both label contributions before mixing; zero-weight labels vanish."""
    weighted = weights[:, None] * probabilities
    mixed = lam * weighted + (1 - lam) * weighted[permutation]
    return -(mixed * logits.float().log_softmax(1)).sum() / weights.sum().clamp_min(1e-8)


def trainable_state(model, *, cpu=True):
    return {name: (p.detach().cpu() if cpu else p.detach()).clone()
            for name, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model, state):
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError("Trainable parameter keys differ from checkpoint")
    with torch.no_grad():
        for name, p in parameters.items():
            if p.shape != state[name].shape:
                raise ValueError(f"Trainable shape mismatch: {name}")
            p.copy_(state[name])


class WeightAverage:
    """EMA or arithmetic mean of trainable weights from a single trajectory."""
    def __init__(self, state, decay=None):
        self.state = {k: v.detach().float().clone() for k, v in state.items()}
        self.decay, self.count = decay, 0

    def update(self, state):
        if self.state.keys() != state.keys():
            raise ValueError("Cannot average differing architectures")
        self.count += 1
        alpha = 1 / self.count if self.decay is None else 1 - self.decay
        for name, value in state.items():
            self.state[name].lerp_(value.detach().to(self.state[name]), alpha)

    def payload(self):
        return dict(state=self.state, decay=self.decay, count=self.count)

    @classmethod
    def restore(cls, payload):
        obj = cls(payload["state"], payload["decay"])
        obj.count = payload["count"]
        return obj


@contextlib.contextmanager
def using_weights(model, state):
    saved = trainable_state(model)
    training = model.training
    load_trainable_state(model, state)
    model.eval()
    try:
        yield model
    finally:
        load_trainable_state(model, saved)
        model.train(training)


@torch.no_grad()
def fit_training_bias(logits, labels, *, iterations=200):
    """Uniform-prior bias fitted on labelled official-training-side predictions.

    Equal mass per observed class avoids imposing the long-tail sample prior.
    This function is never called on test logits by the inference pipeline.
    """
    logits = logits.float()
    classes = logits.shape[1]
    counts = torch.bincount(labels, minlength=classes)
    if len(logits) == 0 or (counts == 0).any():
        raise ValueError("Bias calibration needs every class on the training side")
    sample_mass = 1 / counts[labels].float()
    sample_mass /= sample_mass.sum()
    bias = logits.new_zeros(classes)
    for _ in range(iterations):
        mass = ((logits + bias).softmax(1) * sample_mass[:, None]).sum(0)
        bias -= (mass.clamp_min(1e-8) * classes).log()
        bias -= bias.mean()
    if not torch.isfinite(bias).all():
        raise ValueError("Nonfinite calibration bias")
    return bias.cpu()


def accuracy_report(logits, labels, train_counts):
    predictions = logits.argmax(1).cpu()
    labels = labels.cpu()
    classes = logits.shape[1]
    counts = torch.bincount(labels, minlength=classes)
    correct = torch.bincount(labels[predictions == labels], minlength=classes)
    supported = counts > 0
    per_class = correct.float() / counts.clamp_min(1)
    groups = {}
    for name, mask in [("tail", train_counts < 20),
                       ("middle", (train_counts >= 20) & (train_counts < 100)),
                       ("head", train_counts >= 100)]:
        selected = mask[labels]
        groups[name] = dict(samples=int(selected.sum()), correct=int(
            ((predictions == labels) & selected).sum()))
    return dict(micro=float((predictions == labels).float().mean()),
                macro=float(per_class[supported].mean()), samples=len(labels),
                correct=int(correct.sum()), covered_classes=int(supported.sum()),
                per_class_correct=correct.tolist(), per_class_samples=counts.tolist(), groups=groups)
