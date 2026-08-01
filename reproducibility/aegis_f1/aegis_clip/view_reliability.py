"""Protocol and cache validation for CVRG view-reliability inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F


VIEW_ORDER: tuple[str, str, str, str] = (
    "original_global",
    "original_local",
    "flipped_global",
    "flipped_local",
)
BASE_VIEW_WEIGHTS = torch.tensor(
    [0.30, 0.20, 0.30, 0.20], dtype=torch.float32
)
CVRG_NUM_CLASSES = 500
CVRG_CACHE_FORMAT_VERSION = 1
CVRG_FEATURE_SCHEMA_VERSION = 1
_TRUST_KEYS = {
    "labels",
    "clean_probability",
    "pseudo_label",
    "pseudo_labels",
    "correction_alpha",
}
_REQUIRED_VALIDATION_KEYS = {
    "labels",
    "clean_probability",
    "pseudo_label",
    "correction_alpha",
}


@dataclass(frozen=True)
class CVRGProtocol:
    crop_size: int = 160
    top_k: int = 5
    temperature: float = 1.0
    local_weight: float = 0.4
    flip_weight: float = 0.5
    prior_alignment_strength: float = 1.0


def _require_tensor(
    payload: Mapping[str, Any],
    key: str,
    *,
    rank: int,
    sample_count: int | None = None,
) -> torch.Tensor:
    if key not in payload:
        raise ValueError(f"cache missing required field: {key}")
    value = torch.as_tensor(payload[key])
    if value.ndim != rank:
        raise ValueError(f"{key} must have rank {rank}, got {value.ndim}")
    if sample_count is not None and value.shape[0] != sample_count:
        raise ValueError(f"{key} is not aligned with view_logits")
    if not torch.isfinite(value.float()).all():
        raise ValueError(f"{key} must contain only finite values")
    return value


def _validate_protocol(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("cache protocol must be a mapping")
    expected = asdict(CVRGProtocol())
    actual = {key: value.get(key) for key in expected}
    if actual != expected or set(value) != set(expected):
        raise ValueError(f"cache protocol mismatch: expected {expected}")


def validate_cvrg_cache(
    payload: Mapping[str, Any],
    *,
    require_labels: bool,
    expected_checkpoint_sha256: str | None = None,
) -> int:
    """Validate a CVRG cache and return its aligned sample count.

    Validation caches carry labels and trust metadata. Test caches must be
    explicitly label-free so that a cache cannot accidentally leak evaluation
    targets into the frozen gate or submission path.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("CVRG cache must be a mapping")
    if payload.get("format_version") != CVRG_CACHE_FORMAT_VERSION:
        raise ValueError("unsupported CVRG cache format version")
    split = payload.get("split")
    if split not in {"validation", "test"}:
        raise ValueError("cache split must be validation or test")
    if tuple(payload.get("view_order", ())) != VIEW_ORDER:
        raise ValueError(f"cache view_order must be exactly {VIEW_ORDER}")
    checkpoint_sha256 = payload.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or not checkpoint_sha256:
        raise ValueError("cache checkpoint_sha256 must be a non-empty string")
    if expected_checkpoint_sha256 is not None and (
        checkpoint_sha256 != expected_checkpoint_sha256
    ):
        raise ValueError("checkpoint SHA-256 mismatch")
    split_sha256 = payload.get("split_sha256")
    if not isinstance(split_sha256, str) or not split_sha256:
        raise ValueError("cache split_sha256 must be a non-empty string")
    _validate_protocol(payload.get("protocol"))

    logits = _require_tensor(payload, "view_logits", rank=3)
    sample_count, view_count, class_count = logits.shape
    if view_count != len(VIEW_ORDER) or class_count != CVRG_NUM_CLASSES:
        raise ValueError(
            f"view_logits must have shape [N,4,{CVRG_NUM_CLASSES}]"
        )
    features = _require_tensor(
        payload, "view_features", rank=3, sample_count=sample_count
    )
    if features.shape[1] != len(VIEW_ORDER) or features.shape[2] <= 0:
        raise ValueError("view_features must have shape [N,4,D] with D>0")
    feature_norm = features.float().norm(dim=-1)
    if not torch.allclose(
        feature_norm,
        torch.ones_like(feature_norm),
        atol=2.0e-3,
        rtol=2.0e-3,
    ):
        raise ValueError("view_features must be normalized")
    attention = _require_tensor(
        payload, "orientation_attention", rank=4, sample_count=sample_count
    )
    if attention.shape[1] != 2 or attention.shape[-1] <= 0:
        raise ValueError("orientation_attention must have shape [N,2,H,P]")
    boxes = _require_tensor(
        payload, "crop_boxes", rank=3, sample_count=sample_count
    )
    if boxes.shape[1:] != (2, 4):
        raise ValueError("crop_boxes must have shape [N,2,4]")

    paths = payload.get("paths")
    if not isinstance(paths, (list, tuple)):
        raise ValueError("cache paths must be a list or tuple")
    if len(paths) != sample_count:
        raise ValueError("cache paths are not aligned with view_logits")
    if any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("cache paths must contain non-empty strings")
    if len(set(paths)) != len(paths):
        raise ValueError("cache paths must be unique")

    present_trust_keys = _TRUST_KEYS.intersection(payload)
    if require_labels:
        if split != "validation":
            raise ValueError("labels are only valid for validation caches")
        missing = _REQUIRED_VALIDATION_KEYS - present_trust_keys
        if missing:
            raise ValueError(f"validation cache missing labels/trust fields: {sorted(missing)}")
        labels = _require_tensor(
            payload, "labels", rank=1, sample_count=sample_count
        ).long()
        if (labels < 0).any() or (labels >= CVRG_NUM_CLASSES).any():
            raise ValueError("labels are outside the declared class range")
        clean = _require_tensor(
            payload, "clean_probability", rank=1, sample_count=sample_count
        ).float()
        correction = _require_tensor(
            payload, "correction_alpha", rank=1, sample_count=sample_count
        ).float()
        pseudo = _require_tensor(
            payload, "pseudo_label", rank=1, sample_count=sample_count
        ).long()
        if (clean < 0.0).any() or (clean > 1.0).any():
            raise ValueError("clean_probability must be in [0,1]")
        if (correction < 0.0).any() or (correction > 1.0).any():
            raise ValueError("correction_alpha must be in [0,1]")
        if (pseudo < -1).any() or (pseudo >= CVRG_NUM_CLASSES).any():
            raise ValueError("pseudo_label is outside the declared class range")
    elif present_trust_keys:
        raise ValueError("test caches must be label-free")

    return sample_count



_PAIR_NAMES = ("01", "02", "03", "12", "13", "23")
RELIABILITY_FEATURE_NAMES = (
    "single.max_probability", "single.normalized_entropy", "single.top1_top2_margin",
    "single.top5_probability_mass", "single.energy", "single.logit_l2",
    *(f"pair.{pair}.{metric}" for pair in _PAIR_NAMES for metric in ("js_divergence", "top1_equal", "top5_jaccard")),
    "agreement.top1_fraction",
    "cosine.original_global_local", "cosine.flipped_global_local",
    "cosine.global_original_flipped", "cosine.local_original_flipped",
    "attention.current_entropy", "attention.current_top5_mass",
    "attention.current_crop_center_x", "attention.current_crop_center_y",
    "attention.border_contact", "attention.flip_mapped_center_distance",
    "view.is_original_global", "view.is_original_local", "view.is_flipped_global", "view.is_flipped_local",
)

@dataclass(frozen=True)
class FrozenReliabilityGate:
    feature_names: tuple[str, ...]
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    coefficient: torch.Tensor
    intercept: float
    regularization_c: float
    checkpoint_sha256: str
    validation_cache_sha256: str
    feature_schema_sha256: str
    protocol: CVRGProtocol

def _validate_view_inputs(logits, visual, attention, boxes):
    if logits.ndim != 3 or logits.shape[1:] != (4, CVRG_NUM_CLASSES):
        raise ValueError("view_logits must have shape [N,4,500]")
    n = logits.shape[0]
    if visual.ndim != 3 or visual.shape[:2] != (n, 4) or visual.shape[2] <= 0:
        raise ValueError("view_features must have shape [N,4,D]")
    if attention.ndim != 4 or attention.shape[:2] != (n, 2) or attention.shape[2] <= 0 or attention.shape[3] <= 0:
        raise ValueError("orientation_attention must have shape [N,2,H,P]")
    if boxes.shape != (n, 2, 4):
        raise ValueError("crop_boxes must have shape [N,2,4]")
    for name, value in (("view_logits", logits), ("view_features", visual), ("orientation_attention", attention), ("crop_boxes", boxes)):
        if not torch.isfinite(value.float()).all():
            raise ValueError(f"{name} must contain only finite values")
    return n, int(attention.shape[-1])

def extract_reliability_features(view_logits, view_features, orientation_attention, crop_boxes, *, image_size=224, top_k=5):
    n, patches = _validate_view_inputs(view_logits, view_features, orientation_attention, crop_boxes)
    logits = view_logits.float()
    probs = F.softmax(logits, dim=-1).clamp_min(1e-12)
    vals, inds = probs.topk(top_k, dim=-1)
    top1 = probs.argmax(-1)
    single = torch.stack((probs.max(-1).values, -(probs * probs.log()).sum(-1) / math.log(CVRG_NUM_CLASSES),
                          vals[..., 0] - vals[..., 1], vals.sum(-1), -torch.logsumexp(logits, -1), logits.norm(dim=-1)), -1)
    pairwise = []
    for left, right in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
        p, q = probs[:, left], probs[:, right]
        m = (p + q) * 0.5
        js = 0.5 * ((p * (p / m).log()).sum(-1) + (q * (q / m).log()).sum(-1))
        equal = (top1[:, left] == top1[:, right]).float()
        a, b = inds[:, left], inds[:, right]
        inter = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).sum(-1).float()
        pairwise.extend((js, equal, inter / (2 * top_k - inter).clamp_min(1.0)))
    pairwise = torch.stack(pairwise, -1)[:, None, :].expand(-1, 4, -1)
    agreement = torch.stack([(top1 == top1[:, i:i+1]).sum(-1) for i in range(4)], -1).max(-1).values.float().div(4)[:,None,None].expand(-1,4,1)
    vf = F.normalize(view_features.float(), dim=-1)
    cosine = torch.stack(((vf[:,0]*vf[:,1]).sum(-1),(vf[:,2]*vf[:,3]).sum(-1),(vf[:,0]*vf[:,2]).sum(-1),(vf[:,1]*vf[:,3]).sum(-1)), -1)[:,None,:].expand(-1,4,-1)
    side = math.isqrt(patches)
    if side * side != patches:
        raise ValueError("attention patch count must form a square grid")
    att = orientation_attention.float().mean(2)
    att = att / att.sum(-1, keepdim=True).clamp_min(1e-12)
    ent = -(att.clamp_min(1e-12) * att.clamp_min(1e-12).log()).sum(-1) / math.log(patches)
    mass = att.topk(min(top_k, patches), -1).values.sum(-1)
    rows = torch.arange(side, device=logits.device).repeat_interleave(side).float()
    cols = torch.arange(side, device=logits.device).repeat(side).float()
    cx = (att * ((cols + .5) / side)).sum(-1)
    cy = (att * ((rows + .5) / side)).sum(-1)
    ori = torch.tensor([0,0,1,1], device=logits.device)
    cb = crop_boxes.float()[:, ori]
    border = ((cb <= 0) | (cb >= float(image_size))).any(-1).float()
    oc = (crop_boxes.float()[:,0,:2] + crop_boxes.float()[:,0,2:])*.5 / image_size
    fc = (crop_boxes.float()[:,1,:2] + crop_boxes.float()[:,1,2:])*.5 / image_size
    fd = (oc - torch.stack((1-fc[:,0], fc[:,1]), -1)).norm(dim=-1)[:,None].expand(-1,4)
    attention = torch.stack((ent[:,ori], mass[:,ori], cx[:,ori], cy[:,ori], border, fd), -1)
    identity = torch.eye(4, device=logits.device)[None].expand(n,-1,-1)
    features = torch.cat((single, pairwise, agreement, cosine, attention, identity), -1)
    if features.shape[-1] != 39 or not torch.isfinite(features).all():
        raise RuntimeError("reliability feature extraction produced an invalid schema")
    return features, RELIABILITY_FEATURE_NAMES

def predict_view_reliability(features, gate):
    if features.ndim != 3 or features.shape[1] != 4 or features.shape[-1] != len(gate.feature_names):
        raise ValueError("features do not match frozen gate")
    mean = torch.as_tensor(gate.feature_mean, dtype=torch.float32, device=features.device)
    scale = torch.as_tensor(gate.feature_scale, dtype=torch.float32, device=features.device)
    coef = torch.as_tensor(gate.coefficient, dtype=torch.float32, device=features.device)
    if mean.numel() != features.shape[-1] or scale.numel() != features.shape[-1] or coef.numel() != features.shape[-1] or (scale <= 0).any():
        raise ValueError("frozen gate parameter dimensions are invalid")
    score = torch.einsum("nvf,f->nv", (features.float()-mean)/scale, coef) + float(gate.intercept)
    return torch.sigmoid(score).clamp(1e-4, 1-1e-4)

def compute_dynamic_view_weights(reliability, *, base_weights=BASE_VIEW_WEIGHTS):
    if reliability.ndim != 2 or reliability.shape[1] != 4:
        raise ValueError("reliability must have shape [N,4]")
    base = torch.as_tensor(base_weights, dtype=torch.float32, device=reliability.device)
    if base.shape != (4,) or (base <= 0).any() or not torch.isclose(base.sum(), torch.tensor(1., device=base.device)):
        raise ValueError("base_weights must be a positive simplex")
    r = reliability.float().clamp(1e-4, 1-1e-4)
    return F.softmax(torch.log(base)[None] + torch.log(r) - torch.log1p(-r), 1)

def fuse_dynamic_view_probabilities(view_logits, gate, features):
    if view_logits.ndim != 3 or view_logits.shape[1:] != (4, CVRG_NUM_CLASSES) or features.shape[:2] != view_logits.shape[:2]:
        raise ValueError("view_logits and features have incompatible shapes")
    if torch.count_nonzero(torch.as_tensor(gate.coefficient)) == 0 and float(gate.intercept) == 0:
        from .localization import fuse_global_local_flip_probabilities
        fused = fuse_global_local_flip_probabilities(view_logits[:,0],view_logits[:,1],view_logits[:,2],view_logits[:,3], local_weight=gate.protocol.local_weight, flip_weight=gate.protocol.flip_weight, temperature=gate.protocol.temperature)
        weights = BASE_VIEW_WEIGHTS.to(view_logits.device).expand(view_logits.shape[0],-1)
        return fused, weights, torch.full_like(weights, .5)
    reliability = predict_view_reliability(features, gate)
    weights = compute_dynamic_view_weights(reliability)
    probs = F.softmax(view_logits.float() / gate.protocol.temperature, -1)
    return (probs * weights[...,None]).sum(1).clamp_min(torch.finfo(probs.dtype).tiny).log(), weights, reliability


__all__ = [
    "BASE_VIEW_WEIGHTS",
    "CVRG_CACHE_FORMAT_VERSION",
    "CVRG_FEATURE_SCHEMA_VERSION",
    "CVRG_NUM_CLASSES",
    "CVRGProtocol",
    "FrozenReliabilityGate",
    "RELIABILITY_FEATURE_NAMES",
    "compute_dynamic_view_weights",
    "extract_reliability_features",
    "fuse_dynamic_view_probabilities",
    "predict_view_reliability",
    "VIEW_ORDER",
    "validate_cvrg_cache",
]
