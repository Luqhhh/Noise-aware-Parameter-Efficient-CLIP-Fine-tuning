# Phase 3 A-INFRA Infrastructure Completion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill all remaining infrastructure gaps so B and C can work through public interfaces without touching train.py. The 4 missing weight providers, resolved_config.yaml, artifact_manifest.json, per-class/prediction CSVs, ScheduledLoss tests, and experiment registry updates.

**Architecture:** Each task is independent, touching focused files. Tasks 1–4 add the 4 missing sample weight providers to `common/sample_weighting.py`. Task 5 creates `common/resolved_config.py` and integrates it into train.py. Task 6 wires artifact_manifest.json generation. Task 7 adds per_class_metrics.csv + prediction_records.csv output + extends evaluate() to return per_class_counts. Task 8 registers w2 experiments. Task 9 adds ScheduledLoss unit tests.

**Tech Stack:** Python 3, PyTorch, pandas, pytest, PyYAML

## Global Constraints

- All new providers implement `BaseWeightProvider` (state_dict/load_state_dict for stateful ones)
- Missing samples → fail-closed (raise KeyError)
- Config schema already declares all 7 weight types — no schema changes needed
- All new code uses `from __future__ import annotations`
- No modifications to training loop structure beyond adding new hook points
- No new dependencies beyond what's already in the project

---

### Task 1: Prototype Weight Provider

**Files:**
- Modify: `common/sample_weighting.py:294` (add `PrototypeProvider` class + factory branch)

**Interfaces:**
- Consumes: `BaseWeightProvider` (existing ABC), prototype weights JSON at `manifest_path`
- Produces: `PrototypeProvider(manifest_path, min_weight=0.4, max_weight=1.0, classwise_percentile=True, normalize_by_weight_sum=True, missing_policy="error")`

Loads pre-computed prototype confidence weights from a JSON file (same format as `StaticManifestProvider`: `{image_path: {"weight": float, ...}}`). The weight formula `w_i = 0.4 + 0.6 * c_i` is pre-baked into the JSON. This provider applies clamping to `[min_weight, max_weight]` and logs prototype-specific diagnostics (margin stats if available).

- [ ] **Step 1: Add `PrototypeProvider` class**

In `common/sample_weighting.py`, after `StaticManifestProvider` (after line 124), insert:

```python
# ──────────────────────────────────────────────────────────────────────
# Prototype confidence weighting (stateless)
# ──────────────────────────────────────────────────────────────────────


class PrototypeProvider(BaseWeightProvider):
    """Loads pre-computed prototype confidence weights from a JSON manifest.

    The manifest maps image_path → {"weight": float, ...}.  The weight
    is expected to already encode the prototype confidence transformation
    (e.g. ``0.4 + 0.6 * c_i``).  This provider applies optional clamping
    to ``[min_weight, max_weight]`` and logs diagnostic statistics.
    """

    def __init__(
        self,
        manifest_path: str,
        min_weight: float = 0.4,
        max_weight: float = 1.0,
        classwise_percentile: bool = True,
        normalize_by_weight_sum: bool = True,
        missing_policy: str = "error",
    ):
        if min_weight < 0.0 or max_weight > 1.0 or min_weight >= max_weight:
            raise ValueError(
                f"Invalid weight range: min_weight={min_weight}, "
                f"max_weight={max_weight}"
            )

        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Prototype manifest not found: {path}")
        with open(path) as f:
            raw = json.load(f)

        self._weights: Dict[str, float] = {}
        margins = []
        own_sims = []
        for img_path, entry in raw.items():
            w = float(entry["weight"]) if isinstance(entry, dict) else float(entry)
            w = max(min_weight, min(max_weight, w))
            self._weights[str(img_path)] = w
            if isinstance(entry, dict):
                if "margin" in entry:
                    margins.append(float(entry["margin"]))
                if "own_similarity" in entry:
                    own_sims.append(float(entry["own_similarity"]))

        self._min_weight = min_weight
        self._max_weight = max_weight
        self._classwise_percentile = classwise_percentile
        self._normalize = normalize_by_weight_sum
        self._missing = missing_policy

        # Diagnostics
        w_vals = np.array(list(self._weights.values()))
        logger.info(
            "PrototypeProvider: %d weights loaded from %s | "
            "mean=%.4f median=%.4f min=%.4f max=%.4f | "
            "clamped to [%.2f, %.2f]",
            len(self._weights), manifest_path,
            float(w_vals.mean()), float(np.median(w_vals)),
            float(w_vals.min()), float(w_vals.max()),
            min_weight, max_weight,
        )
        if margins:
            logger.info(
                "PrototypeProvider: margin mean=%.4f median=%.4f (n=%d)",
                float(np.mean(margins)), float(np.median(margins)), len(margins),
            )

    def get_weights(self, sample_paths, labels, epoch, per_sample_loss=None):
        w_vals = []
        for p in sample_paths:
            entry = self._weights.get(p)
            if entry is None:
                if self._missing == "error":
                    raise KeyError(f"Prototype weight missing for: {p}")
                w_vals.append(1.0)
            else:
                w_vals.append(entry)
        return torch.tensor(w_vals, device=labels.device, dtype=torch.float32)
```

- [ ] **Step 2: Add factory branch**

In `build_weight_provider()`, replace the `# Future:` comment at line 294 with:

```python
    if sw_type == "prototype":
        return PrototypeProvider(
            manifest_path=sw["manifest_path"],
            min_weight=sw.get("min_weight", 0.4),
            max_weight=sw.get("max_weight", 1.0),
            classwise_percentile=sw.get("classwise_percentile", True),
            normalize_by_weight_sum=sw.get("normalize_by_weight_sum", True),
            missing_policy=sw.get("missing_weight_policy", "error"),
        )
```

- [ ] **Step 3: Verify import**

```bash
python3 -c "from common.sample_weighting import PrototypeProvider, build_weight_provider; print('OK')"
```
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add common/sample_weighting.py
git commit -m "feat: add PrototypeProvider for pre-computed prototype confidence weights

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: OOF Manifest Weight Provider

**Files:**
- Modify: `common/sample_weighting.py` (add `OOFManifestProvider` class + factory branch)

**Interfaces:**
- Consumes: `BaseWeightProvider`, `ManifestLoader` from `common.manifest_loader`
- Produces: `OOFManifestProvider(manifest_path, min_weight=0.3, max_weight=1.0, missing_policy="error")`

Loads OOF quality scores from a B-delivered CSV manifest via `ManifestLoader`. The `sample_weight` column becomes the per-sample training weight, clamped to `[min_weight, max_weight]`. Exposes `get_training_label()` for future relabel support (returns `original_label` for weight-only mode).

- [ ] **Step 1: Add `OOFManifestProvider` class**

In `common/sample_weighting.py`, after `PrototypeProvider`, add:

```python
# ──────────────────────────────────────────────────────────────────────
# OOF manifest weighting (B-delivered CSV)
# ──────────────────────────────────────────────────────────────────────


class OOFManifestProvider(BaseWeightProvider):
    """Loads per-sample weights from a B-delivered OOF manifest CSV.

    Uses ``ManifestLoader`` for fail-closed validation.  The manifest must
    have 100% coverage and all ``sample_weight`` values in [0, 1].
    """

    def __init__(
        self,
        manifest_path: str,
        min_weight: float = 0.3,
        max_weight: float = 1.0,
        missing_policy: str = "error",
    ):
        from common.manifest_loader import ManifestLoader

        if min_weight < 0.0 or max_weight > 1.0 or min_weight >= max_weight:
            raise ValueError(
                f"Invalid weight range: min_weight={min_weight}, "
                f"max_weight={max_weight}"
            )

        self._loader = ManifestLoader(manifest_path)
        df = self._loader.load()

        self._weights: Dict[str, float] = {}
        self._training_labels: Dict[str, int] = {}
        for _, row in df.iterrows():
            path = str(row["image_path"])
            w = float(row["sample_weight"])
            w = max(min_weight, min(max_weight, w))
            self._weights[path] = w
            # Weight-only mode: keep original label
            self._training_labels[path] = int(row["original_label"])

        self._min_weight = min_weight
        self._max_weight = max_weight
        self._missing = missing_policy
        self._manifest_sha256 = self._loader.sha256

        w_vals = np.array(list(self._weights.values()))
        logger.info(
            "OOFManifestProvider: %d samples | "
            "weight mean=%.4f median=%.4f | "
            "manifest_sha256=%s",
            len(self._weights),
            float(w_vals.mean()), float(np.median(w_vals)),
            self._manifest_sha256[:16],
        )

    def get_weights(self, sample_paths, labels, epoch, per_sample_loss=None):
        w_vals = []
        for p in sample_paths:
            entry = self._weights.get(p)
            if entry is None:
                if self._missing == "error":
                    raise KeyError(f"OOF weight missing for: {p}")
                w_vals.append(1.0)
            else:
                w_vals.append(entry)
        return torch.tensor(w_vals, device=labels.device, dtype=torch.float32)

    def get_training_label(self, image_path: str, original_label: int) -> int:
        """Return the training label for a sample.

        For weight-only manifests, returns *original_label* unchanged.
        """
        return self._training_labels.get(image_path, original_label)
```

- [ ] **Step 2: Add factory branch**

In `build_weight_provider()`, add:

```python
    if sw_type == "oof_manifest":
        return OOFManifestProvider(
            manifest_path=sw["manifest_path"],
            min_weight=sw.get("min_weight", 0.3),
            max_weight=sw.get("max_weight", 1.0),
            missing_policy=sw.get("missing_weight_policy", "error"),
        )
```

- [ ] **Step 3: Verify import**

```bash
python3 -c "from common.sample_weighting import OOFManifestProvider; print('OK')"
```
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add common/sample_weighting.py
git commit -m "feat: add OOFManifestProvider for B-delivered OOF quality weights

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Relabel Manifest Weight Provider

**Files:**
- Modify: `common/sample_weighting.py` (add `RelabelManifestProvider` class + factory branch)

**Interfaces:**
- Consumes: `BaseWeightProvider`, `ManifestLoader`
- Produces: `RelabelManifestProvider(manifest_path, min_weight=0.3, max_weight=1.0, hard_relabel=False, missing_policy="error")`

Like `OOFManifestProvider` but supports both `weight_only` and `hard_relabel` modes. When `hard_relabel=True`, `get_training_label()` returns the relabeled class; otherwise it returns the original label.

- [ ] **Step 1: Add `RelabelManifestProvider` class**

In `common/sample_weighting.py`, after `OOFManifestProvider`, add:

```python
# ──────────────────────────────────────────────────────────────────────
# Relabel manifest weighting (B-delivered CSV)
# ──────────────────────────────────────────────────────────────────────


class RelabelManifestProvider(BaseWeightProvider):
    """Loads per-sample weights from a B-delivered relabel manifest CSV.

    Supports both weight-only (original label kept, weight reduced for
    suspicious samples) and hard relabel (training_label replaces original).
    Uses ``ManifestLoader`` for fail-closed validation.
    """

    def __init__(
        self,
        manifest_path: str,
        min_weight: float = 0.3,
        max_weight: float = 1.0,
        hard_relabel: bool = False,
        missing_policy: str = "error",
    ):
        from common.manifest_loader import ManifestLoader

        if min_weight < 0.0 or max_weight > 1.0 or min_weight >= max_weight:
            raise ValueError(
                f"Invalid weight range: min_weight={min_weight}, "
                f"max_weight={max_weight}"
            )

        self._loader = ManifestLoader(manifest_path)
        df = self._loader.load()

        self._weights: Dict[str, float] = {}
        self._training_labels: Dict[str, int] = {}
        self._hard_relabel = hard_relabel
        n_relabeled = 0

        for _, row in df.iterrows():
            path = str(row["image_path"])
            w = float(row["sample_weight"])
            w = max(min_weight, min(max_weight, w))
            self._weights[path] = w

            if hard_relabel:
                self._training_labels[path] = int(row["training_label"])
            else:
                self._training_labels[path] = int(row["original_label"])

            if int(row["training_label"]) != int(row["original_label"]):
                n_relabeled += 1

        self._min_weight = min_weight
        self._max_weight = max_weight
        self._missing = missing_policy
        self._manifest_sha256 = self._loader.sha256

        w_vals = np.array(list(self._weights.values()))
        logger.info(
            "RelabelManifestProvider: %d samples | hard_relabel=%s | "
            "relabeled=%d (%.2f%%) | "
            "weight mean=%.4f median=%.4f | manifest_sha256=%s",
            len(self._weights), hard_relabel,
            n_relabeled,
            100.0 * n_relabeled / max(len(self._weights), 1),
            float(w_vals.mean()), float(np.median(w_vals)),
            self._manifest_sha256[:16],
        )

    def get_weights(self, sample_paths, labels, epoch, per_sample_loss=None):
        w_vals = []
        for p in sample_paths:
            entry = self._weights.get(p)
            if entry is None:
                if self._missing == "error":
                    raise KeyError(f"Relabel weight missing for: {p}")
                w_vals.append(1.0)
            else:
                w_vals.append(entry)
        return torch.tensor(w_vals, device=labels.device, dtype=torch.float32)

    def get_training_label(self, image_path: str, original_label: int) -> int:
        """Return the (possibly relabeled) training label for a sample."""
        return self._training_labels.get(image_path, original_label)
```

- [ ] **Step 2: Add factory branch**

In `build_weight_provider()`, add:

```python
    if sw_type == "relabel_manifest":
        return RelabelManifestProvider(
            manifest_path=sw["manifest_path"],
            min_weight=sw.get("min_weight", 0.3),
            max_weight=sw.get("max_weight", 1.0),
            hard_relabel=sw.get("hard_relabel", False),
            missing_policy=sw.get("missing_weight_policy", "error"),
        )
```

- [ ] **Step 3: Verify import**

```bash
python3 -c "from common.sample_weighting import RelabelManifestProvider; print('OK')"
```
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add common/sample_weighting.py
git commit -m "feat: add RelabelManifestProvider for B-delivered relabel manifests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Prototype + EMA Loss Hybrid Provider

**Files:**
- Modify: `common/sample_weighting.py` (add `PrototypeEMAProvider` class + factory branch)

**Interfaces:**
- Consumes: `BaseWeightProvider`, prototype weights JSON
- Produces: `PrototypeEMAProvider(num_samples, prototype_manifest_path, momentum=0.9, warmup_epochs=5, min_weight=0.4, max_weight=1.0, proto_weight=0.7, ema_weight=0.3)`

Hybrid: `c_i = 0.7 * c_i^prototype + 0.3 * c_i^ema-loss`, then `w_i = 0.4 + 0.6 * c_i`. Stateful — requires `init_sample_index()`, `state_dict()`, `load_state_dict()`.

- [ ] **Step 1: Add `PrototypeEMAProvider` class**

In `common/sample_weighting.py`, after `RelabelManifestProvider`, add:

```python
# ──────────────────────────────────────────────────────────────────────
# Prototype + EMA loss hybrid weighting (stateful)
# ──────────────────────────────────────────────────────────────────────


class PrototypeEMAProvider(BaseWeightProvider):
    """Hybrid: prototype confidence × EMA-loss confidence → per-sample weight.

    confidence_i = proto_weight * c_i^prototype + ema_weight * c_i^ema
    w_i = min_weight + (max_weight - min_weight) * confidence_i

    Stateful — EMA loss history is written to / restored from checkpoints.
    """

    def __init__(
        self,
        num_samples: int,
        prototype_manifest_path: str,
        momentum: float = 0.9,
        warmup_epochs: int = 5,
        ranking: str = "classwise",
        min_weight: float = 0.4,
        max_weight: float = 1.0,
        proto_weight: float = 0.7,
        ema_weight: float = 0.3,
        high_loss_fraction_epoch_6_15: float = 0.10,
        high_loss_fraction_epoch_16_plus: float = 0.20,
    ):
        if abs(proto_weight + ema_weight - 1.0) > 1e-9:
            raise ValueError(
                f"proto_weight + ema_weight must sum to 1.0, "
                f"got {proto_weight} + {ema_weight}"
            )
        if min_weight < 0.0 or max_weight > 1.0 or min_weight >= max_weight:
            raise ValueError(
                f"Invalid weight range: min_weight={min_weight}, "
                f"max_weight={max_weight}"
            )

        # Load prototype weights
        proto_path = Path(prototype_manifest_path)
        if not proto_path.exists():
            raise FileNotFoundError(
                f"Prototype manifest not found: {proto_path}"
            )
        with open(proto_path) as f:
            raw = json.load(f)

        self._proto_weights: Dict[str, float] = {}
        for img_path, entry in raw.items():
            w = float(entry["weight"]) if isinstance(entry, dict) else float(entry)
            self._proto_weights[str(img_path)] = w

        # EMA state
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        self.ranking = ranking
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.proto_weight = proto_weight
        self.ema_weight = ema_weight
        self.frac_6_15 = high_loss_fraction_epoch_6_15
        self.frac_16_plus = high_loss_fraction_epoch_16_plus

        self.register_buffer(
            "ema_loss", torch.full((num_samples,), float("nan"))
        )
        self._sample_to_idx: Dict[str, int] = {}
        self._label_of_idx: Optional[torch.Tensor] = None

        logger.info(
            "PrototypeEMAProvider: %d prototype weights loaded | "
            "momentum=%.2f warmup=%d proto_w=%.2f ema_w=%.2f",
            len(self._proto_weights), momentum, warmup_epochs,
            proto_weight, ema_weight,
        )

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def init_sample_index(self, sample_paths: list, labels: torch.Tensor):
        """Build path→index mapping (call once before training)."""
        if self._sample_to_idx:
            return
        for i, p in enumerate(sample_paths):
            if p in self._sample_to_idx:
                raise ValueError(f"Duplicate sample path: {p}")
            self._sample_to_idx[p] = i
        self._label_of_idx = labels.clone()
        logger.info(
            "PrototypeEMAProvider: initialised with %d samples",
            len(self._sample_to_idx),
        )

    def get_weights(self, sample_paths, labels, epoch, per_sample_loss=None):
        n = len(sample_paths)
        device = (
            per_sample_loss.device
            if per_sample_loss is not None
            else torch.device("cpu")
        )

        # Warmup: prototype-only weights
        if epoch <= self.warmup_epochs:
            w_vals = []
            for p in sample_paths:
                w_vals.append(self._proto_weights.get(p, 1.0))
            return torch.tensor(w_vals, device=device, dtype=torch.float32)

        # Update EMA
        if per_sample_loss is not None:
            indices = [self._sample_to_idx[p] for p in sample_paths]
            idx_t = torch.tensor(indices, dtype=torch.long)
            new_ema = torch.where(
                self.ema_loss[idx_t].isnan(),
                per_sample_loss.detach().cpu(),
                self.momentum * self.ema_loss[idx_t]
                + (1.0 - self.momentum) * per_sample_loss.detach().cpu(),
            )
            self.ema_loss[idx_t] = new_ema

        frac = self.frac_6_15 if epoch <= 15 else self.frac_16_plus
        weights = torch.ones(n)

        unique_labels = labels.unique()
        for c_label in unique_labels:
            c_mask = labels == c_label
            c_indices = c_mask.nonzero(as_tuple=True)[0]
            if len(c_indices) <= 1:
                continue

            # Prototype confidence: c = (w - 0.4) / 0.6
            c_proto = torch.tensor([
                (self._proto_weights.get(sample_paths[i.item()], 1.0)
                 - self.min_weight)
                / max(self.max_weight - self.min_weight, 1e-8)
                for i in c_indices
            ], dtype=torch.float32)
            c_proto = torch.clamp(c_proto, 0.0, 1.0)

            # EMA confidence: 1 - percentile_rank (lower loss = more confident)
            c_paths = [sample_paths[i.item()] for i in c_indices]
            c_ema_idx = torch.tensor(
                [self._sample_to_idx[p] for p in c_paths], dtype=torch.long
            )
            c_ema_raw = self.ema_loss[c_ema_idx].clone()
            c_ema_raw = torch.where(
                c_ema_raw.isnan(), torch.zeros_like(c_ema_raw), c_ema_raw
            )
            n_class = len(c_indices)
            _, sort_idx = torch.sort(c_ema_raw)
            ranks = torch.zeros(n_class)
            ranks[sort_idx] = torch.arange(n_class, dtype=torch.float32)
            c_ema = 1.0 - ranks / max(n_class - 1, 1)

            # Hybrid confidence → weight
            hybrid = self.proto_weight * c_proto + self.ema_weight * c_ema
            hybrid = torch.clamp(hybrid, 0.0, 1.0)
            c_weights = (
                self.min_weight
                + (self.max_weight - self.min_weight) * hybrid
            )
            weights[c_indices] = c_weights

        return weights.to(device)

    def state_dict(self) -> dict:
        return {
            "ema_loss": self.ema_loss.clone(),
            "sample_to_idx_keys": list(self._sample_to_idx.keys()),
            "sample_to_idx_values": list(self._sample_to_idx.values()),
            "label_of_idx": (
                self._label_of_idx.clone()
                if self._label_of_idx is not None
                else None
            ),
        }

    def load_state_dict(self, d: dict):
        self.ema_loss = d["ema_loss"]
        self._sample_to_idx = dict(
            zip(d["sample_to_idx_keys"], d["sample_to_idx_values"])
        )
        self._label_of_idx = d.get("label_of_idx")
```

- [ ] **Step 2: Add factory branch**

In `build_weight_provider()`, add:

```python
    if sw_type == "prototype_ema_hybrid":
        if num_train_samples <= 0:
            raise ValueError(
                "num_train_samples is required for "
                "prototype_ema_hybrid provider"
            )
        return PrototypeEMAProvider(
            num_samples=num_train_samples,
            prototype_manifest_path=sw["manifest_path"],
            momentum=sw.get("momentum", 0.9),
            warmup_epochs=sw.get("warmup_epochs", 5),
            ranking=sw.get("ranking", "classwise"),
            min_weight=sw.get("min_weight", 0.4),
            max_weight=sw.get("max_weight", 1.0),
            proto_weight=sw.get("proto_weight", 0.7),
            ema_weight=sw.get("ema_weight", 0.3),
        )
```

- [ ] **Step 3: Remove stale `# Future:` comment at line 294**

The comment `# Future: prototype, hybrid, oof_manifest, relabel_manifest` is now replaced by real branches. The final `raise ValueError(...)` fallthrough stays.

- [ ] **Step 4: Verify import**

```bash
python3 -c "from common.sample_weighting import PrototypeEMAProvider; print('OK')"
```
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add common/sample_weighting.py
git commit -m "feat: add PrototypeEMAProvider for hybrid prototype + EMA loss weighting

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Write resolved_config.yaml

**Files:**
- Create: `common/resolved_config.py`
- Modify: `experiments/baseline/train.py` (add import + call after save_config_snapshot)

**Interfaces:**
- Produces: `resolve_config(config: dict) -> dict` — deep-copies config with defaults filled
- Produces: `write_resolved_config(resolved: dict, output_dir: str) -> str`

The `resolved_config.yaml` has a deterministic filename (no timestamp), containing every config key with explicit defaults. This is the canonical config for the experiment.

- [ ] **Step 1: Create `common/resolved_config.py`**

```python
"""Resolved config builder — fills defaults and writes resolved_config.yaml.

Part of A-INFRA-1: Uniform Config Schema.  Every training experiment writes
a deterministic ``resolved_config.yaml`` with all defaults explicitly filled,
so downstream consumers (B's audit, C's hooks, submission tools) see the
complete effective configuration.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

# ── Per-section defaults ──────────────────────────────────────────────
# Keys not present in the user config are filled from this table.

DEFAULTS: Dict[str, Dict[str, Any]] = {
    "experiment": {
        "id": "unknown",
        "parent": None,
        "wave": None,
        "seed": 42,
        "head_type": "linear",
        "augmentation_preset": "a0",
        "mode": "dev",
    },
    "loss": {
        "name": "cross_entropy",
        "q": 0.7,
        "probability_epsilon": 1e-7,
    },
    "sample_weighting": {
        "type": "none",
        "min_weight": 0.4,
        "max_weight": 1.0,
        "normalize_by_weight_sum": True,
        "missing_weight_policy": "error",
    },
    "head_ema": {
        "enabled": False,
        "decay": 0.99,
        "warmup_epochs": 5,
        "selection_source": "raw",
    },
    "peft": {
        "type": "linear_head_only",
    },
    "teacher": {
        "enabled": False,
        "ema_decay": 0.999,
        "confidence_threshold": 0.8,
        "consistency_weight": 1.0,
        "ramp_epochs": 10,
    },
    "train": {
        "amp": False,
        "max_grad_norm": 1.0,
        "num_workers": 4,
        "pin_memory": True,
        "scheduler": "cosine",
    },
    "eval": {
        "batch_size": 256,
    },
    "cache": {
        "enabled": False,
    },
}


def resolve_config(config: dict) -> dict:
    """Return a deep copy of *config* with all defaults filled in.

    Does NOT mutate the input.  Nested sections are merged recursively:
    explicit user values override defaults.  Unknown sections are
    passed through unchanged.
    """
    resolved = copy.deepcopy(config)

    for section, defaults in DEFAULTS.items():
        if section not in resolved:
            resolved[section] = copy.deepcopy(defaults)
        else:
            for key, default_val in defaults.items():
                if key not in resolved[section]:
                    resolved[section][key] = copy.deepcopy(default_val)

    # Also propagate runtime-resolved values into experiment section
    exp = resolved.setdefault("experiment", {})
    exp.setdefault("id", resolved.get("experiment", {}).get("id", "unknown"))

    return resolved


def write_resolved_config(resolved: dict, output_dir: str) -> str:
    """Write ``resolved_config.yaml`` to *output_dir*.

    Returns the path to the written file.
    """
    out = Path(output_dir) / "resolved_config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(
            resolved, f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    logger.info("Resolved config written to %s", out)
    return str(out)
```

- [ ] **Step 2: Integrate into train.py**

In `experiments/baseline/train.py`, add import near other common imports (around line 46):

```python
from common.resolved_config import resolve_config, write_resolved_config
```

After `save_config_snapshot(config, str(save_dir))` (currently line 1446), add:

```python
    # Write resolved config with explicit defaults (A-INFRA-1)
    resolved = resolve_config(config)
    write_resolved_config(resolved, str(save_dir))
```

This places `resolved` in `main()` scope for use by Task 6.

- [ ] **Step 3: Verify import**

```bash
python3 -c "from common.resolved_config import resolve_config, write_resolved_config; print('OK')"
```
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add common/resolved_config.py experiments/baseline/train.py
git commit -m "feat: write resolved_config.yaml with explicit defaults to output dir

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Generate artifact_manifest.json

**Files:**
- Modify: `experiments/baseline/train.py` (add manifest generation after eval_results)

**Interfaces:**
- Consumes: `build_artifact_manifest()`, `write_artifact_manifest()` from `common.artifact_manifest` (existing)
- Consumes: `resolved` dict from Task 5 (in `main()` scope)

- [ ] **Step 1: Add import**

In `experiments/baseline/train.py`, add near other common imports:

```python
from common.artifact_manifest import build_artifact_manifest, write_artifact_manifest
```

- [ ] **Step 2: Add manifest generation**

After `train_logger.info(f"Eval results saved to: {eval_path}")` (currently line 1857), still inside the `if mode in ("dev", "confirm"):` block, add:

```python
        # ── Write artifact manifest (A-INFRA-9) ──
        best_ckpt_path = str(save_dir / "best.pt")
        train_csv_path = str(Path(split_dir) / "train.csv")
        val_csv_path = str(Path(split_dir) / "val.csv")

        manifest = build_artifact_manifest(
            experiment_id=experiment_id,
            parent_experiment=config.get("experiment", {}).get("parent"),
            config=resolved,
            checkpoint_path=best_ckpt_path,
            train_csv=train_csv_path,
            val_csv=val_csv_path,
            extra={
                "best_val_acc": float(best_val_acc),
                "best_raw_acc": float(best_raw_acc),
                "best_ema_acc": (
                    float(best_ema_acc) if ema_enabled else None
                ),
                "best_epoch": dev_best_epoch,
                "sample_weighting_type": config.get(
                    "sample_weighting", {}
                ).get("type", "none"),
            },
        )
        artifact_path = write_artifact_manifest(manifest, str(save_dir))
        train_logger.info(
            "Artifact manifest saved to: %s", artifact_path
        )
```

- [ ] **Step 3: Verify import**

```bash
python3 -c "from experiments.baseline.train import main; print('import OK')"
```
Expected: import OK

- [ ] **Step 4: Commit**

```bash
git add experiments/baseline/train.py
git commit -m "feat: generate artifact_manifest.json after training completion

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Generate per_class_metrics.csv and prediction_records.csv

**Files:**
- Modify: `experiments/baseline/evaluate.py` (add `per_class_counts` to return dict)
- Modify: `experiments/baseline/train.py` (add CSV writing after eval_results)

**Interfaces:**
- Consumes: `per_class_results` from `evaluate_full()` (already computed post-training)
- Produces: `per_class_metrics.csv`, `prediction_records.csv` in save_dir

- [ ] **Step 1: Extend `evaluate()` in evaluate.py to return `per_class_counts`**

In `experiments/baseline/evaluate.py`, in the `evaluate()` function return dict (line 175-185), add `per_class_counts`:

```python
    return {
        "loss": avg_loss,
        "accuracy": micro_acc,
        "macro_accuracy": macro_acc,
        "per_class_accuracy": per_class_acc.cpu().tolist(),
        "per_class_counts": total_per_class.cpu().tolist(),
        "median_per_class_accuracy": median_per_class,
        "bottom_10_percent_accuracy": bottom_10_percent_acc,
        "micro_macro_gap": micro_macro_gap,
        "total_samples": total,
        "correct_samples": correct,
    }
```

(The only change is adding `"per_class_counts": total_per_class.cpu().tolist(),` after `per_class_accuracy`.)

- [ ] **Step 2: Add CSV generation in train.py**

After the artifact manifest block from Task 6, still inside `if mode in ("dev", "confirm"):`, add:

```python
        # ── Write per_class_metrics.csv (A-INFRA-9) ──
        if val_loader is not None and per_class_results:
            import csv as _csv
            per_class_path = save_dir / "per_class_metrics.csv"
            with open(per_class_path, "w", newline="") as f:
                writer = _csv.writer(f)
                writer.writerow(["class_idx", "accuracy", "n_samples"])
                pca = per_class_results.get("per_class_accuracy", [])
                pcc = per_class_results.get("per_class_counts", [])
                for i in range(len(pca)):
                    count = pcc[i] if i < len(pcc) else 0
                    writer.writerow([i, f"{pca[i]:.6f}", count])
            train_logger.info(
                "Per-class metrics saved to: %s", per_class_path
            )

        # ── Write prediction_records.csv (A-INFRA-9) ──
        if val_loader is not None:
            pred_path = save_dir / "prediction_records.csv"
            model.eval()
            with open(pred_path, "w", newline="") as f:
                writer = _csv.writer(f)
                writer.writerow(
                    ["image_path", "true_label", "pred_label", "pred_conf"]
                )
                with torch.no_grad():
                    for batch_data in val_loader:
                        inputs, labels, is_cached, paths = _unpack_batch(
                            batch_data, device
                        )
                        if is_cached:
                            logits = model.forward_features(inputs)
                        else:
                            logits = model(inputs)
                        probs = torch.softmax(logits, dim=1)
                        confs, preds = probs.max(dim=1)
                        for path, tl, pl, cf in zip(
                            paths, labels.cpu(), preds.cpu(), confs.cpu()
                        ):
                            writer.writerow([
                                path, int(tl.item()),
                                int(pl.item()), f"{cf.item():.6f}",
                            ])
            train_logger.info(
                "Prediction records saved to: %s", pred_path
            )
```

Note: `_unpack_batch` and `_forward_inputs` are module-level functions in train.py, so they're accessible here. The `per_class_results` variable already exists from the `evaluate_full()` call.

- [ ] **Step 3: Verify imports**

```bash
python3 -c "from experiments.baseline.evaluate import evaluate; print('evaluate OK')"
python3 -c "from experiments.baseline.train import main; print('train OK')"
```
Expected: Both OK

- [ ] **Step 4: Commit**

```bash
git add experiments/baseline/evaluate.py experiments/baseline/train.py
git commit -m "feat: generate per_class_metrics.csv and prediction_records.csv after training

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Register w2 experiments in phase3_experiments.csv

**Files:**
- Modify: `results/phase3_experiments.csv`

**Interfaces:**
- None — pure data entry

Register the two completed Wave 2 experiments and the B2 baseline.

- [ ] **Step 1: Append rows to phase3_experiments.csv**

The CSV header is: `experiment_id,parent_experiment,wave,priority,commit_sha,config_path,output_dir,split_seed,train_seed,loss_name,loss_parameters,sample_weighting,augmentation,head_ema,trainable_parameters,best_epoch,checkpoint_sha256,train_split_sha256,val_split_sha256,raw_micro,raw_macro,raw_bottom10,trusted_micro,trusted_macro,trusted_class_balanced,trust_weighted_accuracy,rejected_micro,prediction_change_vs_parent,platform_score,platform_delta_vs_ref,platform_delta_vs_parent,status,notes`

Append these two rows:

```
w2_ema_loss,gce_q07,W2,P0,605a262,configs/w2_ema_loss.yaml,outputs/w2_ema_loss/seed42,42,42,gce,q=0.7,ema_loss (momentum=0.9 warmup=5 classwise min=0.4 max=1.0),a0,false,256500,41,572232e5c42e96c3f0df59abe2da433deef293195f253bdbd243e8b0f0751d49,,,,0.6942,,,,,,,,0.593864,0.016467,-0.004486,local_rejected,W2_EMA_LOSS: platform 59.3864% (flat vs gce_q07+TTA 59.4064%); EMA loss provides no generalization gain; no multi-seed
w2_proto_min04,gce_q07,W2,P0,b0dd852,configs/w2_proto_min04.yaml,outputs/w2_proto_min04/seed42,42,42,gce,q=0.7,static_manifest (prototype min=0.4),a0,false,256500,37,1a7f95021efe3491eaf91a2b99b028b2218b63ec031acc3e27196c02f1a99b1b,,,,0.6876,,,,,,,,0.588216,0.010819,-0.010362,closed,W2_PROTO_MIN04: platform 58.82% (-0.58pp vs baseline); prototype weighting HURTS under GCE; CLOSED
```

- [ ] **Step 2: Verify CSV integrity**

```bash
python3 -c "
import csv
with open('results/phase3_experiments.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    print(f'Rows: {len(rows)}')
    for r in rows:
        print(f'  {r[\"experiment_id\"]:20s} status={r[\"status\"]}')
"
```
Expected: Shows 4 rows including the two new entries.

- [ ] **Step 3: Commit**

```bash
git add results/phase3_experiments.csv
git commit -m "chore: register w2_ema_loss and w2_proto_min04 in phase3_experiments.csv

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: ScheduledLoss Unit Tests

**Files:**
- Create: `tests/test_loss_schedule.py`

**Interfaces:**
- Consumes: `ScheduledLoss`, `build_scheduled_loss` from `common.loss_schedule`

Tests cover: phase switching at epoch boundaries, correct loss usage per phase, checkpoint save/restore, schedule validation (gaps, overlaps), epoch out of range.

- [ ] **Step 1: Create `tests/test_loss_schedule.py`**

```python
"""Tests for common.loss_schedule module."""

import torch
import pytest

from common.loss_schedule import (
    ScheduledLoss,
    _validate_schedule,
    build_scheduled_loss,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def ce_gce_schedule():
    """Standard CE → GCE schedule used in Phase 3."""
    return [
        {"start_epoch": 1, "end_epoch": 5, "name": "cross_entropy"},
        {"start_epoch": 6, "end_epoch": 50, "name": "gce", "q": 0.7},
    ]


@pytest.fixture
def logits_targets():
    rng = torch.Generator().manual_seed(42)
    logits = torch.randn(8, 500, generator=rng)
    targets = torch.randint(0, 500, (8,), generator=rng)
    return logits, targets


# ── Schedule validation ───────────────────────────────────────────────


class TestValidateSchedule:
    def test_valid_schedule(self, ce_gce_schedule):
        normalised = _validate_schedule(ce_gce_schedule)
        assert len(normalised) == 2
        assert normalised[0]["start_epoch"] == 1
        assert normalised[0]["end_epoch"] == 5
        assert normalised[1]["start_epoch"] == 6

    def test_empty_schedule_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_schedule([])

    def test_gap_raises(self):
        schedule = [
            {"start_epoch": 1, "end_epoch": 5, "name": "cross_entropy"},
            {"start_epoch": 7, "end_epoch": 50, "name": "gce", "q": 0.7},
        ]
        with pytest.raises(ValueError, match="gap or overlap"):
            _validate_schedule(schedule)

    def test_overlap_raises(self):
        schedule = [
            {"start_epoch": 1, "end_epoch": 6, "name": "cross_entropy"},
            {"start_epoch": 6, "end_epoch": 50, "name": "gce", "q": 0.7},
        ]
        with pytest.raises(ValueError, match="gap or overlap"):
            _validate_schedule(schedule)

    def test_start_gt_end_raises(self):
        schedule = [
            {"start_epoch": 10, "end_epoch": 5, "name": "cross_entropy"},
        ]
        with pytest.raises(ValueError, match="start_epoch"):
            _validate_schedule(schedule)

    def test_missing_start_end_raises(self):
        with pytest.raises(ValueError, match="start_epoch"):
            _validate_schedule([{"name": "cross_entropy"}])


# ── Phase switching ───────────────────────────────────────────────────


class TestPhaseSwitching:
    def test_epoch_5_uses_ce(self, ce_gce_schedule, logits_targets):
        logits, targets = logits_targets
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(5)
        assert sl.current_phase_name == "cross_entropy"

    def test_epoch_6_uses_gce(self, ce_gce_schedule, logits_targets):
        logits, targets = logits_targets
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(6)
        assert sl.current_phase_name == "gce"

    def test_epoch_1_uses_ce(self, ce_gce_schedule, logits_targets):
        logits, targets = logits_targets
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(1)
        assert sl.current_phase_name == "cross_entropy"

    def test_epoch_50_uses_gce(self, ce_gce_schedule, logits_targets):
        logits, targets = logits_targets
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(50)
        assert sl.current_phase_name == "gce"

    def test_epoch_out_of_range_raises(self, ce_gce_schedule):
        sl = ScheduledLoss(ce_gce_schedule)
        with pytest.raises(ValueError, match="not covered"):
            sl.set_epoch(0)

    def test_forward_produces_scalar_with_reduction_none(
        self, ce_gce_schedule, logits_targets
    ):
        """With reduction='none', forward returns per-sample loss."""
        logits, targets = logits_targets
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(1)
        loss = sl(logits, targets)
        assert loss.shape == (logits.size(0),)

    def test_loss_differs_between_phases(
        self, ce_gce_schedule, logits_targets
    ):
        """CE and GCE should produce different loss values."""
        logits, targets = logits_targets
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(1)
        ce_loss = sl(logits, targets).detach().clone()
        sl.set_epoch(6)
        gce_loss = sl(logits, targets).detach().clone()
        assert not torch.allclose(ce_loss, gce_loss), (
            "CE and GCE should produce different losses"
        )


# ── Checkpoint save / restore ─────────────────────────────────────────


class TestCheckpoint:
    def test_state_dict_saves_phase(self, ce_gce_schedule):
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(6)
        sd = sl.state_dict()
        assert "_schedule_phase_idx" in sd
        assert "_schedule_epoch" in sd
        assert sd["_schedule_phase_idx"].item() == 1  # 0-based

    def test_load_state_dict_restores_phase(
        self, ce_gce_schedule, logits_targets
    ):
        logits, targets = logits_targets
        sl1 = ScheduledLoss(ce_gce_schedule)
        sl1.set_epoch(6)
        loss_before = sl1(logits, targets).detach().clone()

        sl2 = ScheduledLoss(ce_gce_schedule)
        sl2.load_state_dict(sl1.state_dict())
        loss_after = sl2(logits, targets).detach().clone()

        assert sl2.current_phase_name == "gce"
        assert sl2.current_epoch == 6
        assert torch.allclose(loss_before, loss_after)

    def test_resume_from_epoch_4(self, ce_gce_schedule):
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(4)
        sd = sl.state_dict()

        sl2 = ScheduledLoss(ce_gce_schedule)
        sl2.load_state_dict(sd)
        assert sl2.current_phase_name == "cross_entropy"
        assert sl2.current_epoch == 4

    def test_resume_from_epoch_6(self, ce_gce_schedule):
        sl = ScheduledLoss(ce_gce_schedule)
        sl.set_epoch(6)
        sd = sl.state_dict()

        sl2 = ScheduledLoss(ce_gce_schedule)
        sl2.load_state_dict(sd)
        assert sl2.current_phase_name == "gce"
        assert sl2.current_epoch == 6


# ── Factory ───────────────────────────────────────────────────────────


class TestBuildScheduledLoss:
    def test_build_from_config(self):
        config = {
            "loss": {
                "schedule": [
                    {"start_epoch": 1, "end_epoch": 3, "name": "cross_entropy"},
                    {"start_epoch": 4, "end_epoch": 10, "name": "gce", "q": 0.7},
                ]
            }
        }
        sl = build_scheduled_loss(config)
        assert isinstance(sl, ScheduledLoss)
        assert len(sl.phases) == 2

    def test_build_missing_schedule_raises(self):
        with pytest.raises(ValueError, match="schedule"):
            build_scheduled_loss({"loss": {"name": "cross_entropy"}})
```

- [ ] **Step 2: Run the tests**

```bash
python3 -m pytest tests/test_loss_schedule.py -v
```
Expected: All 17 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_loss_schedule.py
git commit -m "test: add ScheduledLoss unit tests for phase switching and checkpoint resume

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Execution Checklist (summary)

| # | Task | Files | Verify |
|---|------|-------|--------|
| 1 | PrototypeProvider | `common/sample_weighting.py` | `python3 -c "from common.sample_weighting import PrototypeProvider; print('OK')"` |
| 2 | OOFManifestProvider | `common/sample_weighting.py` | `python3 -c "from common.sample_weighting import OOFManifestProvider; print('OK')"` |
| 3 | RelabelManifestProvider | `common/sample_weighting.py` | `python3 -c "from common.sample_weighting import RelabelManifestProvider; print('OK')"` |
| 4 | PrototypeEMAProvider | `common/sample_weighting.py` | `python3 -c "from common.sample_weighting import PrototypeEMAProvider; print('OK')"` |
| 5 | resolved_config.yaml | `common/resolved_config.py` + `train.py` | `python3 -c "from common.resolved_config import resolve_config; print('OK')"` |
| 6 | artifact_manifest.json | `train.py` | `python3 -c "from experiments.baseline.train import main; print('OK')"` |
| 7 | per_class_metrics.csv + prediction_records.csv | `evaluate.py` + `train.py` | import check |
| 8 | Register w2 experiments | `results/phase3_experiments.csv` | CSV integrity check |
| 9 | ScheduledLoss tests | `tests/test_loss_schedule.py` | `pytest tests/test_loss_schedule.py -v` |

All tasks are independent and can run in any order. Tasks 1–4 share `common/sample_weighting.py` but each adds a self-contained class + factory branch — merge conflicts are trivial (adjacent lines in the factory function).
