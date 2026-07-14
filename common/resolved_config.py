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
