"""Resolve runtime options from CLI and YAML.

Priority:
    explicit CLI > YAML > hard default
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Dict


def _pick(cli_value: Any, yaml_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if yaml_value is not None:
        return yaml_value
    return default


def resolve_runtime_args(
    args: Namespace,
    config: Dict[str, Any],
) -> Namespace:
    exp_cfg = config.get("experiment", {})
    model_cfg = config.get("model", {})

    args.experiment_id = _pick(
        args.experiment_id,
        exp_cfg.get("id"),
        "B0",
    )
    args.mode = _pick(
        args.mode,
        exp_cfg.get("mode"),
        "dev",
    )
    args.augmentation_preset = _pick(
        args.augmentation_preset,
        exp_cfg.get("augmentation_preset"),
        "a0",
    )
    args.head_type = _pick(
        args.head_type,
        exp_cfg.get("head_type"),
        "linear",
    )
    args.use_cached_features = bool(
        _pick(
            args.use_cached_features,
            model_cfg.get("use_cached_features"),
            False,
        )
    )

    # 将最终生效值写回 config，确保 checkpoint 和 snapshot 保存的是 resolved config。
    config.setdefault("experiment", {})
    config["experiment"].update(
        {
            "id": args.experiment_id,
            "mode": args.mode,
            "augmentation_preset": args.augmentation_preset,
            "head_type": args.head_type,
        }
    )
    config.setdefault("model", {})
    config["model"]["use_cached_features"] = args.use_cached_features

    config["runtime"] = {
        "experiment_id": args.experiment_id,
        "mode": args.mode,
        "augmentation_preset": args.augmentation_preset,
        "head_type": args.head_type,
        "use_cached_features": args.use_cached_features,
    }

    return args
