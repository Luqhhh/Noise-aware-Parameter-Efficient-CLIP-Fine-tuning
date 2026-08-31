"""Fit and select the balanced-prior strength on validation logits only.

Rule boundary: the class bias is fitted on the current-stage validation set
and frozen; the test set is never used to fit or select anything.  The output
``prior_config.json`` is consumed by ``infer.py --prior-config`` so official
test inference applies the frozen bias with the validation-selected strength.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aegis_clip.balanced_inference import prediction_metrics
from aegis_clip.prior_alignment import apply_prior_bias, fit_prior_bias
from aegis_clip.runtime import atomic_json_dump, sha256_file


def _parse_strengths(value: str) -> tuple[float, ...]:
    strengths = tuple(float(item) for item in value.split(",") if item.strip())
    if not strengths or any(not 0.0 <= item <= 1.0 for item in strengths):
        raise ValueError("strengths must be comma-separated values in [0, 1]")
    return strengths


def sweep_prior_strength(
    validation_logits_path: str | Path,
    output_dir: str | Path,
    *,
    strengths: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0),
    selector_metric: str = "clean_core_micro",
    clean_core_threshold: float = 0.70,
    iterations: int = 50,
) -> Path:
    validation_path = Path(validation_logits_path).resolve()
    destination = Path(output_dir).resolve()
    payload = torch.load(validation_path, map_location="cpu", weights_only=False)
    required = {
        "logits",
        "labels",
        "clean_probability",
        "pseudo_labels",
        "correction_alpha",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Validation cache missing keys: {sorted(missing)}")
    logits = torch.as_tensor(payload["logits"]).float()
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("Validation logits must be a non-empty [N, C] tensor")
    num_classes = logits.shape[1]
    bias, fit_report = fit_prior_bias(logits, max_iterations=int(iterations))
    metric_arguments = {
        "labels": payload["labels"],
        "clean_probability": payload["clean_probability"],
        "pseudo_labels": payload["pseudo_labels"],
        "correction_alpha": payload["correction_alpha"],
        "num_classes": num_classes,
        "clean_core_threshold": float(clean_core_threshold),
    }
    sweep: dict[str, dict[str, float | int]] = {}
    best_strength = min(strengths)
    best_value = float("-inf")
    for strength in strengths:
        aligned = apply_prior_bias(logits, bias, strength=float(strength))
        metrics = prediction_metrics(aligned.argmax(dim=1), **metric_arguments)
        if selector_metric not in metrics:
            raise ValueError(f"Unknown selector metric: {selector_metric}")
        value = float(metrics[selector_metric])
        sweep[f"strength_{strength:g}"] = {
            "selector_value": value,
            **metrics,
        }
        if value > best_value:
            best_value = value
            best_strength = strength

    destination.mkdir(parents=True, exist_ok=True)
    prior_config = {
        "format_version": 1,
        "method": "validation_fitted_ipf_bias_to_uniform_prior",
        "num_classes": num_classes,
        "bias": bias.tolist(),
        "strength": float(best_strength),
        "selector_metric": selector_metric,
        "selector_value": best_value,
        "clean_core_threshold": float(clean_core_threshold),
        "fit_report": fit_report,
        "sweep": sweep,
        "fitted_on": "current_stage_validation",
        "test_data_used": False,
        "validation_logits": str(validation_path),
        "validation_logits_sha256": sha256_file(validation_path),
    }
    config_path = destination / "prior_config.json"
    atomic_json_dump(prior_config, config_path)
    print(json.dumps(prior_config, ensure_ascii=False, indent=2))
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-logits", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--strengths",
        default="0.0,0.25,0.5,0.75,0.85,0.9,0.95,1.0",
    )
    parser.add_argument("--selector-metric", default="clean_core_micro")
    parser.add_argument("--clean-core-threshold", type=float, default=0.70)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    sweep_prior_strength(
        args.validation_logits,
        args.output_dir,
        strengths=_parse_strengths(args.strengths),
        selector_metric=args.selector_metric,
        clean_core_threshold=args.clean_core_threshold,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
