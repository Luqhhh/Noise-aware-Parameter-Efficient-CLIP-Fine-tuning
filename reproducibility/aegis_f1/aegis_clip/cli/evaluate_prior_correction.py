"""Evaluate one preregistered Prior2Posterior correction without tuning."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from aegis_clip.balanced_inference import (
    effective_model_prior,
    prediction_metrics,
    prior_corrected_logits,
    prior_diagnostics,
)
from aegis_clip.runtime import atomic_json_dump, sha256_file


def evaluate_prior_correction(
    validation_logits_path: str | Path,
    oof_logits_path: str | Path,
    output_dir: str | Path,
    *,
    clean_core_threshold: float = 0.70,
) -> Path:
    validation_path = Path(validation_logits_path).resolve()
    oof_path = Path(oof_logits_path).resolve()
    destination = Path(output_dir).resolve()
    validation = torch.load(validation_path, map_location="cpu", weights_only=False)
    oof = torch.load(oof_path, map_location="cpu", weights_only=False)
    required = {
        "logits",
        "labels",
        "clean_probability",
        "pseudo_labels",
        "correction_alpha",
    }
    missing = required - set(validation)
    if missing:
        raise ValueError(f"Validation cache missing keys: {sorted(missing)}")
    if "logits" not in oof:
        raise ValueError("OOF artifact does not contain logits")
    logits = torch.as_tensor(validation["logits"]).float()
    oof_logits = torch.as_tensor(oof["logits"]).float()
    if logits.ndim != 2 or oof_logits.ndim != 2 or logits.shape[1] != oof_logits.shape[1]:
        raise ValueError("Validation and OOF logits must share [N,C] class dimension")
    num_classes = logits.shape[1]
    source_prior = effective_model_prior(oof_logits, temperature=1.0)
    corrected = prior_corrected_logits(logits, source_prior)
    bare_prediction = logits.argmax(dim=1)
    corrected_prediction = corrected.argmax(dim=1)
    metric_arguments = {
        "labels": validation["labels"],
        "clean_probability": validation["clean_probability"],
        "pseudo_labels": validation["pseudo_labels"],
        "correction_alpha": validation["correction_alpha"],
        "num_classes": num_classes,
        "clean_core_threshold": float(clean_core_threshold),
    }
    bare_metrics = prediction_metrics(bare_prediction, **metric_arguments)
    corrected_metrics = prediction_metrics(corrected_prediction, **metric_arguments)
    metric_names = [
        "raw_micro",
        "raw_macro",
        "trusted_micro",
        "trusted_macro",
        "proxy_micro",
        "proxy_macro",
        "clean_core_micro",
        "clean_core_macro",
    ]
    report = {
        "method": "p2p_cross_fitted_oof_prior_to_uniform",
        "temperature": 1.0,
        "strength": 1.0,
        "num_classes": num_classes,
        "validation_logits": str(validation_path),
        "validation_logits_sha256": sha256_file(validation_path),
        "oof_logits": str(oof_path),
        "oof_logits_sha256": sha256_file(oof_path),
        "source_prior": prior_diagnostics(source_prior),
        "bare": bare_metrics,
        "corrected": corrected_metrics,
        "delta_pp": {
            name: 100.0 * (
                float(corrected_metrics[name]) - float(bare_metrics[name])
            )
            for name in metric_names
        },
        "changed_predictions": int(
            bare_prediction.ne(corrected_prediction).sum()
        ),
        "changed_prediction_fraction": float(
            bare_prediction.ne(corrected_prediction).float().mean()
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    prediction_path = destination / "predictions.pt"
    temporary = prediction_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "format_version": 1,
            "source_prior": source_prior,
            "bare_prediction": bare_prediction,
            "corrected_prediction": corrected_prediction,
            "validation_paths": validation.get("paths"),
        },
        temporary,
    )
    os.replace(temporary, prediction_path)
    report["predictions_sha256"] = sha256_file(prediction_path)
    report_path = destination / "evaluation.json"
    atomic_json_dump(report, report_path)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-logits", required=True)
    parser.add_argument("--oof-logits", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clean-core-threshold", type=float, default=0.70)
    args = parser.parse_args()
    path = evaluate_prior_correction(
        args.validation_logits,
        args.oof_logits,
        args.output_dir,
        clean_core_threshold=args.clean_core_threshold,
    )
    print(path)


if __name__ == "__main__":
    main()
