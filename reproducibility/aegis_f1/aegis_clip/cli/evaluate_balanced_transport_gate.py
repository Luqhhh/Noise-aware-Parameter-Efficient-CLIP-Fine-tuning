"""Evaluate the frozen V1 balanced-prior transport gate on two caches."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch

from aegis_clip.balanced_inference import prediction_metrics
from aegis_clip.balanced_transport import (
    V1_CLEAN_CORE_THRESHOLD,
    V1_ITERATIONS,
    V1_NUM_CLASSES,
    V1_TEMPERATURE,
    balanced_transport_prediction,
    hard_balance_diagnostics,
    paired_change_summary,
    relative_cv_reduction,
    v1_gate_decision,
)
from aegis_clip.runtime import atomic_json_dump, sha256_file


F1_CACHE_SHA256 = "5f927bc9740ec5ce1725a7cfab07fbdc40f3e3dda5213ce59a419092edbf614c"
A2_CACHE_SHA256 = "cea5f7651bf0d24828b7da8b252cb47b52288fa2cb7638a1c6d32359c0c92698"
METRICS = (
    "raw_micro",
    "raw_macro",
    "trusted_micro",
    "trusted_macro",
    "proxy_micro",
    "proxy_macro",
    "clean_core_micro",
    "clean_core_macro",
)


def _evaluate_cache(
    cache_path: str | Path,
    output_dir: str | Path,
    *,
    expected_sha256: str,
    cache_name: str,
) -> dict[str, Any]:
    path = Path(cache_path).resolve()
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{cache_name} cache SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "logits",
        "labels",
        "clean_probability",
        "pseudo_labels",
        "correction_alpha",
        "paths",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{cache_name} cache missing keys: {sorted(missing)}")
    logits = torch.as_tensor(payload["logits"]).float()
    if logits.ndim != 2 or logits.shape[1] != V1_NUM_CLASSES:
        raise ValueError(
            f"{cache_name} logits must have [N,{V1_NUM_CLASSES}] shape"
        )
    if not torch.isfinite(logits).all():
        raise ValueError(f"{cache_name} logits contain non-finite values")
    sample_count = logits.shape[0]
    for key in ("labels", "clean_probability", "pseudo_labels", "correction_alpha"):
        if torch.as_tensor(payload[key]).numel() != sample_count:
            raise ValueError(f"{cache_name} {key} does not align with logits")
    paths = [str(item) for item in payload["paths"]]
    if len(paths) != sample_count or len(paths) != len(set(paths)):
        raise ValueError(f"{cache_name} paths must be aligned and unique")

    baseline_prediction = logits.argmax(dim=1).cpu()
    transport_prediction, sinkhorn = balanced_transport_prediction(
        logits,
        temperature=V1_TEMPERATURE,
        iterations=V1_ITERATIONS,
    )
    metric_arguments = {
        "labels": payload["labels"],
        "clean_probability": payload["clean_probability"],
        "pseudo_labels": payload["pseudo_labels"],
        "correction_alpha": payload["correction_alpha"],
        "num_classes": V1_NUM_CLASSES,
        "clean_core_threshold": V1_CLEAN_CORE_THRESHOLD,
    }
    baseline_metrics = prediction_metrics(baseline_prediction, **metric_arguments)
    transport_metrics = prediction_metrics(transport_prediction, **metric_arguments)
    baseline_balance = hard_balance_diagnostics(
        baseline_prediction, num_classes=V1_NUM_CLASSES
    )
    transport_balance = hard_balance_diagnostics(
        transport_prediction, num_classes=V1_NUM_CLASSES
    )
    labels = torch.as_tensor(payload["labels"]).long()
    clean = torch.as_tensor(payload["clean_probability"]).float()
    report: dict[str, Any] = {
        "protocol": "V1_F1_M1_KNOWN_BALANCED_PRIOR_TRANSPORT",
        "cache_name": cache_name,
        "cache_path": str(path),
        "cache_sha256": actual_sha256,
        "checkpoint": payload.get("checkpoint"),
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "validation_csv": payload.get("validation_csv"),
        "validation_csv_sha256": payload.get("validation_csv_sha256"),
        "parameters": {
            "temperature": V1_TEMPERATURE,
            "sinkhorn_iterations": V1_ITERATIONS,
            "target_prior": "soft_uniform_N_over_500",
            "clean_core_threshold": V1_CLEAN_CORE_THRESHOLD,
            "parameter_scan": False,
        },
        "sample_count": sample_count,
        "num_classes": V1_NUM_CLASSES,
        "test_data_used": False,
        "external_data_used": False,
        "model_parameters_updated": False,
        "bare": {**baseline_metrics, **baseline_balance},
        "transport": {**transport_metrics, **transport_balance},
        "delta_pp": {
            metric: 100.0 * (
                float(transport_metrics[metric]) - float(baseline_metrics[metric])
            )
            for metric in METRICS
        },
        "prediction_count_cv_relative_reduction": relative_cv_reduction(
            float(baseline_balance["prediction_count_cv"]),
            float(transport_balance["prediction_count_cv"]),
        ),
        "paired_raw": paired_change_summary(
            baseline_prediction, transport_prediction, labels
        ),
        "paired_clean_core": paired_change_summary(
            baseline_prediction,
            transport_prediction,
            labels,
            mask=clean >= V1_CLEAN_CORE_THRESHOLD,
        ),
        "sinkhorn": sinkhorn,
    }
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    prediction_path = destination / "predictions.pt"
    temporary = prediction_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "format_version": 1,
            "protocol": report["protocol"],
            "cache_sha256": actual_sha256,
            "baseline_prediction": baseline_prediction,
            "transport_prediction": transport_prediction,
        },
        temporary,
    )
    os.replace(temporary, prediction_path)
    report["predictions_sha256"] = sha256_file(prediction_path)
    atomic_json_dump(report, destination / "evaluation.json")
    return report


def evaluate_v1_gate(
    f1_validation_logits: str | Path,
    a2_validation_logits: str | Path,
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir).resolve()
    f1_report = _evaluate_cache(
        f1_validation_logits,
        destination / "f1_m1",
        expected_sha256=F1_CACHE_SHA256,
        cache_name="f1_m1",
    )
    a2_report = _evaluate_cache(
        a2_validation_logits,
        destination / "a2_m1",
        expected_sha256=A2_CACHE_SHA256,
        cache_name="a2_m1",
    )
    gate = v1_gate_decision({"f1_m1": f1_report, "a2_m1": a2_report})
    gate.update(
        {
            "test_data_used": False,
            "external_data_used": False,
            "model_parameters_updated": False,
            "f1_evaluation_sha256": sha256_file(
                destination / "f1_m1" / "evaluation.json"
            ),
            "a2_evaluation_sha256": sha256_file(
                destination / "a2_m1" / "evaluation.json"
            ),
        }
    )
    gate_path = destination / "gate.json"
    atomic_json_dump(gate, gate_path)
    return gate_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1-validation-logits", required=True)
    parser.add_argument("--a2-validation-logits", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    path = evaluate_v1_gate(
        args.f1_validation_logits,
        args.a2_validation_logits,
        args.output_dir,
    )
    print(path)


if __name__ == "__main__":
    main()

