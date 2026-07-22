"""Compare a fixed inference view against a frozen center-crop cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aegis_clip.balanced_inference import prediction_metrics
from aegis_clip.runtime import atomic_json_dump, sha256_file


def _require_same_validation(reference: dict, candidate: dict) -> None:
    for name in (
        "paths",
        "labels",
        "clean_probability",
        "pseudo_labels",
        "correction_alpha",
    ):
        left, right = reference[name], candidate[name]
        equal = left == right if isinstance(left, list) else torch.equal(left, right)
        if not equal:
            raise ValueError(f"Logit caches differ in validation field {name}")
    if reference.get("checkpoint_sha256") != candidate.get("checkpoint_sha256"):
        raise ValueError("Logit caches were not produced by the same checkpoint")


def _transition_counts(
    source_prediction: torch.Tensor,
    target_prediction: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, int]:
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    source_correct = source_prediction.eq(labels)
    target_correct = target_prediction.eq(labels)
    corrected = mask & ~source_correct & target_correct
    harmed = mask & source_correct & ~target_correct
    return {
        "samples": int(mask.sum()),
        "changed": int((mask & source_prediction.ne(target_prediction)).sum()),
        "corrected": int(corrected.sum()),
        "harmed": int(harmed.sum()),
        "net_correct": int(corrected.sum() - harmed.sum()),
    }


def compare_logit_caches(
    baseline_path: str | Path,
    candidate_path: str | Path,
    output_path: str | Path,
    *,
    clean_core_threshold: float = 0.70,
    global_reference_path: str | Path | None = None,
    attention_reference_path: str | Path | None = None,
) -> Path:
    baseline_path = Path(baseline_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    destination = Path(output_path).resolve()
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    _require_same_validation(baseline, candidate)
    global_reference = None
    if global_reference_path is not None:
        global_reference_path = Path(global_reference_path).resolve()
        global_reference = torch.load(
            global_reference_path, map_location="cpu", weights_only=False
        )
        _require_same_validation(baseline, global_reference)
    attention_reference = None
    if attention_reference_path is not None:
        attention_reference_path = Path(attention_reference_path).resolve()
        attention_reference = torch.load(
            attention_reference_path, map_location="cpu", weights_only=False
        )
        _require_same_validation(baseline, attention_reference)
    baseline_logits = torch.as_tensor(baseline["logits"]).float()
    candidate_logits = torch.as_tensor(candidate["logits"]).float()
    if baseline_logits.shape != candidate_logits.shape:
        raise ValueError("Logit cache shapes do not match")
    metric_arguments = {
        "labels": baseline["labels"],
        "clean_probability": baseline["clean_probability"],
        "pseudo_labels": baseline["pseudo_labels"],
        "correction_alpha": baseline["correction_alpha"],
        "num_classes": baseline_logits.shape[1],
        "clean_core_threshold": float(clean_core_threshold),
    }
    baseline_prediction = baseline_logits.argmax(dim=1)
    candidate_prediction = candidate_logits.argmax(dim=1)
    baseline_metrics = prediction_metrics(
        baseline_prediction, **metric_arguments
    )
    candidate_metrics = prediction_metrics(
        candidate_prediction, **metric_arguments
    )
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
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "baseline_view_mode": baseline.get("view_mode", "unknown"),
        "candidate_view_mode": candidate.get("view_mode", "unknown"),
        "checkpoint_sha256": baseline.get("checkpoint_sha256"),
        "baseline_cache_sha256": sha256_file(baseline_path),
        "candidate_cache_sha256": sha256_file(candidate_path),
        "changed_predictions": int(
            baseline_prediction.ne(candidate_prediction).sum()
        ),
        "changed_prediction_fraction": float(
            baseline_prediction.ne(candidate_prediction).float().mean()
        ),
        "delta_pp": {
            name: 100.0 * (
                float(candidate_metrics[name]) - float(baseline_metrics[name])
            )
            for name in metric_names
        },
    }
    if "global_logits" in candidate:
        global_logits = torch.as_tensor(candidate["global_logits"]).float()
        global_reference_logits = torch.as_tensor(
            global_reference["logits"] if global_reference else baseline_logits
        ).float()
        if global_logits.shape != global_reference_logits.shape:
            raise ValueError("Candidate global logits do not align with baseline")
        report["global_path_max_abs_logit_difference"] = float(
            (global_logits - global_reference_logits).abs().max()
        )
        report["global_path_prediction_agreement"] = float(
            global_logits
            .argmax(dim=1)
            .eq(global_reference_logits.argmax(dim=1))
            .float()
            .mean()
        )
        if global_reference_path is not None:
            report["global_reference_cache_sha256"] = sha256_file(
                global_reference_path
            )
        local_logits = torch.as_tensor(candidate["local_logits"]).float()
        global_prediction = global_logits.argmax(dim=1)
        local_prediction = local_logits.argmax(dim=1)
        local_metrics = prediction_metrics(local_prediction, **metric_arguments)
        report["local"] = local_metrics
        report["local_delta_pp"] = {
            name: 100.0 * (
                float(local_metrics[name]) - float(baseline_metrics[name])
            )
            for name in metric_names
        }
        report["local_global_prediction_agreement"] = float(
            local_prediction.eq(global_prediction).float().mean()
        )
        labels = torch.as_tensor(baseline["labels"]).long()
        clean_core = torch.as_tensor(
            baseline["clean_probability"]
        ).float() >= float(clean_core_threshold)
        report["fusion_transition_raw"] = _transition_counts(
            global_prediction, candidate_prediction, labels
        )
        report["fusion_transition_clean_core"] = _transition_counts(
            global_prediction, candidate_prediction, labels, clean_core
        )
        top_values = torch.topk(global_logits, k=2, dim=1).values
        global_margin = top_values[:, 0] - top_values[:, 1]
        changed = global_prediction.ne(candidate_prediction)
        report["global_margin"] = {
            "changed_mean": float(global_margin[changed].mean()),
            "unchanged_mean": float(global_margin[~changed].mean()),
        }
        if "attention_local_logits" in candidate:
            if attention_reference is None or "local_logits" not in attention_reference:
                raise ValueError(
                    "Attention-local audit requires a reference cache with local_logits"
                )
            attention_logits = torch.as_tensor(
                candidate["attention_local_logits"]
            ).float()
            reference_attention_logits = torch.as_tensor(
                attention_reference["local_logits"]
            ).float()
            if attention_logits.shape != reference_attention_logits.shape:
                raise ValueError("Attention-local logits do not align with reference")
            report["attention_path_max_abs_logit_difference"] = float(
                (attention_logits - reference_attention_logits).abs().max()
            )
            report["attention_path_prediction_agreement"] = float(
                attention_logits
                .argmax(dim=1)
                .eq(reference_attention_logits.argmax(dim=1))
                .float()
                .mean()
            )
            report["attention_reference_cache_sha256"] = sha256_file(
                attention_reference_path
            )
        if "m1_logits" in candidate:
            m1_logits = torch.as_tensor(candidate["m1_logits"]).float()
            if m1_logits.shape != baseline_logits.shape:
                raise ValueError("Nested M1 logits do not align with baseline")
            report["m1_path_max_abs_logit_difference"] = float(
                (m1_logits - baseline_logits).abs().max()
            )
            report["m1_path_prediction_agreement"] = float(
                m1_logits
                .argmax(dim=1)
                .eq(baseline_prediction)
                .float()
                .mean()
            )
            flip_fused_logits = torch.as_tensor(
                candidate["flip_fused_logits"]
            ).float()
            if flip_fused_logits.shape != baseline_logits.shape:
                raise ValueError("Flip-fused logits do not align with baseline")
            flip_prediction = flip_fused_logits.argmax(dim=1)
            flip_metrics = prediction_metrics(
                flip_prediction, **metric_arguments
            )
            report["flip"] = flip_metrics
            report["candidate_vs_flip_delta_pp"] = {
                name: 100.0 * (
                    float(candidate_metrics[name]) - float(flip_metrics[name])
                )
                for name in metric_names
            }
            report["candidate_vs_flip_changed_predictions"] = int(
                candidate_prediction.ne(flip_prediction).sum()
            )
            report["candidate_vs_flip_changed_fraction"] = float(
                candidate_prediction.ne(flip_prediction).float().mean()
            )
            report["candidate_vs_flip_transition_raw"] = _transition_counts(
                flip_prediction, candidate_prediction, labels
            )
            report["candidate_vs_flip_transition_clean_core"] = (
                _transition_counts(
                    flip_prediction,
                    candidate_prediction,
                    labels,
                    clean_core,
                )
            )
    atomic_json_dump(report, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clean-core-threshold", type=float, default=0.70)
    parser.add_argument("--global-reference")
    parser.add_argument("--attention-reference")
    args = parser.parse_args()
    path = compare_logit_caches(
        args.baseline,
        args.candidate,
        args.output,
        clean_core_threshold=args.clean_core_threshold,
        global_reference_path=args.global_reference,
        attention_reference_path=args.attention_reference,
    )
    print(path)


if __name__ == "__main__":
    main()
