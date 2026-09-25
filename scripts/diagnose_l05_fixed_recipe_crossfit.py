#!/usr/bin/env python3
"""Fixed-recipe conditional cross-fit diagnostic for the L05 prior bias.

Reuses the frozen validation branch cache (``l05_val_branch_logits.pt``) and the
existing content-group partition.  For every fold it fits the prior bias on the
complement of that fold (never reading the fold itself) and evaluates the frozen
recipe -- horizontal-flip TTA, ``mean_probabilities``, temperature 1.4, prior
strength 0.60 -- on the held-out fold.  It records per-fold Macro, Micro,
corrections and corruptions, plus the pooled out-of-fold result.

This is deliberately **not** an unbiased end-to-end validation: the checkpoint,
the temperature and the prior strength were all selected on the whole validation
set before this diagnostic existed, so the numbers are a *fixed-recipe conditional
cross-fit diagnostic*.  The script does not re-scan temperature or prior strength
and it never reads the test set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
AEGIS_ROOT = ROOT / "reproducibility" / "aegis_f1"
for location in (str(ROOT), str(AEGIS_ROOT)):
    if location not in sys.path:
        sys.path.insert(0, location)

from analysis.oof.build_folds import assign_group_stratified_folds  # noqa: E402
from aegis_clip.balanced_inference import prediction_metrics  # noqa: E402
from aegis_clip.config import load_config  # noqa: E402
from aegis_clip.runtime import sha256_lines  # noqa: E402
from aegis_clip.tta_prior_binding import (  # noqa: E402
    canonical_cache_paths,
    checkpoint_identity,
    fit_bound_prior,
    fused_validation_logits,
    resolved_recipe,
    validation_cache_identity,
    verify_prior_record,
)


LIMITATION = (
    "Fixed-recipe conditional cross-fit diagnostic only. The checkpoint, the TTA "
    "temperature (1.4) and the prior strength (0.60) were selected on the whole "
    "validation set before this diagnostic. Excluding a fold when fitting the bias "
    "removes that fold's contribution to the bias, but it does not undo the earlier "
    "whole-validation model selection, so these numbers must not be reported as an "
    "unbiased end-to-end validation of the full pipeline."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val-branch-cache", required=True)
    parser.add_argument("--content-groups")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.4)
    parser.add_argument("--prior-strength", type=float, default=0.60)
    parser.add_argument("--fusion", default="mean_probabilities")
    parser.add_argument("--tta", default="horizontal_flip")
    parser.add_argument(
        "--allow-recipe-drift",
        action="store_true",
        help="Record the declared recipe without requiring the frozen L05 point.",
    )
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _subset(payload: dict, key: str, mask: torch.Tensor | None):
    value = payload[key]
    if mask is None:
        return value
    return value[mask]


def _metrics(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    payload: dict,
    num_classes: int,
    mask: torch.Tensor | None = None,
) -> dict:
    return prediction_metrics(
        prediction,
        labels=labels,
        clean_probability=_subset(payload, "clean_probability", mask),
        pseudo_labels=_subset(payload, "pseudo_labels", mask),
        correction_alpha=_subset(payload, "correction_alpha", mask),
        num_classes=int(num_classes),
        clean_core_threshold=0.70,
    )


def _changes(
    raw_prediction: torch.Tensor,
    corrected_prediction: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, int]:
    raw_correct = raw_prediction == labels
    corrected_correct = corrected_prediction == labels
    return {
        "corrections": int(((~raw_correct) & corrected_correct).sum()),
        "corruptions": int((raw_correct & (~corrected_correct)).sum()),
        "changed": int((raw_prediction != corrected_prediction).sum()),
    }


def _fold_sha(assignments: pd.DataFrame) -> str:
    ordered = assignments.sort_values("image_path")
    return sha256_lines(
        f"{row.image_path}:{int(row.label)}:{int(row.fold)}"
        for row in ordered.itertuples(index=False)
    )


def main() -> int:
    args = _parse_args()
    config = load_config(Path(args.config).expanduser().resolve())
    cache_path = Path(args.val_branch_cache).expanduser().resolve()

    checkpoint = checkpoint_identity(args.checkpoint, config)
    cache_identity = validation_cache_identity(cache_path, config, checkpoint)
    recipe = resolved_recipe(
        tta=str(args.tta),
        fusion=str(args.fusion),
        temperature=float(args.temperature),
        prior_strength=float(args.prior_strength),
        enforce_fixed=not bool(args.allow_recipe_drift),
    )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    num_classes = int(cache_identity["num_classes"])
    labels = torch.as_tensor(payload["labels"]).long().cpu()

    fused = fused_validation_logits(
        payload, fusion=recipe["fusion"], temperature=recipe["temperature"]
    )
    raw_prediction = fused.argmax(dim=1)
    raw_full = _metrics(raw_prediction, labels, payload, num_classes)

    center_prediction = torch.as_tensor(payload["original_logits"]).float().argmax(dim=1)
    center_full = _metrics(center_prediction, labels, payload, num_classes)

    # Full-validation fit: reconstructs the production bias and its metrics.
    full_bias, full_report, full_record = fit_bound_prior(
        payload, cache_identity, recipe=recipe, max_iterations=int(args.max_iterations)
    )
    verify_prior_record(full_record, cache_identity, recipe=recipe, bias=full_bias)
    corrected_full = fused + float(recipe["prior_strength"]) * full_bias
    full_fit_metrics = _metrics(corrected_full.argmax(dim=1), labels, payload, num_classes)

    # Content-group folds over the existing validation partition.
    groups_path = (
        Path(args.content_groups).expanduser().resolve()
        if args.content_groups
        else Path(config["data"]["dataset_manifest"]).resolve().parent / "content_groups.csv"
    )
    cache_paths = canonical_cache_paths(payload["paths"])
    with groups_path.open(newline="", encoding="utf-8") as handle:
        group_of = {
            canonical_cache_paths([row["image_path"]])[0]: row["content_group"]
            for row in csv.DictReader(handle)
        }
    missing = [path for path in cache_paths if path not in group_of]
    if missing:
        raise ValueError(f"cache samples missing from content groups: {missing[:3]}")

    frame = pd.DataFrame(
        {
            "image_path": cache_paths,
            "label": labels.tolist(),
            "content_group": [group_of[path] for path in cache_paths],
        }
    )
    frame = frame.sort_values("image_path").reset_index(drop=True)
    frame["sample_id"] = frame["image_path"].map(
        lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    )
    frame["sha256"] = frame["content_group"].astype(str)
    assignments = assign_group_stratified_folds(
        frame[["sample_id", "image_path", "label", "sha256"]],
        n_splits=int(args.folds),
        seed=int(args.fold_seed),
    )
    if assignments.groupby("sha256")["fold"].nunique().max() != 1:
        raise RuntimeError("a content group was split across folds")
    fold_lookup = dict(
        zip(assignments["image_path"].astype(str), assignments["fold"].astype(int))
    )
    fold_tensor = torch.tensor([fold_lookup[path] for path in cache_paths], dtype=torch.long)
    if int(fold_tensor.min()) < 0 or int(fold_tensor.max()) >= int(args.folds):
        raise RuntimeError("fold assignment is incomplete")

    fold_rows: list[dict] = []
    pooled = fused.clone()
    for fold in range(int(args.folds)):
        holdout = fold_tensor == fold
        fit_mask = ~holdout
        bias, report, record = fit_bound_prior(
            payload,
            cache_identity,
            recipe=recipe,
            fit_mask=fit_mask,
            max_iterations=int(args.max_iterations),
        )
        verify_prior_record(record, cache_identity, recipe=recipe, bias=bias)
        corrected_fold = fused[holdout] + float(recipe["prior_strength"]) * bias
        pooled[holdout] = corrected_fold
        fold_metrics = _metrics(
            corrected_fold.argmax(dim=1), labels[holdout], payload, num_classes, mask=holdout
        )
        raw_fold = _metrics(
            raw_prediction[holdout], labels[holdout], payload, num_classes, mask=holdout
        )
        fold_rows.append(
            {
                "fold": fold,
                "holdout_samples": int(holdout.sum()),
                "fit_samples": int(fit_mask.sum()),
                "holdout_content_groups": int(
                    assignments.loc[assignments["fold"] == fold, "sha256"].nunique()
                ),
                "evaluated_classes": int(len(torch.unique(labels[holdout]).tolist())),
                "raw_macro": float(raw_fold["raw_macro"]),
                "raw_micro": float(raw_fold["raw_micro"]),
                "macro": float(fold_metrics["raw_macro"]),
                "micro": float(fold_metrics["raw_micro"]),
                "delta_macro_pp": 100.0
                * (float(fold_metrics["raw_macro"]) - float(raw_fold["raw_macro"])),
                "delta_micro_pp": 100.0
                * (float(fold_metrics["raw_micro"]) - float(raw_fold["raw_micro"])),
                **_changes(
                    raw_prediction[holdout], corrected_fold.argmax(dim=1), labels[holdout]
                ),
                "bias_sha256": record["bias_sha256"],
                "bias_fit_iterations": int(report["iterations"]),
                "fit_scope": record["fit_scope"],
            }
        )

    pooled_metrics = _metrics(pooled.argmax(dim=1), labels, payload, num_classes)
    pooled_changes = _changes(raw_prediction, pooled.argmax(dim=1), labels)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "experiment_id": "RM_V5_L05_CUDA_LOCAL",
        "diagnostic": "fixed_recipe_conditional_crossfit",
        "limitation": LIMITATION,
        "recipe": recipe,
        "recipe_enforced": not bool(args.allow_recipe_drift),
        "folds": int(args.folds),
        "fold_seed": int(args.fold_seed),
        "fold_assignments_sha256": _fold_sha(assignments),
        "content_group_partition": {
            "content_groups": str(groups_path),
            "content_groups_sha256": cache_identity["content_groups_sha256"],
            "content_group_set_sha256": cache_identity["content_group_set_sha256"],
            "distinct_groups": int(assignments["sha256"].nunique()),
        },
        "binding": {
            "cache": cache_identity["cache"],
            "cache_sha256": cache_identity["cache_sha256"],
            "checkpoint": cache_identity["checkpoint"],
            "checkpoint_sha256": cache_identity["checkpoint_sha256"],
            "training_config_sha256": cache_identity["training_config_sha256"],
            "class_mapping_sha256": cache_identity["class_mapping_sha256"],
            "dataset_manifest_sha256": cache_identity["dataset_manifest_sha256"],
            "dataset_fingerprint": cache_identity["dataset_fingerprint"],
            "split_seed": cache_identity["split_seed"],
            "validation_csv_sha256": cache_identity["validation_csv_sha256"],
            "sample_order_sha256": cache_identity["sample_order_sha256"],
            "input_resolution": cache_identity["input_resolution"],
            "preprocessing": cache_identity["preprocessing"],
            "encoder_precision": cache_identity["encoder_precision"],
            "feature_precision": cache_identity["feature_precision"],
            "autocast": cache_identity["autocast"],
        },
        "baseline": {
            "center_raw_macro": float(center_full["raw_macro"]),
            "center_raw_micro": float(center_full["raw_micro"]),
            "fused_no_bias_raw_macro": float(raw_full["raw_macro"]),
            "fused_no_bias_raw_micro": float(raw_full["raw_micro"]),
        },
        "full_validation_fit": {
            "macro": float(full_fit_metrics["raw_macro"]),
            "micro": float(full_fit_metrics["raw_micro"]),
            **_changes(raw_prediction, corrected_full.argmax(dim=1), labels),
            "bias_sha256": full_record["bias_sha256"],
            "bias_fit_iterations": int(full_record["prior_fit_iterations"]),
            "fit_scope": full_record["fit_scope"],
        },
        "conditional_crossfit": {
            "pooled_out_of_fold": {
                "macro": float(pooled_metrics["raw_macro"]),
                "micro": float(pooled_metrics["raw_micro"]),
                "delta_macro_pp": 100.0
                * (float(pooled_metrics["raw_macro"]) - float(raw_full["raw_macro"])),
                "delta_micro_pp": 100.0
                * (float(pooled_metrics["raw_micro"]) - float(raw_full["raw_micro"])),
                **pooled_changes,
            },
            "per_fold": fold_rows,
        },
        "test_data_used": False,
        "temperature_rescanned": False,
        "prior_strength_rescanned": False,
    }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["conditional_crossfit"], ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
