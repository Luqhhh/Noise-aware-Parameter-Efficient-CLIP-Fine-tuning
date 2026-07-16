"""Build raw and trusted validation reports for one candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from common.trusted_subset import (
    build_trusted_subset,
    compute_class_balanced_trusted_accuracy,
    compute_trust_weighted_accuracy,
)


SIGNAL_COLUMNS = {
    "image_path",
    "noisy_label",
    "knn_label_agreement",
    "prototype_supports_noisy_label",
    "prototype_margin",
    "clip_flip_cosine",
    "cross_class_duplicate_conflict",
}
PREDICTION_COLUMNS = {"image_path", "true_label", "pred_label", "pred_conf"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sample_key(path: str) -> str:
    parts = PurePosixPath(str(path).replace("\\", "/")).parts
    if len(parts) < 2:
        raise ValueError(f"Cannot derive class/file key from path: {path}")
    return "/".join(parts[-2:])


def _load_unique_csv(path: Path, required: set[str], kind: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{kind} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["sample_key"] = frame["image_path"].map(stable_sample_key)
    if frame["sample_key"].duplicated().any():
        duplicated = frame.loc[
            frame["sample_key"].duplicated(), "sample_key"
        ].tolist()[:10]
        raise ValueError(f"{kind} contains duplicate sample keys: {duplicated}")
    return frame


def _macro_accuracy(frame: pd.DataFrame, correct_column: str) -> float:
    if frame.empty:
        return float("nan")
    return float(frame.groupby("noisy_label")[correct_column].mean().mean())


def _bottom10_accuracy(frame: pd.DataFrame, correct_column: str) -> float:
    if frame.empty:
        return float("nan")
    per_class = frame.groupby("noisy_label")[correct_column].mean().sort_values()
    count = max(1, int(np.ceil(len(per_class) * 0.10)))
    return float(per_class.iloc[:count].mean())


def evaluate_candidate(
    experiment_id: str,
    prediction_path: Path,
    signal_path: Path,
    output_dir: Path,
    parent_prediction_path: Path | None = None,
) -> dict:
    """Join fixed signals with candidate predictions and write audit artifacts."""
    signals = _load_unique_csv(signal_path, SIGNAL_COLUMNS, "signal metrics")
    predictions = _load_unique_csv(
        prediction_path, PREDICTION_COLUMNS, "prediction records"
    )
    signal_keys = set(signals["sample_key"])
    prediction_keys = set(predictions["sample_key"])
    missing = sorted(signal_keys - prediction_keys)
    extra = sorted(prediction_keys - signal_keys)
    if missing or extra:
        raise ValueError(
            "Prediction/signal coverage mismatch: "
            f"missing={len(missing)} extra={len(extra)}"
        )

    prediction_view = predictions[
        ["sample_key", "true_label", "pred_label", "pred_conf"]
    ]
    merged = signals.merge(
        prediction_view,
        on="sample_key",
        how="inner",
        validate="one_to_one",
    ).reset_index(drop=True)
    if not (
        merged["noisy_label"].astype(int).to_numpy()
        == merged["true_label"].astype(int).to_numpy()
    ).all():
        raise ValueError("Prediction true_label does not match fixed noisy_label")

    merged["correct"] = (
        merged["pred_label"].astype(int) == merged["noisy_label"].astype(int)
    )
    trusted_manifest, trusted_summary = build_trusted_subset(merged)
    trusted_mask = trusted_manifest["trusted_v1"].astype(bool)
    rejected_mask = ~trusted_mask

    weighted = compute_trust_weighted_accuracy(
        merged, merged["correct"].to_numpy(dtype=bool)
    )
    balanced = compute_class_balanced_trusted_accuracy(
        merged, merged["correct"].to_numpy(dtype=bool)
    )

    prediction_change = None
    disagreement = pd.DataFrame(
        columns=[
            "sample_key",
            "noisy_label",
            "candidate_pred",
            "parent_pred",
            "candidate_correct",
            "parent_correct",
        ]
    )
    parent_sha256 = None
    if parent_prediction_path is not None:
        parent = _load_unique_csv(
            parent_prediction_path,
            PREDICTION_COLUMNS,
            "parent prediction records",
        )
        if set(parent["sample_key"]) != signal_keys:
            raise ValueError("Parent prediction coverage does not match signals")
        parent_view = parent[["sample_key", "pred_label"]].rename(
            columns={"pred_label": "parent_pred"}
        )
        compared = merged.merge(
            parent_view, on="sample_key", how="inner", validate="one_to_one"
        )
        changed = (
            compared["pred_label"].astype(int)
            != compared["parent_pred"].astype(int)
        )
        prediction_change = float(changed.mean())
        disagreement = pd.DataFrame(
            {
                "sample_key": compared.loc[changed, "sample_key"],
                "noisy_label": compared.loc[changed, "noisy_label"].astype(int),
                "candidate_pred": compared.loc[changed, "pred_label"].astype(int),
                "parent_pred": compared.loc[changed, "parent_pred"].astype(int),
                "candidate_correct": compared.loc[changed, "correct"].astype(bool),
                "parent_correct": (
                    compared.loc[changed, "parent_pred"].astype(int)
                    == compared.loc[changed, "noisy_label"].astype(int)
                ),
            }
        )
        parent_sha256 = _sha256(parent_prediction_path)

    trusted_frame = trusted_manifest[trusted_mask].copy()
    rejected_frame = trusted_manifest[rejected_mask].copy()
    raw_micro = float(merged["correct"].mean())
    raw_macro = _macro_accuracy(merged, "correct")
    raw_bottom10 = _bottom10_accuracy(merged, "correct")
    trusted_micro = (
        float(trusted_frame["correct"].mean())
        if not trusted_frame.empty
        else float("nan")
    )
    trusted_macro = _macro_accuracy(trusted_frame, "correct")
    rejected_micro = (
        float(rejected_frame["correct"].mean())
        if not rejected_frame.empty
        else None
    )

    per_class = (
        merged.groupby("noisy_label")["correct"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "raw_accuracy"})
    )
    trusted_per_class = (
        trusted_frame.groupby("noisy_label")["correct"]
        .agg(["count", "mean"])
        .rename(
            columns={
                "count": "trusted_count",
                "mean": "trusted_accuracy",
            }
        )
    )
    per_class = per_class.join(trusted_per_class, how="left")
    per_class["trust_weighted_accuracy"] = pd.Series(
        {
            int(class_id): accuracy
            for class_id, accuracy in weighted["per_class_accuracy"].items()
        },
        dtype=float,
    )
    per_class["class_balanced_topk_accuracy"] = pd.Series(
        {
            int(class_id): accuracy
            for class_id, accuracy in balanced["per_class_accuracy"].items()
        },
        dtype=float,
    )
    per_class.index.name = "class_id"

    metrics = {
        "experiment_id": experiment_id,
        "raw_micro": raw_micro,
        "raw_macro": raw_macro,
        "raw_bottom10": raw_bottom10,
        "trusted_micro": trusted_micro,
        "trusted_macro": trusted_macro,
        "trusted_class_balanced": balanced["macro_accuracy"],
        "trust_weighted_accuracy": weighted["accuracy"],
        "rejected_micro": rejected_micro,
        "prediction_change_vs_parent": prediction_change,
        "trusted_coverage": trusted_summary["coverage"],
        "trusted_represented_classes": trusted_summary["represented_classes"],
        "trusted_total_classes": trusted_summary["total_classes"],
        "class_balanced_classes": balanced["num_classes_with_k"],
        "class_balanced_samples_used": balanced["num_samples_used"],
        "trust_effective_samples": weighted["effective_samples"],
        "sample_count": len(merged),
    }
    non_finite = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (float, np.floating))
        and value is not None
        and not np.isfinite(value)
    }
    if non_finite:
        raise ValueError(f"Trusted metrics contain non-finite values: {non_finite}")

    output_dir.mkdir(parents=True, exist_ok=True)
    trusted_manifest.to_csv(output_dir / "trusted_manifest.csv", index=False)
    per_class.reset_index().to_csv(
        output_dir / "per_class_delta.csv", index=False
    )
    disagreement.to_csv(
        output_dir / "prediction_disagreement.csv", index=False
    )
    (output_dir / "trusted_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    audit = {
        "experiment_id": experiment_id,
        "protocol": "trusted_validation_v1_v2_fixed_signals",
        "coverage": len(merged) / len(signals) if len(signals) else 0.0,
        "all_samples_matched_once": len(merged) == len(signals),
        "labels_match": True,
        "fixed_signals_model_agnostic": True,
        "prediction_sha256": _sha256(prediction_path),
        "parent_prediction_sha256": parent_sha256,
        "signal_sha256": _sha256(signal_path),
        "trusted_summary": trusted_summary,
        "metrics": metrics,
    }
    (output_dir / "protocol_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--signals",
        default="outputs/analysis/d3_vs_b2_seed42/sample_metrics.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-predictions")
    args = parser.parse_args()

    metrics = evaluate_candidate(
        experiment_id=args.experiment_id,
        prediction_path=Path(args.predictions),
        signal_path=Path(args.signals),
        output_dir=Path(args.output_dir),
        parent_prediction_path=(
            Path(args.parent_predictions) if args.parent_predictions else None
        ),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
