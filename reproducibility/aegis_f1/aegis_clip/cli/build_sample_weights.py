"""Build the rematch first-round OOF down-weighting sidecar.

Emits, into ``--output-dir``:

* ``oof_folds.csv``       content-group-aware stratified fold assignment
* ``oof_logits.pt``       cross-fitted logits for every ``train_dev`` row
* ``sample_weights.csv``  ``image_path,weight`` -- the trainer's sidecar
* ``manifest.json``       formula, inputs, hashes and diagnostics

The composite reuses ``analysis/oof/quality.py``. The flip-consistency term of
the historical four-term composite is dropped and the remaining three terms are
renormalised, because the rematch harness forbids a second feature cache and so
no flipped features exist for this stage.

Every signal is cross-fitted: a sample is scored by references (linear head,
class prototype, kNN bank) built only from the other folds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aegis_clip.features import canonical_sample_path
from aegis_clip.oof_rebuild import load_oof_inputs, rebuild_oof_logits
from aegis_clip.runtime import atomic_json_dump, environment_manifest, sha256_file


# The fold builder and the quality formula live in the repo-level ``analysis``
# package, which is not on the Aegis path. Bootstrap the repo root so the
# strategy reuses the validated implementations instead of a private copy.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.oof.build_folds import assign_group_stratified_folds  # noqa: E402
from analysis.oof.quality import add_quality_weights, build_sample_quality  # noqa: E402
from common.diagnostic_metrics import (  # noqa: E402
    build_trimmed_class_prototypes,
    chunked_topk_cosine,
    knn_label_metrics,
    prototype_metrics,
)


# Three-term composite, renormalised from the historical 0.35/0.25/0.25/0.15
# split after dropping the flip term: each coefficient is divided by 0.85.
P_ORIGINAL_COEFFICIENT = 0.35 / 0.85
PROTOTYPE_MARGIN_COEFFICIENT = 0.25 / 0.85
KNN_AGREEMENT_COEFFICIENT = 0.25 / 0.85
WEIGHT_FLOOR = 0.3
WEIGHT_SPAN = 0.7
KNN_NEIGHBOURS = 10
# Below this support a class cannot be cross-fitted into usable signals: with
# ``folds`` folds the training partition holds only ~80% of an already tiny
# class, so its percentile ranks are noise. The primary criterion is a macro
# average over the smallest classes, so acting on that noise would confound the
# verdict rather than test the method.
MIN_CLASS_SUPPORT = 10


def _feature_labels(full_train_csv: Path, feature_paths: Path) -> list[int]:
    """Realign full-train labels to the feature cache's own path order."""
    frame = pd.read_csv(
        full_train_csv, dtype={"image_path": str, "label": int}, usecols=["image_path", "label"]
    )
    label_of = {
        canonical_sample_path(path): int(label)
        for path, label in zip(frame["image_path"], frame["label"])
    }
    cached = json.loads(feature_paths.read_text(encoding="utf-8"))
    labels: list[int] = []
    for path in cached:
        key = canonical_sample_path(path)
        if key not in label_of:
            raise ValueError(f"feature cache path is absent from full_train.csv: {key}")
        labels.append(label_of[key])
    return labels


def _duplicate_conflict_flags(frame: pd.DataFrame) -> np.ndarray:
    """Flag every row whose content group carries more than one label."""
    grouped = frame.groupby("content_group")["label"].nunique()
    conflicting = set(grouped[grouped > 1].index)
    return frame["content_group"].isin(conflicting).to_numpy()


def _fold_geometry(
    features: torch.Tensor,
    labels: torch.Tensor,
    folds: np.ndarray,
    num_classes: int,
    device: torch.device,
    k: int,
) -> dict[str, np.ndarray]:
    """Cross-fitted prototype and kNN signals; references never include the query."""
    n_samples = len(folds)
    prototype_own_similarity = np.zeros(n_samples, dtype=np.float64)
    prototype_margin = np.zeros(n_samples, dtype=np.float64)
    prototype_top1 = np.zeros(n_samples, dtype=np.int64)
    knn_agreement = np.zeros(n_samples, dtype=np.float64)
    knn_top1 = np.zeros(n_samples, dtype=np.int64)

    for fold in sorted(set(folds.tolist())):
        holdout = np.flatnonzero(folds == fold)
        reference = np.flatnonzero(folds != fold)
        bank_features = features[reference].to(device)
        bank_labels = labels[reference].to(device)

        prototypes = build_trimmed_class_prototypes(
            bank_features, bank_labels, num_classes
        )
        metrics = prototype_metrics(
            features[holdout].to(device), labels[holdout].to(device), prototypes
        )
        prototype_own_similarity[holdout] = (
            metrics["prototype_label_similarity"].cpu().numpy()
        )
        prototype_margin[holdout] = metrics["prototype_margin"].cpu().numpy()
        prototype_top1[holdout] = metrics["prototype_top1_label"].cpu().numpy()

        neighbour_indices, _ = chunked_topk_cosine(
            features[holdout], features[reference], k, device=str(device)
        )
        neighbour_metrics = knn_label_metrics(
            neighbour_indices.cpu(),
            labels[reference],
            labels[holdout],
            num_classes,
        )
        knn_agreement[holdout] = neighbour_metrics["knn_label_agreement"].numpy()
        knn_top1[holdout] = neighbour_metrics["knn_majority_label"].numpy()
        print(
            f"geometry fold={fold} reference={len(reference)} holdout={len(holdout)}",
            flush=True,
        )

    return {
        "prototype_own_similarity": prototype_own_similarity,
        "prototype_margin": prototype_margin,
        "prototype_top1": prototype_top1,
        "knn_agreement": knn_agreement,
        "knn_top1": knn_top1,
    }


def build_fold_assignments(
    train_csv: Path,
    num_folds: int,
    seed: int,
    num_classes: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Assign content-group-intact, class-stratified OOF folds to ``train_dev``.

    Fails closed if any content group straddles folds or if any fold's training
    partition would be missing a class, since either would make the cross-fitted
    signals unusable for that class.
    """
    frame = pd.read_csv(
        train_csv, dtype={"image_path": str, "label": int, "content_group": str}
    )
    frame = frame.sort_values("image_path").reset_index(drop=True)
    frame["sample_id"] = frame["image_path"].map(canonical_sample_path)
    frame["sha256"] = frame["content_group"]
    if not frame["sample_id"].is_unique:
        raise ValueError("train_dev.csv yields duplicate canonical sample paths")

    labelled = assign_group_stratified_folds(frame, n_splits=num_folds, seed=seed)
    if labelled.groupby("sha256")["fold"].nunique().max() != 1:
        raise ValueError("a content group was split across OOF folds")

    for fold in range(num_folds):
        train_labels = labelled.loc[labelled["fold"] != fold, "label"]
        absent = sorted(set(range(num_classes)) - set(train_labels.tolist()))
        if absent:
            raise ValueError(
                f"fold {fold} training partition misses {len(absent)} classes: "
                f"{absent[:5]}"
            )

    fold_counts = {
        str(fold): int((labelled["fold"] == fold).sum())
        for fold in range(num_folds)
    }
    return labelled, fold_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--full-train-csv", required=True)
    parser.add_argument("--feature-tensor", required=True)
    parser.add_argument("--feature-paths", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--infer-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--knn", type=int, default=KNN_NEIGHBOURS)
    parser.add_argument(
        "--min-class-support",
        type=int,
        default=MIN_CLASS_SUPPORT,
        help=(
            "classes with fewer train_dev samples than this keep weight 1.0; "
            "their cross-fitted signals are not estimable"
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_csv = Path(args.train_csv)
    labelled, fold_counts = build_fold_assignments(
        train_csv, args.folds, args.seed, args.num_classes
    )
    folds = labelled["fold"].to_numpy()

    assignments_path = output / "oof_folds.csv"
    labelled[["sample_id", "image_path", "label", "fold"]].to_csv(
        assignments_path, index=False
    )
    print(f"folds written | sizes={fold_counts}", flush=True)

    feature_labels_path = output / "feature_labels.json"
    feature_labels_path.write_text(
        json.dumps(_feature_labels(Path(args.full_train_csv), Path(args.feature_paths))),
        encoding="utf-8",
    )

    inputs = load_oof_inputs(
        assignments_path,
        args.feature_tensor,
        args.feature_paths,
        feature_labels_path,
    )
    rebuild = rebuild_oof_logits(
        inputs,
        output,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        infer_batch_size=args.infer_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        q=args.q,
        seed=args.seed,
        device=device,
        input_hashes={
            "assignments_sha256": sha256_file(assignments_path),
            "feature_tensor_sha256": sha256_file(args.feature_tensor),
            "feature_paths_sha256": sha256_file(args.feature_paths),
            "feature_labels_sha256": sha256_file(feature_labels_path),
        },
    )

    payload = torch.load(output / "oof_logits.pt", map_location="cpu", weights_only=True)
    logits = payload["logits"].float()
    ordered = payload["image_paths"]
    if len(ordered) != len(inputs.assignments):
        raise RuntimeError("merged OOF logits do not match the assignment count")

    geometry = _fold_geometry(
        inputs.features, inputs.labels, folds, args.num_classes, device, args.knn
    )

    quality = build_sample_quality(
        inputs.assignments,
        logits,
        prototype_own_similarity=geometry["prototype_own_similarity"],
        prototype_margin=geometry["prototype_margin"],
        prototype_top1=geometry["prototype_top1"],
        knn_agreement=geometry["knn_agreement"],
        knn_top1=geometry["knn_top1"],
        # Dropped: no flipped feature cache exists for the rematch stage.
        flip_consistency=np.zeros(len(inputs.assignments)),
        clip_flip_cosine=np.zeros(len(inputs.assignments)),
        duplicate_conflict_flag=_duplicate_conflict_flags(labelled),
    )
    quality = add_quality_weights(quality)

    composite = (
        P_ORIGINAL_COEFFICIENT * quality["p_original_classwise_percentile"]
        + PROTOTYPE_MARGIN_COEFFICIENT
        * quality["prototype_margin_classwise_percentile"]
        + KNN_AGREEMENT_COEFFICIENT * quality["knn_agreement"].clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    quality["quality_downweight"] = composite
    quality["weight"] = (WEIGHT_FLOOR + WEIGHT_SPAN * composite).clip(
        WEIGHT_FLOOR, 1.0
    )

    # Reliability floor: a class too small to cross-fit keeps its weight at 1.0
    # rather than being down-weighted on non-estimable signals.
    class_support = quality.groupby("original_label")["weight"].transform("size")
    quality["class_support"] = class_support.astype(int)
    below_floor = class_support < args.min_class_support
    quality.loc[below_floor, "weight"] = 1.0
    exempt_classes = sorted(
        int(label) for label in quality.loc[below_floor, "original_label"].unique()
    )

    support_rank = (
        quality.groupby("original_label")["class_support"]
        .first()
        .sort_values(kind="stable")
        .index
    )
    tail_classes = set(support_rank[:75].tolist())
    in_tail = quality["original_label"].isin(tail_classes)

    sidecar = output / "sample_weights.csv"
    quality[["image_path", "weight"]].to_csv(sidecar, index=False)

    agrees = quality["oof_top1"] == quality["original_label"]
    manifest = {
        "protocol": "rematch-first-round-oof-continuous-downweighting-v1",
        "formula": {
            "quality_downweight": (
                f"{P_ORIGINAL_COEFFICIENT:.6f} * p_original_classwise_percentile + "
                f"{PROTOTYPE_MARGIN_COEFFICIENT:.6f} * prototype_margin_classwise_percentile + "
                f"{KNN_AGREEMENT_COEFFICIENT:.6f} * knn_agreement"
            ),
            "weight": f"clip({WEIGHT_FLOOR} + {WEIGHT_SPAN} * quality, {WEIGHT_FLOOR}, 1.0)",
            "flip_consistency_term": "dropped; renormalised over 0.35/0.25/0.25/0.15",
            "flip_consistency_reason": (
                "rematch forbids a second feature cache, so no flipped features exist"
            ),
            "scale_invariance": (
                "the trainer optimises a weight-normalised mean, so the global "
                "scale of the weight vector has no effect on the gradient"
            ),
        },
        "external_data": False,
        "test_data_used": False,
        "val_dev_used": False,
        "cross_fitting": {
            "folds": args.folds,
            "group_key": "content_group (decoded rgb sha256)",
            "knn_neighbours": args.knn,
            "references_restricted_to_training_folds": True,
        },
        "oof": {"audit": rebuild["audit"], "manifest": rebuild["manifest"]},
        "diagnostics": {
            "fold_counts": fold_counts,
            "weight_mean": float(quality["weight"].mean()),
            "weight_min": float(quality["weight"].min()),
            "weight_max": float(quality["weight"].max()),
            "weight_std": float(quality["weight"].std()),
            "oof_top1_matches_noisy_label_rate": float(agrees.mean()),
            "weight_mean_when_oof_agrees": float(quality.loc[agrees, "weight"].mean()),
            "weight_mean_when_oof_differs": float(
                quality.loc[~agrees, "weight"].mean()
            ),
            "duplicate_conflict_samples": int(
                quality["duplicate_conflict_flag"].sum()
            ),
            "min_class_support": args.min_class_support,
            "classes_exempt_below_support_floor": exempt_classes,
            "samples_exempt_below_support_floor": int(below_floor.sum()),
            "weight_mean_in_train_dev_tail_75": float(
                quality.loc[in_tail, "weight"].mean()
            ),
            "weight_mean_outside_tail": float(quality.loc[~in_tail, "weight"].mean()),
        },
        "inputs": {
            "train_csv": str(train_csv),
            "train_csv_sha256": sha256_file(train_csv),
            "feature_tensor": str(args.feature_tensor),
            "feature_tensor_sha256": sha256_file(args.feature_tensor),
            "sample_weights_csv": str(sidecar),
            "sample_weights_sha256": sha256_file(sidecar),
        },
        "environment": environment_manifest(),
    }
    atomic_json_dump(manifest, output / "manifest.json")
    print(json.dumps(manifest["diagnostics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
