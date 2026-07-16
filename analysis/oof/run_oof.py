"""Run fixed-epoch GCE linear-head OOF training and quality estimation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, TensorDataset

from analysis.oof.build_folds import _canonical_image_key
from analysis.oof.quality import add_quality_weights, build_sample_quality
from common.clip_utils import encode_frozen_clip_features, load_openai_clip
from common.losses import GCELoss


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_linear_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    q: float,
    seed: int,
    device: torch.device,
) -> tuple[nn.Linear, list[dict]]:
    """Train the frozen-CLIP linear head for a fixed number of epochs."""
    _set_seed(seed)
    features = F.normalize(features.detach().float().cpu(), dim=1)
    labels = labels.detach().long().cpu()
    dataset = TensorDataset(features, labels)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    head = nn.Linear(features.shape[1], num_classes)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)
    head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = GCELoss(q=q, probability_epsilon=1e-7, reduction="mean")
    steps_per_epoch = max(len(loader), 1)
    total_steps = max(epochs * steps_per_epoch, 1)
    warmup_steps = warmup_epochs * steps_per_epoch
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        started = time.time()
        for batch_features, batch_labels in loader:
            if global_step < warmup_steps:
                factor = (global_step + 1) / max(warmup_steps, 1)
            else:
                progress = (global_step - warmup_steps) / max(
                    total_steps - warmup_steps, 1
                )
                factor = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
            optimizer.param_groups[0]["lr"] = lr * factor

            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = head(batch_features)
                loss = criterion(logits, batch_labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_count = len(batch_labels)
            total_loss += float(loss.detach()) * batch_count
            total_correct += int((logits.argmax(dim=1) == batch_labels).sum())
            total_samples += batch_count
            global_step += 1

        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_samples, 1),
            "train_accuracy": total_correct / max(total_samples, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch:02d}/{epochs} "
            f"loss={epoch_record['train_loss']:.6f} "
            f"acc={epoch_record['train_accuracy']:.4f} "
            f"seconds={epoch_record['seconds']:.1f}",
            flush=True,
        )

    return head.cpu(), history


@torch.no_grad()
def infer_logits(
    head: nn.Linear,
    features: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Infer float32 logits in deterministic input order."""
    head = head.to(device).eval()
    outputs = []
    for start in range(0, len(features), batch_size):
        batch = F.normalize(
            features[start : start + batch_size].float(), dim=1
        ).to(device)
        outputs.append(head(batch).float().cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def compute_reference_signals(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    holdout_features: torch.Tensor,
    holdout_labels: torch.Tensor,
    *,
    num_classes: int,
    k_neighbors: int,
    query_batch_size: int,
    reference_chunk_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Compute prototype and exact chunked-kNN signals from training folds only."""
    train_features = F.normalize(train_features.float(), dim=1)
    holdout_features = F.normalize(holdout_features.float(), dim=1)
    train_labels = train_labels.long()
    holdout_labels = holdout_labels.long()

    prototype_sums = torch.zeros(
        num_classes, train_features.shape[1], dtype=torch.float32
    )
    prototype_sums.index_add_(0, train_labels, train_features)
    prototype_counts = torch.bincount(train_labels, minlength=num_classes).clamp_min(1)
    prototypes = F.normalize(prototype_sums / prototype_counts[:, None], dim=1)

    own_similarity_parts = []
    prototype_margin_parts = []
    prototype_top1_parts = []
    knn_agreement_parts = []
    knn_top1_parts = []
    reference_labels_device = train_labels.to(device)
    prototypes_device = prototypes.to(device)

    for query_start in range(0, len(holdout_features), query_batch_size):
        query_end = min(query_start + query_batch_size, len(holdout_features))
        queries = holdout_features[query_start:query_end].to(device)
        query_labels = holdout_labels[query_start:query_end].to(device)

        prototype_similarity = queries @ prototypes_device.T
        own_similarity = prototype_similarity.gather(1, query_labels[:, None]).squeeze(1)
        other_similarity = prototype_similarity.clone()
        other_similarity.scatter_(1, query_labels[:, None], float("-inf"))
        prototype_margin = own_similarity - other_similarity.max(dim=1).values
        prototype_top1 = prototype_similarity.argmax(dim=1)

        best_scores = torch.full(
            (len(queries), k_neighbors),
            float("-inf"),
            device=device,
        )
        best_labels = torch.zeros(
            (len(queries), k_neighbors), dtype=torch.long, device=device
        )
        for reference_start in range(0, len(train_features), reference_chunk_size):
            reference_end = min(
                reference_start + reference_chunk_size, len(train_features)
            )
            reference = train_features[reference_start:reference_end].to(device)
            similarity = queries @ reference.T
            chunk_k = min(k_neighbors, similarity.shape[1])
            chunk_scores, chunk_indices = similarity.topk(chunk_k, dim=1)
            chunk_labels = reference_labels_device[
                reference_start:reference_end
            ][chunk_indices]
            combined_scores = torch.cat((best_scores, chunk_scores), dim=1)
            combined_labels = torch.cat((best_labels, chunk_labels), dim=1)
            best_scores, selected = combined_scores.topk(k_neighbors, dim=1)
            best_labels = combined_labels.gather(1, selected)

        knn_agreement = (best_labels == query_labels[:, None]).float().mean(dim=1)
        vote_counts = F.one_hot(best_labels, num_classes=num_classes).sum(dim=1)
        knn_top1 = vote_counts.argmax(dim=1)

        own_similarity_parts.append(own_similarity.cpu())
        prototype_margin_parts.append(prototype_margin.cpu())
        prototype_top1_parts.append(prototype_top1.cpu())
        knn_agreement_parts.append(knn_agreement.cpu())
        knn_top1_parts.append(knn_top1.cpu())

    return {
        "prototype_own_similarity": torch.cat(own_similarity_parts).numpy(),
        "prototype_margin": torch.cat(prototype_margin_parts).numpy(),
        "prototype_top1": torch.cat(prototype_top1_parts).numpy(),
        "knn_agreement": torch.cat(knn_agreement_parts).numpy(),
        "knn_top1": torch.cat(knn_top1_parts).numpy(),
    }


class _FlippedImageDataset(Dataset):
    def __init__(self, paths: list[str], preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.preprocess(ImageOps.mirror(image.convert("RGB")))


def _load_or_build_flip_features(
    assignments: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    feature_path = output_dir / "flip_features.pt"
    sample_path = output_dir / "flip_feature_sample_ids.json"
    sample_ids = assignments["sample_id"].tolist()
    if feature_path.exists() and sample_path.exists():
        if json.loads(sample_path.read_text(encoding="utf-8")) == sample_ids:
            return torch.load(feature_path, map_location="cpu")

    clip_model, preprocess = load_openai_clip(device)
    clip_model.visual = clip_model.visual.float()
    clip_model.eval()
    dataset = _FlippedImageDataset(assignments["image_path"].tolist(), preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    features = []
    with torch.no_grad():
        for batch_index, images in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            features.append(
                encode_frozen_clip_features(
                    clip_model, images, device, use_amp=False
                ).cpu()
            )
            if batch_index % 25 == 0:
                print(f"flip_features batches={batch_index}/{len(loader)}", flush=True)
    result = torch.cat(features)
    torch.save(result, feature_path)
    sample_path.write_text(json.dumps(sample_ids), encoding="utf-8")
    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _load_strict_features(
    assignments: pd.DataFrame,
    cache_dir: Path,
) -> torch.Tensor:
    features = torch.load(cache_dir / "features.pt", map_location="cpu")
    cache_paths = json.loads((cache_dir / "image_paths.json").read_text(encoding="utf-8"))
    cache_labels = json.loads((cache_dir / "labels.json").read_text(encoding="utf-8"))
    key_to_index = {_canonical_image_key(path): index for index, path in enumerate(cache_paths)}
    indices = []
    for row in assignments.itertuples(index=False):
        key = _canonical_image_key(row.image_path)
        if key not in key_to_index:
            raise ValueError(f"OOF image is missing from feature cache: {row.image_path}")
        cache_index = key_to_index[key]
        if int(cache_labels[cache_index]) != int(row.label):
            raise ValueError(f"Label mismatch for cached OOF sample: {row.image_path}")
        indices.append(cache_index)
    return F.normalize(features[indices].float(), dim=1)


def _duplicate_conflict_flags(assignments: pd.DataFrame, scan_path: Path) -> np.ndarray:
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    conflict_keys = {
        _canonical_image_key(path)
        for group in scan.get("duplicates", [])
        for path in group.get("paths", [])
    }
    return np.array(
        [_canonical_image_key(path) in conflict_keys for path in assignments["image_path"]],
        dtype=bool,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", default="outputs/phase3/oof/fold_assignments.csv")
    parser.add_argument("--cache-dir", default="cache/preliminary/clip_vit_b32_openai")
    parser.add_argument("--duplicate-scan", default="outputs/duplicate_scan.json")
    parser.add_argument("--output-dir", default="outputs/phase3/oof")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--infer-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=500)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--reference-chunk-size", type=int, default=8192)
    parser.add_argument("--flip-batch-size", type=int, default=256)
    parser.add_argument("--flip-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    assignment_path = Path(args.assignments)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments = pd.read_csv(
        assignment_path,
        dtype={"sample_id": str, "image_path": str, "label": int, "fold": int},
    ).sort_values("image_path").reset_index(drop=True)
    if not (cache_dir / "features.pt").exists():
        raise FileNotFoundError(f"Feature cache is incomplete: {cache_dir}")

    strict_features = _load_strict_features(assignments, cache_dir)
    labels = torch.tensor(assignments["label"].to_numpy(copy=True), dtype=torch.long)
    flip_features = _load_or_build_flip_features(
        assignments,
        output_dir,
        device,
        args.flip_batch_size,
        args.flip_workers,
    )
    n_samples = len(assignments)
    merged_logits = torch.empty(n_samples, args.num_classes, dtype=torch.float16)
    signal_names = [
        "prototype_own_similarity",
        "prototype_margin",
        "prototype_top1",
        "knn_agreement",
        "knn_top1",
        "flip_consistency",
        "clip_flip_cosine",
    ]
    signals = {name: np.empty(n_samples) for name in signal_names}
    fold_metrics = []

    for fold in sorted(assignments["fold"].unique()):
        print(f"starting fold={fold}", flush=True)
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        holdout_mask = assignments["fold"].to_numpy() == fold
        train_rows = np.flatnonzero(~holdout_mask)
        holdout_rows = np.flatnonzero(holdout_mask)
        train_features = strict_features[train_rows]
        train_labels = labels[train_rows]
        holdout_features = strict_features[holdout_rows]
        holdout_labels = labels[holdout_rows]

        head, history = train_linear_head(
            train_features,
            train_labels,
            num_classes=args.num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            q=args.q,
            seed=args.seed + int(fold),
            device=device,
        )
        logits = infer_logits(head, holdout_features, args.infer_batch_size, device)
        flip_logits = infer_logits(
            head, flip_features[holdout_rows], args.infer_batch_size, device
        )
        reference_signals = compute_reference_signals(
            train_features,
            train_labels,
            holdout_features,
            holdout_labels,
            num_classes=args.num_classes,
            k_neighbors=args.k_neighbors,
            query_batch_size=args.query_batch_size,
            reference_chunk_size=args.reference_chunk_size,
            device=device,
        )
        for name, values in reference_signals.items():
            signals[name][holdout_rows] = values
        signals["flip_consistency"][holdout_rows] = (
            logits.argmax(dim=1) == flip_logits.argmax(dim=1)
        ).float().numpy()
        signals["clip_flip_cosine"][holdout_rows] = (
            F.normalize(holdout_features, dim=1)
            * F.normalize(flip_features[holdout_rows], dim=1)
        ).sum(dim=1).numpy()
        merged_logits[holdout_rows] = logits.half()

        accuracy = float((logits.argmax(dim=1) == holdout_labels).float().mean())
        fold_record = {
            "fold": int(fold),
            "train_count": int(len(train_rows)),
            "holdout_count": int(len(holdout_rows)),
            "fixed_epochs": args.epochs,
            "holdout_accuracy": accuracy,
            "holdout_used_for_epoch_selection": False,
        }
        fold_metrics.append(fold_record)
        torch.save(
            {
                "state_dict": head.state_dict(),
                "feature_dim": strict_features.shape[1],
                "num_classes": args.num_classes,
                "fold": int(fold),
                "fixed_epochs": args.epochs,
                "q": args.q,
            },
            fold_dir / "linear_head.pt",
        )
        torch.save(
            {
                "row_indices": torch.tensor(holdout_rows),
                "sample_ids": assignments.iloc[holdout_rows]["sample_id"].tolist(),
                "logits": logits.half(),
            },
            fold_dir / "oof_logits.pt",
        )
        (fold_dir / "train_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        (fold_dir / "metrics.json").write_text(
            json.dumps(fold_record, indent=2), encoding="utf-8"
        )
        print(f"completed fold={fold} holdout_accuracy={accuracy:.6f}", flush=True)

    merged_path = output_dir / "oof_logits.pt"
    torch.save(
        {
            "sample_ids": assignments["sample_id"].tolist(),
            "logits": merged_logits,
            "folds": torch.tensor(assignments["fold"].to_numpy(copy=True)),
        },
        merged_path,
    )
    duplicate_flags = _duplicate_conflict_flags(
        assignments, Path(args.duplicate_scan)
    )
    quality = build_sample_quality(
        assignments,
        merged_logits.float(),
        prototype_own_similarity=signals["prototype_own_similarity"],
        prototype_margin=signals["prototype_margin"],
        prototype_top1=signals["prototype_top1"],
        knn_agreement=signals["knn_agreement"],
        knn_top1=signals["knn_top1"],
        flip_consistency=signals["flip_consistency"],
        clip_flip_cosine=signals["clip_flip_cosine"],
        duplicate_conflict_flag=duplicate_flags,
    )
    quality = add_quality_weights(quality)
    quality_path = output_dir / "sample_quality.csv"
    quality.to_csv(quality_path, index=False)
    quality[[
        "sample_id", "image_path", "original_label", "quality", "soft_weight"
    ]].rename(columns={"soft_weight": "weight"}).to_csv(
        output_dir / "oof_soft_weight_manifest.csv", index=False
    )
    quality[[
        "sample_id", "image_path", "original_label", "quality", "discrete_weight"
    ]].rename(columns={"discrete_weight": "weight"}).to_csv(
        output_dir / "oof_discrete_weight_manifest.csv", index=False
    )

    quality["oof_correct"] = quality["oof_top1"] == quality["original_label"]
    class_summary = quality.groupby("original_label").agg(
        sample_count=("sample_id", "count"),
        oof_accuracy=("oof_correct", "mean"),
        mean_p_original=("p_original_label", "mean"),
        mean_quality=("quality", "mean"),
        mean_soft_weight=("soft_weight", "mean"),
        low_weight_fraction=("soft_weight", lambda values: float((values < 0.5).mean())),
    )
    class_summary.to_csv(output_dir / "class_quality_summary.csv")
    warning_classes = class_summary.index[
        class_summary["low_weight_fraction"] > 0.30
    ].astype(int).tolist()
    audit = {
        "sample_count": n_samples,
        "all_samples_have_finite_logits": bool(torch.isfinite(merged_logits).all()),
        "all_samples_have_one_oof_prediction": True,
        "original_validation_used": False,
        "holdout_used_for_epoch_selection": False,
        "oof_accuracy": float(quality["oof_correct"].mean()),
        "prediction_disagreement_rate": float((~quality["oof_correct"]).mean()),
        "soft_weight_min": float(quality["soft_weight"].min()),
        "soft_weight_max": float(quality["soft_weight"].max()),
        "classes_with_over_30pct_weight_below_0_5": warning_classes,
        "fold_metrics": fold_metrics,
    }
    (output_dir / "protocol_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    manifest = {
        "protocol": "W3-0/W3-1 duplicate-aware 3-fold OOF",
        "parent": "b2_gce05",
        "loss": {"name": "gce", "q": args.q},
        "fixed_epochs": args.epochs,
        "fold_seed_base": args.seed,
        "feature_cache": str(cache_dir),
        "fold_assignments_sha256": _sha256(assignment_path),
        "oof_logits_sha256": _sha256(merged_path),
        "sample_quality_sha256": _sha256(quality_path),
        "sample_count": n_samples,
    }
    (output_dir / "oof_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
