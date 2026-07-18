"""Generate OOF zero-weight predictions for d3_strict val fold.

Trains a single linear head on all d3_strict/train.csv cached features,
then predicts on d3_strict/val.csv. Since the model never trained on val,
these ARE valid out-of-fold predictions.

Combines with existing train manifest to produce a full-coverage manifest
(no sample gets default weight 1.0 for "missing from manifest").

Usage:
    python -m analysis.oof.generate_val_oof
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from common.losses import GCELoss


def _canonical_image_key(path: str) -> str:
    """Normalize image path to a canonical key for cache lookup."""
    return str(Path(path).as_posix())


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_head(
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
):
    """Train a frozen-CLIP linear head for a fixed number of epochs."""
    _set_seed(seed)
    features = F.normalize(features.detach().float().cpu(), dim=1)
    labels = labels.detach().long().cpu()
    dataset = TensorDataset(features, labels)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=generator, num_workers=0,
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
    global_step = 0

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
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
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = head(batch_features)
                loss = criterion(logits, batch_labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            n = len(batch_labels)
            total_loss += float(loss.detach()) * n
            total_correct += int((logits.argmax(dim=1) == batch_labels).sum())
            total_samples += n
            global_step += 1

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  epoch={epoch:02d}/{epochs} "
                f"loss={total_loss / max(total_samples, 1):.6f} "
                f"acc={total_correct / max(total_samples, 1):.4f}",
                flush=True,
            )

    return head.cpu()


@torch.no_grad()
def _predict(head, features, device, batch_size=1024):
    """Return softmax probabilities [N, C]."""
    head = head.to(device).eval()
    features = F.normalize(features.float(), dim=1)
    outputs = []
    for start in range(0, len(features), batch_size):
        batch = features[start:start + batch_size].to(device)
        outputs.append(F.softmax(head(batch).float().cpu(), dim=1))
    return torch.cat(outputs, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="outputs/d3_strict/seed42/train.csv")
    parser.add_argument("--val-csv", default="outputs/d3_strict/seed42/val.csv")
    parser.add_argument("--cache-dir", default="cache/preliminary/clip_vit_b32_openai")
    parser.add_argument(
        "--train-manifest",
        default="outputs/phase3/oof/oof_zero_weight_manifest_thresh0.001.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/phase3/oof/oof_zero_weight_manifest_thresh0.001_full.csv",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    # 1. Load feature cache
    cache_dir = Path(args.cache_dir)
    features = torch.load(cache_dir / "features.pt", map_location="cpu")
    cache_paths = json.loads(
        (cache_dir / "image_paths.json").read_text(encoding="utf-8")
    )
    cache_labels = json.loads(
        (cache_dir / "labels.json").read_text(encoding="utf-8")
    )
    key_to_index = {
        _canonical_image_key(p): i for i, p in enumerate(cache_paths)
    }
    print(f"Feature cache: {len(cache_paths)} images, {features.shape[1]}d")

    # 2. Load train and val splits
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    def _resolve(df, split_name):
        indices = []
        labels = []
        missing = []
        for _, row in df.iterrows():
            key = _canonical_image_key(row["image_path"])
            # CSV paths have train_dedup/ or train/ prefix; cache paths are bare
            if key not in key_to_index:
                # Try stripping common prefixes
                for prefix in ("train_dedup/", "train/"):
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                        break
            if key in key_to_index:
                idx = key_to_index[key]
                if int(cache_labels[idx]) != int(row["label"]):
                    raise ValueError(
                        f"Label mismatch for {split_name} sample: {row['image_path']}"
                    )
                indices.append(idx)
                labels.append(int(row["label"]))
            else:
                missing.append(row["image_path"])

        if missing:
            raise RuntimeError(
                f"{len(missing)} {split_name} images not found in feature cache. "
                f"First 5: {missing[:5]}"
            )
        return torch.tensor(indices, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    train_indices, train_labels = _resolve(train_df, "train")
    val_indices, val_labels = _resolve(val_df, "val")
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    # 3. Train head on train features
    print(f"Training head ({args.epochs} epochs, lr={args.lr}, GCE q={args.q})...")
    started = time.time()
    head = _train_head(
        features[train_indices], train_labels,
        num_classes=500,
        epochs=args.epochs,
        batch_size=128,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        q=args.q,
        seed=args.seed,
        device=device,
    )
    print(f"Training complete ({time.time() - started:.0f}s)")

    # 4. Predict on val
    print("Predicting on val...")
    val_probs = _predict(head, features[val_indices], device)
    p_original = val_probs.gather(1, val_labels.unsqueeze(1)).squeeze(1)
    oof_top1 = val_probs.argmax(dim=1)

    # 5. Build val manifest rows
    val_rows = []
    for i, (_, row) in enumerate(val_df.iterrows()):
        p = float(p_original[i])
        w = 0.0 if p < args.threshold else 1.0
        val_rows.append({
            "sample_id": hashlib.sha256(row["image_path"].encode()).hexdigest(),
            "image_path": row["image_path"],
            "original_label": int(row["label"]),
            "training_label": int(row["label"]),
            "sample_weight": w,
            "quality_score": p,
            "source": f"oof_zero_floor_thresh{args.threshold}",
            "oof_top1": int(oof_top1[i]),
            "p_original_label": p,
            "p_top1": float(val_probs[i].max()),
            "prototype_margin": 0.0,
            "knn_agreement": 0.5,
            "flip_consistency": 1.0,
            "duplicate_conflict_flag": False,
        })

    n_zero = sum(1 for r in val_rows if r["sample_weight"] == 0.0)
    print(
        f"Val OOF: {n_zero}/{len(val_rows)} ({100*n_zero/len(val_rows):.1f}%) "
        f"zero-weighted (p < {args.threshold})"
    )

    # 6. Merge with train manifest
    train_manifest = pd.read_csv(args.train_manifest)
    print(f"Train manifest: {len(train_manifest)} entries")

    val_manifest = pd.DataFrame(val_rows)
    combined = pd.concat([train_manifest, val_manifest], ignore_index=True)

    zero_in_train = (train_manifest["sample_weight"] == 0.0).sum()
    zero_in_val = (val_manifest["sample_weight"] == 0.0).sum()
    print(
        f"Combined: {len(combined)} entries "
        f"({len(train_manifest)} train + {len(val_manifest)} val), "
        f"zero-weighted: {zero_in_train} train + {zero_in_val} val "
        f"= {zero_in_train + zero_in_val} total"
    )

    # 7. Write
    output_path = Path(args.output)
    combined.to_csv(output_path, index=False)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
