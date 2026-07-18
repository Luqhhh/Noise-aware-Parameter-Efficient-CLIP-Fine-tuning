"""Compute knn_top1_agreement using OOF fold-safe references.

Each fold's samples query ONLY samples from OTHER folds.
Matches original OOF knn_agreement and knn_top1 computation.

Output: knn_top1_agreement column only (appended to new CSV).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _canonical_key(path: str) -> str:
    return str(Path(path).as_posix())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="cache/preliminary/clip_vit_b32_openai")
    parser.add_argument("--sample-quality", default="outputs/phase/phase3/oof/sample_quality.csv")
    parser.add_argument("--output-csv", default="outputs/phase/phase3/oof/knn_top1_agreement.csv")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--query-batch", type=int, default=256)
    parser.add_argument("--ref-chunk", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load cache
    cache_dir = Path(args.cache_dir)
    features = torch.load(cache_dir / "features.pt", map_location="cpu").float()
    features = F.normalize(features, dim=1)
    cache_paths = json.loads((cache_dir / "image_paths.json").read_text())
    cache_labels = torch.tensor(json.loads((cache_dir / "labels.json").read_text()))
    key_to_idx = {_canonical_key(p): i for i, p in enumerate(cache_paths)}

    # Load quality with fold info
    quality = pd.read_csv(args.sample_quality)
    n = len(quality)

    # Map image_path -> cache index + fold
    cache_indices = []
    folds = quality["fold"].to_numpy()
    for _, row in quality.iterrows():
        key = _canonical_key(row["image_path"])
        for prefix in ("train_dedup/", "train/"):
            if key not in key_to_idx and key.startswith(prefix):
                key = key[len(prefix):]
                break
        cache_indices.append(key_to_idx[key])
    cache_indices = torch.tensor(cache_indices, dtype=torch.long)
    all_labels = cache_labels[cache_indices]

    k = args.k
    knn_top1_agreement = np.empty(n, dtype=np.float32)

    # Pre-compute reference sets per fold: all indices NOT in this fold
    fold_ids = sorted(set(folds))
    fold_mask = {}
    for f in fold_ids:
        fmask = folds != f
        fold_mask[f] = torch.tensor(np.flatnonzero(fmask), dtype=torch.long)
        print(f"  fold={f}: {int(fmask.sum())} reference samples", flush=True)

    features_all = features[cache_indices].to(device)
    labels_all_device = all_labels.to(device)
    query_batch = args.query_batch

    started = time.time()
    for f in fold_ids:
        ref_indices = fold_mask[f]
        ref_features = features_all[ref_indices]
        ref_labels = labels_all_device[ref_indices]
        query_mask = folds == f
        query_idx = np.flatnonzero(query_mask)
        if len(query_idx) == 0:
            continue

        print(f"Processing fold={f}: {len(query_idx)} queries", flush=True)
        fold_started = time.time()

        for qb_start in range(0, len(query_idx), query_batch):
            qb_end = min(qb_start + query_batch, len(query_idx))
            qb_indices = query_idx[qb_start:qb_end]
            queries = features_all[qb_indices]
            q_labels = labels_all_device[qb_indices]
            m = len(queries)

            best_scores = torch.full((m, k), float("-inf"), device=device)
            best_labels = torch.zeros((m, k), dtype=torch.long, device=device)

            for r_start in range(0, len(ref_features), args.ref_chunk):
                r_end = min(r_start + args.ref_chunk, len(ref_features))
                ref = ref_features[r_start:r_end]
                similarity = queries @ ref.T
                chunk_k = min(k, similarity.shape[1])
                chunk_scores, chunk_indices = similarity.topk(chunk_k, dim=1)
                chunk_labels = ref_labels[r_start:r_end][chunk_indices]
                combined_scores = torch.cat([best_scores, chunk_scores], dim=1)
                combined_labels = torch.cat([best_labels, chunk_labels], dim=1)
                best_scores, selected = combined_scores.topk(k, dim=1)
                best_labels = combined_labels.gather(1, selected)

            # knn_top1_agreement
            vote_counts = F.one_hot(best_labels, num_classes=500).sum(dim=1)
            top1 = vote_counts.argmax(dim=1)
            top1_agree = (best_labels == top1[:, None]).float().mean(dim=1)
            knn_top1_agreement[qb_indices] = top1_agree.cpu().numpy()

        elapsed = time.time() - fold_started
        print(f"  fold={f} done in {elapsed:.0f}s", flush=True)

    # Output
    out_df = pd.DataFrame({
        "image_path": quality["image_path"].tolist(),
        "knn_top1_agreement": knn_top1_agreement.tolist(),
    })
    out_df.to_csv(args.output_csv, index=False)

    total_elapsed = time.time() - started
    print(f"Wrote {args.output_csv} ({len(out_df)} rows) in {total_elapsed:.0f}s", flush=True)
    print(f"  knn_top1_agreement mean: {knn_top1_agreement.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
