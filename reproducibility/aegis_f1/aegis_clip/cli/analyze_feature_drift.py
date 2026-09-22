"""Per-class feature drift versus per-class accuracy change, FT against OpenAI.

Motivation: RM-FT beats RM-LP by +9.23pp on local val, and that gain appears
within the first two epochs -- i.e. it is produced by *unfreezing*, not by
learning longer.  The open question is whether FT's gain is "a better
representation" (which should transfer across sources) or "a more thorough fit
of the pool's domain" (which should not).

This tool measures the only local signal with a theoretical link to that
question: how far FT moved each class's features away from the frozen OpenAI
features, and whether the classes it moved most are the classes it improved
most on.  A tight positive relation means FT's gain is concentrated where it
re-shaped the representation -- the domain-fitting reading.  No relation means
the gain is spread independently of how much the features moved.

Caveat, stated up front: this is a weak, anticipatory signal.  It cannot
adjudicate transfer, because (see docs/rematch750_lp_ft_transfer_prereg_20260922.md
section 1) the validation split is a faithful iid sample of the training pool.
It only produces a prior on what the platform's LP arm is about to say.

Reads the OpenAI features from the frozen cache and runs FT over val only; the
test set is not touched.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.config import load_config
from aegis_clip.data import _load_split, resolve_image_path
from aegis_clip.features import FrozenFeatureStore, canonical_sample_path
from aegis_clip.runtime import seed_worker


class ValidationWithReference(Dataset):
    """Val images plus their cached OpenAI feature and class label."""

    def __init__(self, split_csv, image_root, transform, feature_store):
        frame = _load_split(split_csv)
        self.paths = frame["image_path"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.image_root = Path(image_root)
        self.transform = transform
        self.feature_store = feature_store

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        relative_path = self.paths[index]
        absolute_path = resolve_image_path(self.image_root, relative_path)
        with Image.open(io.BytesIO(absolute_path.read_bytes())) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "images": tensor,
            "reference_features": self.feature_store.get(relative_path),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "path": canonical_sample_path(relative_path),
        }


@torch.no_grad()
def collect(model, loader, device, use_amp):
    """Return per-sample FT feature, OpenAI feature, label and correctness."""
    model.eval()
    feats, refs, labels, correct = [], [], [], []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, enabled=use_amp and device.type == "cuda"
        ):
            logits, features = model(images=images, return_features=True)
        labels.append(batch["label"].clone())
        refs.append(batch["reference_features"].float().clone())
        feats.append(F.normalize(features.float(), dim=1).cpu())
        correct.append(
            (logits.float().argmax(dim=1).cpu() == batch["label"]).long()
        )
    return (
        torch.cat(feats),
        F.normalize(torch.cat(refs), dim=1),
        torch.cat(labels),
        torch.cat(correct),
    )


def average_ranks(v):
    """Ranks starting at 0, ties sharing their mean rank.

    A plain argsort would hand tied values distinct ranks in index order, which
    fabricates a relation that is not in the data -- a constant column would
    come out as strongly correlated with whatever it is compared against.
    """
    order = v.argsort()
    ordered = v[order]
    ranks = torch.empty(len(v), dtype=torch.float64)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and bool(ordered[j + 1] == ordered[i]):
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx = average_ranks(x)
    ry = average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="FT best.pt")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ft-evaluation", required=True,
                        help="FT logs/evaluation_epoch_*.json (has per_class)")
    parser.add_argument("--lp-evaluation", required=True,
                        help="RM-LP logs/evaluation_epoch_*.json (has per_class)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="defaults to evaluation.batch_size, which is what produced the "
             "recorded metrics; under autocast a different batch shape can make "
             "cuDNN pick other kernels and flip a boundary image, so the anchor "
             "will not reproduce with an arbitrary value",
    )
    parser.add_argument("--anchor-micro", type=float, default=None,
                        help="recorded FT val micro; the run is rejected if the "
                             "forward here does not reproduce it")
    parser.add_argument("--anchor-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    device = torch.device(args.device)
    config = load_config(args.config)
    model, preprocess, _ = build_from_checkpoint(
        args.checkpoint, device, config_override=config
    )
    feature_config = config["features"]
    feature_store = FrozenFeatureStore(
        feature_config["tensor_path"],
        feature_config["paths_path"],
        feature_config.get("manifest_path"),
        expected_dim=int(config["model"].get("feature_dim", 512)),
    )
    workers = int(config["train"].get("num_workers", 4))
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(config.get("evaluation", {}).get("batch_size", 128))
    )
    loader = DataLoader(
        ValidationWithReference(
            config["data"]["val_csv"], config["data"]["train_root"],
            preprocess, feature_store,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        timeout=int(config["train"].get("loader_timeout", 120 if workers else 0)),
        pin_memory=bool(config["train"].get("pin_memory", True)),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
    )

    feats, refs, labels, correct = collect(
        model, loader, device, bool(config["train"].get("amp", True))
    )
    micro = float(correct.float().mean())
    print(f"val micro (this forward) = {micro!r} over {len(labels)} samples")
    if args.anchor_micro is not None:
        if abs(micro - args.anchor_micro) > args.anchor_tolerance:
            raise SystemExit(
                f"ANCHOR FAILED: {micro!r} != recorded {args.anchor_micro!r}; "
                "this is not the same forward pass"
            )
        print(f"anchor OK (recorded {args.anchor_micro!r})")

    # drift = 1 - cosine(FT feature, frozen OpenAI feature); both L2-normalised.
    drift = 1.0 - (feats * refs).sum(dim=1)
    print(f"mean feature drift = {float(drift.mean())!r}")

    ft_eval = json.loads(Path(args.ft_evaluation).read_text(encoding="utf-8"))
    lp_eval = json.loads(Path(args.lp_evaluation).read_text(encoding="utf-8"))
    ft_by_class = {int(r["label"]): r for r in ft_eval["per_class"]}
    lp_by_class = {int(r["label"]): r for r in lp_eval["per_class"]}

    rows = []
    for label in sorted(ft_by_class):
        mask = labels == label
        n = int(mask.sum())
        if n == 0:
            continue
        ft_row, lp_row = ft_by_class[label], lp_by_class[label]
        if int(ft_row["val_samples"]) != n:
            raise SystemExit(
                f"class {label}: evaluation json has {ft_row['val_samples']} val "
                f"samples but this forward saw {n}"
            )
        rows.append({
            "label": label,
            "val_samples": n,
            "train_samples": int(ft_row["train_samples"]),
            "drift": float(drift[mask].mean()),
            "ft_recall": float(ft_row["recall"]),
            "lp_recall": float(lp_row["recall"]),
            "delta_recall": float(ft_row["recall"]) - float(lp_row["recall"]),
        })

    d = torch.tensor([r["drift"] for r in rows], dtype=torch.float64)
    dd = torch.tensor([r["delta_recall"] for r in rows], dtype=torch.float64)
    w = torch.tensor([r["val_samples"] for r in rows], dtype=torch.float64)
    pearson = float(
        ((d - (d * w).sum() / w.sum()) * (dd - (dd * w).sum() / w.sum())).sum()
        / (
            ((d - (d * w).sum() / w.sum()) ** 2 * w).sum().sqrt()
            * ((dd - (dd * w).sum() / w.sum()) ** 2 * w).sum().sqrt()
        ).clamp_min(1e-12)
    )
    print(f"classes={len(rows)}  pearson(drift, delta_recall)={pearson:+.4f}  "
          f"spearman={spearman(d, dd):+.4f}")

    # Equal-count quintiles by drift, so the relation can be read without a fit.
    order = d.argsort()
    k = len(order) // 5
    quintiles = []
    print("\n  drift 五分位       类数  val数   mean_drift  mean_drecall  dMicro")
    for i in range(5):
        idx = order[i * k:(i + 1) * k] if i < 4 else order[4 * k:]
        n_val = float(w[idx].sum())
        d_micro = float((dd[idx] * w[idx]).sum() / n_val)
        quintiles.append({
            "quintile": i + 1,
            "classes": len(idx),
            "val_samples": int(n_val),
            "mean_drift": float(d[idx].mean()),
            "mean_delta_recall": float(dd[idx].mean()),
            "delta_micro": d_micro,
        })
        print(f"  Q{i + 1}                 {len(idx):<5} {int(n_val):<6} "
              f"{float(d[idx].mean()):10.4f}  {float(dd[idx].mean()):+11.4f}  "
              f"{d_micro * 100:+7.2f}pp")

    Path(args.output).write_text(
        json.dumps({
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "val_micro_this_forward": micro,
            "mean_feature_drift": float(drift.mean()),
            "classes": len(rows),
            "pearson_drift_vs_delta_recall": pearson,
            "spearman_drift_vs_delta_recall": spearman(d, dd),
            "drift_quintiles": quintiles,
            "per_class": rows,
            "interpretation": (
                "positive relation => FT's gain concentrates where it moved the "
                "representation most (domain-fitting reading); ~0 => the gain is "
                "independent of how much the features moved"
            ),
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
