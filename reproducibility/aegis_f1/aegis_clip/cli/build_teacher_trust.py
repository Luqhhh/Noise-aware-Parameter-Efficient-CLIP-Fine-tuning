"""Build a bounded teacher-augmented trust bundle from official train images."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.data import OnlineImageDataset
from aegis_clip.features import FrozenFeatureStore, canonical_sample_path
from aegis_clip.runtime import atomic_json_dump, seed_worker, sha256_file
from aegis_clip.teacher_trust import augment_teacher_trust


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--base-trust", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--minimum-confidence", type=float, default=0.90)
    parser.add_argument("--minimum-margin", type=float, default=0.75)
    parser.add_argument("--maximum-clean-probability", type=float, default=0.60)
    parser.add_argument("--admission-clean-probability", type=float, default=0.65)
    parser.add_argument("--correction-alpha", type=float, default=0.50)
    parser.add_argument("--maximum-class-fraction", type=float, default=0.08)
    parser.add_argument(
        "--teacher-logits-cache",
        help=(
            "Optional reusable center/flip teacher-logit cache. If the file "
            "exists it is verified and reused; otherwise it is created atomically."
        ),
    )
    parser.add_argument("--overwrite-teacher-logits-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    train_csv = Path(args.train_csv).resolve()
    base_trust_path = Path(args.base_trust).resolve()
    destination = Path(args.output).resolve()
    audit_path = Path(args.audit_output).resolve()
    if (destination.exists() or audit_path.exists()) and not args.overwrite:
        raise FileExistsError("teacher trust output exists; pass --overwrite")

    checkpoint_sha256 = sha256_file(checkpoint_path)
    train_csv_sha256 = sha256_file(train_csv)
    cache_path = (
        Path(args.teacher_logits_cache).resolve()
        if args.teacher_logits_cache
        else None
    )
    reuse_cache = bool(
        cache_path is not None
        and cache_path.exists()
        and not args.overwrite_teacher_logits_cache
    )
    if reuse_cache:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        required_cache = {
            "center_logits",
            "flip_logits",
            "labels",
            "paths",
            "checkpoint_sha256",
            "train_csv_sha256",
        }
        missing_cache = required_cache - set(cache)
        if missing_cache:
            raise ValueError(
                f"teacher logits cache is missing fields: {sorted(missing_cache)}"
            )
        if cache["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError("teacher logits cache checkpoint SHA-256 mismatch")
        if cache["train_csv_sha256"] != train_csv_sha256:
            raise ValueError("teacher logits cache train CSV SHA-256 mismatch")
        center_logits = torch.as_tensor(cache["center_logits"]).float().cpu()
        flip_logits = torch.as_tensor(cache["flip_logits"]).float().cpu()
        labels = torch.as_tensor(cache["labels"]).long().flatten().cpu()
        paths = [canonical_sample_path(path) for path in cache["paths"]]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, preprocess, checkpoint = build_from_checkpoint(checkpoint_path, device)
        config = checkpoint["config"]
        features = config["features"]
        feature_store = FrozenFeatureStore(
            features["tensor_path"],
            features["paths_path"],
            features.get("manifest_path"),
            expected_dim=int(config["model"].get("feature_dim", 512)),
        )
        dataset = OnlineImageDataset(
            train_csv,
            config["data"]["train_root"],
            preprocess,
            feature_store,
            trust_bundle=None,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
            persistent_workers=int(args.num_workers) > 0,
            worker_init_fn=seed_worker,
        )
        model.eval()
        center_parts: list[torch.Tensor] = []
        flip_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        paths = []
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            center_parts.append(model(images=images).float().cpu())
            flip_parts.append(model(images=torch.flip(images, dims=[3])).float().cpu())
            label_parts.append(batch["label"].long().cpu())
            paths.extend(canonical_sample_path(path) for path in batch["path"])
        center_logits = torch.cat(center_parts)
        flip_logits = torch.cat(flip_parts)
        labels = torch.cat(label_parts)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            torch.save(
                {
                    "center_logits": center_logits,
                    "flip_logits": flip_logits,
                    "labels": labels,
                    "paths": paths,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "train_csv": str(train_csv),
                    "train_csv_sha256": train_csv_sha256,
                },
                cache_temporary,
            )
            os.replace(cache_temporary, cache_path)

    if center_logits.ndim != 2 or center_logits.shape != flip_logits.shape:
        raise ValueError("teacher logits cache must contain equal [N,C] tensors")
    if center_logits.shape[0] != labels.numel() or labels.numel() != len(paths):
        raise ValueError("teacher logits cache sample dimensions are inconsistent")
    if not torch.isfinite(center_logits).all() or not torch.isfinite(flip_logits).all():
        raise ValueError("teacher logits cache contains non-finite values")
    base = torch.load(base_trust_path, map_location="cpu", weights_only=False)
    base_paths = [canonical_sample_path(path) for path in base["paths"]]
    if len(paths) != len(set(paths)) or len(base_paths) != len(set(base_paths)):
        raise ValueError("teacher split and base trust paths must both be unique")
    teacher_index = {path: index for index, path in enumerate(paths)}
    if set(teacher_index) != set(base_paths):
        missing = sorted(set(base_paths) - set(teacher_index))
        extra = sorted(set(teacher_index) - set(base_paths))
        raise ValueError(
            "teacher split and base trust path sets differ: "
            f"missing={len(missing)} extra={len(extra)}"
        )
    reorder = torch.tensor(
        [teacher_index[path] for path in base_paths], dtype=torch.long
    )
    center_logits = center_logits[reorder]
    flip_logits = flip_logits[reorder]
    labels = labels[reorder]
    output, audit = augment_teacher_trust(
        base,
        labels,
        center_logits,
        flip_logits,
        temperature=args.temperature,
        minimum_confidence=args.minimum_confidence,
        minimum_margin=args.minimum_margin,
        maximum_clean_probability=args.maximum_clean_probability,
        admission_clean_probability=args.admission_clean_probability,
        correction_alpha=args.correction_alpha,
        maximum_class_fraction=args.maximum_class_fraction,
    )
    output.setdefault("metadata", {})["teacher_augmentation"] = {
        "method": "two_view_bounded_teacher_trust_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "train_csv": str(train_csv),
        "train_csv_sha256": sha256_file(train_csv),
        "base_trust": str(base_trust_path),
        "base_trust_sha256": sha256_file(base_trust_path),
        **audit,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(output, temporary)
    os.replace(temporary, destination)
    audit.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "train_csv": str(train_csv),
            "train_csv_sha256": sha256_file(train_csv),
            "base_trust": str(base_trust_path),
            "base_trust_sha256": sha256_file(base_trust_path),
            "output": str(destination),
            "output_sha256": sha256_file(destination),
            "teacher_logits_cache": str(cache_path) if cache_path else None,
            "teacher_logits_cache_sha256": (
                sha256_file(cache_path) if cache_path else None
            ),
            "teacher_logits_cache_reused": reuse_cache,
        }
    )
    atomic_json_dump(audit, audit_path)
    print(audit)


if __name__ == "__main__":
    main()
