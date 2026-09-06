"""Build the audited SCOPE-K2 Pass-A parent cache."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from aegis_clip.checkpoint import build_from_checkpoint
from aegis_clip.data import TestImageDataset, TrustBundle, load_class_mapping, resolve_image_path
from aegis_clip.features import canonical_sample_path
from aegis_clip.local_feature_adapter import load_local_feature_adapter
from aegis_clip.local_inference import adapted_dual_local_view_logits
from aegis_clip.localization import extract_attention_crops, forward_with_last_block_attention
from aegis_clip.part_token_adapter import load_part_token_adapter
from aegis_clip.prior_alignment import align_logits_to_prior
from aegis_clip.runtime import atomic_json_dump, seed_worker, set_seed, sha256_file
from aegis_clip.scope_cache import (
    atomic_save_scope_cache,
    formal_row_binding_hash,
    pack_crop_boxes,
    scope_parent_batch_scores,
    stable_top2,
    tensor_sha256,
    validate_parent_cache,
)
from aegis_clip.scope_evidence import validate_scope_parent_model
from aegis_clip.scope_protocol import (
    PARENT_BRANCH_ORDER,
    load_scope_protocol,
    verify_scope_assets,
)


class ValidationImages(Dataset):
    def __init__(self, split_csv: Path, train_root: Path, transform: Any, trust: TrustBundle) -> None:
        frame = pd.read_csv(split_csv).reset_index(drop=True)
        if not {"image_path", "label"}.issubset(frame) or frame["image_path"].duplicated().any():
            raise ValueError("SCOPE validation split is malformed")
        paths = [canonical_sample_path(value) for value in frame["image_path"].astype(str)]
        if paths != sorted(paths):
            raise ValueError("SCOPE validation paths must be canonical-sorted")
        self.paths = paths
        self.labels = frame["label"].astype(int).tolist()
        self.train_root = train_root
        self.transform = transform
        self.trust = trust
        trust.verify_coverage(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        corrupt = False
        try:
            with Image.open(resolve_image_path(self.train_root, path)) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception:
            corrupt = True
            tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
        trust = self.trust.values_for(path, self.labels[index])
        return {
            "images": tensor, "path": path, "corrupt": corrupt,
            "label": torch.tensor(self.labels[index], dtype=torch.int64),
            "clean_probability": trust["clean_probability"].float(),
            "pseudo_label": trust["pseudo_label"].to(torch.int64),
            "correction_alpha": trust["correction_alpha"].float(),
        }


def _sha256_lines(values: list[str]) -> str:
    hasher = hashlib.sha256()
    for value in values:
        hasher.update(str(value).encode("utf-8") + b"\n")
    return hasher.hexdigest()


def _source_tree_sha256(project_root: Path) -> str:
    paths = sorted((project_root / "aegis_clip").glob("scope_*.py"))
    paths += sorted((project_root / "aegis_clip" / "cli").glob("*scope*.py"))
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.relative_to(project_root).as_posix().encode("utf-8") + b"\0")
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _dirty_diff_sha256(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"], cwd=repository_root,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _lineage(protocol: Any, split_sha: str) -> dict[str, str]:
    project_root = Path(__file__).parents[2]
    repository_root = Path(__file__).parents[4]
    return {
        "checkpoint_sha256": protocol.assets.checkpoint_sha256,
        "split_sha256": split_sha,
        "class_to_idx_sha256": protocol.assets.class_to_idx_sha256,
        "idx_to_class_sha256": protocol.assets.idx_to_class_sha256,
        "trust_bundle_sha256": protocol.assets.trust_bundle_sha256,
        "exact_group_sha256": protocol.assets.group_artifact_sha256,
        "protocol_sha256": sha256_file(protocol.config_path),
        "code_sha256": _source_tree_sha256(project_root),
        "dirty_diff_sha256": _dirty_diff_sha256(repository_root),
        "lockfile_sha256": sha256_file(project_root / "uv.lock"),
    }


@torch.inference_mode()
def build_scope_parent_cache(
    checkpoint: str | Path,
    config: str | Path,
    output: str | Path,
    split: str,
    device: str | torch.device,
    batch_size: int,
    num_workers: int,
    *,
    limit: int | None = None,
) -> Path:
    protocol = load_scope_protocol(config)
    verify_scope_assets(protocol)
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if int(batch_size) != int(protocol.fixed["execution"]["batch_size"]):
        raise ValueError("formal SCOPE parent cache requires batch_size=128")
    if int(num_workers) < 0 or (limit is not None and int(limit) <= 0):
        raise ValueError("num_workers/limit is invalid")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if checkpoint_path != protocol.assets.checkpoint or sha256_file(checkpoint_path) != protocol.assets.checkpoint_sha256:
        raise ValueError("SCOPE checkpoint path or SHA-256 mismatch")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("SCOPE CUDA execution requested but CUDA is unavailable")

    model, preprocess, checkpoint_payload = build_from_checkpoint(checkpoint_path, target_device)
    model.eval()
    validate_scope_parent_model(model, checkpoint_payload)
    local_adapter = load_local_feature_adapter(checkpoint_payload, target_device)
    part_adapter = load_part_token_adapter(checkpoint_payload, target_device)
    pool = checkpoint_payload["part_token_adapter"]["spec"]["part_pool_spec"]
    _, idx_to_class = load_class_mapping(protocol.assets.class_to_idx)
    if len(idx_to_class) != 500:
        raise ValueError("SCOPE parent requires exactly 500 classes")
    set_seed(int(protocol.fixed["crossfit"]["seed"]), deterministic=True)

    if split == "validation":
        dataset: Dataset = ValidationImages(
            protocol.assets.val_csv, protocol.assets.train_root, preprocess,
            TrustBundle(protocol.assets.trust_bundle),
        )
        paths = list(dataset.paths)
    else:
        dataset = TestImageDataset(protocol.assets.test_root, preprocess)
        paths = [canonical_sample_path(path.name) for path in dataset.paths]
    if limit is not None:
        dataset = Subset(dataset, range(min(int(limit), len(dataset))))
        paths = paths[: int(limit)]
    expected = protocol.fixed["execution"][
        "expected_validation_samples" if split == "validation" else "expected_test_samples"
    ]
    if limit is None and len(paths) != int(expected):
        raise ValueError(f"SCOPE {split} row count mismatch: {len(paths)} != {expected}")
    if paths != sorted(paths):
        raise ValueError("SCOPE formal paths are not canonical-sorted")
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers),
        timeout=120 if int(num_workers) else 0, pin_memory=target_device.type == "cuda",
        persistent_workers=int(num_workers) > 0, worker_init_fn=seed_worker,
    )

    parent_spec = protocol.fixed["parent"]
    crop_sizes = tuple(int(value) for value in parent_spec["crop_sizes"])
    all_scores: list[torch.Tensor] = []
    all_constituents: list[torch.Tensor] = []
    all_top1: list[torch.Tensor] = []
    all_boxes: list[torch.Tensor] = []
    all_corrupt: list[torch.Tensor] = []
    diagnostics: dict[str, list[torch.Tensor]] = {
        field: [] for field in ("label", "clean_probability", "pseudo_label", "correction_alpha")
    }
    for batch in tqdm(loader, desc=f"SCOPE parent {split}"):
        images = batch["images"].to(target_device, non_blocking=True)
        flipped = torch.flip(images, dims=(3,))
        global_o, attention_o = forward_with_last_block_attention(model, images)
        global_f, attention_f = forward_with_last_block_attention(model, flipped)
        local_o: list[torch.Tensor] = []
        local_f: list[torch.Tensor] = []
        boxes_o: list[list[tuple[int, int, int, int]]] = []
        boxes_f: list[list[tuple[int, int, int, int]]] = []
        for crop_size in crop_sizes:
            crop_o, box_o = extract_attention_crops(
                images, attention_o, crop_size=crop_size, top_k=int(parent_spec["local_top_k"])
            )
            crop_f, box_f = extract_attention_crops(
                flipped, attention_f, crop_size=crop_size, top_k=int(parent_spec["local_top_k"])
            )
            local_o.append(adapted_dual_local_view_logits(
                model, local_adapter, part_adapter, crop_o,
                part_top_patches=int(pool["top_patches"]),
                part_temperature=float(pool["temperature"]),
            ))
            local_f.append(adapted_dual_local_view_logits(
                model, local_adapter, part_adapter, crop_f,
                part_top_patches=int(pool["top_patches"]),
                part_temperature=float(pool["temperature"]),
            ))
            boxes_o.append(box_o)
            boxes_f.append(box_f)
        scores = scope_parent_batch_scores(
            global_o, local_o, global_f, local_f,
            global_temperature=float(parent_spec["global_temperature"]),
            local_temperature=float(parent_spec["local_temperature"]),
            local_scale_weights=parent_spec["local_scale_weights"],
            local_weight=float(parent_spec["local_weight"]),
            flip_weight=float(parent_spec["flip_weight"]),
        )
        all_scores.append(scores.fused_log_scores.cpu())
        all_constituents.append(scores.constituent_scores.cpu())
        all_top1.append(scores.constituent_top1.cpu())
        all_boxes.append(pack_crop_boxes(boxes_o, boxes_f))
        all_corrupt.append(torch.as_tensor(batch["corrupt"], dtype=torch.bool).cpu())
        if split == "validation":
            for field in diagnostics:
                diagnostics[field].append(torch.as_tensor(batch[field]).cpu())

    raw_log_scores = torch.cat(all_scores).float()
    prior = parent_spec["prior"]
    aligned_scores, report, applied_bias = align_logits_to_prior(
        raw_log_scores, strength=float(prior["strength"]),
        max_iterations=int(prior["max_iterations"]),
        tolerance=float(prior["tolerance"]), damping=float(prior["damping"]),
        return_applied_bias=True,
    )
    aligned_scores = aligned_scores.float().cpu()
    applied_bias = applied_bias.float().cpu()
    if not torch.equal(aligned_scores, raw_log_scores.cpu() + applied_bias):
        raise ValueError("balanced-prior did not produce one shared class bias")
    candidates, candidate_scores = stable_top2(aligned_scores)
    rows = torch.arange(len(paths), dtype=torch.int64)
    output_path = Path(output)
    prior_path = output_path.with_suffix(".prior.json")
    atomic_json_dump(dict(report), prior_path)
    split_sha = protocol.assets.val_csv_sha256 if split == "validation" else _sha256_lines(paths)
    constituents = torch.cat(all_constituents).float()
    payload: dict[str, Any] = {
        "schema": protocol.fixed["schemas"]["parent"], "split": split,
        "paths": paths, "formal_row_id": rows,
        "formal_row_binding_sha256": formal_row_binding_hash(rows, paths),
        "candidate_indices": candidates, "candidate_parent_log_scores": candidate_scores,
        "parent_margin": candidate_scores[:, 1] - candidate_scores[:, 0],
        "parent_predictions": candidates[:, 0].clone(),
        "constituent_scores": constituents,
        "constituent_top1": torch.cat(all_top1).to(torch.int64),
        "constituent_order": list(PARENT_BRANCH_ORDER),
        "constituent_scores_sha256": tensor_sha256(constituents),
        "crop_boxes": torch.cat(all_boxes).to(torch.int64),
        "corrupt": torch.cat(all_corrupt).bool(),
        "prior_bias": applied_bias.to(torch.float64),
        "prior_iterations": int(report["iterations"]),
        "prior_report_sha256": sha256_file(prior_path),
        "aligned_log_scores_shape": list(aligned_scores.shape),
        "aligned_log_scores_dtype": "float32",
        "aligned_log_scores_sha256": tensor_sha256(aligned_scores),
        "lineage": _lineage(protocol, split_sha),
        "_run_metadata": {
            "argv": list(sys.argv), "cwd": os.getcwd(), "formal": limit is None,
            "limit": limit, "python": sys.version, "torch": torch.__version__,
            "numpy": np.__version__, "sklearn": sklearn.__version__,
        },
    }
    if split == "validation":
        payload.update({field: torch.cat(values) for field, values in diagnostics.items()})
        payload["label"] = payload["label"].to(torch.int64)
        payload["pseudo_label"] = payload["pseudo_label"].to(torch.int64)
        payload["clean_probability"] = payload["clean_probability"].float()
        payload["correction_alpha"] = payload["correction_alpha"].float()
    validate_parent_cache(payload, protocol, split)
    manifest = atomic_save_scope_cache(payload, output_path)
    atomic_json_dump(
        {"cache_sha256": manifest.sha256, "semantic_sha256": manifest.semantic_sha256,
         "rows": len(paths), "split": split, "formal": limit is None},
        output_path.with_suffix(".run.json"),
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    build_scope_parent_cache(
        args.checkpoint, args.config, args.output, args.split, args.device,
        args.batch_size, args.num_workers, limit=args.limit,
    )


if __name__ == "__main__":
    main()
