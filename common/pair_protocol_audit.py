"""
Experiment pair protocol audit.

Compares two experiment configs, their effective training samples, class mappings,
and checkpoints to determine whether they form a valid paired comparison suitable
for causal attribution of performance differences.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import yaml

from common.utils import load_config

logger = logging.getLogger(__name__)


def _sha256_hex(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# Fields allowed to differ between reference and candidate configs
ALLOWED_DIFFERENCES = {
    "experiment.id",
    "output.log_dir",
    "output.submission_dir",
    "train.save_dir",
    "loss.name",
    "loss.q",
    "loss.probability_epsilon",
    "loss.epsilon",
    "loss.reduction",
}

# Fields that must be identical
REQUIRED_IDENTICAL = [
    "experiment.mode",
    "experiment.head_type",
    "experiment.augmentation_preset",
    "data.seed",
    "data.split_seed",
    "data.train_seed",
    "data.split_dir",
    "data.class_mapping_path",
    "model.clip_model_name",
    "model.feature_dim",
    "model.freeze_clip",
    "model.num_classes",
    "model.unfreeze_last_n_blocks",
    "model.train_ln_post",
    "model.train_visual_proj",
    "train.batch_size",
    "train.epochs",
    "train.lr",
    "train.weight_decay",
    "train.scheduler",
    "train.warmup_epochs",
    "train.min_lr_ratio",
    "train.early_stop_patience",
    "train.max_grad_norm",
    "train.amp",
]


def _nested_get(d: dict, dotted_key: str, default=object()):
    """Get a nested dict value by dotted key, e.g. 'data.seed'."""
    keys = dotted_key.split(".")
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            if default is not object():
                return default
            raise KeyError(dotted_key)
    return d


@dataclass(frozen=True)
class PairAuditResult:
    paired_valid: bool
    causal_claim_allowed: bool
    allowed_differences: List[str]
    unexpected_differences: List[dict]
    warnings: List[str]
    hashes: Dict[str, str]
    counts: Dict[str, int]
    resolved_paths: Dict[str, List[str]]
    sample_classification: str = "not_checked"  # identical_effective_samples | different_effective_samples | ...
    max_visual_abs_diff: float = 0.0
    extra: dict = field(default_factory=dict)


def _resolve_effective_samples(split_csv_path: str) -> dict:
    """Resolve the actual training samples from a split CSV.

    Returns dict with: count, missing, paths, sha256_set, label_set, sha256_label_set.
    """
    csv_path = Path(split_csv_path)
    if not csv_path.exists():
        return {"count": 0, "missing": 0, "paths": [], "error": "CSV not found"}

    df = pd.read_csv(csv_path)
    result = {"count": len(df), "missing": 0, "paths": [], "sha256_set": set(),
              "label_set": set(), "sha256_label_set": set()}

    cwd = Path.cwd()
    for _, row in df.iterrows():
        img_path = Path(row["image_path"])
        if not img_path.is_absolute():
            img_path = cwd / img_path
        abs_path = str(img_path.resolve())
        result["paths"].append(abs_path)

        if img_path.exists():
            sha = _sha256_hex(img_path)
            result["sha256_set"].add(sha)
            result["sha256_label_set"].add((sha, int(row["label"])))
        else:
            result["missing"] += 1

        result["label_set"].add(int(row["label"]))

    return result


def _compare_config_fields(ref_config: dict, cand_config: dict) -> Tuple[List[str], List[dict]]:
    """Compare config fields. Returns (allowed_diffs, unexpected_diffs)."""
    allowed = []
    unexpected = []

    for key_path in REQUIRED_IDENTICAL:
        try:
            ref_val = _nested_get(ref_config, key_path)
            cand_val = _nested_get(cand_config, key_path)
        except KeyError:
            unexpected.append({"field": key_path, "issue": "missing_in_one_config",
                               "ref": "KeyError", "cand": "KeyError"})
            continue

        if ref_val != cand_val:
            if key_path in ALLOWED_DIFFERENCES:
                allowed.append(key_path)
            else:
                unexpected.append({
                    "field": key_path,
                    "ref_value": str(ref_val),
                    "cand_value": str(cand_val),
                })

    return allowed, unexpected


def _compare_visual_encoders(ref_ckpt: dict, cand_ckpt: dict) -> float:
    """Compare visual encoder weights. Returns max absolute difference."""
    ref_state = ref_ckpt.get("model_state_dict", ref_ckpt)
    cand_state = cand_ckpt.get("model_state_dict", cand_ckpt)

    visual_keys = [k for k in ref_state if k.startswith("visual.")]
    if not visual_keys:
        return 0.0

    max_diff = 0.0
    for k in visual_keys:
        if k in cand_state:
            diff = (ref_state[k].float() - cand_state[k].float()).abs().max().item()
            max_diff = max(max_diff, diff)

    return max_diff


def audit_experiment_pair(
    reference_config_path: str,
    candidate_config_path: str,
    reference_ckpt_path: str,
    candidate_ckpt_path: str,
    output_path: str,
) -> PairAuditResult:
    """Run full protocol audit between two experiments.

    Args:
        reference_config_path: Path to reference experiment YAML config.
        candidate_config_path: Path to candidate experiment YAML config.
        reference_ckpt_path: Path to reference checkpoint .pt file.
        candidate_ckpt_path: Path to candidate checkpoint .pt file.
        output_path: Path to write audit JSON.

    Returns:
        PairAuditResult with full audit details.
    """
    warnings = []
    allowed_diffs = []
    unexpected_diffs = []
    hashes = {}
    counts = {}

    # Load configs
    ref_config = load_config(reference_config_path)
    cand_config = load_config(candidate_config_path)

    # -- A. Split and class mapping --
    ref_split_dir = Path(ref_config["data"]["split_dir"])
    cand_split_dir = Path(cand_config["data"]["split_dir"])

    for name, path in [("ref_val", ref_split_dir / "val.csv"),
                        ("cand_val", cand_split_dir / "val.csv"),
                        ("ref_train", ref_split_dir / "train.csv"),
                        ("cand_train", cand_split_dir / "train.csv")]:
        if path.exists():
            hashes[name] = _sha256_hex(path)
        else:
            hashes[name] = "MISSING"
            warnings.append(f"{name} not found at {path}")

    # Validation CSV must be identical
    if hashes.get("ref_val") and hashes.get("cand_val"):
        if hashes["ref_val"] != hashes["cand_val"]:
            unexpected_diffs.append({
                "field": "val.csv",
                "ref_sha256": hashes["ref_val"],
                "cand_sha256": hashes["cand_val"],
            })

    # Class mapping
    class_mapping_path = ref_config["data"].get("class_mapping_path", ref_config["data"]["split_dir"])
    for fname in ["class_to_idx.json", "idx_to_class.json"]:
        p = Path(class_mapping_path) / fname
        if p.exists():
            hashes[fname] = _sha256_hex(p)

    # -- B. Effective training samples --
    ref_samples = _resolve_effective_samples(str(ref_split_dir / "train.csv"))
    cand_samples = _resolve_effective_samples(str(cand_split_dir / "train.csv"))

    counts["ref_train_samples"] = ref_samples["count"]
    counts["cand_train_samples"] = cand_samples["count"]
    counts["ref_train_missing"] = ref_samples.get("missing", 0)
    counts["cand_train_missing"] = cand_samples.get("missing", 0)

    # Classify sample relationship
    if ref_samples.get("sha256_set") and cand_samples.get("sha256_set"):
        if ref_samples["sha256_set"] == cand_samples["sha256_set"]:
            if ref_samples.get("sha256_label_set") == cand_samples.get("sha256_label_set"):
                sample_class = "identical_effective_samples"
            else:
                sample_class = "same_images_different_labels"
        else:
            overlap = ref_samples["sha256_set"] & cand_samples["sha256_set"]
            if len(overlap) == min(len(ref_samples["sha256_set"]), len(cand_samples["sha256_set"])):
                sample_class = "same_images_different_paths"
            else:
                sample_class = "different_effective_samples"
    else:
        sample_class = "could_not_compute"

    if sample_class != "identical_effective_samples":
        warnings.append(f"Training samples classified as: {sample_class}")
    if ref_samples.get("missing", 0) > 0 or cand_samples.get("missing", 0) > 0:
        warnings.append("Missing training files detected")

    # -- C. Config comparison --
    allowed_diffs, unexpected_diffs = _compare_config_fields(ref_config, cand_config)

    # train_dir is a special case -- warn if different but effective samples same
    if ref_config["data"].get("train_dir") != cand_config["data"].get("train_dir"):
        warnings.append(
            f"data.train_dir differs: ref={ref_config['data']['train_dir']}, "
            f"cand={cand_config['data']['train_dir']}. "
            f"Deferring to effective sample audit."
        )

    # -- D. Checkpoint comparison --
    ref_ckpt = torch.load(reference_ckpt_path, map_location="cpu", weights_only=False)
    cand_ckpt = torch.load(candidate_ckpt_path, map_location="cpu", weights_only=False)

    hashes["ref_checkpoint"] = _sha256_hex(Path(reference_ckpt_path))
    hashes["cand_checkpoint"] = _sha256_hex(Path(candidate_ckpt_path))

    ref_state = ref_ckpt.get("model_state_dict", {})
    cand_state = cand_ckpt.get("model_state_dict", {})

    ref_keys = set(ref_state.keys())
    cand_keys = set(cand_state.keys())
    if ref_keys != cand_keys:
        unexpected_diffs.append({
            "field": "checkpoint.state_dict_keys",
            "only_in_ref": sorted(ref_keys - cand_keys),
            "only_in_cand": sorted(cand_keys - ref_keys),
        })

    # Classifier shape
    for ckpt_name, state in [("ref", ref_state), ("cand", cand_state)]:
        if "classifier.weight" in state:
            w = state["classifier.weight"]
            counts[f"{ckpt_name}_classifier_weight_shape"] = list(w.shape)
        if "classifier.bias" in state:
            counts[f"{ckpt_name}_classifier_bias_shape"] = list(state["classifier.bias"].shape)

    counts["ref_checkpoint_epoch"] = ref_ckpt.get("epoch", -1)
    counts["cand_checkpoint_epoch"] = cand_ckpt.get("epoch", -1)

    # Visual encoder comparison
    max_visual_abs_diff = _compare_visual_encoders(ref_ckpt, cand_ckpt)
    if max_visual_abs_diff > 0:
        warnings.append(f"Visual encoder weights differ: max_abs_diff={max_visual_abs_diff:.6f}")
    else:
        logger.info("Visual encoder weights are identical (max_abs_diff=0).")

    # -- E. Determine validity --
    paired_valid = (
        sample_class in ("identical_effective_samples", "same_images_different_paths")
        and len([d for d in unexpected_diffs if "checkpoint" not in d.get("field", "")]) == 0
        and hashes.get("ref_val") == hashes.get("cand_val")
        and hashes.get("ref_val") is not None
        and hashes.get("ref_val") != "MISSING"
    )

    causal_claim_allowed = paired_valid and max_visual_abs_diff == 0

    result = PairAuditResult(
        paired_valid=paired_valid,
        causal_claim_allowed=causal_claim_allowed,
        allowed_differences=allowed_diffs,
        unexpected_differences=unexpected_diffs,
        warnings=warnings,
        hashes=hashes,
        counts=counts,
        resolved_paths={},
        sample_classification=sample_class,
        max_visual_abs_diff=max_visual_abs_diff,
    )

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "paired_valid": result.paired_valid,
            "causal_claim_allowed": result.causal_claim_allowed,
            "allowed_differences": result.allowed_differences,
            "unexpected_differences": result.unexpected_differences,
            "warnings": result.warnings,
            "hashes": result.hashes,
            "counts": result.counts,
            "sample_classification": result.sample_classification,
            "max_visual_abs_diff": result.max_visual_abs_diff,
        }, f, indent=2, default=str)

    return result
