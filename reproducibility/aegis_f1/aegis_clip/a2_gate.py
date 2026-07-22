"""Build a leakage-audited A2-to-LoRA causal gate from two fixed splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from aegis_clip.features import canonical_sample_path
from aegis_clip.runtime import atomic_json_dump, sha256_file


def prepare_a2_gate(
    *,
    a2_train_csv: str | Path,
    a2_val_csv: str | Path,
    aegis_train_csv: str | Path,
    aegis_val_csv: str | Path,
    content_groups_json: str | Path,
    trust_bundle_path: str | Path,
    a2_rejected_paths: str | Path,
    output_dir: str | Path,
    clean_threshold: float = 0.70,
    expected_classes: int = 500,
) -> dict[str, Any]:
    """Create disjoint adaptation/evaluation splits and a union trust bundle.

    The adaptation split is ``A2 train ∩ Aegis validation``.  The evaluation
    split is ``A2 validation ∩ Aegis train``.  This makes the newly learned
    LoRA parameters blind to the evaluation images while retaining A2's own
    strict validation boundary.  ``A2 validation ∩ Aegis validation`` is kept
    as a small cross-audit set and is never used for checkpoint selection.
    """
    if not 0.0 <= float(clean_threshold) <= 1.0:
        raise ValueError("clean_threshold must be in [0,1]")
    sources = {
        "a2_train": Path(a2_train_csv),
        "a2_val": Path(a2_val_csv),
        "aegis_train": Path(aegis_train_csv),
        "aegis_val": Path(aegis_val_csv),
    }
    frames = {name: _indexed_split(path) for name, path in sources.items()}
    _verify_shared_labels(frames)

    adapt_keys = set(frames["a2_train"].index) & set(frames["aegis_val"].index)
    evaluation_keys = set(frames["a2_val"].index) & set(
        frames["aegis_train"].index
    )
    cross_audit_keys = set(frames["a2_val"].index) & set(
        frames["aegis_val"].index
    )
    if not adapt_keys or not evaluation_keys or not cross_audit_keys:
        raise ValueError("Two-split construction produced an empty partition")
    _verify_disjoint(
        adapt=adapt_keys,
        evaluation=evaluation_keys,
        cross_audit=cross_audit_keys,
    )

    with Path(content_groups_json).open("r", encoding="utf-8") as handle:
        raw_groups = json.load(handle)
    groups = {canonical_sample_path(path): str(group) for path, group in raw_groups.items()}
    all_gate_keys = adapt_keys | evaluation_keys | cross_audit_keys
    missing_groups = sorted(all_gate_keys - set(groups))
    if missing_groups:
        raise ValueError(f"Content groups miss {len(missing_groups)} paths")
    adapt_groups = {groups[path] for path in adapt_keys}
    evaluation_groups = {groups[path] for path in evaluation_keys}
    _verify_disjoint(adapt=adapt_groups, evaluation=evaluation_groups)
    conflicting_cross_audit = {
        path
        for path in cross_audit_keys
        if groups[path] in adapt_groups or groups[path] in evaluation_groups
    }
    cross_audit_keys -= conflicting_cross_audit
    if not cross_audit_keys:
        raise ValueError("Cross-audit partition is empty after content isolation")
    group_sets = {
        "adapt": adapt_groups,
        "evaluation": evaluation_groups,
        "cross_audit": {groups[path] for path in cross_audit_keys},
    }
    _verify_disjoint(**group_sets)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_paths = {
        "adapt": output / "adapt_train.csv",
        "evaluation": output / "evaluation.csv",
        "cross_audit": output / "cross_audit.csv",
    }
    _frame_for_keys(adapt_keys, frames["a2_train"]).to_csv(
        split_paths["adapt"], index=False
    )
    _frame_for_keys(evaluation_keys, frames["a2_val"]).to_csv(
        split_paths["evaluation"], index=False
    )
    _frame_for_keys(cross_audit_keys, frames["a2_val"]).to_csv(
        split_paths["cross_audit"], index=False
    )

    rejected = {
        canonical_sample_path(line)
        for line in Path(a2_rejected_paths).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    trust = torch.load(trust_bundle_path, map_location="cpu", weights_only=False)
    trust_paths = [canonical_sample_path(path) for path in trust["paths"]]
    path_to_index = {path: index for index, path in enumerate(trust_paths)}
    missing_rejected = sorted(rejected - set(path_to_index))
    if missing_rejected:
        raise ValueError(f"Trust bundle misses {len(missing_rejected)} A2 rejects")
    clean_probability = torch.as_tensor(
        trust["clean_probability"], dtype=torch.float32
    ).clone()
    rejected_indices = torch.tensor(
        [path_to_index[path] for path in sorted(rejected)], dtype=torch.long
    )
    clean_probability[rejected_indices] = 0.0
    combined_trust = dict(trust)
    combined_trust["clean_probability"] = clean_probability
    combined_trust["metadata"] = {
        **dict(trust.get("metadata", {})),
        "a2_union_gate": {
            "method": "cvt_clean_core_union_a2_high_precision_rejects",
            "a2_rejected_count": len(rejected),
            "clean_threshold": float(clean_threshold),
            "source_trust_sha256": sha256_file(trust_bundle_path),
            "source_rejected_paths_sha256": sha256_file(a2_rejected_paths),
        },
    }
    combined_trust_path = output / "cvt_a2_union.pt"
    torch.save(combined_trust, combined_trust_path)

    adapt_selected = [
        path
        for path in adapt_keys
        if float(clean_probability[path_to_index[path]]) >= float(clean_threshold)
    ]
    selected_classes = {
        int(frames["a2_train"].loc[path, "label"]) for path in adapt_selected
    }
    if len(selected_classes) != int(expected_classes):
        raise ValueError(
            "Clean adaptation supervision does not cover all classes: "
            f"{len(selected_classes)}/{expected_classes}"
        )

    manifest: dict[str, Any] = {
        "method": "a2_disjoint_lora_gate_v1",
        "partitions": {
            "adapt": len(adapt_keys),
            "evaluation": len(evaluation_keys),
            "cross_audit": len(cross_audit_keys),
        },
        "content_group_counts": {
            name: len(values) for name, values in group_sets.items()
        },
        "cross_audit_content_conflicts_removed": len(conflicting_cross_audit),
        "adapt_clean_supervision": len(adapt_selected),
        "adapt_clean_supervision_classes": len(selected_classes),
        "a2_rejected_total": len(rejected),
        "a2_rejected_in_adapt": len(rejected & adapt_keys),
        "clean_threshold": float(clean_threshold),
        "sources": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
        "content_groups_sha256": sha256_file(content_groups_json),
        "trust_bundle_sha256": sha256_file(trust_bundle_path),
        "a2_rejected_paths_sha256": sha256_file(a2_rejected_paths),
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in split_paths.items()
        },
        "combined_trust": {
            "path": str(combined_trust_path.resolve()),
            "sha256": sha256_file(combined_trust_path),
        },
        "test_data_used": False,
        "external_data_used": False,
    }
    atomic_json_dump(manifest, output / "manifest.json")
    return manifest


def prepare_a2_fullfit(
    *,
    a2_train_csv: str | Path,
    a2_val_csv: str | Path,
    content_groups_json: str | Path,
    trust_bundle_path: str | Path,
    a2_rejected_paths: str | Path,
    output_dir: str | Path,
    clean_threshold: float = 0.70,
    expected_classes: int = 500,
) -> dict[str, Any]:
    """Build a conservative fixed full-fit set for the platform-positive A2 head.

    A2's original strict-train samples are replayed except for its physical
    blacklist.  Previously held-out A2 validation samples are added only when
    their independent OOF clean probability reaches the frozen threshold.
    Rejected examples are removed from the CSV rather than merely zero-weighted,
    so pixel MixUp cannot reintroduce their noisy targets.
    """
    if not 0.0 <= float(clean_threshold) <= 1.0:
        raise ValueError("clean_threshold must be in [0,1]")
    sources = {
        "a2_train": Path(a2_train_csv),
        "a2_val": Path(a2_val_csv),
    }
    frames = {name: _indexed_split(path) for name, path in sources.items()}
    _verify_shared_labels(frames)
    train_keys = set(frames["a2_train"].index)
    val_keys = set(frames["a2_val"].index)
    _verify_disjoint(a2_train=train_keys, a2_val=val_keys)

    with Path(content_groups_json).open("r", encoding="utf-8") as handle:
        raw_groups = json.load(handle)
    groups = {canonical_sample_path(path): str(group) for path, group in raw_groups.items()}
    missing_groups = sorted((train_keys | val_keys) - set(groups))
    if missing_groups:
        raise ValueError(f"Content groups miss {len(missing_groups)} paths")
    train_groups = {groups[path] for path in train_keys}
    val_groups = {groups[path] for path in val_keys}
    overlapping_groups = train_groups & val_groups
    val_content_conflicts = {
        path for path in val_keys if groups[path] in overlapping_groups
    }

    rejected = {
        canonical_sample_path(line)
        for line in Path(a2_rejected_paths).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not rejected <= train_keys:
        raise ValueError("A2 rejected paths must be a subset of A2 strict train")
    trust = torch.load(trust_bundle_path, map_location="cpu", weights_only=False)
    trust_paths = [canonical_sample_path(path) for path in trust["paths"]]
    path_to_index = {path: index for index, path in enumerate(trust_paths)}
    missing_trust = sorted((train_keys | val_keys) - set(path_to_index))
    if missing_trust:
        raise ValueError(f"Trust bundle misses {len(missing_trust)} A2 paths")
    original_clean = torch.as_tensor(
        trust["clean_probability"], dtype=torch.float32
    ).clone()
    added_val = {
        path
        for path in val_keys
        if path not in val_content_conflicts
        and float(original_clean[path_to_index[path]]) >= float(clean_threshold)
    }
    replay_train = train_keys - rejected
    fullfit_keys = replay_train | added_val
    labels = pd.concat([frames["a2_train"], frames["a2_val"]])
    selected_classes = {int(labels.loc[path, "label"]) for path in fullfit_keys}
    if len(selected_classes) != int(expected_classes):
        raise ValueError(
            f"Full-fit supervision covers {len(selected_classes)}/{expected_classes} classes"
        )

    combined_clean = original_clean.clone()
    for path in replay_train:
        combined_clean[path_to_index[path]] = 1.0
    for path in rejected:
        combined_clean[path_to_index[path]] = 0.0
    combined_trust = dict(trust)
    combined_trust["clean_probability"] = combined_clean
    combined_trust["metadata"] = {
        **dict(trust.get("metadata", {})),
        "a2_fixed_fullfit": {
            "method": "a2_replay_plus_oof_clean_heldout",
            "a2_replay_count": len(replay_train),
            "added_heldout_count": len(added_val),
            "a2_rejected_count": len(rejected),
            "clean_threshold": float(clean_threshold),
            "source_trust_sha256": sha256_file(trust_bundle_path),
            "source_rejected_paths_sha256": sha256_file(a2_rejected_paths),
        },
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fullfit_csv = output / "fullfit_train.csv"
    diagnostic_val_csv = output / "diagnostic_val.csv"
    _frame_for_keys(fullfit_keys, labels).to_csv(fullfit_csv, index=False)
    _frame_for_keys(val_keys, frames["a2_val"]).to_csv(
        diagnostic_val_csv, index=False
    )
    combined_trust_path = output / "a2_fixed_fullfit_trust.pt"
    torch.save(combined_trust, combined_trust_path)

    manifest: dict[str, Any] = {
        "method": "a2_fixed_fullfit_v1",
        "a2_train_original": len(train_keys),
        "a2_replay_after_reject": len(replay_train),
        "a2_val_original": len(val_keys),
        "a2_val_added_clean": len(added_val),
        "a2_val_added_fraction": len(added_val) / len(val_keys),
        "a2_val_content_conflicts_excluded": len(val_content_conflicts),
        "a2_train_val_overlapping_content_groups": len(overlapping_groups),
        "fullfit_train": len(fullfit_keys),
        "fullfit_classes": len(selected_classes),
        "a2_rejected": len(rejected),
        "clean_threshold": float(clean_threshold),
        "content_groups": {
            "a2_train": len(train_groups),
            "a2_val": len(val_groups),
            "fullfit": len({groups[path] for path in fullfit_keys}),
        },
        "sources": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
        "content_groups_sha256": sha256_file(content_groups_json),
        "trust_bundle_sha256": sha256_file(trust_bundle_path),
        "a2_rejected_paths_sha256": sha256_file(a2_rejected_paths),
        "outputs": {
            "fullfit_train": {
                "path": str(fullfit_csv.resolve()),
                "sha256": sha256_file(fullfit_csv),
            },
            "diagnostic_val": {
                "path": str(diagnostic_val_csv.resolve()),
                "sha256": sha256_file(diagnostic_val_csv),
            },
            "trust": {
                "path": str(combined_trust_path.resolve()),
                "sha256": sha256_file(combined_trust_path),
            },
        },
        "validation_overlap_with_training": True,
        "selection_policy": "last_epoch",
        "test_data_used": False,
        "external_data_used": False,
    }
    atomic_json_dump(manifest, output / "manifest.json")
    return manifest


def _indexed_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"image_path", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Split {path} misses columns: {sorted(missing)}")
    frame = frame[["image_path", "label"]].copy()
    frame["canonical_path"] = frame["image_path"].map(canonical_sample_path)
    if frame["canonical_path"].duplicated().any():
        raise ValueError(f"Split {path} contains duplicate canonical paths")
    frame["label"] = frame["label"].astype(int)
    return frame.set_index("canonical_path", drop=True)


def _verify_shared_labels(frames: dict[str, pd.DataFrame]) -> None:
    names = list(frames)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left, right = frames[left_name], frames[right_name]
            shared = sorted(set(left.index) & set(right.index))
            if not shared:
                continue
            mismatch = [
                path
                for path in shared
                if int(left.loc[path, "label"]) != int(right.loc[path, "label"])
            ]
            if mismatch:
                raise ValueError(
                    f"Label mismatch between {left_name} and {right_name}: {mismatch[0]}"
                )


def _verify_disjoint(**partitions: set[str]) -> None:
    names = list(partitions)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = partitions[left] & partitions[right]
            if overlap:
                raise ValueError(
                    f"Partitions {left}/{right} overlap by {len(overlap)} entries"
                )


def _frame_for_keys(keys: set[str], source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for path in sorted(keys, key=lambda value: (int(source.loc[value, "label"]), value)):
        label = int(source.loc[path, "label"])
        rows.append(
            {
                "image_path": f"train/{path}",
                "class_name": f"{label:04d}",
                "label": label,
            }
        )
    return pd.DataFrame(rows, columns=["image_path", "class_name", "label"])
