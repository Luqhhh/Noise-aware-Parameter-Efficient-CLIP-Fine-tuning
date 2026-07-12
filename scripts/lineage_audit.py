#!/usr/bin/env python3
"""
Lineage audit for competition submission checkpoint.

Traces the full chain from the submission checkpoint back to raw data:
  checkpoint → eval_results.json → config → split_dir → CSV files
  → parent checkpoint → parent eval_results → ... → raw data

Produces lineage_audit.json with full traceability.

Usage:
    python3 scripts/lineage_audit.py --ckpt outputs/f1_strict/seed42/checkpoints/best.pt
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_checkpoint(ckpt_path: Path) -> dict:
    """Trace one checkpoint back through its lineage."""
    info = {"checkpoint_path": str(ckpt_path)}

    if not ckpt_path.exists():
        info["error"] = f"Checkpoint not found: {ckpt_path}"
        return info

    info["ckpt_sha256"] = sha256_file(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    info["ckpt_epoch"] = checkpoint.get("epoch")
    info["ckpt_best_val_acc"] = checkpoint.get("best_val_acc")
    info["ckpt_config_present"] = "config" in checkpoint

    # Check for eval_results.json alongside checkpoint
    eval_path = ckpt_path.parent / "eval_results.json"
    if eval_path.exists():
        eval_data = json.loads(eval_path.read_text())
        info["eval_results"] = {
            "experiment_id": eval_data.get("experiment_id"),
            "best_val_acc": eval_data.get("best_val_acc"),
            "dev_best_epoch": eval_data.get("dev_best_epoch"),
            "actual_epochs_run": eval_data.get("actual_epochs_run"),
            "max_epochs": eval_data.get("max_epochs"),
            "early_stopped": eval_data.get("early_stopped"),
            "stopped_at_epoch": eval_data.get("stopped_at_epoch"),
            "git_commit": eval_data.get("git_commit"),
            "config_path": eval_data.get("config_path"),
            "init_checkpoint": eval_data.get("init_checkpoint"),
            "split_seed": eval_data.get("split_seed"),
            "train_seed": eval_data.get("train_seed"),
            "epoch0_val_acc": eval_data.get("epoch0_val_acc"),
            "epoch0_delta": eval_data.get("epoch0_delta"),
            "macro_accuracy": eval_data.get("macro_accuracy"),
            "micro_macro_gap": eval_data.get("micro_macro_gap"),
            "bottom_10_percent_accuracy": eval_data.get(
                "bottom_10_percent_accuracy"
            ),
        }
        info["init_checkpoint"] = eval_data.get("init_checkpoint")
    else:
        info["eval_results"] = "NOT FOUND"
        info["init_checkpoint"] = checkpoint.get("init_checkpoint")

    # Check for split_lineage_audit.json
    split_audit_path = ckpt_path.parent.parent / "split_lineage_audit.json"
    if split_audit_path.exists():
        info["split_lineage_audit"] = json.loads(
            split_audit_path.read_text()
        )

    # Check for config snapshot
    for f in ckpt_path.parent.iterdir():
        if f.name.startswith("config_snapshot_"):
            info["config_snapshot"] = str(f)
            break

    return info


def trace_splits(split_dir: Path, experiment_id: str) -> dict:
    """Trace split CSVs for an experiment."""
    info = {"experiment_id": experiment_id, "split_dir": str(split_dir)}

    for csv_name in ["train.csv", "val.csv"]:
        csv_path = split_dir / csv_name
        if csv_path.exists():
            lines = len(csv_path.read_text().strip().split("\n")) - 1
            info[f"{csv_name}_rows"] = lines
            info[f"{csv_name}_sha256"] = sha256_file(csv_path)
        else:
            info[f"{csv_name}_rows"] = "NOT FOUND"

    # Check for cleaning_report.json
    cleaning_path = split_dir / "cleaning_report.json"
    if cleaning_path.exists():
        info["cleaning_report"] = json.loads(cleaning_path.read_text())

    # Check for class_to_idx.json
    c2i_path = split_dir / "class_to_idx.json"
    if c2i_path.exists():
        c2i = json.loads(c2i_path.read_text())
        info["num_classes"] = len(c2i)

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Lineage audit for competition submission checkpoint."
    )
    parser.add_argument(
        "--ckpt",
        default="outputs/f1_strict/seed42/checkpoints/best.pt",
        help="Path to submission checkpoint."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for lineage_audit.json (default: alongside checkpoint)."
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    print(f"Tracing lineage for: {ckpt_path}")

    lineage = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "submission_checkpoint": str(ckpt_path),
    }

    # Step 1: Trace the submission checkpoint
    print("Step 1: Trace submission checkpoint...")
    ckpt_info = trace_checkpoint(ckpt_path)
    lineage["submission"] = ckpt_info

    # Step 2: Trace the parent checkpoint (if any)
    init_ckpt = ckpt_info.get("init_checkpoint")
    if init_ckpt:
        print(f"Step 2: Trace parent checkpoint: {init_ckpt}...")
        parent_path = Path(init_ckpt)
        lineage["parent"] = trace_checkpoint(parent_path)

        # Step 2b: Trace further up (grandparent)
        parent_init = lineage["parent"].get("init_checkpoint")
        if parent_init:
            print(f"Step 2b: Trace grandparent checkpoint: {parent_init}...")
            lineage["grandparent"] = trace_checkpoint(Path(parent_init))

    # Step 3: Trace split CSVs
    print("Step 3: Trace split CSVs...")
    for level_name in ["submission", "parent", "grandparent"]:
        if level_name not in lineage:
            continue
        level = lineage[level_name]
        eval_r = level.get("eval_results", {})
        if isinstance(eval_r, dict) and "config_path" in eval_r:
            config_path = Path(eval_r["config_path"])
            if config_path.exists():
                import yaml
                config = yaml.safe_load(config_path.read_text())
                split_dir = Path(config["data"]["split_dir"])
                level["split_info"] = trace_splits(
                    split_dir,
                    eval_r.get("experiment_id", "unknown"),
                )

    # Step 4: Integrity checks
    print("Step 4: Integrity checks...")
    checks = []

    # Check: submission val matches parent val
    if "submission" in lineage and "parent" in lineage:
        sub_split = lineage["submission"].get("split_info", {})
        par_split = lineage["parent"].get("split_info", {})
        if sub_split.get("val.csv_sha256") and par_split.get("val.csv_sha256"):
            checks.append({
                "check": "submission_val == parent_val",
                "result": "PASS" if sub_split["val.csv_sha256"]
                           == par_split["val.csv_sha256"] else "FAIL",
                "detail": f"sub={sub_split['val.csv_sha256'][:16]}... "
                          f"par={par_split['val.csv_sha256'][:16]}...",
            })

    # Check: child_val not in parent_train (split audit)
    if "submission" in lineage:
        split_audit = lineage["submission"].get("split_lineage_audit", {})
        if split_audit:
            checks.append({
                "check": "no_val_leakage_from_parent_train",
                "result": "PASS" if split_audit.get("protocol_valid")
                          else "FAIL",
                "detail": f"child_val_in_parent_train="
                          f"{split_audit.get('child_val_in_parent_train', '?')}",
            })

    lineage["integrity_checks"] = checks

    # Write output
    output_path = args.output
    if not output_path:
        output_path = ckpt_path.parent / "lineage_audit.json"
    else:
        output_path = Path(output_path)

    output_path.write_text(json.dumps(lineage, indent=2))
    print(f"\nLineage audit complete!")
    print(f"  Output: {output_path}")
    chain = [k for k in ['grandparent', 'parent', 'submission'] if k in lineage]
    print(f"  Checkpoint chain: {' → '.join(chain)}")
    for check in checks:
        status = "✅" if check["result"] == "PASS" else "❌"
        print(f"  {status} {check['check']}: {check['result']}")

    # Print submission trace
    print(f"\n{'='*60}")
    print("Submission Trace Summary")
    print(f"{'='*60}")
    sub = lineage.get("submission", {}).get("eval_results", {})
    if isinstance(sub, dict):
        exp_id = sub.get('experiment_id', '?')
        print(f"  Experiment:    {exp_id}")
        best = sub.get('best_val_acc')
        if best is not None:
            print(f"  Best Val Acc:  {best:.4f}")
        else:
            print(f"  Best Val Acc:  ?")
        commit = sub.get('git_commit', '?')
        print(f"  Git Commit:    {commit[:12]}...")
        print(f"  Init from:     {ckpt_info.get('init_checkpoint', 'none')}")


if __name__ == "__main__":
    main()
