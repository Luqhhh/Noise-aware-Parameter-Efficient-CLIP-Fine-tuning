#!/usr/bin/env python3
"""Batch-rename experiments from old messy IDs to clean structured IDs.

Run from repo root:
    python3 tools/rename_experiments.py --dry-run   # preview
    python3 tools/rename_experiments.py             # execute
"""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ── Mapping table ──
# (old_config, new_config, new_exp_id, old_output_dir, new_output_dir)
# split_dir will be updated automatically if it references an old output dir.
RENAMES = [
    # Reference & noise-robust
    ("d3_strict.yaml",           "ref.yaml",              "ref",              "d3_strict",           "ref"),
    ("b2_gce07.yaml",            "gce_q07.yaml",          "gce_q07",          "b2_gce07",            "gce_q07"),
    ("r3_d3_proto_weight.yaml",  "pw_v1.yaml",            "pw_v1",            "r3_d3_proto_weight",  "pw_v1"),
    # Cosine head
    ("c0_cosine_scale.yaml",     "cos_s10_fixed.yaml",    "cos_s10_fixed",    None, None),
    ("c1_cosine_scale.yaml",     "cos_s10_learn.yaml",    "cos_s10_learn",    None, None),
    ("c2_cosine_scale.yaml",     "cos_s20_learn.yaml",    "cos_s20_learn",    None, None),
    # Dropout
    ("d4a_dropout.yaml",         "drop_p03.yaml",         "drop_p03",         "d4a",                 "drop_p03"),
    ("d4b_dropout.yaml",         "drop_p05.yaml",         "drop_p05",         "d4b",                 "drop_p05"),
    ("d4c_dropout.yaml",         "drop_p07.yaml",         "drop_p07",         "d4c",                 "drop_p07"),
    # Base / CE baselines
    ("e0_strict.yaml",           "base_ce.yaml",          "base_ce",          "e0_strict",           "base_ce"),
    ("b0_regression.yaml",       "base_b0.yaml",          "base_b0",          "b0",                  "base_b0"),
    ("baseline.yaml",            "baseline.yaml",         "baseline",         "baseline",            "baseline"),  # keep name, update id
    # Cosine + hyper
    ("e1_hyper_search.yaml",     "cos_hyper.yaml",        "cos_hyper",        "e1",                  "cos_hyper"),
    # Augmentation
    ("e2_augmentation.yaml",     "aug_a1.yaml",           "aug_a1",           "e2",                  "aug_a1"),
    ("e2b_equal_lr.yaml",        "aug_a1_lr5e3.yaml",     "aug_a1_lr5e3",     "e2b",                 "aug_a1_lr5e3"),
    ("e3_augmentation.yaml",     "aug_a2.yaml",           "aug_a2",           "e3",                  "aug_a2"),
    ("e4_augmentation.yaml",     "aug_a3.yaml",           "aug_a3",           "e4",                  "aug_a3"),
    # Cosine + aug combo
    ("e5_combined.yaml",         "cos_a3.yaml",           "cos_a3",           "e5",                  "cos_a3"),
    # Fine-tune / unfreeze
    ("f0_strict.yaml",           "ft_frozen.yaml",        "ft_frozen",        "f0_strict",           "ft_frozen"),
    ("f1_strict.yaml",           "ft_lnpost.yaml",        "ft_lnpost",        "f1_strict",           "ft_lnpost"),
]

# ── Old output dir → new output dir for split_dir rewriting ──
OUTPUT_DIR_MAP = {}
for _, _, _, old_out, new_out in RENAMES:
    if old_out and new_out:
        OUTPUT_DIR_MAP[old_out] = new_out


def _update_path(path_str: str) -> str:
    """Rewrite a path like outputs/d3_strict/seed42 → outputs/ref/seed42."""
    if not path_str:
        return path_str
    # Split into parts, replace first matching segment
    parts = path_str.split("/")
    new_parts = []
    replaced = False
    for p in parts:
        if not replaced and p in OUTPUT_DIR_MAP:
            new_parts.append(OUTPUT_DIR_MAP[p])
            replaced = True
        else:
            new_parts.append(p)
    return "/".join(new_parts)


def process_configs(dry_run: bool):
    """Rename config files and update their content."""
    config_dir = REPO / "configs"
    for old_name, new_name, new_id, old_out, new_out in RENAMES:
        old_path = config_dir / old_name
        new_path = config_dir / new_name
        if not old_path.exists():
            print(f"  SKIP (not found): {old_name}")
            continue

        # Read content
        with open(old_path) as f:
            content = f.read()

        # Update experiment ID
        import re
        content = re.sub(
            r'id:\s*\S+',
            f'id: {new_id}',
            content,
            count=1,
        )

        # Update output paths
        for old_dir, new_dir in OUTPUT_DIR_MAP.items():
            content = content.replace(f"outputs/{old_dir}", f"outputs/{new_dir}")

        if dry_run:
            print(f"  [{old_name} → {new_name}] id={new_id}")
        else:
            with open(old_path, 'w') as f:
                f.write(content)
            if old_name != new_name:
                old_path.rename(new_path)
            print(f"  ✓ {old_name} → {new_name}  (id={new_id})")


def move_outputs(dry_run: bool):
    """Rename output directories."""
    outputs_dir = REPO / "outputs"
    for old_name, new_name, new_id, old_out, new_out in RENAMES:
        if old_out is None or new_out is None:
            continue
        if old_out == new_out:
            continue

        old_path = outputs_dir / old_out
        new_path = outputs_dir / new_out
        if not old_path.exists():
            print(f"  SKIP (not found): outputs/{old_out}")
            continue
        if new_path.exists():
            print(f"  SKIP (target exists): outputs/{new_out}")
            continue

        if dry_run:
            print(f"  [outputs/{old_out} → outputs/{new_out}]")
        else:
            old_path.rename(new_path)
            print(f"  ✓ outputs/{old_out} → outputs/{new_out}")


def fix_split_dir_refs(dry_run: bool):
    """Update split_dir in configs that reference renamed output dirs."""
    config_dir = REPO / "configs"
    for cfg_file in sorted(config_dir.glob("*.yaml")):
        with open(cfg_file) as f:
            content = f.read()

        new_content = content
        for old_dir, new_dir in OUTPUT_DIR_MAP.items():
            # Match split_dir: outputs/old_dir/...
            pattern = f"outputs/{old_dir}/"
            if pattern in new_content:
                new_content = new_content.replace(pattern, f"outputs/{new_dir}/")

        if new_content != content:
            if dry_run:
                print(f"  [fix split_dir in {cfg_file.name}]")
            else:
                with open(cfg_file, 'w') as f:
                    f.write(new_content)
                print(f"  ✓ fixed split_dir in {cfg_file.name}")


def update_tool_references(dry_run: bool):
    """Update experiment names in tool docstrings and examples."""
    replacements = {
        "d3_strict": "ref",
        "D3_STRICT": "ref",
        "b2_gce07": "gce_q07",
        "B2_GCE07": "gce_q07",
        "B3_PROTO_STATIC": "pw_v1",
        "r3_d3_proto_weight": "pw_v1",
        "e0_strict": "base_ce",
        "E0_STRICT": "base_ce",
        "f0_strict": "ft_frozen",
        "F0_STRICT": "ft_frozen",
        "f1_strict": "ft_lnpost",
        "F1_STRICT": "ft_lnpost",
    }

    tool_dirs = [REPO / "tools", REPO / "scripts",
                 REPO / "common", REPO / "tests"]
    for tool_dir in tool_dirs:
        if not tool_dir.exists():
            continue
        for py_file in sorted(tool_dir.rglob("*.py")):
            if py_file.name == "rename_experiments.py":
                continue  # skip self
            with open(py_file) as f:
                content = f.read()
            new_content = content
            for old, new in replacements.items():
                if old in new_content:
                    new_content = new_content.replace(old, new)
            if new_content != content:
                if dry_run:
                    print(f"  [update refs in {py_file.relative_to(REPO)}]")
                else:
                    with open(py_file, 'w') as f:
                        f.write(new_content)
                    print(f"  ✓ updated refs in {py_file.relative_to(REPO)}")


def main():
    p = argparse.ArgumentParser(description="Batch rename experiments")
    p.add_argument("--dry-run", action="store_true", help="Preview only")
    args = p.parse_args()

    if args.dry_run:
        print("=== DRY RUN (no changes) ===\n")

    print("1. Config files:")
    process_configs(args.dry_run)

    print("\n2. Output directories:")
    move_outputs(args.dry_run)

    print("\n3. Update tool references:")
    update_tool_references(args.dry_run)

    if args.dry_run:
        print("\n=== DRY RUN complete. Run without --dry-run to execute. ===")
    else:
        print("\n=== Rename complete ===")
        print("Review and run:")
        print("  git diff --stat")
        print("  git add -A && git commit -m 'refactor: clean experiment naming scheme'")


if __name__ == "__main__":
    main()
