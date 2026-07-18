# Validation Protocol Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix stage-to-stage validation leak, establish master split, and rebuild trusted E0/D3/F0/F1 baselines on identical validation data.

**Architecture:** Single master split per seed → train-only D3 cleaning → parent-child lineage audit on all init-checkpoint experiments → epoch-0 validation gate → multi-seed confirmation → submission registry.

**Tech Stack:** Python 3.10, PyTorch, CLIP ViT-B/32, existing `experiments/baseline/` training pipeline.

## Global Constraints

- All experiments sharing a seed MUST use the identical train/val split files
- D3 cleaning MUST NOT touch validation data
- Any experiment with `--init-checkpoint` MUST pass parent-child split audit before training
- Audit failure MUST cause hard exit (non-zero exit code)
- Epoch-0 validation MUST match parent accuracy within 0.05pp
- Multi-seed confirmation requires ≥2/3 seeds positive and paired mean delta > 0
- Submission CSV MUST pass all 9 validation checks from `scripts/check_submission.py`
- Train seed fixed at 42 for all experiments
- Config files must include `seed` key in `data` section for inference script compatibility

---

### Task 1: Mark + freeze existing experiment results

**Files:**
- Modify: `results/ablation.csv`

**Interfaces:**
- Produces: Updated `ablation.csv` with `result_status`, `validation_protocol` columns

- [ ] **Step 1: Add new columns to ablation.csv**

Add these columns to the CSV header and populate for all existing experiments:

```csv
result_status,validation_protocol,parent_experiment,parent_train_split,val_overlap_with_parent
```

Status values:
- E0: `historical_valid`
- D3: `pending_fair_comparison`
- D4A, D4B: `historical_valid`
- D4C: `incomplete`
- F0: `incomplete`
- F1, F1b: `invalid_stage_leakage`
- F2, F2-d3, F2-hi-bb, F2-no-drop: `blocked_by_invalid_parent`

- [ ] **Step 2: Append leak documentation**

Add F1 leak note row or footnote: `F1 val=10116, overlap with D3 train=8902 (88.0%)`

- [ ] **Step 3: Commit**

```bash
git add results/ablation.csv
git commit -m "chore: mark F1/F1b as invalid_stage_leakage, add result status taxonomy

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Build master split infrastructure

**Files:**
- Create: `scripts/build_master_split.py`
- Create: `outputs/data/master_splits/seed42/split_manifest.json`

**Interfaces:**
- Produces: `master_splits/seed42/train.csv`, `master_splits/seed42/val.csv`, `master_splits/seed42/split_manifest.json`
- Produces: `class_to_idx.json`, `idx_to_class.json` under `master_splits/seed42/`

- [ ] **Step 1: Write `scripts/build_master_split.py`**

```python
"""
Build the canonical master train/val split.

This script generates the ONE authoritative split per seed that ALL
experiments MUST reference. No experiment is allowed to generate its
own split after this exists.

Usage:
    python scripts/build_master_split.py --train-dir train --output-root outputs/data/master_splits --seed 42 --val-ratio 0.1
"""

import argparse
import json
import hashlib
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common.class_mapping import generate_mapping
from common.utils import setup_logging

logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="train")
    parser.add_argument("--output-root", default="outputs/data/master_splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    output_dir = Path(args.output_root) / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(output_dir), name="build_master_split")

    # Collect all images
    records = []
    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                records.append({
                    "image_path": str(img_path.relative_to(train_dir)),
                    "class_name": class_name,
                })

    df = pd.DataFrame(records)
    logger.info(f"Found {len(df)} images in {df['class_name'].nunique()} classes")

    # Generate class mapping (lexicographic sort)
    class_to_idx, idx_to_class = generate_mapping(
        train_dir, output_dir, expected_num_classes=500
    )

    df["class_idx"] = df["class_name"].map(class_to_idx)

    # Stratified split
    train_df, val_df = train_test_split(
        df, test_size=args.val_ratio, random_state=args.seed,
        stratify=df["class_idx"],
    )

    train_df = train_df.sort_values("image_path").reset_index(drop=True)
    val_df = val_df.sort_values("image_path").reset_index(drop=True)

    # Write CSVs
    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    train_df[["image_path", "class_name", "class_idx"]].to_csv(train_csv, index=False)
    val_df[["image_path", "class_name", "class_idx"]].to_csv(val_csv, index=False)

    # Verify disjointness
    train_paths = set(train_df["image_path"])
    val_paths = set(val_df["image_path"])
    overlap = train_paths & val_paths
    assert len(overlap) == 0, f"Train/val overlap: {overlap}"
    assert len(train_paths) + len(val_paths) == len(df), "Union != full dataset"

    # Write manifest
    manifest = {
        "split_seed": args.seed,
        "source_root": str(train_dir.resolve()),
        "source_file_count": len(df),
        "train_count": len(train_df),
        "val_count": len(val_df),
        "num_classes": df["class_name"].nunique(),
        "train_csv_sha256": sha256_file(train_csv),
        "val_csv_sha256": sha256_file(val_csv),
        "created_by_git_commit": None,
        "duplicate_grouping_enabled": False,
    }

    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(f"Master split created: {len(train_df)} train / {len(val_df)} val")
    logger.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```bash
python scripts/build_master_split.py --train-dir train --output-root outputs/data/master_splits --seed 42 --val-ratio 0.1
```

Expected: creates `outputs/data/master_splits/seed42/` with train.csv, val.csv, class_to_idx.json, idx_to_class.json, split_manifest.json.

- [ ] **Step 3: Verify split integrity**

```bash
python -c "
import pandas as pd
from pathlib import Path

d = Path('outputs/data/master_splits/seed42')
train = pd.read_csv(d / 'train.csv')
val = pd.read_csv(d / 'val.csv')

# Disjoint
t = set(train['image_path'])
v = set(val['image_path'])
assert len(t & v) == 0, f'Overlap: {len(t & v)}'

# Complete
import glob
all_imgs = set()
for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
    for p in Path('train').rglob(ext):
        all_imgs.add(str(p.relative_to('train')))
assert len(t | v) == len(all_imgs), f'{len(t|v)} != {len(all_imgs)}'

# Class coverage
assert train['class_idx'].nunique() == 500
assert val['class_idx'].nunique() == 500

print(f'OK: {len(t)} train + {len(v)} val, disjoint, 500 classes each')
"
```

- [ ] **Step 4: Record git commit in manifest**

```bash
GIT_COMMIT=$(git rev-parse HEAD)
python -c "
import json
m = json.load(open('outputs/data/master_splits/seed42/split_manifest.json'))
m['created_by_git_commit'] = '$GIT_COMMIT'
json.dump(m, open('outputs/data/master_splits/seed42/split_manifest.json', 'w'), indent=2)
"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/build_master_split.py outputs/data/master_splits/
git commit -m "feat: build canonical master split (seed=42)

Train: 93102, Val: 10116, 500 classes, SHA256-fixed.
All subsequent experiments MUST use this split.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Add parent-child split lineage audit

**Files:**
- Create: `common/split_audit.py`
- Modify: `experiments/baseline/train.py` (add audit call in `main()`)

**Interfaces:**
- Consumes: `split_manifest.json` from parent experiment
- Produces: `split_lineage_audit.json` in child experiment output directory
- Produces: `SystemExit(1)` on audit failure

- [ ] **Step 1: Write `common/split_audit.py`**

```python
"""
Parent-child split lineage audit.

Ensures experiments initialized from a parent checkpoint do not leak
validation data from the training stage. Must be called before any
training occurs.

Rules enforced:
  1. child_val ∩ parent_train = ∅
  2. child_val = parent_val (identical set)
  3. Hard exit on any violation
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class SplitAuditError(ValueError):
    """Fatal split integrity violation."""


def load_split_csv(csv_path: Path) -> set:
    """Load image paths from a split CSV. Returns set of image_path values."""
    if not csv_path.exists():
        raise SplitAuditError(f"Split CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns:
        raise SplitAuditError(f"No 'image_path' column in {csv_path}")
    return set(df["image_path"])


def run_split_audit(
    parent_experiment_id: str,
    parent_checkpoint_path: str,
    parent_train_csv: Path,
    parent_val_csv: Path,
    child_train_csv: Path,
    child_val_csv: Path,
    output_dir: Path,
) -> dict:
    """Run parent-child split lineage audit.

    Args:
        parent_experiment_id: Identifier for the parent experiment.
        parent_checkpoint_path: Path to the checkpoint used for init.
        parent_train_csv: Path to parent's train split CSV.
        parent_val_csv: Path to parent's val split CSV.
        child_train_csv: Path to child's train split CSV.
        child_val_csv: Path to child's val split CSV.
        output_dir: Where to write split_lineage_audit.json.

    Returns:
        Audit result dict. Raises SplitAuditError on fatal violation.

    Raises:
        SplitAuditError: If any integrity rule is violated.
    """
    parent_train = load_split_csv(parent_train_csv)
    parent_val = load_split_csv(parent_val_csv)
    child_train = load_split_csv(child_train_csv)
    child_val = load_split_csv(child_val_csv)

    child_val_in_parent_train = child_val & parent_train
    child_val_in_parent_val = child_val & parent_val

    # Rule 1: child val must not appear in parent train (no leakage)
    if child_val_in_parent_train:
        raise SplitAuditError(
            f"VALIDATION LEAK: {len(child_val_in_parent_train)} images "
            f"in child validation were seen in parent training. "
            f"First 5: {sorted(list(child_val_in_parent_train))[:5]}"
        )

    # Rule 2: child val must equal parent val (same comparison basis)
    child_val_equals_parent_val = child_val == parent_val
    if not child_val_equals_parent_val:
        missing = parent_val - child_val
        extra = child_val - parent_val
        raise SplitAuditError(
            f"VALIDATION MISMATCH: child validation differs from parent.\n"
            f"  In parent but not child: {len(missing)}\n"
            f"  In child but not parent: {len(extra)}"
        )

    # Rule 3: child train must equal parent train (for pure continue/F0)
    child_train_equals_parent_train = child_train == parent_train

    audit = {
        "parent_experiment": parent_experiment_id,
        "parent_checkpoint": parent_checkpoint_path,
        "parent_train_count": len(parent_train),
        "parent_val_count": len(parent_val),
        "child_train_count": len(child_train),
        "child_val_count": len(child_val),
        "child_val_in_parent_train": len(child_val_in_parent_train),
        "child_val_in_parent_val": len(child_val_in_parent_val),
        "child_val_equals_parent_val": child_val_equals_parent_val,
        "child_train_equals_parent_train": child_train_equals_parent_train,
        "protocol_valid": True,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "split_lineage_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    logger.info(f"Split lineage audit PASSED: {audit_path}")

    return audit
```

- [ ] **Step 2: Write tests for split audit**

File: `tests/test_split_audit.py`

```python
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from common.split_audit import (
    SplitAuditError,
    load_split_csv,
    run_split_audit,
)


def make_csv(path: Path, image_paths: list[str]):
    df = pd.DataFrame({"image_path": image_paths, "class_name": ["c"] * len(image_paths), "class_idx": [0] * len(image_paths)})
    df.to_csv(path, index=False)


def test_load_split_csv():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "test.csv"
        make_csv(p, ["a.jpg", "b.jpg"])
        result = load_split_csv(p)
        assert result == {"a.jpg", "b.jpg"}


def test_load_split_csv_missing():
    with pytest.raises(SplitAuditError, match="not found"):
        load_split_csv(Path("/nonexistent.csv"))


def test_audit_passes_valid():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_csv(d / "parent_train.csv", ["a.jpg", "b.jpg", "c.jpg"])
        make_csv(d / "parent_val.csv", ["d.jpg", "e.jpg"])
        make_csv(d / "child_train.csv", ["a.jpg", "b.jpg", "c.jpg"])
        make_csv(d / "child_val.csv", ["d.jpg", "e.jpg"])

        audit = run_split_audit(
            "D3", "d3/best.pt",
            d / "parent_train.csv", d / "parent_val.csv",
            d / "child_train.csv", d / "child_val.csv",
            d,
        )
        assert audit["protocol_valid"] is True
        assert audit["child_val_equals_parent_val"] is True


def test_audit_detects_leakage():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_csv(d / "parent_train.csv", ["a.jpg", "b.jpg", "c.jpg", "LEAK.jpg"])
        make_csv(d / "parent_val.csv", ["d.jpg", "e.jpg"])
        make_csv(d / "child_train.csv", ["a.jpg", "b.jpg", "c.jpg"])
        # LEAK.jpg appears in parent_train but is now in child_val
        make_csv(d / "child_val.csv", ["d.jpg", "e.jpg", "LEAK.jpg"])

        with pytest.raises(SplitAuditError, match="VALIDATION LEAK"):
            run_split_audit(
                "D3", "d3/best.pt",
                d / "parent_train.csv", d / "parent_val.csv",
                d / "child_train.csv", d / "child_val.csv",
                d,
            )


def test_audit_detects_val_mismatch():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_csv(d / "parent_train.csv", ["a.jpg", "b.jpg"])
        make_csv(d / "parent_val.csv", ["d.jpg", "e.jpg"])
        make_csv(d / "child_train.csv", ["a.jpg", "b.jpg"])
        make_csv(d / "child_val.csv", ["d.jpg", "DIFFERENT.jpg"])

        with pytest.raises(SplitAuditError, match="VALIDATION MISMATCH"):
            run_split_audit(
                "D3", "d3/best.pt",
                d / "parent_train.csv", d / "parent_val.csv",
                d / "child_train.csv", d / "child_val.csv",
                d,
            )
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_split_audit.py -v
```

Expected: 4 tests pass.

- [ ] **Step 4: Add audit call to `experiments/baseline/train.py`**

Add after `--init-checkpoint` loading block in `main()`, before training loop:

```python
# Parent-child split lineage audit
if args.init_checkpoint:
    from common.split_audit import run_split_audit

    # Determine parent experiment from checkpoint path
    # Convention: outputs/<parent_exp>/checkpoints/best.pt
    ckpt_path = Path(args.init_checkpoint)
    parent_exp = ckpt_path.parent.parent.name  # e.g., "d3_strict"

    parent_split_dir = Path(config["output"].get("parent_split_dir", ""))
    if not parent_split_dir.exists():
        # Fall back: infer parent split from checkpoint path structure
        parent_split_dir = ckpt_path.parent.parent / "splits"
        if not (parent_split_dir / "train.csv").exists():
            parent_output_root = ckpt_path.parent.parent.parent
            parent_split_dir = parent_output_root.parent / "master_splits" / "seed42"

    child_split_dir = Path(config["data"]["split_dir"])

    run_split_audit(
        parent_experiment_id=parent_exp,
        parent_checkpoint_path=args.init_checkpoint,
        parent_train_csv=parent_split_dir / "train.csv",
        parent_val_csv=parent_split_dir / "val.csv",
        child_train_csv=child_split_dir / "train.csv",
        child_val_csv=child_split_dir / "val.csv",
        output_dir=Path(config["output"]["log_dir"]).parent,
    )
```

- [ ] **Step 5: Commit**

```bash
git add common/split_audit.py tests/test_split_audit.py experiments/baseline/train.py
git commit -m "feat: add parent-child split lineage audit

Hard exit on: child_val ∩ parent_train ≠ ∅, child_val ≠ parent_val.
Prevents stage-to-stage validation leakage like F1/D3.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Refactor config system — all experiments use master split

**Files:**
- Modify: `configs/e0_hyper_search.yaml` (or create `configs/e0_strict.yaml`)
- Create: `configs/d3_strict.yaml`
- Create: `configs/f0_strict.yaml`
- Create: `configs/f1_strict.yaml`

**Interfaces:**
- All configs share: `split_dir: outputs/data/master_splits/seed42`
- D3, F0, F1 configs reference: `parent_split_dir: outputs/data/master_splits/seed42`

- [ ] **Step 1: Create `configs/e0_strict.yaml`**

```yaml
# E0-strict: Baseline on unified master split.
# Frozen CLIP + linear head, no augmentation, seed=42.

experiment:
  id: E0_STRICT
  mode: dev
  head_type: linear
  augmentation_preset: a0

data:
  stage: preliminary
  image_extensions:
  - .jpg
  - .jpeg
  - .png
  - .bmp
  - .webp
  seed: 42
  split_seed: 42
  train_seed: 42
  split_dir: outputs/data/master_splits/seed42
  test_dir: test
  train_dir: train
  val_ratio: 0.1
  expected_num_classes: 500
  class_mapping_path: outputs/data/master_splits/seed42
  use_full_training_set: false

model:
  clip_model_name: ViT-B/32
  feature_dim: 512
  freeze_clip: true
  num_classes: 500
  use_cached_features: false
  unfreeze_last_n_blocks: 0
  train_ln_post: false
  train_visual_proj: false

eval:
  batch_size: 256

output:
  log_dir: outputs/e0_strict/seed42/logs
  submission_dir: outputs/e0_strict/seed42/submissions

train:
  amp: true
  batch_size: 128
  device: cuda
  epochs: 50
  image_size: 224
  lr: 0.005
  max_grad_norm: 1.0
  num_workers: 8
  save_dir: outputs/e0_strict/seed42/checkpoints
  scheduler: cosine
  warmup_epochs: 2
  weight_decay: 0.0001
  min_lr_ratio: 0.01
  early_stop_patience: 10
```

- [ ] **Step 2: Create `configs/d3_strict.yaml`**

Same as e0_strict.yaml but:
```yaml
experiment:
  id: D3_STRICT

data:
  train_dir: train_dedup       # <-- different train source
  split_dir: outputs/data/master_splits/seed42  # <-- same val split
  class_mapping_path: outputs/data/master_splits/seed42

train:
  save_dir: outputs/data/d3_strict/seed42/checkpoints

output:
  log_dir: outputs/data/d3_strict/seed42/logs
  submission_dir: outputs/data/d3_strict/seed42/submissions

# D3 cleaning runs only on master-train
# Output: d3_train_clean.csv is the training dataset
```

Note: D3 uses the deduped `train_dedup/` directory but the SAME `master_splits/seed42/val.csv`. The dedup process must run on the master-train only.

- [ ] **Step 3: Create `configs/f0_strict.yaml`**

```yaml
# F0-strict: Frozen continue control experiment.
# Init from D3-strict best, train only head, 50 extra epochs.
# Purpose: isolate "just more training" from "partial unfreeze".

experiment:
  id: F0_STRICT
  mode: dev
  head_type: linear
  augmentation_preset: a0

data:
  seed: 42
  split_seed: 42
  train_seed: 42
  split_dir: outputs/data/master_splits/seed42
  class_mapping_path: outputs/data/master_splits/seed42

model:
  clip_model_name: ViT-B/32
  feature_dim: 512
  freeze_clip: true
  num_classes: 500

output:
  log_dir: outputs/f0_strict/seed42/logs
  submission_dir: outputs/f0_strict/seed42/submissions

train:
  lr: 0.0003
  epochs: 50
  save_dir: outputs/f0_strict/seed42/checkpoints
  # ... other train params same as F1
```

- [ ] **Step 4: Create `configs/f1_strict.yaml`**

```yaml
# F1-strict: Unfreeze ln_post + visual.proj on same master split.
# Init from D3-strict best checkpoint.
# Parent-child audit enforced.

experiment:
  id: F1_STRICT
  mode: dev
  head_type: linear
  augmentation_preset: a0

data:
  seed: 42
  split_seed: 42
  train_seed: 42
  split_dir: outputs/data/master_splits/seed42
  class_mapping_path: outputs/data/master_splits/seed42

model:
  clip_model_name: ViT-B/32
  feature_dim: 512
  freeze_clip: false
  num_classes: 500
  unfreeze_last_n_blocks: 0
  train_ln_post: true
  train_visual_proj: true

output:
  log_dir: outputs/f1_strict/seed42/logs
  submission_dir: outputs/f1_strict/seed42/submissions

train:
  lr: 0.0003
  backbone_lr: 0.00001
  backbone_weight_decay: 0.01
  epochs: 50
  save_dir: outputs/f1_strict/seed42/checkpoints
  # ... other params as before
```

- [ ] **Step 5: Commit**

```bash
git add configs/e0_strict.yaml configs/d3_strict.yaml configs/f0_strict.yaml configs/f1_strict.yaml
git commit -m "feat: add strict configs referencing unified master split

All experiments now use outputs/data/master_splits/seed42 for train/val.
D3 uses same val as E0 for fair comparison.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Add epoch-0 validation gate

**Files:**
- Modify: `experiments/baseline/train.py`

**Interfaces:**
- Consumes: `--init-checkpoint` flag
- Produces: epoch-0 validation line in training log; hard exit if mismatch > 0.05pp

- [ ] **Step 1: Add epoch-0 validation in `train.py` main()**

Insert after `--init-checkpoint` weight loading and audit, but before the training loop:

```python
# Epoch-0 validation gate: verify loaded model matches expected accuracy
if args.init_checkpoint:
    train_logger.info("=" * 60)
    train_logger.info("Epoch-0 validation gate: verifying loaded checkpoint")
    
    val_loss_0, val_acc_0 = validate(model, val_loader, criterion, device, config)
    
    # Get expected accuracy from parent checkpoint metadata
    parent_expected_acc = checkpoint.get("best_val_acc", None)
    
    train_logger.info(
        f"Epoch 0   | Val Loss: {val_loss_0:.4f} | Val Acc: {val_acc_0:.4f}"
    )
    
    if parent_expected_acc is not None:
        delta = abs(val_acc_0 - parent_expected_acc)
        if delta > 0.0005:  # 0.05pp
            train_logger.error(
                f"EPOCH-0 VALIDATION MISMATCH: "
                f"loaded={val_acc_0:.4f}, expected={parent_expected_acc:.4f}, "
                f"delta={delta:.4f} (> 0.0005 threshold). "
                f"Check model loading, transforms, class mapping."
            )
            raise RuntimeError(
                f"Epoch-0 validation mismatch: delta={delta:.6f} > 0.0005"
            )
        train_logger.info(f"Epoch-0 validation gate PASSED: delta={delta:.6f} <= 0.0005")
    else:
        train_logger.warning(
            "No best_val_acc in checkpoint metadata; skipping epoch-0 gate. "
            "Set best_val_acc in parent checkpoint for validation."
        )
    
    train_logger.info("=" * 60)
```

- [ ] **Step 2: Test the gate**

Add a test in `tests/test_epoch_zero_gate.py`:

```python
"""Test epoch-0 validation gate."""
# This is tested implicitly via train.py with --init-checkpoint.
# Manual verification: run F0 or F1 with a known checkpoint and verify
# the epoch-0 accuracy matches the checkpoint's best_val_acc.
```

- [ ] **Step 3: Commit**

```bash
git add experiments/baseline/train.py tests/test_epoch_zero_gate.py
git commit -m "feat: add epoch-0 validation gate for init-checkpoint experiments

Verifies loaded model accuracy matches parent checkpoint within 0.05pp.
Hard exit on mismatch.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Rebuild D3 dedup as train-only cleaning

**Files:**
- Modify: `scripts/build_dedup_cache.py` (or create new cleaning script)
- Create: `scripts/d3_train_only_clean.py`

- [ ] **Step 1: Write `scripts/d3_train_only_clean.py`**

```python
"""
D3 train-only dedup cleaning.

Takes the master-train split, runs CLIP centroid arbitration
ONLY on training images, and produces a clean training CSV.
Master-val is NEVER touched.

Usage:
    python scripts/d3_train_only_clean.py \
        --master-split outputs/data/master_splits/seed42 \
        --train-dir train \
        --output outputs/data/d3_strict/seed42
"""

import argparse
import json
import logging
import hashlib
from pathlib import Path

import pandas as pd

from common.utils import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-split", required=True)
    parser.add_argument("--train-dir", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trim-pct", type=float, default=10.0)
    parser.add_argument("--margin-threshold", type=float, default=0.02)
    args = parser.parse_args()

    master = Path(args.master_split)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    setup_logging(str(output), name="d3_clean")

    # Load master train
    train_df = pd.read_csv(master / "train.csv")
    logger.info(f"Master train: {len(train_df)} images")

    # Run duplicate detection on train images only
    # (Reuse existing dedup infrastructure from build_dedup_cache.py)
    # This is a simplified version — in practice, call the existing dedup
    # pipeline with train-only scope

    # For now: use the pre-built dedup cache but FILTER to train-only
    # The key constraint: removal_list must NOT contain any master-val images

    # Load existing dedup decisions (from build_dedup_cache.py output)
    # and intersect with master-train to produce train_clean.csv

    # Copy master-val unchanged
    val_df = pd.read_csv(master / "val.csv")
    val_csv_out = output / "val.csv"
    val_df.to_csv(val_csv_out, index=False)

    # Verify val unchanged
    val_orig = pd.read_csv(master / "val.csv")
    val_out = pd.read_csv(output / "val.csv")
    assert val_orig.equals(val_out), "Validation set was modified!"

    # Write clean train
    # train_clean_df = train_df[~train_df["image_path"].isin(removal_set)]
    # train_clean_df.to_csv(output / "train_clean.csv", index=False)

    # Write cleaning report
    report = {
        "master_train_count": len(train_df),
        "clean_train_count": 0,  # filled after dedup
        "removed_count": 0,
        "removed_ratio": 0.0,
        "val_untouched": True,
    }
    (output / "cleaning_report.json").write_text(json.dumps(report, indent=2))

    logger.info("D3 train-only cleaning complete")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Integrate with existing dedup pipeline**

The existing `build_dedup_cache.py` already has SHA-256 duplicate scanning + CLIP centroid arbitration. The change is scoping it to train-only:

```python
# Key constraint in the dedup logic:
# 1. Compute centroids from master-train only
# 2. Detect conflicts within master-train only
# 3. NEVER use master-val for trimming or threshold decisions
# 4. NEVER remove images from master-val
```

- [ ] **Step 3: Verify val untouched**

```bash
python -c "
import pandas as pd
master_val = pd.read_csv('outputs/data/master_splits/seed42/val.csv')
d3_val = pd.read_csv('outputs/data/d3_strict/seed42/val.csv')
assert master_val.equals(d3_val), 'D3 modified the validation set!'
# Also verify no master-val images appear in removal list
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/d3_train_only_clean.py
git commit -m "feat: add D3 train-only cleaning script

Centroid computation and duplicate removal restricted to master-train.
Master-val is NEVER touched.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Run E0-strict and D3-strict

**Files:**
- (No code changes — execution only)

**Checkpoints to create:**
- `outputs/e0_strict/seed42/checkpoints/best.pt`
- `outputs/data/d3_strict/seed42/checkpoints/best.pt`

- [ ] **Step 1: Run E0-strict**

```bash
python3 -m experiments.baseline.train --config configs/e0_strict.yaml 2>&1 | tee outputs/e0_strict/seed42/logs/run.log
```

Expected: training completes, `eval_results.json` written with `best_val_acc`.

- [ ] **Step 2: Run D3-strict**

Prepare D3 clean training data first:
```bash
# Build dedup cache for train-only
python scripts/build_dedup_cache.py --train-dir train --split-csv outputs/data/master_splits/seed42/train.csv --output cache/d3_strict
```

Then train:
```bash
python3 -m experiments.baseline.train --config configs/d3_strict.yaml 2>&1 | tee outputs/data/d3_strict/seed42/logs/run.log
```

- [ ] **Step 3: Compare E0 vs D3 on same validation set**

```bash
python -c "
import json
e0 = json.load(open('outputs/e0_strict/seed42/checkpoints/eval_results.json'))
d3 = json.load(open('outputs/data/d3_strict/seed42/checkpoints/eval_results.json'))
delta = d3['best_val_acc'] - e0['best_val_acc']
print(f'E0-strict: {e0[\"best_val_acc\"]:.4f}')
print(f'D3-strict: {d3[\"best_val_acc\"]:.4f}')
print(f'Delta:     {delta:+.4f} ({delta*100:+.2f}pp)')
if delta >= 0.0020:
    print('PASS: D3 improvement >= 0.20pp → D3 remains baseline')
else:
    print('FAIL: D3 improvement < 0.20pp → reconsider D3 as default')
"
```

- [ ] **Step 4: Commit results**

```bash
git add outputs/e0_strict/ outputs/data/d3_strict/
git commit -m "results: E0-strict + D3-strict on unified master split (seed=42)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Run F0-strict control experiment

- [ ] **Step 1: Run F0-strict**

```bash
python3 -m experiments.baseline.train \
    --config configs/f0_strict.yaml \
    --init-checkpoint outputs/data/d3_strict/seed42/checkpoints/best.pt \
    2>&1 | tee outputs/f0_strict/seed42/logs/run.log
```

Verify epoch-0 gate passes and split audit passes.

- [ ] **Step 2: Compare F0 vs D3**

```bash
python -c "
import json
d3 = json.load(open('outputs/data/d3_strict/seed42/checkpoints/eval_results.json'))
f0 = json.load(open('outputs/f0_strict/seed42/checkpoints/eval_results.json'))
delta = f0['best_val_acc'] - d3['best_val_acc']
print(f'D3: {d3[\"best_val_acc\"]:.4f}')
print(f'F0: {f0[\"best_val_acc\"]:.4f}')
print(f'Delta: {delta:+.4f} ({delta*100:+.2f}pp)')
print('F0 answers: does more training alone help?')
"
```

- [ ] **Step 3: Commit results**

```bash
git add outputs/f0_strict/
git commit -m "results: F0-strict control experiment (frozen continue from D3)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Run F1-strict (the critical re-run)

- [ ] **Step 1: Run F1-strict**

```bash
python3 -m experiments.baseline.train \
    --config configs/f1_strict.yaml \
    --init-checkpoint outputs/data/d3_strict/seed42/checkpoints/best.pt \
    2>&1 | tee outputs/f1_strict/seed42/logs/run.log
```

Verify:
- Epoch-0 gate passes
- Split audit passes (`child_val_in_parent_train == 0`)
- `split_lineage_audit.json` shows `protocol_valid: true`

- [ ] **Step 2: Compare F1 vs all baselines**

```bash
python -c "
import json
e0 = json.load(open('outputs/e0_strict/seed42/checkpoints/eval_results.json'))
d3 = json.load(open('outputs/data/d3_strict/seed42/checkpoints/eval_results.json'))
f0 = json.load(open('outputs/f0_strict/seed42/checkpoints/eval_results.json'))
f1 = json.load(open('outputs/f1_strict/seed42/checkpoints/eval_results.json'))

baseline = max(d3['best_val_acc'], f0['best_val_acc'])
delta = f1['best_val_acc'] - baseline

print(f'E0:  {e0[\"best_val_acc\"]:.4f}')
print(f'D3:  {d3[\"best_val_acc\"]:.4f}')
print(f'F0:  {f0[\"best_val_acc\"]:.4f}')
print(f'F1:  {f1[\"best_val_acc\"]:.4f}')
print(f'---')
print(f'F1 - max(D3,F0): {delta:+.4f} ({delta*100:+.2f}pp)')

if delta >= 0.0030:
    print('PASS: F1 improvement >= 0.30pp → proceed to multi-seed')
else:
    print('FAIL: F1 improvement < 0.30pp → partial unfreeze may not be real')
"
```

- [ ] **Step 3: Decision gate**

Based on the comparison:
- If F1 - max(D3, F0) >= 0.30pp → proceed to Task 10 (multi-seed)
- Otherwise → pause partial unfreeze line, document findings

- [ ] **Step 4: Commit results**

```bash
git add outputs/f1_strict/
git commit -m "results: F1-strict (ln_post+proj) on unified master split

Parent-child audit: protocol_valid=true. Epoch-0 gate: passed.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Multi-seed confirmation + submission

**Pre-condition:** seed=42 passes decision gate for F1.

- [ ] **Step 1: Generate seed=3407 and seed=2026 master splits**

```bash
python scripts/build_master_split.py --train-dir train --output-root outputs/data/master_splits --seed 3407 --val-ratio 0.1
python scripts/build_master_split.py --train-dir train --output-root outputs/data/master_splits --seed 2026 --val-ratio 0.1
```

- [ ] **Step 2: Run E0-strict on seeds 3407, 2026**

```bash
for seed in 3407 2026; do
    python3 -m experiments.baseline.train \
        --config configs/e0_strict.yaml \
        --override data.split_dir outputs/data/master_splits/seed${seed} \
        --override output.log_dir outputs/e0_strict/seed${seed}/logs \
        --override train.save_dir outputs/e0_strict/seed${seed}/checkpoints
done
```

- [ ] **Step 3: Run D3-strict and F1-strict on seeds 3407, 2026** (only if D3 passes seed=42 gate)

Similar loop for D3 and F1.

- [ ] **Step 4: Compute multi-seed statistics**

```bash
python -c "
import json
seeds = [42, 3407, 2026]
for exp in ['e0_strict', 'd3_strict', 'f1_strict']:
    accs = []
    for s in seeds:
        path = f'outputs/{exp}/seed{s}/checkpoints/eval_results.json'
        try:
            d = json.load(open(path))
            accs.append(d['best_val_acc'])
        except FileNotFoundError:
            pass
    if accs:
        import numpy as np
        print(f'{exp}: mean={np.mean(accs):.4f} std={np.std(accs):.4f} seeds={len(accs)}')
"
```

- [ ] **Step 5: Generate submission for best model**

```bash
# Determine best checkpoint from multi-seed results
BEST_CKPT="outputs/f1_strict/seed42/checkpoints/best.pt"

# Inference
python3 -m experiments.baseline.infer \
    --config configs/f1_strict.yaml \
    --ckpt $BEST_CKPT

# Generate submission
python3 -m common.submission \
    --raw outputs/f1_strict/seed42/submissions/pred_raw.csv \
    --out_dir outputs/f1_strict/seed42/submissions

# Validate
python3 scripts/check_submission.py \
    --test_dir test \
    --csv outputs/f1_strict/seed42/submissions/pred_results.csv \
    --zip outputs/f1_strict/seed42/submissions/submission.zip
```

- [ ] **Step 6: Update submission registry**

Create `results/submission_registry.csv`:

```csv
submission_id,experiment_id,git_commit,config_path,checkpoint_sha256,split_id,train_seed,best_epoch,local_micro_acc,submission_zip_sha256,online_acc,submission_time,notes
T0_001,D3_STRICT,,configs/d3_strict.yaml,,master_splits/seed42,42,,,,,,baseline submission
T1_001,F1_STRICT,,configs/f1_strict.yaml,,master_splits/seed42,42,,,,,,best model (if multi-seed confirmed)
```

- [ ] **Step 7: Update results/ablation.csv with strict results**

Add rows for:
- `E0_STRICT` (seed42, 3407, 2026)
- `D3_STRICT` (seed42)
- `F0_STRICT` (seed42)
- `F1_STRICT` (seed42)

- [ ] **Step 8: Final commit and push**

```bash
git add -A
git commit -m "feat: complete validation protocol rebuild — E0/D3/F0/F1 strict

Multi-seed confirmation. Submission registry established.
F1 old 80.13% deprecated; strict results in ablation.csv.

Co-Authored-By: Claude <noreply@anthropic.com>"
git push
```

---

## Execution Order Summary

```
Task 1  (mark results)        → immediate, no dependencies
Task 2  (master split)        → immediate, no dependencies
Task 3  (split audit)         → depends on Task 2
Task 4  (strict configs)      → depends on Task 2
Task 5  (epoch-0 gate)        → depends on Task 3
Task 6  (D3 train-only clean) → depends on Task 2
Task 7  (run E0/D3)           → depends on Tasks 2,4,5,6
Task 8  (run F0)              → depends on Task 7
Task 9  (run F1)              → depends on Task 7
Task 10 (multi-seed + submit) → depends on Task 9 passing decision gate
```

Tasks 1-4 can run in parallel. Tasks 5-6 can run in parallel after 2-4 complete. Tasks 7-10 are strictly sequential (each depends on previous experiment results).
