# Baseline Optimization Design

**Date**: 2026-07-09
**Status**: approved
**Scope**: Fix 6 bugs (P0×3, P1×3) + 5 structural improvements in the noise-label FGVC baseline

## Module A: P0 Bug Fixes

### A1 — Inference Label Mapping (P0)

**Root cause**: `experiments/baseline/infer.py:87` formats `pred_idx` directly as 4-digit string (`f"{int(pred_idx):04d}"`), ignoring `idx_to_class.json`. If class folders are named `0001`–`0500` but mapped to indices 0–499, submission labels will be off by one for all classes.

**Fix**: Pass `idx_to_class` dict into `run_inference()`, look up `pred_class_name = idx_to_class[str(int(pred_idx))]`, use `pred_class_name.zfill(4)` as fallback.

**Files**: `experiments/baseline/infer.py`

**Acceptance criteria**:
1. `pred_label` value comes from `idx_to_class` lookup, not raw index formatting
2. With class folders `0001`–`0500`, submission labels are `0001`–`0500` (not shifted to `0000`–`0499`)
3. Backward compatible: with class folders `0000`–`0499`, behavior unchanged

### A2 — Script Naming Consistency (P0)

**Root cause**: `train.py:389` error message references `scripts/split_train_val.py`; `split_data.py` docstring also references `split_train_val.py`. The actual script is `scripts/split_data.py`.

**Fix**: Replace all `split_train_val.py` references with `split_data.py`.

**Files**: `experiments/baseline/train.py`, `scripts/split_data.py`

**Acceptance criteria**: `grep -r "split_train_val" .` returns zero matches

### A3 — Submission Checker File-Name Validation (P0)

**Root cause**: `check_submission.py:177` computes `all_ok = all(not e.startswith("❌") for e in errors[1:])`, skipping `errors[0]` which is the file-name check. A CSV named `wrong_name.csv` would not cause a failure.

**Fix**: Change `errors[1:]` to `errors`.

**Files**: `scripts/check_submission.py`

**Acceptance criteria**: If CSV filename is not `pred_results.csv`, the script exits with code 1

---

## Module B: P1 Bug Fixes

### B1 — check_data.py log_dir Default (P1)

**Root cause**: `check_data.py:61` sets `default="outputs/logs"`. Since argparse default is not `None`, the config's `output.log_dir` value is never used.

**Fix**: Change `default="outputs/logs"` to `default=None`.

**Files**: `scripts/check_data.py`

**Acceptance criteria**: With `--config configs/baseline.yaml`, logs go to `outputs/baseline/logs/` not `outputs/logs/`

### B2 — AMP Device-Aware Autocast (P1)

**Root cause**: `train.py`, `evaluate.py`, `infer.py` all use hardcoded `autocast('cuda')` and `GradScaler('cuda', ...)`. On CPU-only machines this raises an error even though the rest of the code falls back to CPU.

**Fix**: Replace all occurrences:
- `autocast('cuda')` → `autocast(device_type=device.type, enabled=use_amp)`
- `GradScaler('cuda', enabled=...)` → `GradScaler(device=device.type, enabled=...)`

**Files**: `experiments/baseline/train.py`, `experiments/baseline/evaluate.py`, `experiments/baseline/infer.py`

**Acceptance criteria**:
1. CUDA environment: AMP works as before
2. CPU environment: no error from `autocast('cuda')`

### B3 — Config Path De-hardcoding (P1)

**Root cause**: `configs/baseline.yaml` contains hardcoded absolute paths (`/home/lux1/noise/train`, `/home/lux1/noise/test`), making it non-portable.

**Fix**:
1. Change paths to `data/preliminary/train` and `data/preliminary/test`
2. Create `configs/baseline.example.yaml` as reference
3. Add `configs/baseline.yaml` to `.gitignore` (so local changes aren't committed)

**Files**: `configs/baseline.yaml`, `.gitignore`

**Acceptance criteria**: README instructs users to copy/edit `baseline.yaml` with their own paths

---

## Module C: Structural Improvements

### C1 — Test Suite (`tests/`)

Create:
- `tests/__init__.py`
- `tests/test_label_mapping.py` — verifies `class_to_idx` ↔ `idx_to_class` bidirectionally, verifies `infer.py` pred_label uses `idx_to_class`
- `tests/test_split_data.py` — verifies train+val covers all samples, split ratio is correct
- `tests/test_submission.py` — verifies submission output format `name.jpg, 0001`

**Acceptance criteria**: `python -m pytest tests/ -v` all pass

### C2 — Tool Scripts (`tools/`)

Create:
- `tools/make_tiny_dataset.py` — generates 5 classes × 4 images each
- `tools/run_smoke_test.sh` — end-to-end pipeline on tiny dataset

**Acceptance criteria**: `bash tools/run_smoke_test.sh` runs start-to-finish without errors

### C3 — Experiment Registry (`results/ablation.csv`)

Create CSV with headers: `exp_id,method,backbone,head,freeze_clip,lr,batch_size,epochs,val_acc,online_acc,ckpt_path,notes`

Seed with baseline record if available.

**Acceptance criteria**: File exists with correct headers

### C4 — Smoke Test Script

`tools/run_smoke_test.sh` steps:
1. `python tools/make_tiny_dataset.py` — generate tiny train + test
2. `python scripts/check_data.py --train_dir ... --test_dir ...`
3. `python scripts/split_data.py --train_dir ...`
4. `python -m experiments.baseline.train --config configs/baseline.yaml` (1 epoch)
5. `python -m experiments.baseline.evaluate --config ... --ckpt ...`
6. `python -m experiments.baseline.infer --config ... --ckpt ...`
7. `python -m common.submission --raw ... --out_dir ...`
8. `python scripts/check_submission.py --test_dir ... --csv ... --zip ...`
9. Each step fails the script on error (`set -e`)

**Acceptance criteria**: Clean-room run passes all 9 steps

### C5 — Code Formatting

Run `black` and `isort` on all Python source directories. Verify with `black --check` and `isort --check`.

Also verify: `python -m py_compile` on all .py files, YAML parse check.

**Acceptance criteria**: `black --check` and `isort --check` report no changes needed

---

## Implementation Strategy

Execute in order: A (P0) → B (P1) → C (structural), because:
- A fixes are correctness-critical and must land first
- B fixes improve robustness without changing behavior
- C adds new files/tooling and depends on A+B being stable

Two-agent workflow: **Implementer** agent applies all changes; **Verifier** agent runs acceptance criteria independently.
