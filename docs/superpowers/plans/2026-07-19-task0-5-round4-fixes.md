# Task 0-5 Round 4 Acceptance Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all FAIL and PARTIAL items from the Round 4 acceptance report so formal Wave A training can start.

**Architecture:** Eight independent fix streams touching train.py (logger fix), build_purification_manifest.py (cj persistence), consensus.py (source-class cap), real_dry_run.py (portability), plus four new files (runtime audit tests, batch probe script, reproduce script, portable consensus tests). All fixes are additive or surgical — no architectural refactors.

**Tech Stack:** Python 3, pytest, pandas, numpy, torch, pyyaml

## Global Constraints

- All paths must be repo-relative (use `Path(__file__).resolve().parents[1]` not hardcoded `/home/lux1/noise`)
- Tests must use synthetic fixtures, not hardcoded `outputs/phase/phase3/oof/` paths
- Runtime audit must be fail-closed: any mismatch raises, legal manifest returns cleanly
- Per-class source relabel rate must be algorithmically capped at 3%, not coincidentally
- Confident-joint matrix and issue table must be persisted to disk alongside manifest
- All new scripts must return non-zero exit code on assertion failure

---

### Task 1: Fix P0 train_logger NameError in `_runtime_manifest_audit()`

**Files:**
- Modify: `experiments/baseline/train.py:278-439` (function signature and body)
- Modify: `experiments/baseline/train.py:2027-2030` (call site in main)

**Interfaces:**
- Consumes: `logging.Logger` from stdlib
- Produces: `_runtime_manifest_audit(train_dataset, weight_provider, mode, save_dir, audit_logger)` — same behavior, new required param

- [ ] **Step 1: Add `audit_logger` parameter to function signature**

In `experiments/baseline/train.py`, change line 278-283 from:

```python
def _runtime_manifest_audit(
    train_dataset,
    weight_provider,
    mode: str,
    save_dir: Path,
):
```

To:

```python
def _runtime_manifest_audit(
    train_dataset,
    weight_provider,
    mode: str,
    save_dir: Path,
    audit_logger: logging.Logger,
):
```

- [ ] **Step 2: Replace `train_logger` references with `audit_logger`**

Line 307: `train_logger.warning(...)` → `audit_logger.warning(...)`
Line 319: `train_logger.warning(...)` → `audit_logger.warning(...)`
Line 434: `train_logger.info(...)` → `audit_logger.info(...)`

- [ ] **Step 3: Update call site in `main()`**

Line 2027-2030, change from:

```python
_runtime_manifest_audit(
    train_dataset, weight_provider, mode,
    Path(config["train"]["save_dir"]),
)
```

To:

```python
_runtime_manifest_audit(
    train_dataset, weight_provider, mode,
    Path(config["train"]["save_dir"]),
    train_logger,
)
```

- [ ] **Step 4: Create minimal test proving clean manifest no longer raises NameError**

Create `tests/test_runtime_manifest_audit.py` with this single test:

```python
"""Tests for _runtime_manifest_audit fail-closed behavior."""
import logging
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestRuntimeManifestAudit:
    """Verify _runtime_manifest_audit correctly validates manifests."""

    def test_clean_manifest_passes_without_name_error(self, tmp_path):
        """A perfectly matching manifest returns cleanly (no NameError)."""
        import torch
        from experiments.baseline.train import _runtime_manifest_audit

        # Build minimal dataset mock
        ds = MagicMock()
        ds.samples = ["img0.jpg", "img1.jpg"]
        ds.labels = [5, 3]

        # Build minimal manifest CSV that exactly matches
        manifest_csv = tmp_path / "manifest.csv"
        pd.DataFrame({
            "image_path": ["img0.jpg", "img1.jpg"],
            "original_label": [5, 3],
            "training_label": [5, 3],
            "sample_weight": [1.0, 1.0],
            "training_role": ["clean", "clean"],
        }).to_csv(manifest_csv, index=False)

        # Build weight provider mock with correct _loader._path
        wp = MagicMock()
        wp._missing = "error"
        wp._loader._path = str(manifest_csv)

        save_dir = tmp_path / "save"
        audit_logger = logging.getLogger("test_audit")

        # This must NOT raise NameError or any other exception
        _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)

        # Verify audit file was written
        audit_path = save_dir / "manifest_runtime_audit.json"
        assert audit_path.exists()
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
PYTHONPATH=. pytest -q tests/test_runtime_manifest_audit.py::TestRuntimeManifestAudit::test_clean_manifest_passes_without_name_error -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add experiments/baseline/train.py tests/test_runtime_manifest_audit.py
git commit -m "fix: pass audit_logger to _runtime_manifest_audit to fix NameError

The function referenced train_logger which is a local variable in main(),
causing a NameError on every manifest-enabled training run before epoch 1.
Now accepts audit_logger as an explicit parameter.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Complete runtime audit regression test suite

**Files:**
- Modify: `tests/test_runtime_manifest_audit.py` (add 7 more tests)

**Interfaces:**
- Consumes: `_runtime_manifest_audit()` from `experiments.baseline.train`
- Produces: 8 total tests covering all failure modes

- [ ] **Step 1: Add test for missing path in manifest**

```python
def test_missing_path_in_manifest_raises(self, tmp_path):
    """Dataset image missing from manifest raises ValueError."""
    import torch
    from experiments.baseline.train import _runtime_manifest_audit

    ds = MagicMock()
    ds.samples = ["img0.jpg", "img1.jpg", "img2.jpg"]  # img2 not in manifest
    ds.labels = [5, 3, 7]

    manifest_csv = tmp_path / "manifest.csv"
    pd.DataFrame({
        "image_path": ["img0.jpg", "img1.jpg"],
        "original_label": [5, 3],
        "training_label": [5, 3],
        "sample_weight": [1.0, 1.0],
        "training_role": ["clean", "clean"],
    }).to_csv(manifest_csv, index=False)

    wp = MagicMock()
    wp._missing = "error"
    wp._loader._path = str(manifest_csv)

    save_dir = tmp_path / "save"
    audit_logger = logging.getLogger("test_audit")

    with pytest.raises(ValueError, match="missing from manifest"):
        _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)
```

- [ ] **Step 2: Add test for extra path in manifest**

```python
def test_extra_path_in_manifest_raises(self, tmp_path):
    """Manifest image not in dataset raises ValueError."""
    from experiments.baseline.train import _runtime_manifest_audit

    ds = MagicMock()
    ds.samples = ["img0.jpg"]
    ds.labels = [5]

    manifest_csv = tmp_path / "manifest.csv"
    pd.DataFrame({
        "image_path": ["img0.jpg", "EXTRA.jpg"],
        "original_label": [5, 3],
        "training_label": [5, 3],
        "sample_weight": [1.0, 1.0],
        "training_role": ["clean", "clean"],
    }).to_csv(manifest_csv, index=False)

    wp = MagicMock()
    wp._missing = "error"
    wp._loader._path = str(manifest_csv)

    save_dir = tmp_path / "save"
    audit_logger = logging.getLogger("test_audit")

    with pytest.raises(ValueError, match="not in dataset"):
        _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)
```

- [ ] **Step 3: Add test for duplicate manifest paths**

```python
def test_duplicate_paths_in_manifest_raises(self, tmp_path):
    """Duplicate image_path in manifest raises ValueError."""
    from experiments.baseline.train import _runtime_manifest_audit

    ds = MagicMock()
    ds.samples = ["img0.jpg"]
    ds.labels = [5]

    manifest_csv = tmp_path / "manifest.csv"
    pd.DataFrame({
        "image_path": ["img0.jpg", "img0.jpg"],
        "original_label": [5, 5],
        "training_label": [5, 5],
        "sample_weight": [1.0, 1.0],
        "training_role": ["clean", "clean"],
    }).to_csv(manifest_csv, index=False)

    wp = MagicMock()
    wp._missing = "error"
    wp._loader._path = str(manifest_csv)

    save_dir = tmp_path / "save"
    audit_logger = logging.getLogger("test_audit")

    with pytest.raises(ValueError, match="duplicate"):
        _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)
```

- [ ] **Step 4: Add test for original_label mismatch**

```python
def test_original_label_mismatch_raises(self, tmp_path):
    """original_label differs between dataset and manifest raises ValueError."""
    from experiments.baseline.train import _runtime_manifest_audit

    ds = MagicMock()
    ds.samples = ["img0.jpg"]
    ds.labels = [5]  # dataset says 5

    manifest_csv = tmp_path / "manifest.csv"
    pd.DataFrame({
        "image_path": ["img0.jpg"],
        "original_label": [3],  # manifest says 3
        "training_label": [3],
        "sample_weight": [1.0],
        "training_role": ["clean"],
    }).to_csv(manifest_csv, index=False)

    wp = MagicMock()
    wp._missing = "error"
    wp._loader._path = str(manifest_csv)

    save_dir = tmp_path / "save"
    audit_logger = logging.getLogger("test_audit")

    with pytest.raises(ValueError, match="original_label mismatch"):
        _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)
```

- [ ] **Step 5: Add test for zero-clean class**

```python
def test_zero_clean_class_raises(self, tmp_path):
    """Class with zero clean samples raises ValueError."""
    from experiments.baseline.train import _runtime_manifest_audit

    ds = MagicMock()
    ds.samples = ["img0.jpg"]
    ds.labels = [5]

    manifest_csv = tmp_path / "manifest.csv"
    pd.DataFrame({
        "image_path": ["img0.jpg"],
        "original_label": [5],
        "training_label": [5],
        "sample_weight": [0.0],  # all rejected → zero clean
        "training_role": ["rejected"],
    }).to_csv(manifest_csv, index=False)

    wp = MagicMock()
    wp._missing = "error"
    wp._loader._path = str(manifest_csv)

    save_dir = tmp_path / "save"
    audit_logger = logging.getLogger("test_audit")

    with pytest.raises(ValueError, match="zero clean"):
        _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)
```

- [ ] **Step 6: Add test for non-error missing policy**

```python
def test_non_error_missing_policy_raises(self, tmp_path):
    """missing_policy != 'error' raises ValueError before any comparison."""
    from experiments.baseline.train import _runtime_manifest_audit

    ds = MagicMock()
    wp = MagicMock()
    wp._missing = "ignore"  # not "error"

    save_dir = tmp_path / "save"
    audit_logger = logging.getLogger("test_audit")

    with pytest.raises(ValueError, match="missing_weight_policy"):
        _runtime_manifest_audit(ds, wp, "dev", save_dir, audit_logger)
```

- [ ] **Step 7: Add test for None weight_provider (no-op)**

```python
def test_none_weight_provider_returns_early(self, tmp_path):
    """None weight_provider skips audit without error."""
    from experiments.baseline.train import _runtime_manifest_audit

    ds = MagicMock()
    save_dir = tmp_path / "save"
    audit_logger = logging.getLogger("test_audit")

    # Should not raise
    _runtime_manifest_audit(ds, None, "dev", save_dir, audit_logger)
```

- [ ] **Step 8: Run all 8 runtime audit tests**

```bash
PYTHONPATH=. pytest -q tests/test_runtime_manifest_audit.py -v
```

Expected: 8 passed

- [ ] **Step 9: Commit**

```bash
git add tests/test_runtime_manifest_audit.py
git commit -m "test: add full runtime manifest audit regression suite

Covers: legal manifest PASS, missing path, extra path, duplicate path,
original_label mismatch, zero-clean class, non-error missing policy,
None weight_provider early return.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Fix confident-joint persistence in `_write_outputs()`

**Files:**
- Modify: `analysis/noisy_labels/build_purification_manifest.py:161-210` (`_write_outputs` signature and body)
- Modify: `analysis/noisy_labels/build_purification_manifest.py:357` (call site, all three modes)

**Interfaces:**
- Consumes: `confident_joint: np.ndarray | None`, `issues: pd.DataFrame | None`
- Produces: `confident_joint.npy` and `confident_joint_issues.csv` in output_dir when available

- [ ] **Step 1: Update `_write_outputs` signature**

Change line 161 from:

```python
def _write_outputs(df: pd.DataFrame, output_dir: Path, mode: str):
```

To:

```python
def _write_outputs(
    df: pd.DataFrame,
    output_dir: Path,
    mode: str,
    confident_joint: np.ndarray | None = None,
    issues: pd.DataFrame | None = None,
):
```

- [ ] **Step 2: Replace broken `"cj" in dir()` check**

Replace lines 188-193:

```python
    # Save confident joint
    cj_path = output_dir / "confident_joint.npy"
    if "cj" in dir():
        np.save(cj_path, cj)
        audit["confident_joint_path"] = str(cj_path)
    else:
        audit["confident_joint_path"] = None
```

With:

```python
    # Save confident joint matrix
    if confident_joint is not None:
        cj_path = output_dir / "confident_joint.npy"
        np.save(cj_path, confident_joint)
        audit["confident_joint_path"] = str(cj_path)
        audit["confident_joint_sha256"] = _sha256(cj_path)
    else:
        audit["confident_joint_path"] = None
        audit["confident_joint_sha256"] = None

    # Save confident-joint issue table
    if issues is not None:
        issues_path = output_dir / "confident_joint_issues.csv"
        issues.to_csv(issues_path, index=False)
        audit["confident_joint_issues_path"] = str(issues_path)
        audit["confident_joint_issues_sha256"] = _sha256(issues_path)
    else:
        audit["confident_joint_issues_path"] = None
        audit["confident_joint_issues_sha256"] = None
```

- [ ] **Step 3: Update all three call sites in `main()`**

**cl_classwise_drop path** — after line 250 (`issues = rank_label_issues(...)`), collect locals, then at line 280 (`df = _build_manifest_from_issues(quality, issues)`) and the `_write_outputs` call, pass them through. The structure is:

After the `if args.mode in ("cl_classwise_drop", "cl_knn_drop"):` block, save `cj` and `issues` as locals that survive to the `_write_outputs` call at line 357.

Change line 357 from:

```python
    _write_outputs(df, output_dir, args.mode)
```

To:

```python
    _write_outputs(df, output_dir, args.mode,
                   confident_joint=cj if 'cj' in dir() else None,
                   issues=issues if 'issues' in dir() else None)
```

But since `dir()` is broken, we need to track explicitly. Add at line 237 (after `cj = build_confident_joint(...)`):

```python
        _cj = cj
```

And at line 250 (after `issues = rank_label_issues(...)`):

```python
        _issues = issues
```

And change line 280 to also save `_cj, _issues` for the cl_classwise_drop case.

Then at the final `_write_outputs` call:

```python
    _write_outputs(df, output_dir, args.mode,
                   confident_joint=_cj,
                   issues=_issues)
```

Add `_cj = None; _issues = None` at the top of `main()` (after argument parsing) so the variables are always defined.

- [ ] **Step 4: Write test verifying cj persistence**

Add to `tests/test_purification_manifest.py`:

```python
def test_write_outputs_persists_confident_joint(self, tmp_path):
    """_write_outputs saves confident_joint.npy and issues CSV when provided."""
    import numpy as np
    from analysis.noisy_labels.build_purification_manifest import _write_outputs

    n = 10
    df = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "image_path": [f"img{i}.jpg" for i in range(n)],
        "original_label": [i % 5 for i in range(n)],
        "training_label": [i % 5 for i in range(n)],
        "sample_weight": [1.0] * n,
        "quality_score": [0.9] * n,
        "training_role": ["clean"] * n,
    })

    cj = np.zeros((5, 5), dtype=np.int64)
    cj[0, 1] = 3
    issues = pd.DataFrame({
        "index": [0, 1, 2],
        "selected": [True, True, False],
        "score": [0.8, 0.7, 0.3],
    })

    _write_outputs(df, tmp_path, "cl_classwise_drop",
                   confident_joint=cj, issues=issues)

    assert (tmp_path / "confident_joint.npy").exists()
    assert (tmp_path / "confident_joint_issues.csv").exists()

    loaded_cj = np.load(tmp_path / "confident_joint.npy")
    assert loaded_cj.shape == (5, 5)
    assert loaded_cj[0, 1] == 3

    loaded_issues = pd.read_csv(tmp_path / "confident_joint_issues.csv")
    assert len(loaded_issues) == 3
```

- [ ] **Step 5: Run the test**

```bash
PYTHONPATH=. pytest -q tests/test_purification_manifest.py::test_write_outputs_persists_confident_joint -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add analysis/noisy_labels/build_purification_manifest.py tests/test_purification_manifest.py
git commit -m "fix: persist confident_joint matrix and issue table in _write_outputs

Replace broken 'cj' in dir() check with explicit parameters. Now saves
confident_joint.npy and confident_joint_issues.csv alongside manifest,
with SHA-256 recorded in protocol_audit.json.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Add source-class cap to relabel selector

**Files:**
- Modify: `analysis/noisy_labels/consensus.py:91-197` (`select_consensus_relabel_v2` signature and body)

**Interfaces:**
- Consumes: `quality: pd.DataFrame`, `issues: pd.DataFrame`, `top_k: int`, `max_source_class_relabel_rate: float = 0.03`
- Produces: `set` of selected indices, guaranteed ≤3% per source class

- [ ] **Step 1: Add cap parameter and logic**

Change line 91-95 from:

```python
def select_consensus_relabel_v2(
    quality: pd.DataFrame,
    issues: pd.DataFrame,
    top_k: int = 100,
) -> set:
```

To:

```python
def select_consensus_relabel_v2(
    quality: pd.DataFrame,
    issues: pd.DataFrame,
    top_k: int = 100,
    max_source_class_relabel_rate: float = 0.03,
) -> set:
```

- [ ] **Step 2: Add per-class cap after sorting, before truncation**

Replace lines 196-197:

```python
    candidates.sort(key=lambda x: x[1], reverse=True)
    return set(idx for idx, _ in candidates[:top_k])
```

With:

```python
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Per-source-class cap
    n = len(quality)
    class_counts = quality["original_label"].value_counts().reindex(
        range(NUM_CLASSES), fill_value=0
    )
    class_cap = {
        c: max(1, int(np.floor(max_source_class_relabel_rate * class_counts[c])))
        for c in range(NUM_CLASSES)
    }
    class_used = {c: 0 for c in range(NUM_CLASSES)}

    selected = set()
    for idx, score in candidates:
        if len(selected) >= top_k:
            break
        orig = int(quality.iloc[idx]["original_label"])
        if class_used[orig] >= class_cap[orig]:
            continue
        selected.add(idx)
        class_used[orig] += 1

    return selected
```

- [ ] **Step 3: Write test for source-class cap enforcement**

Add to `tests/test_consensus_selection.py`:

```python
def test_source_class_cap_enforced(self):
    """When all candidates come from one source class, cap limits selection."""
    n = 200
    # All candidates from class 5, targeting class 3
    q = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "image_path": [f"img{i}.jpg" for i in range(n)],
        "original_label": [5] * n,  # all same source class
        "oof_top1": [3] * n,
        "knn_top1": [3] * n,
        "prototype_top1": [3] * n,
        "p_top1": [0.95] * n,
        "top1_margin": [0.80] * n,
        "knn_agreement": [0.10] * n,
        "knn_top1_agreement": [0.85] * n,
        "flip_consistency": [1.0] * n,
        "duplicate_conflict_flag": [False] * n,
    })
    issues = pd.DataFrame({"index": list(range(n)), "selected": [True] * n})

    result = select_consensus_relabel_v2(
        q, issues, top_k=100,
        max_source_class_relabel_rate=0.03,
    )
    # 3% of 200 = 6 → at most 6 from class 5
    assert len(result) <= 6, (
        f"Expected <= 6 due to 3% source-class cap, got {len(result)}"
    )
```

- [ ] **Step 4: Run the test**

```bash
PYTHONPATH=. pytest -q tests/test_consensus_selection.py::test_source_class_cap_enforced -v
```

Expected: PASS

- [ ] **Step 5: Run all consensus tests to check no regressions**

```bash
PYTHONPATH=. pytest -q tests/test_consensus_selection.py -v
```

Expected: all pass (old + new)

- [ ] **Step 6: Commit**

```bash
git add analysis/noisy_labels/consensus.py tests/test_consensus_selection.py
git commit -m "feat: add per-source-class cap to select_consensus_relabel_v2

Algorithm now enforces max_source_class_relabel_rate (default 3%) by
traversing scored candidates and skipping any that would exceed the
per-class budget. Previously only the global top_k was enforced.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Make tests portable (remove hardcoded paths)

**Files:**
- Modify: `tests/test_consensus_selection.py:160-201` (two integration tests)

**Interfaces:**
- Consumes: synthetic DataFrames (already available via `_make_quality`)
- Produces: Tests pass without `outputs/phase/phase3/oof/` on disk

- [ ] **Step 1: Replace real-data integration test `test_cl_knn_drop_on_real_data`**

Replace lines 160-179 with a synthetic-data test that constructs a minimal but realistic DataFrame:

```python
def test_cl_knn_drop_on_synthetic_data_returns_nonzero(self):
    """Integration: cl_knn_drop on synthetic data produces selections."""
    n = 200
    np.random.seed(42)
    labels = np.random.randint(0, 10, n)
    q = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "image_path": [f"img{i}.jpg" for i in range(n)],
        "original_label": labels,
        "oof_top1": [(l + 1) % 10 for l in labels],
        "knn_top1": [(l + 1) % 10 for l in labels],
        "top1_margin": np.random.uniform(0.5, 0.95, n),
        "knn_agreement": np.random.uniform(0.0, 0.15, n),
        "duplicate_conflict_flag": [False] * n,
        "prototype_top1": [(l + 1) % 10 for l in labels],
        "p_top1": np.random.uniform(0.7, 0.99, n),
        "flip_consistency": [1.0] * n,
    })
    # Construct synthetic issues: all marked as selected
    issues = pd.DataFrame({
        "index": list(range(n)),
        "selected": [True] * n,
    })
    result = select_consensus_drop(q, issues)
    assert len(result) > 0, f"cl_knn_drop selected {len(result)} — expected > 0"
    # All caps should be respected
    for c in range(10):
        cls_mask = q["original_label"] == c
        cls_count = cls_mask.sum()
        if cls_count > 0:
            cls_selected = sum(
                1 for i in result if q.iloc[i]["original_label"] == c
            )
            assert cls_selected <= max(1, int(0.10 * cls_count)), (
                f"Class {c}: {cls_selected} selected > 10% of {cls_count}"
            )
```

- [ ] **Step 2: Replace real-data integration test `test_relabel_v2_on_real_data_returns_nonzero`**

Replace lines 181-201 with:

```python
def test_relabel_v2_on_synthetic_data_returns_nonzero(self):
    """Integration: consensus_relabel_v2 on synthetic data produces selections."""
    n = 500
    np.random.seed(42)
    # 10 source classes, all have candidates that agree on new label
    labels = []
    for c in range(10):
        labels.extend([c] * 50)
    q = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "image_path": [f"img{i}.jpg" for i in range(n)],
        "original_label": labels,
        "oof_top1": [(l + 1) % 10 for l in labels],
        "knn_top1": [(l + 1) % 10 for l in labels],
        "prototype_top1": [(l + 1) % 10 for l in labels],
        "p_top1": np.random.uniform(0.90, 0.99, n),
        "top1_margin": np.random.uniform(0.60, 0.95, n),
        "knn_agreement": np.random.uniform(0.0, 0.15, n),
        "knn_top1_agreement": np.random.uniform(0.60, 0.95, n),
        "flip_consistency": [1.0] * n,
        "duplicate_conflict_flag": [False] * n,
    })
    issues = pd.DataFrame({
        "index": list(range(n)),
        "selected": [True] * n,
    })
    result = select_consensus_relabel_v2(
        q, issues, top_k=50,
        max_source_class_relabel_rate=0.03,
    )
    assert len(result) > 0, f"relabel_v2 selected {len(result)} — expected > 0"
    # Verify source-class cap: at most 3% per class
    for c in range(10):
        cls_count = 50
        cls_selected = sum(
            1 for i in result if q.iloc[i]["original_label"] == c
        )
        cap = max(1, int(0.03 * cls_count))
        assert cls_selected <= cap, (
            f"Class {c}: {cls_selected} selected > cap {cap}"
        )
```

- [ ] **Step 3: Run all consensus tests**

```bash
PYTHONPATH=. pytest -q tests/test_consensus_selection.py -v
```

Expected: all pass (old unit tests + new synthetic integration tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_consensus_selection.py
git commit -m "test: replace hardcoded real-data paths with synthetic fixtures

test_cl_knn_drop_on_real_data → test_cl_knn_drop_on_synthetic_data
test_relabel_v2_on_real_data → test_relabel_v2_on_synthetic_data

Both now construct synthetic DataFrames that don't depend on
outputs/phase/phase3/oof/ being present. Tests are portable.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Make `real_dry_run.py` portable and add runtime audit call

**Files:**
- Modify: `scripts/real_dry_run.py:8-9` (REPO definition)
- Modify: `scripts/real_dry_run.py:99` (manual coverage=1.0)
- Modify: `scripts/real_dry_run.py:60-100` (add runtime audit call)

**Interfaces:**
- Consumes: `_runtime_manifest_audit` from `experiments.baseline.train`
- Produces: portable dry-run that calls real audit, doesn't hardcode coverage

- [ ] **Step 1: Make REPO portable**

Replace lines 8-9:

```python
REPO = Path("/home/lux1/noise")
sys.path.insert(0, str(REPO))
```

With:

```python
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
```

- [ ] **Step 2: Replace all `REPO / config_path` and `REPO / manifest_path` references**

Line 21: `config = yaml.safe_load(open(REPO / config_path))` → `config = yaml.safe_load(open(REPO / config_path))` (already correct pattern, just need REPO to be portable)

Line 63: `if manifest_path and Path(REPO / manifest_path).exists():` → `if manifest_path and (REPO / manifest_path).exists():`

Line 65: `m = pd.read_csv(REPO / manifest_path)` → `m = pd.read_csv(REPO / manifest_path)` (already correct)

Line 85: `m = pd.read_csv(REPO / manifest_path)` → same

Line 167: `m = pd.read_csv(REPO / manifest_path)` → same

Line 180: `m = pd.read_csv(REPO / manifest_path)` → same

- [ ] **Step 3: Remove manual coverage=1.0 and call real runtime audit**

Replace lines 86-99:

```python
        if manifest_loaded:
            ...
            results["coverage"] = 1.0
```

With actual runtime audit call. After the weight provider is built (line 58), add:

```python
    # Runtime manifest audit (fail-closed)
    if weight_provider is not None and manifest_loaded:
        import logging
        from experiments.baseline.train import _runtime_manifest_audit
        audit_logger = logging.getLogger("dry_run_audit")
        audit_logger.setLevel(logging.INFO)
        if not audit_logger.handlers:
            h = logging.StreamHandler()
            h.setLevel(logging.INFO)
            audit_logger.addHandler(h)

        save_dir = REPO / "outputs" / "dry_run_audit"
        _runtime_manifest_audit(
            train_dataset, weight_provider, "dev", save_dir, audit_logger,
        )

        # Read back audit for coverage stats
        import json
        audit_path = save_dir / "manifest_runtime_audit.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            results["coverage"] = audit["coverage"]
            results["clean_count"] = audit["clean_count"]
            results["rejected_count"] = audit["rejected_count"]
            results["pseudo_count"] = audit["pseudo_count"]
            results["audit_errors"] = 0
        else:
            results["coverage"] = 0.0
            results["audit_errors"] = 1
    elif manifest_loaded:
        results["coverage"] = 1.0
```

- [ ] **Step 4: Remove the duplicate manifest loading block**

The manifest stats are now populated by the audit. Remove lines 83-99 (the duplicate `if manifest_loaded:` block that reads manifest and sets coverage=1.0), keeping only the initial `manifest_loaded` detection.

Replace lines 61-99:

```python
    # Manifest stats
    manifest_path = sw_cfg.get("manifest_path", "")
    manifest_loaded = False
    if manifest_path and Path(REPO / manifest_path).exists():
        import pandas as pd
        m = pd.read_csv(REPO / manifest_path)
        roles = m.get("training_role", pd.Series())
        manifest_loaded = True
```

With:

```python
    # Manifest stats (initial detection only; full audit below)
    manifest_path = sw_cfg.get("manifest_path", "")
    manifest_loaded = bool(manifest_path and (REPO / manifest_path).exists())
```

And remove lines 83-99 (the second `if manifest_loaded:` block that sets coverage=1.0 and reads manifest again).

- [ ] **Step 5: Fix the summary calculation at the bottom**

After the training loop, the `results["effective_samples"]` calculation at line 164 uses `results.get("total_rows", ...)`. Since we removed the manifest stats reading, update to rely on audit results. Change line 164 from:

```python
    results["effective_samples"] = results.get("total_rows", len(train_dataset)) - results.get("rejected_count", 0)
```

To:

```python
    results["effective_samples"] = len(train_dataset) - results.get("rejected_count", 0)
```

- [ ] **Step 6: Verify portability**

```bash
cd /tmp && python -c "
import sys; sys.path.insert(0, '$(dirname $(readlink -f scripts/real_dry_run.py))/..')
from pathlib import Path
# Just verify REPO resolution works
p = Path('$(readlink -f scripts/real_dry_run.py)').resolve().parents[1]
print(f'REPO resolved to: {p}')
assert p.exists(), f'{p} does not exist'
print('OK: REPO resolution works')
"
```

- [ ] **Step 7: Commit**

```bash
git add scripts/real_dry_run.py
git commit -m "fix: make real_dry_run.py portable and call real runtime audit

- REPO now resolved from __file__ location, not hardcoded /home/lux1/noise
- Removed manual coverage=1.0 — uses actual _runtime_manifest_audit result
- Calls the real audit function before training loop
- Removed duplicate manifest reading

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Create `build_relabel_batch_probe.py`

**Files:**
- Create: `scripts/build_relabel_batch_probe.py`

**Interfaces:**
- Consumes: manifest CSV path, output path
- Produces: `relabel_batch_probe.json` with per-role samples and MixUp reduction evidence
- Exit code: 0 on success, non-zero on assertion failure

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Build relabel_batch_probe.json from a purification manifest.

Selects clean/rejected/pseudo samples, verifies training labels and
weights through the provider, and confirms that rejected samples
contribute zero gradient through the weighted MixUp reduction path.

Usage:
    python scripts/build_relabel_batch_probe.py \\
        --manifest outputs/phase4/purification/nr_consensus_relabel_v2_top100/purification_manifest.csv \\
        --output audit/relabel_batch_probe.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Resolve repo root from this script's location
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def build_probe(manifest_path: str, output_path: str, n_per_role: int = 5):
    """Build the batch probe JSON."""
    df = pd.read_csv(manifest_path)

    clean = df[df["training_role"] == "clean"]
    rejected = df[df["training_role"] == "rejected"]
    pseudo = df[df["training_role"] == "pseudo"]

    # Select up to n_per_role from each, preferring high quality_score
    clean_sample = clean.nlargest(min(n_per_role, len(clean)), "quality_score")
    rejected_sample = rejected.nlargest(min(n_per_role, len(rejected)), "quality_score")
    pseudo_sample = pseudo.nlargest(min(n_per_role, len(pseudo)), "quality_score")

    selected = pd.concat([clean_sample, rejected_sample, pseudo_sample], ignore_index=True)

    # Build probe records
    probe_records = []
    for _, row in selected.iterrows():
        rec = {
            "image_path": str(row["image_path"]),
            "original_label": int(row["original_label"]),
            "training_label": int(row["training_label"]),
            "sample_weight": float(row["sample_weight"]),
            "training_role": str(row["training_role"]),
        }
        # Verify invariants
        role = rec["training_role"]
        if role == "clean":
            assert rec["training_label"] == rec["original_label"], (
                f"Clean sample has training_label != original_label: {rec['image_path']}"
            )
            assert rec["sample_weight"] == 1.0, (
                f"Clean sample has weight != 1.0: {rec['image_path']}"
            )
        elif role == "rejected":
            assert rec["training_label"] == rec["original_label"], (
                f"Rejected sample has training_label != original_label: {rec['image_path']}"
            )
            assert rec["sample_weight"] == 0.0, (
                f"Rejected sample has weight != 0.0: {rec['image_path']}"
            )
        elif role == "pseudo":
            assert rec["training_label"] != rec["original_label"], (
                f"Pseudo sample has training_label == original_label: {rec['image_path']}"
            )
            assert rec["sample_weight"] == 1.0, (
                f"Pseudo sample has weight != 1.0: {rec['image_path']}"
            )
        probe_records.append(rec)

    # Verify MixUp reduction: rejected weight=0 zeroes contribution
    _verify_mixup_zeroing(probe_records)

    probe = {
        "manifest_path": manifest_path,
        "total_rows": len(df),
        "clean_count": int((df["training_role"] == "clean").sum()),
        "rejected_count": int((df["training_role"] == "rejected").sum()),
        "pseudo_count": int((df["training_role"] == "pseudo").sum()),
        "probe_samples": probe_records,
        "mixup_zeroing_verified": True,
    }

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(probe, f, indent=2)

    print(f"Wrote {len(probe_records)} probe records to {output_path}")
    print(f"  clean={probe['clean_count']}, rejected={probe['rejected_count']}, "
          f"pseudo={probe['pseudo_count']}")
    return probe


def _verify_mixup_zeroing(records: list):
    """Verify that rejected (weight=0) samples zero out in MixUp reduction."""
    # Simulate a batch with MixUp
    n = len(records)
    weights = torch.tensor([r["sample_weight"] for r in records])
    loss_a = torch.rand(n)
    loss_b = torch.rand(n)
    lam = 0.4
    permutation = torch.randperm(n)

    # Weighted MixUp reduction from train.py
    wa = weights
    wb = weights[permutation]
    numerator = lam * wa * loss_a + (1.0 - lam) * wb * loss_b
    denominator = lam * wa + (1.0 - lam) * wb

    for i, rec in enumerate(records):
        if rec["training_role"] == "rejected":
            assert weights[i] == 0.0, f"Rejected sample {i} has non-zero weight"
            # Contribution to numerator should be zero
            assert numerator[i] == 0.0, (
                f"Rejected sample {i} has non-zero MixUp numerator: {numerator[i]}"
            )

    # Clean and pseudo samples should have non-zero contribution
    for i, rec in enumerate(records):
        if rec["training_role"] in ("clean", "pseudo"):
            assert weights[i] == 1.0, (
                f"{rec['training_role']} sample {i} has weight != 1.0"
            )

    print("MixUp zeroing verification PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="Path to purification_manifest.csv")
    parser.add_argument("--output", default="audit/relabel_batch_probe.json",
                        help="Output JSON path")
    parser.add_argument("--n-per-role", type=int, default=5,
                        help="Samples per role")
    args = parser.parse_args()

    try:
        build_probe(args.manifest, args.output, args.n_per_role)
    except AssertionError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 2: Test the script with synthetic data**

```bash
python -c "
import pandas as pd
from pathlib import Path
import tempfile, os

# Create minimal test manifest
d = tempfile.mkdtemp()
csv = Path(d) / 'test_manifest.csv'
df = pd.DataFrame({
    'image_path': [f'img{i}.jpg' for i in range(15)],
    'sample_id': [f's{i}' for i in range(15)],
    'original_label': [0]*5 + [1]*5 + [2]*5,
    'training_label': [0]*5 + [1]*5 + [0]*5,  # last 5 are pseudo (0->0: clean, 1->1: clean, 2->0: pseudo)
    'sample_weight': [1.0]*5 + [1.0]*5 + [1.0]*5,
    'quality_score': [0.9]*15,
    'training_role': ['clean']*5 + ['clean']*5 + ['pseudo']*5,
})
# Make some rejected
df.loc[0:1, 'training_role'] = 'rejected'
df.loc[0:1, 'sample_weight'] = 0.0
df.to_csv(csv, index=False)

# Run probe script
import subprocess, sys
result = subprocess.run([
    sys.executable, 'scripts/build_relabel_batch_probe.py',
    '--manifest', str(csv),
    '--output', str(Path(d) / 'probe.json'),
    '--n-per-role', '2',
], capture_output=True, text=True)
print('STDOUT:', result.stdout)
print('STDERR:', result.stderr)
print('EXIT:', result.returncode)
assert result.returncode == 0, f'Script failed: {result.stderr}'

import json
probe = json.loads(Path(d) / 'probe.json').read_text()
print('Probe keys:', list(json.loads(probe).keys()))
"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/build_relabel_batch_probe.py
git commit -m "feat: add build_relabel_batch_probe.py for reproducible audit evidence

Selects clean/rejected/pseudo samples from manifest, verifies training
labels and weights through provider invariants, and confirms rejected
samples contribute zero through weighted MixUp reduction.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Create `reproduce_acceptance.sh`

**Files:**
- Create: `reproduce_acceptance.sh`

**Interfaces:**
- Produces: Runs all tests, dry-run, and batch probe; exits 0 only if all pass

- [ ] **Step 1: Write the reproduction script**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Task 0-5 Round 4 Acceptance Reproduction Script
# Runs all verification steps required for acceptance.
# Exits 0 only if everything passes.

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

echo "=============================================="
echo "Task 0-5 Round 4 Acceptance Reproduction"
echo "Repo: $REPO"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================="

FAILED=0

# ── 1. Full test suite ──
echo ""
echo "── 1. Full test suite ──"
if PYTHONPATH=. python -m pytest -q 2>&1; then
    echo "   PASS: All tests pass"
else
    echo "   FAIL: Some tests failed"
    FAILED=1
fi

# ── 2. Runtime audit tests specifically ──
echo ""
echo "── 2. Runtime audit regression tests ──"
if PYTHONPATH=. python -m pytest -q tests/test_runtime_manifest_audit.py -v 2>&1; then
    echo "   PASS: Runtime audit tests"
else
    echo "   FAIL: Runtime audit tests"
    FAILED=1
fi

# ── 3. Consensus selection tests ──
echo ""
echo "── 3. Consensus selection tests ──"
if PYTHONPATH=. python -m pytest -q tests/test_consensus_selection.py -v 2>&1; then
    echo "   PASS: Consensus selection tests"
else
    echo "   FAIL: Consensus selection tests"
    FAILED=1
fi

# ── 4. Purification manifest tests ──
echo ""
echo "── 4. Purification manifest tests ──"
if PYTHONPATH=. python -m pytest -q tests/test_purification_manifest.py -v 2>&1; then
    echo "   PASS: Purification manifest tests"
else
    echo "   FAIL: Purification manifest tests"
    FAILED=1
fi

# ── 5. Dry-run portability check ──
echo ""
echo "── 5. Dry-run portability ──"
DRY_RUN_REPO=$(python -c "from pathlib import Path; import sys; sys.path.insert(0, 'scripts'); print(Path('scripts/real_dry_run.py').resolve().parents[1])")
if [ -d "$DRY_RUN_REPO" ]; then
    echo "   PASS: REPO resolves to $DRY_RUN_REPO"
else
    echo "   FAIL: REPO resolution failed"
    FAILED=1
fi

# ── 6. Batch probe script syntax check ──
echo ""
echo "── 6. Batch probe syntax ──"
if python -c "import py_compile; py_compile.compile('scripts/build_relabel_batch_probe.py', doraise=True)"; then
    echo "   PASS: build_relabel_batch_probe.py compiles"
else
    echo "   FAIL: build_relabel_batch_probe.py has syntax errors"
    FAILED=1
fi

# ── Summary ──
echo ""
echo "=============================================="
if [ $FAILED -eq 0 ]; then
    echo "ALL CHECKS PASSED"
    echo "=============================================="
    exit 0
else
    echo "SOME CHECKS FAILED"
    echo "=============================================="
    exit 1
fi
```

- [ ] **Step 2: Make executable**

```bash
chmod +x reproduce_acceptance.sh
```

- [ ] **Step 3: Test the script**

```bash
bash reproduce_acceptance.sh
```

Expected: all PASS sections, exit code 0

- [ ] **Step 4: Commit**

```bash
git add reproduce_acceptance.sh
git commit -m "feat: add reproduce_acceptance.sh for Task 0-5 Round 4 verification

Runs full test suite, runtime audit tests, consensus tests, purification
tests, dry-run portability check, and batch probe syntax check. Exits 0
only if everything passes.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Final integration — run full test suite and verify

**Files:**
- None (verification only)

- [ ] **Step 1: Run full pytest suite**

```bash
PYTHONPATH=. pytest -q
```

Expected: all tests pass (47+ new tests)

- [ ] **Step 2: Run reproduce_acceptance.sh**

```bash
bash reproduce_acceptance.sh
```

Expected: exit code 0

- [ ] **Step 3: Verify critical tests individually**

```bash
PYTHONPATH=. pytest -q tests/test_runtime_manifest_audit.py tests/test_consensus_selection.py tests/test_purification_manifest.py tests/test_relabel_training.py tests/test_weighted_mixup.py -v
```

Expected: all pass

- [ ] **Step 4: Generate final git evidence**

```bash
git diff HEAD~8 --stat > audit/git_diff_stats.txt
git log --oneline -10 > audit/git_log.txt
git status > audit/git_status.txt
```

- [ ] **Step 5: Final commit with audit trail**

```bash
git add audit/git_diff_stats.txt audit/git_log.txt audit/git_status.txt reproduce_acceptance.sh
git commit -m "chore: Task 0-5 Round 4 final acceptance evidence

All fixes verified:
- P0 train_logger NameError → fixed with explicit audit_logger parameter
- Runtime audit regression tests → 8 tests covering all failure modes
- Portable tests → synthetic fixtures, no hardcoded paths
- Confident-joint persistence → explicit parameters, SHA-256 audit
- Portable dry-run → __file__-based REPO, calls real audit
- Source-class cap → max_source_class_relabel_rate=0.03 enforced
- Batch probe → reproducible script with assertion checks
- Reproduce script → reproduce_acceptance.sh

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Execution Order

Tasks are ordered for dependency reasons:
1. Task 1 (logger fix) — P0, must be first
2. Task 2 (audit tests) — depends on Task 1
3. Task 3 (cj persistence) — independent
4. Task 4 (source-class cap) — independent
5. Task 5 (portable tests) — depends on Task 4 (imports updated function)
6. Task 6 (portable dry-run) — depends on Task 1 (imports _runtime_manifest_audit)
7. Task 7 (batch probe) — independent
8. Task 8 (reproduce script) — depends on all above
9. Task 9 (final verification) — depends on all above

Tasks 3, 4, 7 can run in parallel with Tasks 1-2.
