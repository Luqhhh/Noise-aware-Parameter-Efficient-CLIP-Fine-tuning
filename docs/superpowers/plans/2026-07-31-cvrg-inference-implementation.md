# CVRG Inference Reliability Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a validation-trained, cross-fitted reliability gate that dynamically weights the existing four CLIP views per image, preserves the exact static M1+Flip baseline when the residual gate is zero, and permits test inference only after the preregistered W060/W050 promotion gate passes.

**Architecture:** Cache the four unchanged view streams and their low-dimensional, class-agnostic evidence; fit one shared binary logistic correctness model with deterministic nested grouped cross-fitting; evaluate out-of-fold dynamic fusion against the fixed baseline after balanced-prior alignment; freeze tensor-only gate artifacts only after promotion; and integrate the frozen gate at the existing four-view fusion point in the audited inference CLI.

**Tech Stack:** Python 3.12, PyTorch 2.13, scikit-learn 1.9 (`LogisticRegression`, `StratifiedGroupKFold`), NumPy, pandas, pytest, existing Aegis CLIP cache/runtime/evaluation/submission utilities.

## Global Constraints

- Preserve every unrelated tracked and untracked worktree change. Stage and commit only files named in the current task.
- One checkpoint per candidate. Never combine W060 and W050 predictions.
- No test labels, test-set fitting, online updates, pseudolabels, neighbors, or batch-dependent reliability features.
- View order is immutable: `original_global`, `original_local`, `flipped_global`, `flipped_local`.
- Baseline weights are immutable: `[0.30, 0.20, 0.30, 0.20]`, equivalent to local weight 0.4 and flip weight 0.5.
- Local protocol is immutable: crop 160, top 5 final-block attention patches, temperature 1.0.
- Dynamic fusion precedes `align_logits_to_prior(..., strength=1.0)`.
- All validation comparisons are OOF. The full-validation refit occurs only after the overall promotion decision passes.
- Fail closed on schema, hash, shape, finiteness, fold, metric, checkpoint, or protocol mismatch.
- Persist tensor-only model parameters; never pickle a scikit-learn estimator.
- Run CPU unit tests first. Do not require model weights or test data for the default test suite.

## File Structure

### Create

- `reproducibility/aegis_f1/aegis_clip/view_reliability.py` — protocol constants, cache validation, feature extraction, frozen gate representation, dynamic fusion, serialization.
- `reproducibility/aegis_f1/aegis_clip/cvrg_crossfit.py` — deterministic nested cross-fitting, regularization selection, OOF prediction generation, promotion decision.
- `reproducibility/aegis_f1/aegis_clip/cli/cache_cvrg_views.py` — audited validation/test four-view cache generation.
- `reproducibility/aegis_f1/aegis_clip/cli/evaluate_cvrg_gate.py` — W060/W050 OOF evaluation, artifacts, promotion, post-promotion refit.
- `reproducibility/aegis_f1/tests/test_view_reliability.py` — pure feature, validation, fusion, serialization tests.
- `reproducibility/aegis_f1/tests/test_cvrg_crossfit.py` — fold isolation, nested selection, determinism, promotion tests.
- `reproducibility/aegis_f1/tests/test_cvrg_cli.py` — cache payload and evaluator artifact contract tests using synthetic tensors.

### Modify

- `reproducibility/aegis_f1/aegis_clip/cli/infer.py` — load a promoted frozen gate and replace only the static four-view fusion call.
- `reproducibility/aegis_f1/pyproject.toml` — register the two new console entry points.
- `docs/superpowers/specs/2026-07-31-cvrg-inference-design.md` — keep the approved design synchronized only if implementation reveals a contract correction.
- Result documentation under `reproducibility/aegis_f1/reports/` — update only after real W060/W050 evaluation, never with synthetic results.

---

### Task 1: Define the frozen protocol and cache contract

**Files:**
- Create: `reproducibility/aegis_f1/aegis_clip/view_reliability.py`
- Create: `reproducibility/aegis_f1/tests/test_view_reliability.py`

- [ ] **Step 1: Write failing protocol/cache validation tests**

Add tests that establish immutable order, base weights, required validation fields, label-free test caches, unique paths, aligned first dimensions, 500 classes, finite tensors, and exact protocol fields.

```python
from aegis_clip.view_reliability import (
    BASE_VIEW_WEIGHTS,
    CVRGProtocol,
    VIEW_ORDER,
    validate_cvrg_cache,
)

def test_protocol_constants_match_preregistered_baseline() -> None:
    assert VIEW_ORDER == (
        "original_global",
        "original_local",
        "flipped_global",
        "flipped_local",
    )
    assert torch.equal(BASE_VIEW_WEIGHTS, torch.tensor([0.30, 0.20, 0.30, 0.20]))
    assert CVRGProtocol().crop_size == 160
    assert CVRGProtocol().top_k == 5
    assert CVRGProtocol().temperature == 1.0

def test_validation_cache_requires_labels_and_rejects_nonfinite_logits() -> None:
    payload = make_valid_cache(samples=3, classes=500, split="validation")
    del payload["labels"]
    with pytest.raises(ValueError, match="labels"):
        validate_cvrg_cache(payload, require_labels=True)
```

Also cover:
- `view_logits: [N,4,500]`
- `view_features: [N,4,D]` and unit-norm tolerance
- `orientation_attention: [N,2,H,P]`
- `crop_boxes: [N,2,4]`
- `paths: list[str]`
- validation-only `labels`, `clean_probability`, `pseudo_label`, `correction_alpha`
- `checkpoint_sha256`, `split_sha256`, `view_order`, `protocol`
- test payload rejects any `labels` or trust fields.

- [ ] **Step 2: Run the tests and confirm the expected failure**

Run:

```bash
cd reproducibility/aegis_f1
python3 -m pytest tests/test_view_reliability.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'aegis_clip.view_reliability'`.

- [ ] **Step 3: Implement the minimal protocol types and validator**

Use these public types and signatures:

```python
VIEW_ORDER: tuple[str, str, str, str] = (
    "original_global",
    "original_local",
    "flipped_global",
    "flipped_local",
)
BASE_VIEW_WEIGHTS = torch.tensor([0.30, 0.20, 0.30, 0.20], dtype=torch.float32)
CVRG_NUM_CLASSES = 500
CVRG_CACHE_FORMAT_VERSION = 1
CVRG_FEATURE_SCHEMA_VERSION = 1

@dataclass(frozen=True)
class CVRGProtocol:
    crop_size: int = 160
    top_k: int = 5
    temperature: float = 1.0
    local_weight: float = 0.4
    flip_weight: float = 0.5
    prior_alignment_strength: float = 1.0

def validate_cvrg_cache(
    payload: Mapping[str, Any],
    *,
    require_labels: bool,
    expected_checkpoint_sha256: str | None = None,
) -> int:
    """Validate and return sample count; raise ValueError on any mismatch."""
```

Keep validation pure and CPU-only. Do not silently cast malformed shapes or accept alternate view orders.

- [ ] **Step 4: Run the focused test and confirm it passes**

Run the same pytest command. Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit only the Task 1 files**

```bash
git add reproducibility/aegis_f1/aegis_clip/view_reliability.py reproducibility/aegis_f1/tests/test_view_reliability.py
git diff --cached --check
git commit -m "feat: define CVRG cache protocol"
```

---

### Task 2: Implement the reliability feature schema and exact dynamic fusion

**Files:**
- Modify: `reproducibility/aegis_f1/aegis_clip/view_reliability.py`
- Modify: `reproducibility/aegis_f1/tests/test_view_reliability.py`

- [ ] **Step 1: Add failing hand-computed feature tests**

Define the schema as exactly 39 columns per view row:

- 6 current-view score features: maximum probability, normalized entropy, top-1/top-2 margin, top-5 probability mass, negative log-sum-exp energy, logit L2 norm.
- 18 ordered-context fields for the six unordered view pairs: Jensen-Shannon divergence, top-1 equality, top-5 Jaccard.
- 1 four-view top-1 agreement count normalized to `[0,1]`.
- 4 feature cosines: original global/local, flipped global/local, global original/flip, local original/flip.
- 6 orientation/attention fields: normalized attention entropy, attention top-5 mass, normalized crop center x/y, border-contact flag, flip-mapped center distance.
- 4 view-type one-hot fields.

The six pair names are fixed lexicographically by view index: `01, 02, 03, 12, 13, 23`. Assert the exact name tuple and `sha256_lines(feature_names)`.

Add tests for:
- uniform logits have entropy 1.0 and margin 0.0;
- a one-hot-like pair has a hand-computed JS divergence;
- identical top-5 sets have Jaccard 1.0;
- horizontal flip maps x to `1-x`;
- view one-hot rows are correct;
- no feature name contains label, class ID, path, neighbor, pseudolabel, or raw-logit fields;
- all outputs are finite.

- [ ] **Step 2: Add failing gate/fusion tests**

Use this public contract:

```python
@dataclass(frozen=True)
class FrozenReliabilityGate:
    feature_names: tuple[str, ...]
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    coefficient: torch.Tensor
    intercept: float
    regularization_c: float
    checkpoint_sha256: str
    validation_cache_sha256: str
    feature_schema_sha256: str
    protocol: CVRGProtocol

def extract_reliability_features(
    view_logits: torch.Tensor,
    view_features: torch.Tensor,
    orientation_attention: torch.Tensor,
    crop_boxes: torch.Tensor,
    *,
    image_size: int = 224,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Return [N,4,F] float32 features and exact schema names."""

def predict_view_reliability(
    features: torch.Tensor,
    gate: FrozenReliabilityGate,
) -> torch.Tensor:
    """Return clipped [N,4] probabilities in [1e-4,1-1e-4]."""

def compute_dynamic_view_weights(
    reliability: torch.Tensor,
    *,
    base_weights: torch.Tensor = BASE_VIEW_WEIGHTS,
) -> torch.Tensor:
    """Return finite non-negative [N,4] weights whose rows sum to one."""

def fuse_dynamic_view_probabilities(
    view_logits: torch.Tensor,
    gate: FrozenReliabilityGate,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fused log scores, dynamic weights, and reliability."""
```

Required assertions:
- every weight row sums to one;
- higher reliability raises that view's weight while holding others fixed;
- invalid shapes/finiteness fail;
- a zero coefficient and zero intercept gate takes an explicit fast path through `fuse_global_local_flip_probabilities(..., local_weight=0.4, flip_weight=0.5, temperature=1.0)`;
- fast-path output is `torch.equal`, not merely close, to the existing baseline output;
- serialization round-trip preserves predictions exactly.

- [ ] **Step 3: Run the focused tests and confirm failure**

```bash
cd reproducibility/aegis_f1
python3 -m pytest tests/test_view_reliability.py -q
```

Expected: failures identify missing schema, feature, gate, fusion, and serialization functions.

- [ ] **Step 4: Implement features, frozen inference, and tensor-only serialization**

Implement:
- numerically stable `softmax`, entropy, JS, top-k, and cosine calculations in float32;
- attention normalization with denominator clamps;
- constant feature scales replaced by 1.0;
- manual standardized linear score `((x-mean)/scale) @ coefficient + intercept`;
- reliability clip `[1e-4, 1-1e-4]`;
- residual weights `softmax(log(w0) + logit(r))`;
- explicit zero-residual baseline branch;
- `frozen_gate_to_payload`, `frozen_gate_from_payload`, `atomic_torch_save`, and `load_frozen_gate` using only primitive metadata and tensors.

The loader must verify the adjacent `final_gate_manifest.json`, gate SHA-256, feature schema, checkpoint binding, protocol, and promoted status.

- [ ] **Step 5: Run tests and commit**

```bash
cd reproducibility/aegis_f1
python3 -m pytest tests/test_view_reliability.py -q
git add aegis_clip/view_reliability.py tests/test_view_reliability.py
git diff --cached --check
git commit -m "feat: add CVRG reliability features and fusion"
```

---

### Task 3: Implement deterministic nested grouped cross-fitting

**Files:**
- Create: `reproducibility/aegis_f1/aegis_clip/cvrg_crossfit.py`
- Create: `reproducibility/aegis_f1/tests/test_cvrg_crossfit.py`

- [ ] **Step 1: Write failing split, selection, and determinism tests**

Use synthetic image-level data with four view rows per image and enough examples per class/fold. Test:

- every image receives exactly one outer fold;
- no image group appears in both train and holdout for any outer or inner fold;
- all four rows of one image share a fold;
- outer folds are deterministic for a fixed seed;
- candidate C values are exactly `(0.01, 0.1, 1.0)`;
- inner selection minimizes mean view-correctness Brier score and resolves ties to the smaller C;
- each OOF prediction comes from a gate that did not train on that image;
- rerunning produces byte-identical fold IDs, selected C values, probabilities, and audit JSON;
- a one-class correctness target, missing class support, nonfinite input, or incomplete assignment fails closed.

- [ ] **Step 2: Run and confirm expected failure**

```bash
cd reproducibility/aegis_f1
python3 -m pytest tests/test_cvrg_crossfit.py -q
```

Expected: collection fails because `aegis_clip.cvrg_crossfit` does not exist.

- [ ] **Step 3: Implement the cross-fitting API**

```python
@dataclass(frozen=True)
class CVRGFitConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    c_candidates: tuple[float, ...] = (0.01, 0.1, 1.0)
    seed: int = 42
    maximum_iterations: int = 1000

@dataclass(frozen=True)
class CrossFitResult:
    oof_reliability: torch.Tensor       # [N,4]
    outer_fold_id: torch.Tensor         # [N]
    selected_c_by_outer_fold: tuple[float, ...]
    inner_brier_by_outer_fold: tuple[dict[str, float], ...]
    feature_schema_sha256: str

def make_image_folds(
    labels: torch.Tensor,
    groups: Sequence[str],
    *,
    folds: int,
    seed: int,
) -> torch.Tensor: ...

def fit_frozen_gate(
    features: torch.Tensor,             # [N,4,F]
    view_logits: torch.Tensor,           # [N,4,C]
    labels: torch.Tensor,                # [N]
    image_indices: torch.Tensor,
    *,
    c: float,
    seed: int,
    metadata: GateFitMetadata,
) -> FrozenReliabilityGate: ...

def select_regularization_c(
    features: torch.Tensor,
    view_logits: torch.Tensor,
    labels: torch.Tensor,
    groups: Sequence[str],
    image_indices: torch.Tensor,
    *,
    config: CVRGFitConfig,
    seed_offset: int,
) -> tuple[float, dict[str, float]]: ...

def cross_fit_reliability(
    features: torch.Tensor,
    view_logits: torch.Tensor,
    labels: torch.Tensor,
    groups: Sequence[str],
    *,
    config: CVRGFitConfig = CVRGFitConfig(),
) -> CrossFitResult: ...
```

Implementation details:
- split image rows with `StratifiedGroupKFold`, using noisy class label for stratification and path as group;
- flatten only the selected images to `4 * n_images` view rows after splitting;
- target is `1[argmax(view_logits[v]) == label]`;
- fit unweighted binary `LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed)`;
- standardize using training rows only and export mean/scale/coef/intercept;
- select C by mean inner-holdout Brier score; compare rounded-free float values, with explicit smaller-C tie break;
- after OOF promotion, select final C using deterministic five-fold grouped CV over the full validation set, then fit all validation images exactly once.

- [ ] **Step 4: Run tests and commit**

```bash
cd reproducibility/aegis_f1
python3 -m pytest tests/test_cvrg_crossfit.py -q
git add aegis_clip/cvrg_crossfit.py tests/test_cvrg_crossfit.py
git diff --cached --check
git commit -m "feat: add CVRG nested cross fitting"
```

---

### Task 4: Build audited four-view caches

**Files:**
- Create: `reproducibility/aegis_f1/aegis_clip/cli/cache_cvrg_views.py`
- Create: `reproducibility/aegis_f1/tests/test_cvrg_cli.py`

- [ ] **Step 1: Write failing cache payload/manifest tests**

Test pure helpers with synthetic batches, not a real CLIP model:

```python
def test_validation_payload_preserves_exact_view_order_and_trust_fields() -> None: ...
def test_test_payload_is_label_free() -> None: ...
def test_manifest_binds_checkpoint_split_protocol_and_payload_hashes() -> None: ...
def test_cache_and_online_feature_extraction_are_equal() -> None: ...
```

Assert:
- logits are stacked as `[original_global, original_local, flipped_global, flipped_local]`;
- features use the same order and are normalized;
- attention is `[original, flipped]`;
- crop boxes are integer `[N,2,4]`;
- paths are aligned and unique;
- corrupt images abort publication;
- output uses atomic replacement and refuses overwrite by default;
- `view_cache_manifest.json` contains cache SHA-256, checkpoint/config/split hashes, feature schema hash, protocol, sample count, class count, and `contains_labels`.

- [ ] **Step 2: Run and confirm failure**

```bash
cd reproducibility/aegis_f1
python3 -m pytest tests/test_cvrg_cli.py -q
```

Expected: missing `cache_cvrg_views` helpers.

- [ ] **Step 3: Implement cache generation by reusing existing forward paths**

Public CLI:

```text
aegis-cache-cvrg-views \
  --checkpoint CHECKPOINT \
  --config CONFIG \
  --split validation|test \
  --output CACHE.pt \
  --batch-size N \
  --device cuda \
  [--overwrite]
```

Implementation rules:
- reuse `build_from_checkpoint`, `select_inference_preprocess`, `OnlineImageDataset`/validation trust loading, `TestImageDataset`, `seed_worker`, and deterministic `set_seed`;
- use `forward_features_with_last_block_attention` for original and flipped globals;
- use `extract_attention_crops` and `model(..., return_features=True)` for original and flipped locals;
- save `orientation_attention` and `crop_boxes` before fusion;
- compute the feature schema once through `extract_reliability_features`;
- validation payload includes canonical singular `pseudo_label`; test payload contains no label/trust key;
- run `validate_cvrg_cache` immediately before atomic save;
- hash the finished cache, then atomically write `view_cache_manifest.json`.

- [ ] **Step 4: Run tests and commit**

```bash
cd reproducibility/aegis_f1
python3 -m pytest tests/test_cvrg_cli.py tests/test_view_reliability.py -q
git add aegis_clip/cli/cache_cvrg_views.py tests/test_cvrg_cli.py
git diff --cached --check
git commit -m "feat: add audited CVRG view cache"
```

---

### Task 5: Evaluate W060/W050 and enforce the preregistered promotion gate

**Files:**
- Create: `reproducibility/aegis_f1/aegis_clip/cli/evaluate_cvrg_gate.py`
- Modify: `reproducibility/aegis_f1/aegis_clip/cvrg_crossfit.py`
- Modify: `reproducibility/aegis_f1/tests/test_cvrg_crossfit.py`
- Modify: `reproducibility/aegis_f1/tests/test_cvrg_cli.py`

- [ ] **Step 1: Write failing metric and promotion tests**

Build synthetic cache pairs that separately exercise every decision:

- W060 raw micro delta at least +0.20 percentage points;
- W060 clean-core micro delta at least +0.20 pp;
- at least 4/5 W060 folds have non-negative raw delta;
- no W060 fold is below -0.10 pp raw;
- W050 raw and clean-core micro deltas are strictly positive;
- both checkpoints keep trusted/proxy/raw/clean-core macro deltas at least -0.05 pp;
- raw and clean-core wrong-to-correct counts exceed correct-to-wrong;
- class count is 500, all artifacts finite, hashes deterministic;
- any failure returns status `closed_no_test_inference`;
- failed promotion writes no `final_gate.pt`;
- successful promotion writes distinct W060 and W050 gates and never combines them.

- [ ] **Step 2: Run and confirm failure**

```bash
cd reproducibility/aegis_f1
python3 -m pytest tests/test_cvrg_crossfit.py tests/test_cvrg_cli.py -q
```

Expected: missing evaluation and promotion functions.

- [ ] **Step 3: Implement OOF evaluation and promotion decision**

Add:

```python
PROMOTED = "promoted_for_test_inference"
CLOSED = "closed_no_test_inference"
METRICS = (
    "raw_micro", "raw_macro", "trusted_micro", "trusted_macro",
    "proxy_micro", "proxy_macro", "clean_core_micro", "clean_core_macro",
)

def paired_change_summary(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    labels: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> dict[str, int]: ...

def cvrg_promotion_decision(
    w060_report: Mapping[str, Any],
    w050_report: Mapping[str, Any],
) -> dict[str, Any]: ...
```

Evaluation sequence per checkpoint:
1. Validate expected cache SHA-256 supplied on the command line.
2. Extract `[N,4,F]` features.
3. Generate OOF reliability with nested selection.
4. Produce OOF dynamic fused scores and exact static baseline scores.
5. Apply `align_logits_to_prior(..., strength=1.0)` once to each full validation score matrix.
6. Compute metrics with `balanced_inference.prediction_metrics`.
7. Slice the already aligned predictions by outer fold for fold deltas.
8. Save fold IDs, OOF reliability/weights/predictions, metric report, paired changes, and hashes.
9. Evaluate the overall W060/W050 decision.
10. Only when promoted, select final C with five-fold grouped CV on all validation images and fit separate full-validation W060 and W050 gates.

- [ ] **Step 4: Implement the CLI and exact artifacts**

```text
aegis-evaluate-cvrg-gate \
  --w060-validation-cache W060.pt \
  --w060-cache-sha256 HEX \
  --w050-validation-cache W050.pt \
  --w050-cache-sha256 HEX \
  --output-dir OUTPUT
```

Write:

```text
OUTPUT/
  feature_schema.json
  promotion_gate.json
  w060/
    fold_assignment.csv
    oof_predictions.pt
    oof_evaluation.json
    final_gate.pt                 # promoted only
    final_gate_manifest.json      # promoted only
  w050/
    fold_assignment.csv
    oof_predictions.pt
    oof_evaluation.json
    final_gate.pt                 # promoted only
    final_gate_manifest.json      # promoted only
```

Every JSON uses `atomic_json_dump`; every tensor artifact uses `atomic_torch_save`. The root gate includes both evaluation hashes, explicit no-test/no-external/no-model-update flags, every threshold and observed value, and the final status.

- [ ] **Step 5: Run tests and commit**

```bash
cd reproducibility/aegis_f1
python3 -m pytest tests/test_cvrg_crossfit.py tests/test_cvrg_cli.py tests/test_balanced_inference.py -q
git add aegis_clip/cvrg_crossfit.py aegis_clip/cli/evaluate_cvrg_gate.py tests/test_cvrg_crossfit.py tests/test_cvrg_cli.py
git diff --cached --check
git commit -m "feat: enforce CVRG promotion gate"
```

---

### Task 6: Integrate only promoted gates into audited inference

**Files:**
- Modify: `reproducibility/aegis_f1/aegis_clip/cli/infer.py`
- Modify: `reproducibility/aegis_f1/tests/test_view_reliability.py`
- Modify: `reproducibility/aegis_f1/tests/test_cvrg_cli.py`

- [ ] **Step 1: Write failing inference contract tests**

Extract argument validation into a pure helper so tests do not invoke a model:

```python
def validate_cvrg_inference_arguments(args: argparse.Namespace) -> None: ...
```

Test that `--view-reliability-gate` requires:
- promoted adjacent manifest and matching gate SHA-256;
- matching checkpoint SHA-256;
- `--local-view attention_crop`;
- `--tta horizontal_flip`;
- `--tta-fusion mean_probabilities`;
- crop 160, top-k 5, temperature 1.0, local weight 0.4, flip weight 0.5;
- `--prior-alignment-strength 1.0`;
- both risk acknowledgements;
- no multiprototype/adapter/other TTA mode.

Also test mismatched schema/checkpoint/promotion hashes fail before a dataset or model forward is requested.

- [ ] **Step 2: Run and confirm failure**

```bash
cd reproducibility/aegis_f1
python3 -m pytest tests/test_cvrg_cli.py tests/test_view_reliability.py -q
```

Expected: missing inference gate option and validation helper.

- [ ] **Step 3: Add the gate option and replace only the fusion point**

Add:

```python
parser.add_argument(
    "--view-reliability-gate",
    help="Promoted tensor-only CVRG final_gate.pt; requires the frozen four-view protocol",
)
```

Load and validate the gate before constructing the test loader. In the existing stacked branch:
- call `forward_features_with_last_block_attention` for both global views;
- request normalized features from both local forwards;
- preserve crop boxes returned by `extract_attention_crops`;
- stack logits/features/attention/boxes in the cache order;
- call `extract_reliability_features` and `fuse_dynamic_view_probabilities`;
- leave the no-gate branch byte-for-byte behaviorally unchanged;
- retain prior alignment after concatenation.

Accumulate only label-free diagnostics: per-view mean/min/max weights, mean reliability, fraction of images whose largest weight differs from the largest baseline weight, gate/schema/promotion hashes.

- [ ] **Step 4: Extend submission manifest auditing**

Under `extra_manifest["cvrg"]`, record:
- gate path and SHA-256;
- final gate manifest SHA-256;
- promotion gate SHA-256 and status;
- checkpoint/cache/schema hashes bound by the gate;
- view order/base weights;
- dynamic-weight diagnostics;
- `validation_supervised_frozen=True`;
- `test_data_used_for_fitting=False`;
- `model_parameters_updated=False`.

Keep `inference_mode` explicit, for example:
`attention_crop_flip:cvrg_v1:topk=5:crop=160:balanced_prior=1`.

- [ ] **Step 5: Run regression tests and commit**

```bash
cd reproducibility/aegis_f1
python3 -m pytest tests/test_view_reliability.py tests/test_cvrg_cli.py tests/test_localization.py tests/test_submission.py -q
git add aegis_clip/cli/infer.py tests/test_view_reliability.py tests/test_cvrg_cli.py
git diff --cached --check
git commit -m "feat: integrate promoted CVRG inference"
```

---

### Task 7: Register commands, verify the full suite, and run the real gate

**Files:**
- Modify: `reproducibility/aegis_f1/pyproject.toml`
- Modify only after real evaluation: relevant files under `reproducibility/aegis_f1/reports/`

- [ ] **Step 1: Register console scripts**

Add:

```toml
aegis-cache-cvrg-views = "aegis_clip.cli.cache_cvrg_views:main"
aegis-evaluate-cvrg-gate = "aegis_clip.cli.evaluate_cvrg_gate:main"
```

Do not add a dependency; scikit-learn is already pinned.

- [ ] **Step 2: Run static and unit verification**

```bash
cd reproducibility/aegis_f1
python3 -m compileall -q aegis_clip
python3 -m pytest tests/test_view_reliability.py tests/test_cvrg_crossfit.py tests/test_cvrg_cli.py -q
python3 -m pytest tests/test_localization.py tests/test_prior_alignment.py tests/test_balanced_inference.py tests/test_submission.py -q
python3 -m pytest -q
```

Expected: all commands exit 0. If the full suite has a pre-existing failure, record the exact failing test and prove the focused/new tests still pass; do not conceal or rewrite unrelated work.

- [ ] **Step 3: Run deterministic artifact verification**

Generate the same synthetic evaluation twice in separate temporary output directories and compare:

```bash
sha256sum run_a/feature_schema.json run_b/feature_schema.json
sha256sum run_a/w060/oof_predictions.pt run_b/w060/oof_predictions.pt
sha256sum run_a/w050/oof_predictions.pt run_b/w050/oof_predictions.pt
sha256sum run_a/promotion_gate.json run_b/promotion_gate.json
```

Expected: corresponding SHA-256 values match. If `torch.save` container metadata prevents byte-identical files, compare canonical tensor payload hashes stored inside the manifests and make the JSON hashes deterministic; do not weaken prediction equality.

- [ ] **Step 4: Commit command registration**

```bash
git add pyproject.toml
git diff --cached --check
git commit -m "chore: register CVRG commands"
```

- [ ] **Step 5: Cache and evaluate the real W060/W050 validation candidates**

Run the new cache command separately for W060 and W050, compute cache SHA-256 values, then run the evaluator once with both frozen hashes. Use repository-approved checkpoint/config paths discovered at execution time; do not guess them in code or edit constants to match newly produced files.

- [ ] **Step 6: Honor the gate result**

If `promotion_gate.json` says `closed_no_test_inference`:
- stop;
- do not cache or infer on test data;
- do not refit, tune thresholds, rescan C values, or weaken metrics;
- report the failed conditions and OOF deltas.

If it says `promoted_for_test_inference`:
- verify both final gate manifests and hashes;
- run W060 and W050 test inference separately with their corresponding checkpoints/gates and frozen arguments;
- validate each submission with the existing submission checker;
- never ensemble or choose between them using test-derived evidence.

- [ ] **Step 7: Update reports only with real artifacts**

Add exact commands, checkpoint/cache/gate hashes, OOF metrics, fold deltas, paired changes, promotion status, and submission audit results. Clearly distinguish W060 from W050 and state that no test labels or fitting were used.

- [ ] **Step 8: Final verification and handoff**

```bash
git status --short
git log --oneline --decorate -8
```

Confirm:
- unrelated local changes remain untouched;
- each commit contains only its task files;
- no final gate exists when promotion failed;
- every claimed metric is traceable to a hashed artifact.
