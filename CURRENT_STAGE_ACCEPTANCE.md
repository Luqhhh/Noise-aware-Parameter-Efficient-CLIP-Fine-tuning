# CURRENT STAGE ACCEPTANCE — 2026-07-12

## Modified Files

| File | Change |
|---|---|
| `experiments/baseline/train.py` | Added `--allow-overwrite` flag, `_prepare_fresh_run_artifacts()` guard, fixed CSV header logic (`log_header = not log_file.exists()`), reload best.pt before post-eval, added micro/macro consistency hard-checks, expanded `eval_results.json` with `post_eval_checkpoint`, `post_eval_checkpoint_epoch`, `post_eval_micro_accuracy`, `post_eval_macro_accuracy`, `micro_macro_gap` |
| `experiments/baseline/evaluate.py` | Flexible seed access (`train_seed` → `split_seed` → `seed` → 42), `--output-json` flag, streaming SHA-256 for checkpoint and val CSV, default output renamed to `reeval_best.json`, expanded output JSON |
| `scripts/build_submission_manifest.py` | NEW: Builds auditable `submission_manifest.json` with streaming SHA-256 hashes, label validation (`^\d{4}$`), prediction count check, ZIP-internal CSV hash match, checkpoint best_val_acc vs reeval micro consistency |
| `scripts/register_submission.py` | NEW: Registers submission from manifest into `results/submission_registry.csv`, rejects duplicate ZIP hashes |
| `tests/test_run_artifact_guard.py` | NEW: 7 tests — fresh-run refusal, `--allow-overwrite` removal, resume bypass, missing checkpoint |
| `tests/test_best_checkpoint_post_eval.py` | NEW: 4 tests — best.pt reload vs in-memory divergence, strict load, missing checkpoint |
| `tests/test_metric_consistency.py` | NEW: 7 tests — micro-macro gap = micro − macro (1e-10 precision), balanced/imbalanced/single-class/random, bottom-10% aggregation |
| `tests/test_submission_manifest.py` | NEW: 11 tests — SHA-256 streaming vs full read, ZIP CSV hash match, extra/missing file detection, label format validation, prediction count, duplicate rejection, manifest schema completeness |
| `results/ablation.csv` | Updated E0_STRICT → `pending_clean_rerun`, D3_STRICT → `valid_seed42_pending_multiseed`, F0_STRICT → `control_complete_no_gain`, F1_STRICT → `below_gain_threshold`, old F1/F1b/F2 → `invalid_stage_leakage`, all with reeval metrics |
| `README.md` | Updated test count (147), added local vs platform distinction, D3 platform submission table, reeval-based strict protocol table |
| `results/submission_registry.csv` | NEW: D3_STRICT_20260712_123554 registered with all hashes and platform 57.3397% |
| `outputs/d3_strict/seed42/checkpoints/reeval_best.json` | NEW: D3 reevaluation from best.pt |
| `outputs/f0_strict/seed42/checkpoints/reeval_best.json` | NEW: F0 reevaluation from best.pt |
| `outputs/f1_strict/seed42/checkpoints/reeval_best.json` | NEW: F1 reevaluation from best.pt |
| `outputs/d3_strict/seed42/submissions/submission_manifest.json` | NEW: D3 submission manifest |

## Test Results

```
147 passed, 8 warnings in 67.33s
```

All existing tests (111) + new tests (36) pass.

### New Test Breakdown

| Test File | Tests | Key Coverage |
|---|---|---|
| `test_run_artifact_guard.py` | 7 | fresh-run refusal, allow-overwrite, resume bypass, missing checkpoint |
| `test_best_checkpoint_post_eval.py` | 4 | best.pt reload correctness, strict load, missing file |
| `test_metric_consistency.py` | 7 | micro-macro gap identity (1e-10), edge cases |
| `test_submission_manifest.py` | 11 | SHA-256, ZIP validation, labels, duplicates, schema |

## E0/D3/F0/F1 Final Consistency Metrics

### D3_STRICT (reeval_best.json)

| Metric | Value |
|---|---|
| micro_accuracy | 0.7065723148507174 |
| macro_accuracy | 0.7060997486114502 |
| micro_macro_gap | 0.00047256623926716923 |
| Verify: micro − macro | 0.0004725662392672 ✓ |
| Verify: micro == best_val_acc | 0.7065723148507174 == 0.7065723148507174 ✓ |
| checkpoint_epoch | 49 |
| val_samples | 10,316 |
| checkpoint_sha256 | 45cbfb1eed38eed7efcfb014063082c3067a70407b0d52a152633aeed675cda3 |
| val_csv_sha256 | 70a63d5a5f358a9f1ea5613c45c0904d1a30cbab8cd8241db70363d5de417c2c |

### F0_STRICT (reeval_best.json)

| Metric | Value |
|---|---|
| micro_accuracy | 0.7063784412563009 |
| macro_accuracy | 0.7059394717216492 |
| micro_macro_gap | 0.00043896953465172306 |
| Verify: micro − macro | 0.0004389695346517 ✓ |
| checkpoint_epoch | 5 |
| checkpoint_sha256 | eb846bd50fc4697073b96c2dcd521bc66427a9013c7da4aebbebc74dedd6073c |

### F1_STRICT (reeval_best.json)

| Metric | Value |
|---|---|
| micro_accuracy | 0.7078324932144242 |
| macro_accuracy | 0.7074584364891052 |
| micro_macro_gap | 0.00037405672531898304 |
| Verify: micro − macro | 0.00037405672531898304 ✓ |
| checkpoint_epoch | 4 |
| checkpoint_sha256 | 57119016f6b59a0bc3bec0518eb6834099c244e4f56bd8278fe6e4d37f9d225f |

### E0_STRICT

**Status: Clean rerun in progress.** Legacy run archived to:
`outputs/archive/e0_strict_seed42_legacy_20260712_203451/`

## D3 − E0 Delta

**Pending** — will be computed after E0 clean rerun completes.

## D3 Local-Platform Gap

| Metric | Value |
|---|---|
| Local micro (strict validation) | 0.7065723148507174 (70.6572%) |
| Platform online accuracy | 0.573397 (57.3397%) |
| **Gap** | **0.1331753148507174 (13.3175pp)** |

## Unresolved Issues

1. **E0_STRICT clean rerun incomplete** — training started 2026-07-12 20:35 CST; estimated completion ~2-3 hours (50 epochs with early stopping patience=10)
2. **D3 − E0 paired delta** — cannot be computed until E0 completes
3. **D3 platform submission** — pred_results.csv has 24,966 predictions (1 fewer than test set's 24,967 images); likely 1 corrupted test image skipped by inference; platform accepted 24,966
4. **split_seed/train_seed** — not stored in submission manifest (manifest built from reeval_best.json which doesn't include these); registry fields are empty for D3_STRICT
5. **Multi-seed confirmation** — D3_STRICT status is `valid_seed42_pending_multiseed`; seeds 3407/2026 not yet evaluated

## Verification Checklist

- [x] Non-resume fresh run cannot write to existing run directory
- [x] train_log.csv has only one header (fresh run deletes stale CSV)
- [x] Post-training metrics computed from best.pt
- [x] best_val_acc == post_eval_micro_accuracy
- [x] micro_macro_gap == micro − macro
- [x] D3/F0/F1 reeval_best.json generated
- [ ] E0_STRICT clean rerun completed (IN PROGRESS)
- [ ] E0 no manual stop (pending completion)
- [x] E0 and D3 share same master-val (both use outputs/master_splits/seed42/val.csv)
- [x] submission_manifest contains checkpoint/CSV/ZIP SHA-256 hashes
- [x] Platform 57.3397% registered
- [x] pytest 147/147 passed
