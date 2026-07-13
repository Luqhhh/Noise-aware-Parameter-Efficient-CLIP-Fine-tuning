# Experiment Pair Audit & Trusted Validation Framework

**Date:** 2026-07-13
**Status:** Approved

## Overview

Build a comprehensive analysis framework to audit paired experiments (D3_STRICT vs B2_GCE07), construct model-agnostic trusted validation subsets, and evaluate whether noisy-label validation systematically underestimates robust loss functions like GCE.

## Architecture

### New files

```
common/
├── diagnostic_metrics.py    # Per-sample metrics, chunked kNN, robust prototypes
├── pair_protocol_audit.py   # Experiment pair comparison and audit
└── trusted_subset.py        # V1 model-agnostic trusted subset rules

tools/
├── audit_experiment_pair.py          # CLI: run protocol audit
├── export_feature_bank.py            # CLI: export frozen CLIP features
├── analyze_checkpoint_disagreement.py # CLI: four-group disagreement analysis
├── build_trusted_subset.py           # CLI: build trusted manifest
├── evaluate_dual_validation.py       # CLI: raw + trusted evaluation
└── summarize_disagreement.py         # CLI: generate findings.md

tests/
├── test_diagnostic_metrics.py
├── test_chunked_knn.py
├── test_pair_protocol_audit.py
├── test_trusted_subset.py
└── test_dual_validation.py
```

### Dependency graph

```
diagnostic_metrics.py  ← pure computation, no I/O
pair_protocol_audit.py ← depends on utils, dataset (for path resolution)
trusted_subset.py      ← depends on nothing model-specific

audit_experiment_pair.py      ← pair_protocol_audit, utils
export_feature_bank.py        ← clip_utils, reuses _CacheImageDataset pattern
analyze_checkpoint_disagreement.py ← diagnostic_metrics, pair_protocol_audit
build_trusted_subset.py       ← trusted_subset
evaluate_dual_validation.py   ← model, evaluation patterns
summarize_disagreement.py     ← reads all JSON/CSV outputs
```

## Key Design Decisions

### 1. kNN: Chunked exact top-k, never full matrix

`chunked_topk_cosine(query, bank, k=20, query_chunk=256, bank_chunk=8192)`:
- For 10,316 val × ~93K train, peak memory ~32MB per chunk
- Returns indices and similarities; caller derives label metrics
- Test verifies equality with brute-force on small matrices

### 2. Feature bank fast-path

When audit confirms D3/B2 visual encoders are byte-identical:
1. Read `val_feature_bank.pt` once
2. Load classifier weight/bias from each checkpoint
3. Compute logits via `F.linear(features, weight, bias)`
4. Verify against full model on 32 samples (max diff < 1e-6)
Otherwise fall back to full model inference and record `shared_feature_fast_path: false`.

### 3. Trusted subset V1: Model-agnostic only

Rules use only:
- `knn_label_agreement >= 0.60`
- `prototype_supports_noisy_label == True`
- `prototype_margin >= 0.02`
- `clip_flip_cosine >= 0.90`
- `cross_class_duplicate_conflict == False`

Never reads D3/B2 logits, confidence, margin, or correctness.
Tests verify: deleting model columns → identical output.

### 4. Dual validation: Same trusted manifest for both models

- Single `trusted_manifest.csv` used for both D3 and B2 evaluation
- Paired bootstrap: 10,000 iterations, class-stratified
- Reports: micro, macro (present classes + all classes), median, bottom-10%

### 5. No new dependencies

Everything uses existing stack: torch, pandas, numpy, PIL, tqdm, pytest.

## Implementation Phases (Incremental)

### Phase 1: Common modules
1. `common/diagnostic_metrics.py` + tests
2. `common/pair_protocol_audit.py` + tests
3. `common/trusted_subset.py` + tests

### Phase 2: Tools
4. `tools/audit_experiment_pair.py`
5. `tools/export_feature_bank.py`
6. `tools/analyze_checkpoint_disagreement.py`
7. `tools/build_trusted_subset.py`
8. `tools/evaluate_dual_validation.py`
9. `tools/summarize_disagreement.py`

### Phase 3: Pipeline execution
10. Run protocol audit → verify paired_valid
11. Export feature banks
12. Run disagreement analysis → sample_metrics.csv, group_summary.json
13. Build trusted subset → trusted_manifest.csv
14. Run dual validation for D3 and B2
15. Generate findings.md

## Acceptance Criteria

- All existing + new tests pass (`pytest -q`)
- Protocol audit has clear paired_valid conclusion
- 10,316 validation samples in exactly one of four groups
- D3/B2 raw metrics match `reeval_best.json` exactly
- kNN uses chunked computation, no full N×M matrix
- Trusted V1 does not read any D3/B2 logits/confidence/margin
- Findings report clearly distinguishes correlation from causal evidence
