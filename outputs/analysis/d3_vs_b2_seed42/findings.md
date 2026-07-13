# Disagreement Analysis Findings

*Generated from: reference=D3_STRICT, candidates=B2_GCE07, B3_PROTO_STATIC*
*Seed: 42 only. Multi-seed validation pending.*

## 1. Protocol Audit

| Check | B2_GCE07 | B3_PROTO_STATIC |
|-------|----------|-----------------|
| paired_valid | ✅ True | ✅ True |
| causal_claim_allowed | ✅ True | ✅ True |
| sample_classification | identical_effective_samples | identical_effective_samples |
| max_visual_abs_diff | 0.0 | 0.0 |
| unexpected_differences | 0 | 0 |

*Both candidates are strictly paired with D3_STRICT. Visual encoders are byte-identical.*

## 2. Platform Scores

| Experiment | Platform Score | vs D3 |
|------------|---------------|-------|
| D3_STRICT | ~57.34% | — |
| B2_GCE07 | **58.9578%** | **+1.62pp** |
| B3_PROTO_STATIC | **58.0526%** | **+0.71pp** |

*Both noise-robust methods achieve positive platform gains despite negative local raw deltas.*

## 3. Known Issue: Feature-Bank vs Reeval Discrepancy

The feature-bank encoding produces a small (1-3 sample) discrepancy vs the official `reeval_best.json`:

| Model | reeval_best.json | Feature-bank raw | Diff |
|-------|-----------------|-----------------|------|
| D3 | 70.6572% (7289) | 70.6282% (7286) | 3 samples |
| B2 | 69.5909% (7179) | 69.5812% (7178) | 1 sample |
| B3 | ~70.1919% (~7241) | 70.2113% (7243) | ~2 samples |

**Root cause confirmed:** Fast-path `F.linear` vs full model `forward_features` produces **zero prediction mismatches** on all 10,316 samples. The discrepancy comes from the feature bank encoding (`encode_frozen_clip_features`) producing slightly different features than on-the-fly encoding (`model.encode_image`). This is under investigation.

**Impact:** The trusted delta for B2 vs D3 is only 3 samples — the same magnitude as the feature-bank encoding noise. Until the encoding path is unified, **trusted subset results are reported as supporting evidence but cannot be the sole model selection criterion**.

## 4. Raw Noisy-Label Validation Accuracy

*Measured via feature-bank fast-path. See Section 3 for reeval comparison.*

| Model | Correct | Total | Micro Accuracy | Macro | Median | Bottom 10% |
|-------|---------|-------|---------------|-------|--------|------------|
| **D3_STRICT** | 7286 | 10316 | 0.7063 | 0.7058 | 0.7619 | 0.3103 |
| **B2_GCE07** | 7178 | 10316 | 0.6958 | 0.6953 | 0.7500 | 0.2837 |
| **B3_PROTO_STATIC** | 7243 | 10316 | 0.7021 | 0.7016 | 0.7500 | 0.3206 |

| Delta | Micro | Macro | Median | Bottom 10% |
|-------|-------|-------|--------|------------|
| B2 - D3 | **-1.05pp** | -1.05pp | -1.19pp | -2.66pp |
| B3 - D3 | **-0.42pp** | -0.42pp | -1.19pp | +1.03pp |

*Both B2 and B3 show negative local raw deltas vs D3. Under a raw-noisy-label-only evaluation, both would appear to regress.*

## 5. Trusted Validation Accuracy

Trusted subset: V1 model-agnostic rules (kNN agreement ≥0.60, prototype supports label, prototype margin ≥0.02, CLIP flip cosine ≥0.90, no cross-class conflict).

| Metric | Value |
|--------|-------|
| Trusted samples | 2,073 / 10,316 |
| Coverage | 20.09% |
| Represented classes | 336 / 500 |
| Missing classes | 164 |

| Model | Correct | Trusted Total | Micro Accuracy |
|-------|---------|--------------|---------------|
| **D3_STRICT** | 2068 | 2073 | 99.76% |
| **B2_GCE07** | 2071 | 2073 | **99.90%** |
| **B3_PROTO_STATIC** | 2068 | 2073 | 99.76% |

| Delta | Micro | Samples |
|-------|-------|---------|
| B2 - D3 | **+0.14pp** | +3 |
| B3 - D3 | 0.00pp | 0 |

**⚠️ Caveat:** The trusted subset accuracy is near ceiling (99.76-99.90%), and B2's advantage is only 3 samples — identical in magnitude to the feature-bank encoding discrepancy (Section 3). Additionally, 164 classes have zero trusted samples. These results provide **directional support** but are not yet conclusive.

## 6. Rejected Subset Diagnostic

*"Rejected" = samples not meeting trusted V1 rules. These are lower-consistency samples — they may include noisy labels, hard ambiguous cases, or both. Do not conflate "rejected" with "confirmed noisy."*

| Model | Accuracy | Rejected Samples |
|-------|----------|-----------------|
| **D3_STRICT** | 63.30% | 8243 |
| **B2_GCE07** | 61.96% | 8243 |
| **B3_PROTO_STATIC** | 62.78% | 8243 |

| Delta | Micro |
|-------|-------|
| B2 - D3 | **-1.34pp** |
| B3 - D3 | -0.52pp |

*Both methods lose accuracy on lower-consistency samples. B2's loss is larger, which is consistent with GCE reducing overfitting to unreliable labels.*

## 7. Four-Group Composition (B2 vs D3)

| Group | Count | Percentage |
|-------|-------|------------|
| both_correct | 6841 | 66.31% |
| D3_only_correct | 445 | 4.31% |
| B2_only_correct | 337 | 3.27% |
| both_wrong | 2693 | 26.11% |

**D3_only - B2_only = 108**

## 8. Key Metric Comparison: D3_only_correct vs both_correct

| Metric | D3_only (n=445) | both_correct (n=6841) | Delta |
|--------|-----------------|----------------------|-------|
| knn_label_agreement | 0.1935 | 0.5363 | **-0.3429** |
| prototype_margin | 0.0080 | 0.0204 | **-0.0125** |
| clip_flip_cosine | 0.9760 | 0.9821 | -0.0061 |
| prototype_supports_noisy_label | 0.1685 | 0.7579 | **-0.5894** |

*D3-only samples have significantly lower label consistency (4.5× lower prototype label support, 2.8× lower kNN agreement). This suggests they are more likely to contain noisy labels or hard ambiguous samples, though individual samples may also be genuinely clean but visually difficult.*

## 9. B2 Predictions in D3_only_correct Region

| Metric | B2 Prediction | Noisy Label | Delta |
|--------|--------------|-------------|-------|
| kNN support | 0.1535 | 0.1935 | -0.0400 |
| prototype similarity | 0.8362 | 0.8228 | **+0.0134** |

*B2's alternative predictions in the D3_only region have higher prototype similarity than the original noisy labels. kNN support does not favor B2's predictions in this region.*

## 10. Multi-Experiment Summary

| Experiment | Method | Platform Δ | Raw Δ | Trusted Δ | Rejected Δ |
|------------|--------|-----------|-------|-----------|------------|
| B2_GCE07 | Generalized Cross Entropy (q=0.7) | **+1.62pp** | -1.05pp | **+0.14pp** (3 samples) | -1.34pp |
| B3_PROTO_STATIC | Prototype-Confidence Sample Weighting | **+0.71pp** | -0.47pp | 0.00pp | -0.52pp |

### Observed Pattern

In seed=42, across B2 and B3, both noise-robust methods show a consistent pattern:
1. **Platform-positive**: both beat D3 on the platform test set
2. **Local-negative on raw noisy labels**: both lose to D3 on raw validation accuracy
3. **No degradation on trusted samples**: B2 slightly gains (+3 samples), B3 is flat
4. **Losses concentrated on lower-consistency samples**: the rejected subset absorbs the performance drop

## 11. Conclusion

**Observed (seed=42, B2 & B3): raw noisy-label validation accuracy shows a reversed model ranking relative to the platform clean test set.**

This is evidence consistent with the hypothesis that noisy-label validation underestimates models that avoid overfitting to label noise. However, the following limitations prevent a stronger claim:

- Only a single random seed (42) — multi-seed validation required
- Only two noise-robust methods tested (GCE + prototype weighting)
- Trusted subset covers only 20% of samples and 336/500 classes
- Trusted accuracy is near ceiling; B2's advantage is 3 samples — same magnitude as feature-bank encoding noise
- The "rejected" subset is defined by consistency rules, not ground-truth label verification; some rejected samples may be genuinely clean but hard

**Next steps:**
1. Unify feature bank encoding with on-the-fly encoding to eliminate the 1-3 sample discrepancy
2. Extend to seeds 43 and 44 for multi-seed confirmation
3. Evaluate additional noise-robust methods (label smoothing, mixup, etc.) to test pattern generality
4. If multi-seed confirms, upgrade conclusion to: "GCE consistently trades noisy-label agreement for improved clean-test generalization"

**Current recommendation:** B2_GCE07 remains the strongest candidate (platform +1.62pp). Trusted subset provides directional support but should not be used as a standalone model selection criterion until the encoding discrepancy is resolved and multi-seed results are available.
