# V2 F1+M1 Converged Known-Balanced-Prior Transport Protocol

## Status

- Experiment: `V2_F1_M1_CONVERGED_KNOWN_BALANCED_PRIOR_TRANSPORT`
- Version label: `V2_closed_v1`
- Status: **CLOSED / NO TEST INFERENCE**
- Date: 2026-07-22
- Execution boundary: CPU-only validation-cache inference; no training, test-set inference, external data, or submission packaging.

## Motivation and protected inference

V1 used 100 fixed Sinkhorn updates. It produced positive F1+M1 accuracy deltas but failed its numerical row-marginal gate on both validation caches, while A2+M1 also missed its clean-core promotion gate. V1 is permanently closed and is not reinterpreted.

V2 tests one narrower, falsifiable question: **does completing the same predeclared transport solve to a fixed numerical tolerance change enough hard assignments to satisfy the unchanged cross-checkpoint accuracy gate?** The experiment is a numerical repair, not a parameter search. Failure closes this transport direction for the fixed M1 caches.

## Immutable inputs

| Cache | Path | SHA-256 | Shape |
|---|---|---|---:|
| F1+M1 validation logits | `outputs/M1_F1_TRANSFER/seed42/f1_attention_local_global.pt` | `5f927bc9740ec5ce1725a7cfab07fbdc40f3e3dda5213ce59a419092edbf614c` | 10,316 x 500 |
| A2+M1 validation logits | `outputs/M1_A2_TRANSFER/seed42/a2_attention_local_global.pt` | `cea5f7651bf0d24828b7da8b252cb47b52288fa2cb7638a1c6d32359c0c92698` | 10,322 x 500 |

The evaluator must fail closed on any input hash, shape, alignment, uniqueness, or finiteness mismatch.

## Frozen method

- Input score: the existing M1 fused log-probability tensor, unchanged.
- Target marginal: soft uniform `N / 500` for every class.
- Temperature: `1.0`.
- Scaling: log-space Sinkhorn row and column updates.
- Minimum iterations: `100`.
- Convergence check interval: every `10` completed row/column update cycles, including iteration 100.
- Stop only when both maximum row absolute error and maximum column absolute error are at most `1e-5`.
- Maximum iterations: `2000`; reaching the maximum without convergence is a failed numerical gate.
- Hard prediction: row-wise argmax of the final allocation.
- Parameter scan: none.
- Baseline: row-wise argmax of the same fixed M1 logits.

No result-dependent change to tolerance, iteration cap, temperature, target prior, hardening rule, cache, metric, or gate is allowed.

## Frozen promotion gate

All checks must pass simultaneously:

| Checkpoint | Check | Required |
|---|---|---:|
| F1+M1 | clean-core micro delta | >= +0.20 pp |
| A2+M1 | clean-core micro delta | >= +0.10 pp |
| Both | trusted macro delta | >= -0.05 pp |
| Both | raw micro delta | >= -0.10 pp |
| Both | hard prediction-count CV relative reduction | >= 20% |
| Both | empty predicted classes | 0 |
| Both | maximum Sinkhorn row absolute error | <= 1e-5 |
| Both | maximum Sinkhorn column absolute error | <= 1e-5 |
| Both | solver convergence flag | true |

Any failed check yields `closed_no_test_inference`. Passing yields only `eligible_for_compliance_review`; it does not authorize test inference or packaging because batch-transductive balancing still needs captain/organizer confirmation.

## Reproducibility and reporting

- Run the exact fixed command once, then repeat it unchanged.
- Require byte-identical SHA-256 hashes for both evaluation files, both prediction files, and the gate file.
- Report all eight validation metric deltas, paired wrong-to-correct/correct-to-wrong counts, exact two-sided McNemar results, hard-balance diagnostics, convergence iteration and errors.
- Preserve failures and anomalies; do not select only the favorable checkpoint.

## Material passport

- Test data used: no.
- External data used: no.
- Model parameters updated: no.
- Team live worktree modified: no.
- Expected output root: `outputs/V2_F1_M1_CONVERGED_KNOWN_BALANCED_PRIOR_TRANSPORT/seed42/`.

## Results

### Strict gate outcome

- `passed`: `false`
- `decision`: `closed_no_test_inference`
- Failed checks:
  1. `a2_m1_clean_core_micro_delta_pp`: `+0.069308 pp < +0.10 pp`
  2. `a2_m1_sinkhorn_maximum_column_absolute_error`: `1.335144e-5 > 1e-5`
  3. `a2_m1_sinkhorn_converged`: `false` at the frozen 2,000-iteration cap
- No test-set inference or submission package was produced.

### Accuracy and balance

| Cache | Raw micro | Raw macro | Trusted micro | Trusted macro | Proxy micro | Proxy macro | Clean-core micro | Clean-core macro | Count-CV reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F1+M1 | +0.300503 pp | +0.308442 pp | +0.228548 pp | +0.368792 pp | +0.290644 pp | +0.405127 pp | +0.229853 pp | +0.433755 pp | 50.026% |
| A2+M1 | -0.077504 pp | -0.066203 pp | -0.003421 pp | +0.140721 pp | -0.003421 pp | +0.140721 pp | +0.069308 pp | +0.226033 pp | 51.130% |

Both candidates predicted all 500 classes. F1 hard prediction-count CV changed from `0.224218` to `0.112051`; A2 changed from `0.227356` to `0.111110`.

### Paired changes and exact tests

| Cache / scope | Changed | Wrong -> correct | Correct -> wrong | Net correct | Exact two-sided McNemar p |
|---|---:|---:|---:|---:|---:|
| F1 raw | 866 | 207 | 176 | +31 | 0.125183 |
| F1 clean core | 413 | 131 | 114 | +17 | 0.306679 |
| A2 raw | 901 | 200 | 208 | -8 | 0.728975 |
| A2 clean core | 380 | 121 | 116 | +5 | 0.795063 |

The paired evidence is not statistically persuasive. The F1 direction is positive, but selecting it alone after observing A2 would violate the preregistered cross-checkpoint protection.

### Numerical diagnostics

| Cache | Iterations | Converged | Max row error | Max column error |
|---|---:|---:|---:|---:|
| F1+M1 | 250 | yes | `8.702278e-6` | `9.536743e-6` |
| A2+M1 | 2,000 | no | `5.960464e-7` | `1.335144e-5` |

The A2 residual is a float32 column-sum precision floor for the fractional `N/500` target. It is not used to excuse the result: the independent A2 clean-core accuracy gate also fails, and its raw paired net is negative.

## Reproducibility record

The frozen command was executed unchanged three times. Hashes were explicitly captured after runs two and three and matched byte-for-byte:

| Artifact | SHA-256 |
|---|---|
| F1 evaluation | `da20b112ebae6678e23248728e0ddd8ede07e703aa993d93af6b94edc9cb1f9e` |
| F1 predictions | `4597916028324a50171078b319a9dc3abe82ddfac4422c9ec6e4ef3e922f4ddc` |
| A2 evaluation | `e6b78da70b54ea01a14ce1fd753e7cc2641d82c2bbde9b1e797dd537ed1f419b` |
| A2 predictions | `9cbeaf64462688a96b023723308a9a846112456e699fcc576e284db3e1584e7e` |
| Gate | `0a41657a8d20e43714da524ceb9b66f8b35d3ee6cc82f7740daaf1bb74630ca8` |

## Fallacy audit

| Risk | Assessment |
|---|---|
| Simpson's paradox | CAUTION: aggregate and clean-core results differ; both are reported for both caches. |
| Ecological fallacy | NOTE: class-count balance is not evidence that individual predictions are correct. |
| Berkson / selection bias | CAUTION: clean-core is trust-selected; raw metrics are retained as a safety check. |
| Collider bias | CAUTION: trust-derived masks and proxy labels share model-derived signals. |
| Base-rate neglect | PASS: the official balanced-test statement motivated the fixed prior, but accuracy gates prevent balance-only promotion. |
| Regression to the mean | N/A: no repeated noisy training measurement was averaged into a claim. |
| Survivorship bias | CAUTION: V1 and V2 failures are retained rather than hiding the A2 cross-check. |
| Multiple comparisons / look-elsewhere | PASS: no temperature, tolerance, iteration, prior, or checkpoint scan occurred. |
| Garden of forking paths | PASS: inputs, solver, metrics, and gates were committed before execution. |
| Correlation implies causation | CAUTION: validation changes cannot establish platform-score causality. |
| Reverse causality | N/A for the deterministic fixed-cache comparison. |

## Final interpretation

V2 confirms that V1's main limitation was not incomplete row-marginal convergence. The hard predictions and accuracy deltas are effectively unchanged after numerical completion: F1 remains modestly positive, while A2 remains below the promotion threshold and slightly negative on raw micro accuracy. This fixed known-balanced-prior transport direction is therefore closed for M1; pursuing it further would become result-conditioned tuning rather than a clean numerical repair.

