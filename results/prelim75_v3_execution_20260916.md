# PRELIM75 v3 execution record — 2026-09-16

## Status

- Plan: `PRELIM75_V3_20260915`
- Reviewed parent commit: `b4d6ab5af9a809886c461d91733c5b7e6210cbae`
- Implementation commits: `0bea559`, `7027f2c`
- Implemented: yes
- D0 cache and cohort: complete, gate passed
- S0/S1 training: complete, fixed epoch 3 outputs retained
- Submission packages: both generated and validated
- Platform feedback: S0 = 68.2982%; S1 = null
- Current decision: S0 is the real winner over V1 while S1 remains pending. No KD-effect claim is made.

No package was uploaded automatically and no commit was pushed.

## Exact execution

Configuration: `configs/prelim75_v3.yaml`

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -u scripts/run_prelim75_v3_queue.py --config configs/prelim75_v3.yaml --execute
```

The fixed queue completed with exit code 0 in 15202.5526 seconds (4.22 hours), within the eight-hour resource ceiling. It produced one teacher cache and exactly two three-epoch candidates. Both candidates independently loaded V1 and used sample epochs 4/5/6, 9678 optimizer updates, seed 42, effective batch 32, and no OOM fallback.

An initial fail-closed epoch-0 check exposed a representation mismatch: the existing fusion helper returns log probabilities. The comparison was corrected to the same log-probability scale without changing the fixed inference protocol or thresholds. The failed, zero-sample attempt is preserved at `outputs/prelim75_v3_20260915_failed_epoch0_fusion_20260916/`. The repaired epoch-0 check had maximum absolute error `1.9073486328125e-06` and zero argmax disagreements.

## D0 result

| Item | Result |
|---|---:|
| Official train rows | 103218 |
| Test rows admitted | 0 |
| Old zero-supervision rows | 26264 |
| Old zero-supervision unique groups | 26107 |
| Zero-only canonical representatives | 25957 |
| Rows / groups excluded by positive-supervision groups | 151 / 150 |
| Rejected by view disagreement | 10344 |
| Rejected by confidence | 5167 |
| Rejected by margin | 22 |
| Eligible before class caps | 10424 |
| Rejected by class caps | 1756 |
| Selected groups after caps | 8668 |
| Covered teacher classes | 494 |
| Feasibility gate | passed (required 512 groups / 100 classes) |

Teacher artifact hashes:

- Manifest: `93e35940e19bbc371a30e67f0c12a429ecd48e88ca784be0983d4c8eacfc86e4`
- Probabilities: `309c7c00efbdb177677fefd63b56a288ddbc350663bc0b2fc8df48a07e5435bd`
- Boxes: `9041665389e51b8da6af21b457a6d8a69eb365495d17558e9a5dedb4a7ed2ff1`

## Candidate results

| Candidate | S0 | S1 |
|---|---:|---:|
| Checkpoint SHA-256 | `04cb95ad...eec029` | `68144cf8...301140` |
| Final loss | 0.1538147 | 0.2216972 |
| Final anchor | 0.0139667 | 0.0146444 |
| Final weighted KD | 0 | 0.0567851 |
| Raw overlap diagnostic | 82.7743% | 82.4932% |
| Clean-core overlap diagnostic | 95.3349% | 95.0075% |
| Raw delta vs V1 | +0.9888pp | +0.7076pp |
| Clean-core delta vs V1 | +1.2413pp | +0.9139pp |
| Engineering stop | no | no |
| Platform score | **68.2982%** | null |

The overlap diagnostics are engineering checks only because the 10316-row set overlaps training. They were not used to select an epoch. S1's lower overlap diagnostic does not substitute for its platform result.

S0 platform deltas are +0.7050pp versus V1 (67.5932%) and -2.054666pp versus the historical prior-0.90 result (70.352866%). Exact correct count and upload time were not provided and remain null. As S0 does not contain the new KD term, its gain cannot establish that recovery distillation works; S1 feedback is required for the planned S1-versus-S0 decision.

## Delivered artifacts

| Candidate | Repository ZIP | Windows desktop ZIP | ZIP SHA-256 | CSV SHA-256 |
|---|---|---|---|---|
| S0 | `outputs/prelim75_v3_20260915/S0/submission/submission.zip` | `C:\\Users\\lqh22\\Desktop\\prelim75_v3_S0_submission.zip` | `f6925507dd51f5a25a17805a1be1322d0e37b1aafa6eb15aba782c3ff7b419db` | `54d69cce616a550587885249a90ab888ab763e1b9c3d85c85606dfb97a0c7cdf` |
| S1 | `outputs/prelim75_v3_20260915/S1/submission/submission.zip` | `C:\\Users\\lqh22\\Desktop\\prelim75_v3_S1_submission.zip` | `4d508a09b6ff5476ee36faec45e4af476d1c4a5df12989ed40acb5edf466c438` | `95f212bad389916485a70257a625f0f1d9e42279ed231f46a037a9b04a07cbf7` |

Both packages contain all 24967 test filenames exactly once, use four-digit labels, have 500 predicted classes, and have byte-identical inner/outer CSV files. The existing submission validator exited 0 for both. Final inference used one student checkpoint and did not access the teacher cache or checkpoint.

## Verification and implementation scope

- New/relevant tests: 21 passed.
- Full `reproducibility/aegis_f1` suite: 461 passed, 1 skipped.
- Root test suite: 424 passed, 1 skipped.
- Strict reload verified the visual state, shared head, O3 and PTA adapters; no teacher asset is required for final inference.
- S1 KD-only gradients reached the permitted visual projection, O3-up and PTA-up parameters.
- V2 source/plan checks were retained; no `strict=False` loading workaround was introduced.

Implementation commit `0bea559` added the v3 config, recovery module and CLIs, fixed queue, and tests. Commit `7027f2c` corrected only the epoch-0 fusion comparison scale and added output-directory ignore coverage. These commits are local on `main`; `origin/main` was not pushed.
