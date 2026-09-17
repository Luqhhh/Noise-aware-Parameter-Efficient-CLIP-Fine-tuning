# PRELIM75 v4 execution record — 2026-09-17

## Status

- Plan: `PRELIM75_V4_20260916`
- Reviewed parent commit: `74b0ab8f75c1f31293cf288fbe235a725d855021`
- Implementation commit: `971272f4d733611dadd17de32763e991997a690b`
- Implemented: yes
- Trusted-support gate: passed
- C0/C1 training: complete, fixed epoch 9 outputs retained
- Submission packages: both generated and independently revalidated
- Platform feedback: pending for both candidates
- Automatic upload/push/C2: none

No score is inferred from the overlapping local diagnostic. The selected v3 S0 fallback remains 68.2982% until real platform feedback is supplied.

## Exact execution

Configuration: `configs/prelim75_v4.yaml`

The fixed queue was started with:

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -u scripts/run_prelim75_v4_queue.py --config configs/prelim75_v4.yaml --execute
```

C0 completed normally. The interactive process was then externally terminated during the first C1 attempt at sample epoch 7, optimizer step 901 (28,832 rows, beta 0.0697674). No checkpoint was created and the partial result was not eligible for delivery. Its log and state are preserved under `outputs/prelim75_v4_20260916/C1_interrupted_epoch7_step901_20260917/` and `interruption_report.json`.

C1 was restarted independently from the registered S0 parent with the frozen source snapshot and the same fixed recipe:

```bash
PYTHONPATH=outputs/prelim75_v4_20260916/training_source \
  /usr/bin/python3 -u -m aegis_clip.cli.train_prelim75_v4 \
  --config configs/prelim75_v4.yaml --action train --candidate C1
```

The restart did not resume partial weights, modify the recipe, or add a candidate. C1 evaluation and delivery used the same entry point with `--action evaluate` and `--action deliver`. Completed formal training remained exactly C0/C1 × 3 epochs; the failed attempt contributed 901 disclosed extra optimizer steps. Measured compute-action time including that partial attempt was approximately 13,156.6 seconds (3.65 hours), below the eight-hour ceiling.

## Trusted-support result

| Item | Result |
|---|---:|
| Official train rows | 103,218 |
| Positive-weight rows | 76,954 |
| Rows after clean probability ≥ 0.90 | 58,584 |
| Rows after exact original one-hot target | 55,337 |
| Eligible rows | 54,227 |
| Eligible unique groups | 54,131 |
| Eligible classes | 497 |
| Conflicting groups | 1,032 |
| Eligible original-weight mass | 71.7267% |
| Gate | passed (required 512 groups / 100 classes) |

The mask used only the original official training assets. Teacher probabilities, recovery weights, and the v3 recovery mask were not read. Mask SHA-256: `32a3d75eea361e8d8fe863bb9f918647b7dcbcee3f91897512b981efc740a456`; original-supervision SHA-256: `7fc6dcf0efb87e0a402114a7ff7ef9421e7f1c8fedcc06db10ba6d6485c606c4`.

The isolated worst-case C1 smoke passed at effective batch 32 without accumulation or OOM fallback. It created no optimizer and consumed no formal RNG state. C0 and restarted C1 had the same first effective batch, beta 0, loss components, and permitted gradient norms within the fixed tolerance; batch identity SHA-256 was `31915ba8a0af2b8517d2eebf65dc214fdfa72b2be4506dfcccc47aa007cf6d91`.

## Candidate results

| Candidate | C0 | C1 |
|---|---:|---:|
| Completed epochs / updates | 3 / 9,678 | 3 / 9,678 |
| Final loss | 0.13095349 | 0.13511070 |
| Final global GCE | 0.09339486 | 0.09254977 |
| Final global mixed loss | 0.09339486 | 0.09953671 |
| Final local GCE | 0.12525275 | 0.12518578 |
| Final anchor | 0.01240773 | 0.01265718 |
| Raw overlap diagnostic | 83.229935% | 83.239627% |
| Clean-core overlap diagnostic | 95.744103% | 95.812303% |
| Raw delta vs S0 | +0.455606pp | +0.465298pp |
| Clean-core delta vs S0 | +0.409222pp | +0.477421pp |
| Engineering stop | no | no |
| Platform score | pending | pending |

The 10,316-row diagnostic overlaps training and was used only for the fixed obvious-regression check. Neither candidate was selected from this table, and the small C1–C0 difference is not treated as evidence of platform improvement.

Checkpoint SHA-256 values:

- C0: `48d4f4ccec8758f93b126e059790107981c2359a94963d5d0147d49f05613f8a`
- C1: `e1ff4161c926b87e64b31ac2be437e6df6492d2e8ef10dd851ee7453ed70a565`

Both strict reload checks included the visual state, shared classifier, O3, and PTA. Final inference recomputed candidate attention and did not access training boxes, the trusted mask, trust data, teacher probabilities, or a teacher checkpoint.

## Delivered artifacts

| Candidate | Repository ZIP | Desktop copy | ZIP SHA-256 | CSV SHA-256 |
|---|---|---|---|---|
| C0 | `outputs/prelim75_v4_20260916/C0/submission/submission.zip` | `submission_prelim75_v4_C0_20260917.zip` | `b85d680781407bcf90b9b8b37bc1540387c42296d4e1ba97b78a10720d7e27a0` | `a298295ee86c5d1959b77b76cdcf229a7ebda7eab8edac653e44f91987695032` |
| C1 | `outputs/prelim75_v4_20260916/C1/submission/submission.zip` | `submission_prelim75_v4_C1_20260917.zip` | `45d92bb649ace9e1edf33bda305e545637e8f27f3477303a72ee99b899d9e873` | `fb02542fe6de2aeb7d4fc0da917ae490f921a3888a2af1e8ba0b6c1966fec11a` |

Both archives contain only `pred_results.csv`, cover all 24,967 test filenames exactly once, use four-digit labels in range 0000–0499, and have byte-identical inner/outer CSV files. The existing validator exited 0 for both. Each prediction happened to cover all 500 classes; coverage was reported, not forced. Desktop copies were hash-checked against the repository archives.

## Verification and remaining decision

- New v4 tests: 10 passed.
- All preliminary-75 tests: 31 passed.
- Full `reproducibility/aegis_f1` suite: 472 passed (8 warnings).
- Root repository test directory: 425 passed (12 warnings).
- Plain repository-wide pytest discovery also entered `.claude/worktrees/baseline-improvements` and hit duplicate `tests` module names; explicit root suites above passed and this unrelated discovery collision was not hidden.

Platform accuracy, exact correct counts, and upload timestamps remain null. Apply the fixed decision only after the user reports C0 and C1 scores: retain S0 if neither exceeds 68.2982%; retain C0 if it is best; retain C1 only if it exceeds both S0 and C0. A +0.30pp C1 margin is the engineering return gate, not a significance claim. No further training, beta scan, prior, C2, upload, or push is authorized by this execution.
