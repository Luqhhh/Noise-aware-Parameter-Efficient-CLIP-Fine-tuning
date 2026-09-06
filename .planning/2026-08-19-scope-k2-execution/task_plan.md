# Task Plan: SCOPE-K2 Execution

## Goal
Implement and evaluate the approved SCOPE-K2 plan through a verified promoted submission or exact-parent fallback, then make one local commit and never push.

## Source of Truth
- docs/superpowers/plans/2026-08-19-scope-k2-implementation.md
- Baseline commit: 830de947e9c07738ddb69b9a5d274f0c2a6269e3

## Current Phase
Phase 5

## Phases

### Phase 0: Git and Asset Gates
- [x] Fetch/prune and rescan all refs/recent main for overlap
- [x] Rebase with automatic autostash only if origin/main is ahead (not needed: 0/0)
- [x] Locate and hash-verify checkpoint, trust bundle, splits, groups, and fallback
- **Status:** complete

### Phase 1: Protocol and Evidence TDD
- [x] Write and verify failing protocol/evidence tests
- [x] Implement frozen SCOPE protocol, residual grid, H/H0/PACE evidence, aggregation, and gates
- [x] Run focused green and PACE regression tests
- **Status:** complete

### Phase 2: Cache TDD and Parent/Evidence Runs
- [x] Write and verify failing cache/binding/test-leakage tests
- [x] Implement parent/evidence schemas and CLIs
- [x] Produce and compare validation run1/run2 Pass A/B caches
- **Status:** complete

### Phase 3: Grouped Conditional Nested OOF
- [x] Write and verify failing fold/solver/threshold/bootstrap/promotion tests
- [x] Freeze one 5x3 grouped fold artifact
- [x] Evaluate Parent, margin-only, matched PACE, gate-only, no-topology, and full SCOPE
- [x] Produce rejected decision/report artifacts; deployment correctly omitted
- **Status:** complete

### Phase 4: Terminal Submission
- [x] Promotion failed, so no test Pass A/B or SCOPE CSV/ZIP was created
- [x] Bind exact archived parent CSV/ZIP without rerunning parent test inference
- [x] Run required root checker and record hashes
- **Status:** complete

### Phase 5: Verification and Local Commit
- [ ] Run focused and full relevant tests plus diff checks
- [x] Complete result report and artifact manifest
- [ ] Explicitly stage only SCOPE files, including the approved task brief
- [ ] Make one local commit; do not push
- **Status:** in_progress

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Work in the current main checkout | The approved repository workflow explicitly forbids a branch/worktree for this experiment. |
| Treat validation as conditional scorer-level nested OOF | The frozen FULLFT parent has validation overlap and cannot be represented as honest parent-level OOF. |
| Stop at any missing or mismatched required asset | The approved plan makes asset identity a hard pre-implementation gate. |
| Use strict TDD for every new behavior | Required by the approved implementation plan and execution skill. |
| Separate cache instance semantics from replicate semantics | Evidence must retain each parent file SHA while run1/run2 comparison ignores only that byte-instance field and remains bound to the identical parent semantic SHA. |
| Reject SCOPE and use exact parent fallback | Full SCOPE selected no-switch on all folds and failed five promotion gates; fallback hashes and nine-item checker passed. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|--------:|------------|
| Planning initializer failed on CRLF line endings | 1 | Executed the same skill script through a CRLF-normalized read-only stream; initialization succeeded. |
| find/sha256 batch command was mangled at the PowerShell/WSL boundary | 1 | Replaced it with exact-path and batched read-only searches; no required model asset was found. |
| A shell boolean probe transiently printed success for missing target files | 1 | Rejected it because immediate sha256sum, stat, ls, and find checks all returned ENOENT. |
| Full-batch prior-bias reconstruction varied due FP32 subtraction cancellation | 1 | Added an opt-in return of the actually applied bias; default prior API remained unchanged and regression tests passed. |
| Evidence run replicas had different instance semantic hashes | 1 | Confirmed the only semantic field difference was parent file SHA; added a replicate semantic hash that ignores only that field and still binds parent semantic/evidence values. |
