# Findings & Decisions

## Requirements
- Execute docs/superpowers/plans/2026-08-19-scope-k2-implementation.md exactly.
- Preserve every pre-existing dirty file.
- Do not continue PACE formal execution or platform testing.
- Only generate SCOPE test artifacts after all promotion gates pass.
- End with a checker-passing SCOPE submission or exact archived parent fallback.
- Commit locally once at the terminal checkpoint; never push.

## Known Starting State
- main and origin/main were both 830de947 at plan authoring time.
- The worktree contained 21 tracked modifications and 57 untracked path entries before the task brief was added.
- Required FULLFT_DUAL checkpoint and trust bundle were previously absent in WSL.
- Exact parent fallback CSV/ZIP/manifest were present and hash-verified.
- In-progress PACE implementation is unverified and may only be reused behind regression tests.
- 2026-08-19 execution preflight re-fetched all refs: main and origin/main remain exactly 830de947 with ahead/behind 0/0, so no pull or autostash was needed.
- A precise content scan of all 17 local/remote/stash refs found zero SCOPE-K2, Spatially Coherent, patch-set verifier, counterfactual patch, or patch masking implementations.
- The only adjacent committed match is the known PACE pairwise-evidence plan; it is an ablation dependency, not a SCOPE topology implementation.
- The task brief named the split manifest as manifest.json, but the existing hash-matching artifact is split_manifest.json; the task brief was corrected without changing the frozen hash.
- Checkpoint searches found no F1_FLAT_FULL_FT_R3MS directory or matching best.pt under /home, C:/Users/clairvoyant, or D:.
- The clean-core trust bundle filename was also absent under /home, C:/Users/clairvoyant, and D:.
- Train/val/class-map/split-manifest/group and all three parent fallback artifacts match their frozen SHA-256 values.
- Repository reports contain only obsolete /home/lux1/noise creation paths and build commands; no downloadable URI, Git LFS object, release asset, or local recovery source is recorded.
- Final target-directory checks with sha256sum, stat, ls, and find confirm that both required model files are absent. A prior boolean-only probe contradicted those strong checks and was discarded as a shell-boundary artifact.
- The user restored both assets at the frozen paths. SHA-256 now matches exactly: parent f72b0104... and trust 6868041c.... Phase 0 is unlocked.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Full parent uses eight local views; evidence uses frozen six views | Resolves the parent/evidence protocol mismatch without retuning. |
| SCOPE H uses 49 nodes and 84 fixed four-neighbor edges | Adds topology with no learned spatial parameter. |
| Matched PACE and no-topology share candidates, views, weights, gates, and folds | Isolates the topology contribution. |
| Beta and threshold are scorer-level conditional crossfit parameters | Prevents calibration leakage while acknowledging parent overlap. |
| Beta regularizer gradient is `(sum(data_gradient)+beta)/n` | This exactly differentiates the preregistered `beta^2/(2n)` term; the older dirty PACE helper added beta once per row and was not reused. |
| Outer application uses mapped refit threshold, never inner numeric gamma | Preserves the frozen inner switch fraction without allowing holdout score distribution into mapping. |
| Bootstrap strata include both outer fold and group-majority label | Matches the preregistration; whole duplicate groups are resampled together with PCG64(42). |
| SCOPE is rejected | Its evidence was not useful enough to move beta away from zero or pass the precision/Wilson threshold; strict comparison to PACE/no-topology also failed. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Existing active plan referred to PACE reassessment | Created and activated an isolated SCOPE-K2 execution plan. |
| Generic executing-plans guidance prefers worktrees | Follow the explicit approved repository rule to work directly on main. |
| PowerShell mangled git for-each-ref format and broad regex audit commands | Replaced them with git show-ref plus parameterized per-ref git grep calls; the refined 17-ref scan completed. |
| Task brief referenced a nonexistent manifest.json path | Located split_manifest.json with the exact frozen d8cf... hash and corrected only the filename in the brief. |
| Parent and trust checkpoints have no available recovery source | Stop at the approved hard gate; request exact-hash restoration before any SCOPE implementation, cache generation, or formal test. |
| Parent/evidence byte files differ across deterministic runs | `torch.save` bytes and parent file bindings are instance-specific; exact tensor/scalar/path semantics were separately hashed and matched. |
| All methods produced no-switch | Confirmed eligibility was nonempty and fit did not fail; this is the preregistered beta=0 plus threshold safety outcome, not a cache/gate omission. |

## Resources
- docs/superpowers/plans/2026-08-19-scope-k2-implementation.md
- results/fullft_dual_adapters_20260806.md
- results/f1_flat_mlp_lora_selftrain_r3_multiscale_prep_20260805.md
- reproducibility/aegis_f1/protocol_artifacts/pace_k2_r2_parttoken/content_groups.json
