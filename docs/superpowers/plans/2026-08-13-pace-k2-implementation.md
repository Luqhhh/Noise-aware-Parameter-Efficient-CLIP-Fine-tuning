# PACE-K2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved PACE-K2 pairwise antisymmetric patch-evidence reranker as an auditable single-checkpoint experiment, freeze its exact-duplicate protocol artifact, evaluate it with conditional nested OOF, and produce either a validated PACE submission or a hash-verified parent fallback.

**Architecture:** Keep the platform-best parent inference unchanged and add independent protocol, cache, evidence, cross-fit, evaluation, and submission components. Pass A freezes parent candidates and crop boxes; Pass B computes only six compact pairwise evidence scalars per image. A deterministic shared nonnegative coefficient and tagged abstention policy are cross-fitted on validation data, embedded into one composite checkpoint only after both promotion gates pass, and applied read-only at test time.

**Tech Stack:** Python 3.12, PyTorch 2.13, NumPy 2.5.1, pandas 3.0.3, scikit-learn 1.9.0, OpenAI CLIP ViT-B/32, PyYAML, pytest, atomic JSON/Torch artifacts, repository submission checker.

## Global Constraints

- Treat [the approved design](../specs/2026-08-12-pace-k2-design.md) at commit `84f4705`, SHA-256 `668eca56452b3e2dc55ae1e8a9fceea626765f518b65688a8454d697e8f64fc9`, as the source of truth.
- Work directly on `main`; never create a branch and never push. Stage only PACE files and make local commits.
- Before each execution batch, fetch `origin/main`, verify behind count is zero, and scan all refs for overlapping PACE, pairwise patch, or candidate-evidence work. With a mixed worktree use only `git pull --rebase --autostash origin main`; never run `stash pop`.
- Preserve every unrelated modified or untracked file. Do not alter default `aegis_clip.cli.infer` behavior or existing platform-best assets.
- Use TDD: write the named failing test, observe the expected failure, implement the minimum behavior, rerun focused tests, then the broader suite.
- Fail closed on missing fields, shape/dtype/order mismatch, non-finite values, hash mismatch, corrupt mismatch, test-label leakage, or invalid tagged-threshold state.
- Formal inference is CUDA FP32 with batch size 128 and the fixed parent fusion/prior/PartToken protocol.
- The one-time `group_artifact_frozen` checkpoint has no submission. Commit and stop before Pass A so the user can push and teammates can synchronize.
- After synchronization, do not pause until the experiment has a checker-validated PACE submission, a checker-validated parent fallback, or the explicit external-artifact blocker permitted by the design.
- Run project commands from `reproducibility/aegis_f1` with `PYTHONPATH=$PWD` and the locked environment.

## File Structure

### Create

- `aegis_clip/pace_protocol.py`: protocol loading, asset preflight, exact groups, formal row IDs, terminal states.
- `aegis_clip/candidate_patch_evidence.py`: classifier-space checks, antisymmetric evidence, eligibility, decisions.
- `aegis_clip/pace_cache.py`: parent/evidence schemas, validators, atomic publication.
- `aegis_clip/pace_crossfit.py`: folds, beta, tagged thresholds, nested OOF, metrics, gates.
- `aegis_clip/cli/prepare_pace_group_artifact.py`
- `aegis_clip/cli/cache_pace_parent.py`
- `aegis_clip/cli/cache_pace_evidence.py`
- `aegis_clip/cli/evaluate_pace_k2.py`
- `aegis_clip/cli/infer_pace_submission.py`
- `configs/pace_k2_r2_parttoken.yaml`
- `protocol_artifacts/pace_k2_r2_parttoken/content_groups.json`
- `protocol_artifacts/pace_k2_r2_parttoken/group_artifact_report.json`
- `tests/test_pace_protocol.py`
- `tests/test_candidate_patch_evidence.py`
- `tests/test_pace_cache.py`
- `tests/test_pace_crossfit.py`
- `tests/test_pace_submission.py`
- `results/pace_k2_r2_parttoken_20260813.md`

### Modify

- `aegis_clip/local_inference.py`: opt-in raw projected patch return; default scored/PartToken branches unchanged.
- `aegis_clip/checkpoint.py`: atomically embed validated PACE payload into a verified parent copy.
- `pyproject.toml`: register five PACE CLIs.
- Existing local inference, PartToken, checkpoint, and submission tests for regressions.
- `README.md` only after a terminal experiment result.

---

## Task 1: Freeze the exact-duplicate protocol artifact

**Files:**

- Create: `aegis_clip/pace_protocol.py`
- Create: `aegis_clip/cli/prepare_pace_group_artifact.py`
- Create: `configs/pace_k2_r2_parttoken.yaml`
- Create: `tests/test_pace_protocol.py`
- Modify: `tests/test_config.py`
- Create at formal execution: both tracked `protocol_artifacts/pace_k2_r2_parttoken` JSON files.

**Interfaces:**

- `load_pace_protocol(path: Path) -> PaceProtocol`
- `verify_protocol_assets(protocol, require_model_assets: bool) -> PreflightAudit`
- `build_exact_group_artifact(protocol, hash_workers: int) -> GroupArtifactAudit`
- `freeze_group_sha_in_config(config_path: Path, digest: str) -> None`
- The builder never imports or calls `prepare_stage`.

- [x] **Step 1: Write failing protocol tests.** Assert expected-hash mismatches, missing/extra paths, cross-split duplicate groups, malformed digests, and output paths outside the tracked protocol directory fail. Assert two independent tiny builds are byte-identical and only the one-time builder accepts `state: bootstrap_unfrozen, sha256: null`.

  ```bash
  cd reproducibility/aegis_f1
  PYTHONPATH=$PWD pytest -q tests/test_pace_protocol.py
  ```

  Expected: import failure for `aegis_clip.pace_protocol`.

- [x] **Step 2: Implement frozen protocol dataclasses and strict YAML loading.** Reject unknown keys and enforce all constants from the design. The initial config includes:

  ```yaml
  protocol_id: pace_k2_r2_parttoken_v1
  group_artifact:
    state: bootstrap_unfrozen
    sha256: null
    output_path: ../protocol_artifacts/pace_k2_r2_parttoken/content_groups.json
    expected_total_rows: 103218
    expected_unique_groups: 101980
    expected_duplicate_extras: 1238
  parent_assets:
    checkpoint_sha256: 26916fd3ec96311dcab7a637f416ad3455cf7c78087844d408a38958f168962a
    prediction_csv_sha256: c6ed3e6a7f63c49a9b821f0e09222a153d926702de6f4c42505781aa7ae89fdd
    submission_zip_sha256: 6333375eea0f0b7575b833de16daf89c897df521c9eaa3f64a71e546c5ec4dc6
  ```

- [x] **Step 3: Implement deterministic double construction.** Read existing hash-verified train/val CSV rows, canonicalize to `class/filename`, hash official bytes in a pool, restore sorted key order, use `atomic_json_dump`, compare two temporary files byte-for-byte, then validate 103,218 rows, 101,980 groups, 1,238 duplicate extras, full coverage, and zero cross-split overlap.

- [x] **Step 4: Freeze config and tracked-path state.** Atomically change only group state and digest; refuse a second different digest. Treat `git check-ignore --quiet` success as an error.

- [x] **Step 5: Run focused regressions.**

  ```bash
  cd reproducibility/aegis_f1
  PYTHONPATH=$PWD pytest -q tests/test_pace_protocol.py tests/test_prepare_stage.py tests/test_config.py tests/test_features.py
  ```

- [x] **Step 6: Run the formal freeze.**

  ```bash
  cd reproducibility/aegis_f1
  PYTHONPATH=$PWD python3 -m aegis_clip.cli.prepare_pace_group_artifact --config configs/pace_k2_r2_parttoken.yaml --hash-workers 8
  git check-ignore -q protocol_artifacts/pace_k2_r2_parttoken/content_groups.json
  ```

  Expected checker exit is 1 (not ignored), and the report contains all required counts and identical double-build hashes.

- [x] **Step 7: Commit and stop.** Stage only Task 1 code/tests/config/map/report plus this plan, inspect the cached diff, and commit `experiment: freeze PACE-K2 duplicate protocol`. Report the map SHA, checks, files, commit and origin status, then stop before Pass A.

---

## Task 2: Implement classifier-space antisymmetric patch evidence

**Files:** Create `candidate_patch_evidence.py` and its tests; modify `local_inference.py` and its tests.

**Interfaces:**

- `native_visual_forward_with_patch_features(..., return_projected_patch_values=False)`
- `validate_pace_parent_model(model, checkpoint_payload) -> ClassifierSpaceAudit`
- `pairwise_view_evidence(cls, patches, weight, candidates, tail_size=7) -> Tensor`
- `aggregate_view_evidence(per_view) -> EvidenceSummary`
- `pace_eligibility(parent, evidence) -> EligibilityAudit`
- `apply_pace_decision(parent_top2, eligible, eta, threshold) -> Tensor`

- [ ] **Step 1: Add failing tests** for exact native logits/features, optional raw patch values, one Identity adaptation, classifier/PEFT gates, common-vector invariance, bias independence, norm below `1e-12`, stable ties, canonical sign reversal, separately recomputed `1e-6` antisymmetry, and non-finite rejection.
- [ ] **Step 2: Expose an opt-in fourth return** with projected `h_vp` before normalization, without changing the default three values or adding a transformer call. Keep normalized patches unchanged for PartToken pooling.
- [ ] **Step 3: Implement evidence** in FP32 from normalized weight difference and patch-minus-base-CLS residual; stable-sort by value then patch index and compute the top/bottom seven mean midpoint.
- [ ] **Step 4: Implement six-view aggregation and eligibility** using exact view order and `0.5*(0.45,0.50,0.05)`; require conflict, at least 4/6 positives, positive total/orientations/each leave-one-scale result, and non-corrupt state.
- [ ] **Step 5: Verify.**

  ```bash
  cd reproducibility/aegis_f1
  PYTHONPATH=$PWD pytest -q tests/test_candidate_patch_evidence.py tests/test_local_inference.py tests/test_part_token_adapter.py
  ```

- [ ] **Step 6: Commit** `experiment: add PACE-K2 patch evidence`.

---

## Task 3: Define fail-closed cache contracts

**Files:** Create `pace_cache.py` and `tests/test_pace_cache.py`.

**Interfaces:**

- `validate_parent_cache(payload, protocol, split) -> int`
- `validate_evidence_cache(payload, parent, protocol, split) -> int`
- `atomic_save_pace_cache(payload, destination) -> CacheManifest`
- `load_pace_cache(path, expected_sha256=None) -> dict`
- `formal_row_binding_hash(ids, paths) -> str`

- [ ] **Step 1: Add failing mutation tests** for every design-section-20 field: IDs/path binding, candidates, N×4×500 branches, N×2×3×4 boxes, view order, dtype/finite/range, validation diagnostics, test-label exclusion, corrupt/candidate/hash mismatch, and accidental raw patch storage.
- [ ] **Step 2: Implement schemas and validators.** Validation-only keys are exactly `label`, `clean_probability`, `pseudo_label`, `correction_alpha`; derived masks are not persisted.
- [ ] **Step 3: Implement temporary-save, reload-and-validate, `os.replace`, and sorted JSON manifest publication.**
- [ ] **Step 4: Verify** with `PYTHONPATH=$PWD pytest -q tests/test_pace_cache.py tests/test_submission.py`.
- [ ] **Step 5: Commit** `experiment: define PACE-K2 cache contracts`.


---

## Task 4: Build and audit Pass A parent caches

**Files:** Create `cli/cache_pace_parent.py`; modify cache tests and `pyproject.toml`.

**Interface:** `build_pace_parent_cache(checkpoint, config_path, output, split, device, batch_size, num_workers) -> Path`; CLI `aegis-cache-pace-parent`.

- [ ] **Step 1: Add failing synthetic tests** for formal row order, stable top-2 ties, four exact branch formulas, post-prior margins, frozen boxes, corrupt propagation, and test-label exclusion.
- [ ] **Step 2: Implement a PACE-only parent callable** using exact operations from `cli.infer`: native/flipped global, three attention crops per orientation, PartToken local logits, per-view softmax before branch/final fusion, full-batch prior, stable log-softmax, top-2. Do not change default inference.
- [ ] **Step 3: Add identity gates** for expected hashes, `visual_lora_mlp_lora`, linear/Identity classifier space, batch 128, FP32, fixed PartToken, branch-score hashes, and prior report. Validation requires identical two-run hashes. Test requires audited parent CSV/ZIP/manifest hashes.
- [ ] **Step 4: Verify and smoke test.**

  ```bash
  cd reproducibility/aegis_f1
  PYTHONPATH=$PWD pytest -q tests/test_pace_cache.py tests/test_prior_alignment.py tests/test_localization.py
  PYTHONPATH=$PWD python -m aegis_clip.cli.cache_pace_parent --config configs/pace_k2_r2_parttoken.yaml --checkpoint /home/lux1/noise/reproducibility/aegis_f1/outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_PART_TOKEN_ADAPTER_CROP112_FP32/seed42/best.pt --split validation --output outputs/pace_k2/r2_parttoken_crop112/cache/validation_parent_smoke.pt --batch-size 128 --num-workers 0 --device cuda --limit 256
  ```

  The smoke cache is marked non-formal because `limit` is set.

- [ ] **Step 5: Commit** `experiment: add PACE-K2 parent cache pass`.

---

## Task 5: Build Pass B compact evidence caches

**Files:** Create `cli/cache_pace_evidence.py`; modify cache tests and `pyproject.toml`.

**Interface:** `build_pace_evidence_cache(checkpoint, parent_cache, output, split, device, batch_size, num_workers) -> Path`; CLI `aegis-cache-pace-evidence`.

- [ ] **Step 1: Add failing tests** for pixel-identical crop replay, frozen candidates/order, compact-vs-raw evidence, no raw tensor publication, exact validation diagnostics, absent test diagnostics, and whole-cache failure on any Pass A/B mismatch.
- [ ] **Step 2: Implement frozen crop replay.** Validate parent cache, reconstruct six views only from cached boxes, run the native patch hook once per view, keep parent normalized patches separate, and feed PACE base CLS/patches through one feature-map adaptation.
- [ ] **Step 3: Publish compact evidence:** N×6 float64 view evidence, aggregate float64, support count, orientation/leave-one-scale flags, eligibility prerequisites, classifier audit, candidate/path bindings and lineage hashes.
- [ ] **Step 4: Verify and smoke test.**

  ```bash
  cd reproducibility/aegis_f1
  PYTHONPATH=$PWD pytest -q tests/test_candidate_patch_evidence.py tests/test_pace_cache.py
  PYTHONPATH=$PWD python -m aegis_clip.cli.cache_pace_evidence --config configs/pace_k2_r2_parttoken.yaml --checkpoint /home/lux1/noise/reproducibility/aegis_f1/outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_PART_TOKEN_ADAPTER_CROP112_FP32/seed42/best.pt --parent-cache outputs/pace_k2/r2_parttoken_crop112/cache/validation_parent_smoke.pt --split validation --output outputs/pace_k2/r2_parttoken_crop112/cache/validation_evidence_smoke.pt --batch-size 128 --num-workers 0 --device cuda
  ```

- [ ] **Step 5: Commit** `experiment: add PACE-K2 evidence cache pass`.

---

## Task 6: Implement deterministic grouped nested cross-fit

**Files:** Create `pace_crossfit.py` and `tests/test_pace_crossfit.py`.

**Interfaces:**

- `freeze_fold_artifact(labels, groups, paths, protocol, destination) -> FoldAudit`
- `fit_shared_beta(margin, evidence, target, row_ids, solver) -> BetaFit`
- `select_oof_threshold(scores, eligible, labels, candidates, policy) -> OOFThreshold`
- `map_deployed_threshold(oof, refit_scores) -> DeployedThreshold`
- `run_conditional_nested_oof(parent, evidence, folds, method) -> MethodOOFResult`
- `cluster_bootstrap_delta(parent_correct, method_correct, groups, labels, protocol) -> Interval`

- [ ] **Step 1: Add failing tests** for duplicate-safe SGKF, exact holdout coverage, inner containment, beta boundaries/failures, fixed accumulation, all/finite/no records, single-tie all-switch pass/fail, Wilson, integer mapping/ties, score-unavailable reasons, and refit drift.
- [ ] **Step 2: Freeze authoritative folds** from canonical-path-sorted formal IDs, per-image noisy labels, exact SHA groups, outer 5 and inner 3 SGKF at seed 42. Persist before fitting and validate no leakage.
- [ ] **Step 3: Implement beta** in CPU float64 with ascending row IDs, stable sigmoid, explicit ordered derivative sum, beta=0 boundary, doubling through `2**20`, and 100 bisections or width `<=1e-12`.
- [ ] **Step 4: Implement tagged thresholds** with JSON nulls, float64 score hashes, distinct-value cuts, W/L/N and Wilson, fixed utility/ties, and integer-only finite refit mapping.
- [ ] **Step 5: Implement three methods.** Margin-only uses conflict gate; gate-only uses full eligibility with margin; PACE uses full eligibility and `m_q + beta*E`. Incomplete inner OOF closes the outer procedure.
- [ ] **Step 6: Implement metrics** through `prediction_metrics`, W/L/N, Wilson, McNemar, coverage, oracle, Brier/log-loss/ECE, and 10,000-draw PCG64 exact-group bootstrap.
- [ ] **Step 7: Verify.**

  ```bash
  cd reproducibility/aegis_f1
  PYTHONPATH=$PWD pytest -q tests/test_pace_crossfit.py tests/test_cvrg_crossfit.py tests/test_balanced_inference.py
  ```

- [ ] **Step 8: Commit** `experiment: add PACE-K2 nested cross-fit`.

---

## Task 7: Evaluate gates and materialize one composite checkpoint

**Files:** Create `cli/evaluate_pace_k2.py`; modify `checkpoint.py`, tests and `pyproject.toml`.

**Interfaces:**

- `evaluate_pace_k2(protocol, parent_cache, evidence_cache, group_map, output_dir) -> EvaluationOutcome`
- `embed_pace_payload(parent_checkpoint, pace_payload, destination) -> Path`
- CLI `aegis-evaluate-pace-k2`.

- [ ] **Step 1: Add failing gate tests** for every section-15 condition, deterministic reruns, final OOF no-switch, inner/full beta failure, finite mapping failure, zero switches, and successful all/finite deployment.
- [ ] **Step 2: Orchestrate validation** for parent, three methods and oracle; write fold predictions and complete JSON; rerun and require identical folds, beta, thresholds, predictions and report hash.
- [ ] **Step 3: Implement two gates.** Outer gate is the full conjunction. Only after it passes, run final 3-fold OOF/full refit. Final gate passes only for audited all/finite with positive refit count; unavailable gates remain null after runtime failure.
- [ ] **Step 4: Embed atomically.** Preserve every parent model tensor exactly, add one validated `pace_k2` payload, reload with `build_from_checkpoint`, and audit state identity.
- [ ] **Step 5: Verify** with `PYTHONPATH=$PWD pytest -q tests/test_pace_crossfit.py tests/test_part_token_adapter.py tests/test_checkpoint.py`.
- [ ] **Step 6: Commit** `experiment: evaluate and freeze PACE-K2 payload`.

---

## Task 8: Implement terminal-state submission and verified fallback

**Files:** Create `cli/infer_pace_submission.py` and `tests/test_pace_submission.py`; modify `pyproject.toml`.

**Interfaces:**

- `run_pace_submission(protocol, checkpoint, parent_cache, evidence_cache, output_dir) -> SubmissionOutcome`
- `restore_verified_parent_submission(protocol, source_dir, output_dir, reason) -> SubmissionOutcome`
- CLI `aegis-infer-pace-submission`.

- [ ] **Step 1: Add failing tests** for exclusive terminal states, runtime failure overriding historical gates, no-switch/map-failure fallback, exact parent hashes, incomplete-asset isolation, test-label absence, checker failure, ZIP contents, 24,967 unique names, and deterministic decisions.
- [ ] **Step 2: Implement read-only PACE decisions.** Require successful composite gates, validated test caches, CPU float64 product-then-add eta, fixed eligibility/tagged threshold, and `create_submission` with complete lineage.
- [ ] **Step 3: Implement runtime close/fallback.** Save first stage/reason, stop later work, isolate temporary assets, keep unavailable gates null, and restore only exact parent CSV/ZIP/manifest. Missing/different assets yield `external_artifact_blocker`.
- [ ] **Step 4: Run independent audit and deterministic rerun.** Set `pace_success` only after generation, nine checks, ZIP/CSV audit, and decision-hash rerun all pass.
- [ ] **Step 5: Verify** with `PYTHONPATH=$PWD pytest -q tests/test_pace_submission.py tests/test_submission.py tests/test_audit_submission.py`.
- [ ] **Step 6: Commit** `experiment: add PACE-K2 submission fallback`.

---

## Task 9: Run the formal experiment to a terminal state

**Files:** Create caches/folds/evaluation/checkpoint/submission under `outputs/pace_k2/r2_parttoken_crop112` and `results/pace_k2_r2_parttoken_20260813.md`.

- [ ] **Step 1: Re-synchronize after group freeze.** Fetch/pull with automatic autostash if needed, rescan refs, exact-match synchronized group SHA, and verify all formal assets. Missing assets follow preflight-close; never substitute.
- [ ] **Step 2: Build validation Pass A twice and Pass B once.**

  ```bash
  cd reproducibility/aegis_f1
  PYTHONPATH=$PWD python -m aegis_clip.cli.cache_pace_parent --config configs/pace_k2_r2_parttoken.yaml --checkpoint /home/lux1/noise/reproducibility/aegis_f1/outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_PART_TOKEN_ADAPTER_CROP112_FP32/seed42/best.pt --split validation --output outputs/pace_k2/r2_parttoken_crop112/cache/validation_parent_run1.pt --batch-size 128 --num-workers 4 --device cuda
  PYTHONPATH=$PWD python -m aegis_clip.cli.cache_pace_parent --config configs/pace_k2_r2_parttoken.yaml --checkpoint /home/lux1/noise/reproducibility/aegis_f1/outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_PART_TOKEN_ADAPTER_CROP112_FP32/seed42/best.pt --split validation --output outputs/pace_k2/r2_parttoken_crop112/cache/validation_parent_run2.pt --batch-size 128 --num-workers 4 --device cuda
  PYTHONPATH=$PWD python -m aegis_clip.cli.cache_pace_evidence --config configs/pace_k2_r2_parttoken.yaml --checkpoint /home/lux1/noise/reproducibility/aegis_f1/outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_PART_TOKEN_ADAPTER_CROP112_FP32/seed42/best.pt --parent-cache outputs/pace_k2/r2_parttoken_crop112/cache/validation_parent_run1.pt --split validation --output outputs/pace_k2/r2_parttoken_crop112/cache/validation_evidence.pt --batch-size 128 --num-workers 4 --device cuda
  ```

  Require exact two-run parent prediction/log-score/prior/branch hashes.

- [ ] **Step 3: Run conditional nested OOF twice.**

  ```bash
  PYTHONPATH=$PWD python -m aegis_clip.cli.evaluate_pace_k2 --config configs/pace_k2_r2_parttoken.yaml --checkpoint /home/lux1/noise/reproducibility/aegis_f1/outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_PART_TOKEN_ADAPTER_CROP112_FP32/seed42/best.pt --parent-cache outputs/pace_k2/r2_parttoken_crop112/cache/validation_parent_run1.pt --evidence-cache outputs/pace_k2/r2_parttoken_crop112/cache/validation_evidence.pt --output-dir outputs/pace_k2/r2_parttoken_crop112/evaluation/run1
  PYTHONPATH=$PWD python -m aegis_clip.cli.evaluate_pace_k2 --config configs/pace_k2_r2_parttoken.yaml --checkpoint /home/lux1/noise/reproducibility/aegis_f1/outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_PART_TOKEN_ADAPTER_CROP112_FP32/seed42/best.pt --parent-cache outputs/pace_k2/r2_parttoken_crop112/cache/validation_parent_run1.pt --evidence-cache outputs/pace_k2/r2_parttoken_crop112/cache/validation_evidence.pt --output-dir outputs/pace_k2/r2_parttoken_crop112/evaluation/run2
  ```

  Require identical fold/beta/threshold/prediction/report hashes and record parent, ablations, PACE, oracle and validation-overlap limitation.

- [ ] **Step 4: Follow the frozen gate.** If both gates pass, keep the composite and build formal test Pass A/B. If either fails, skip test evidence and restore exact parent submission. Runtime failure stops at the first failure.
- [ ] **Step 5: Produce and audit the terminal submission.**

  ```bash
  PYTHONPATH=$PWD python -m aegis_clip.cli.infer_pace_submission --config configs/pace_k2_r2_parttoken.yaml --checkpoint outputs/pace_k2/r2_parttoken_crop112/checkpoints/pace_k2_composite.pt --parent-cache outputs/pace_k2/r2_parttoken_crop112/cache/test_parent.pt --evidence-cache outputs/pace_k2/r2_parttoken_crop112/cache/test_evidence.pt --output-dir outputs/pace_k2/r2_parttoken_crop112/submission
  PYTHONPATH=$PWD python -m aegis_clip.cli.audit_submission --test-root /home/lux1/noise/test --csv outputs/pace_k2/r2_parttoken_crop112/submission/pred_results.csv --zip outputs/pace_k2/r2_parttoken_crop112/submission/submission.zip
  ```

  Closed gates use the CLI verified-parent fallback mode.

- [ ] **Step 6: Run full verification:** `PYTHONPATH=$PWD pytest -q`, `git diff --check`, and targeted `git status`.
- [ ] **Step 7: Write the measured report** with every design-section-22 metric, exact commands/config/environment/hashes, terminal outcome, submission/checker paths, and limitations.
- [ ] **Step 8: Commit and pause.** Stage only PACE/result files, commit `experiment: evaluate PACE-K2 reranking`, report ID/commands/metrics/files/commit/submission/checker/origin status, and do not push.

---

## Plan Self-Review Checklist

- [x] Every design component has an owning module, callable interface, test, and formal command.
- [x] Group freeze is separated from model execution and forces synchronization before Pass A.
- [x] Parent defaults remain unchanged; new behavior is opt-in through independent PACE CLIs.
- [x] Label boundaries, groups, row IDs, corrupt flags, candidates, and hashes are checked at every cache boundary.
- [x] Beta, tagged thresholds, integer refit mapping, ablations, statistics and both gates match the approved formulas.
- [x] Every success/closed path reaches one terminal outcome and a validated submission or explicit external blocker.
- [x] No TODO, TBD, ellipsis placeholder, unbound interface, or unspecified hyperparameter remains.
- [x] Focused tests precede the full suite, and each task ends in an intentional local-only commit.
