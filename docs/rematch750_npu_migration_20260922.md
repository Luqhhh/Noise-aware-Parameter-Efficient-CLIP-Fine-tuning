# REMATCH750 NPU migration (2026-09-22)

Status: **two-epoch migration acceptance passed**. Backend implementation was
pushed in `cb64838`; full dataset parity, cache, 20-epoch LP, two-epoch FT,
checkpoint reload and CSV/ZIP validation completed. No platform upload.

The user authorized migration to `vllm-lqh-86` (Ascend 910B2). This is an
engineering migration, not a new noise-handling experiment. Keep the existing
CUDA baseline and submission packages. Do not upload migration predictions.

## Fixed scope

- Single allocated NPU 0, ARM64, CANN 9.0.0, Python 3.11.6,
  torch 2.7.1 / torch_npu 2.7.1.post4 / torchvision 0.22.1.
- Isolated environment `/workspace/noise-npu-venv`; code/data `/workspace/noise`.
- Official OpenAI CLIP ViT-B/32 and this stage's 148,695 train / 37,444 test images.
- Rebuild decoded-content groups and require every dataset-manifest field except
  machine-specific image roots to equal the original GPU manifest. This includes
  all nine manifest file hashes and the 133,815 / 14,880 independent split.
- Rebuild FP32 center-crop feature cache on NPU; train fresh RM_LP for 20 epochs;
  train RM_FT_NPU from that NPU LP. The original eight-epoch schedule remains
  fixed, but the user explicitly shortened acceptance to the saved epoch-2
  checkpoint on 2026-09-22. Do not claim an eight-epoch FT reproduction. Preserve all recipe settings,
  including batch sizes, schedules, losses, augmentation and selector.
- Rebuilding LP and changing backend/library versions means the resulting FT is
  an NPU baseline, not a strict single-variable scientific comparison to GPU FT.
- Single best checkpoint, 224 center crop, no TTA/prior/ensemble/test adaptation.
  Generate and independently validate a 37,444-row CSV/ZIP before completion.

## Backend changes

`device.py` lazily imports torch_npu for explicit NPU requests, selects the device,
and rejects unavailable accelerators. CUDA calls preserve existing behavior;
legacy non-rematch CLI CPU fallback remains explicit. NPU AMP is used consistently
in training, validation and inference. NPU seeding enables deterministic algorithms;
checkpoint save/resume includes NPU RNG state. CUDA does not import torch_npu.

NPU uses AdamW with `foreach=False` and disables pinned DataLoader memory. These
are execution choices, not changes to optimizer hyperparameters or sampling.
Cache and inference CLI default to the configured device. The legacy `rematch run`
CUDA LP/FT/LT queue rejects NPU use; invoke explicit NPU configs instead.

GPU dependency pins remain untouched. Install the separate
`reproducibility/aegis_f1/requirements-npu.txt`, following the
[official compatibility matrix](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md).
Use `PYTHONPATH` rather than installing the CUDA-only pyproject into this environment.

## Reproduction

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTHONPATH=reproducibility/aegis_f1
/workspace/noise-npu-venv/bin/python -m pip install -r reproducibility/aegis_f1/requirements-npu.txt
AEGIS_TEST_NPU=1 /workspace/noise-npu-venv/bin/python -m pytest reproducibility/aegis_f1/tests/test_npu_hardware.py -q
/workspace/noise-npu-venv/bin/python -u scripts/run_rematch_npu_migration.py --execute
```

The original queue is bounded to prepare/verify/cache/LP/FT/infer and stops on errors or
split mismatches. Logs: `outputs/rematch750_npu/migration/`. Execution report:
`results/rematch750_npu_migration_execution.json`. Data transfer must finish first.
Existing prepared assets/runs are not overwritten or silently resumed.

## User-approved two-epoch cutoff

The user chose two-epoch acceptance after the full run had started. The finishing
controller below replaces the original supervisor while leaving the trainer
running. It waits for both epoch-2 checkpoints and binding files to be saved,
stops that trainer, audits the saved epoch-2 state, reloads and re-evaluates it,
then invokes normal inference and submission validation. The original config
continues to declare the eight-epoch schedule; actual completed epochs are
recorded separately in the execution report. No checkpoint metadata is rewritten.

```bash
# PIDs below are from this run; on replay use the new supervisor/trainer PIDs.
/workspace/noise-npu-venv/bin/python -u scripts/finish_rematch_npu_acceptance.py \
  --supervisor-pid 928002 --trainer-pid 938229
```

Same-epoch GPU reference: epoch 2 macro 68.255639%, micro 69.348121%,
recorded with its source log hash in
[reference evidence](../results/rematch750_npu_epoch2_reference.json).
Do not compare this acceptance checkpoint to GPU epoch 8 as an accuracy regression.

## Validation so far

- CUDA/CPU regression: 654 passed, two opt-in hardware cases skipped, two known historical scope-asset failures.
- Device/rematch checks after CLI updates: 23 passed, one opt-in hardware skip.
- Actual NPU tests: full FT and LoRA official CLIP + AMP + AdamW updates,
  optimizer/scheduler restore, exact NPU RNG continuation and exact restored-model
  outputs: 2 passed (37.03 s).
- CPU DataLoader after NPU initialization: 4 workers passed on the actual NPU
  (1 passed, 2 deselected, 15.82 s).
- Full 148,695 train / 37,444 test decode audit and dataset parity passed: all
  non-root manifest fields and all nine asset hashes match the GPU baseline.
  Preparation took 588.40 s; subsequent verification took 8.56 s.
- Source: upstream `cb6483877bf4faa06d6af0e1087e7ce4bb7a4bc4`. The remote
  mirror is a snapshot repository (`1a33bd1f7129198c1449c2d257e220afd16304df`),
  not an upstream clone. The [source manifest](../results/rematch750_npu_source_manifest.json)
  binds the runtime/config/runner files to the upstream commit; all 181 hashes
  were verified remotely.
- Hardware evidence: [validation record](../results/rematch750_npu_hardware_validation.json).
- Earlier synthetic batch-32 probe: 1.88 GiB peak tensor allocation; three warm
  steps 0.083–0.127 s. Not an end-to-end training speed claim.

- FP32 full-training feature cache completed in 959.8 s. Four CPU loader workers
  were close to full utilization; input processing is a suspected bottleneck.
- NPU LP: 20 epochs in 272.7 s; selected epoch 20, macro 62.6175%, micro
  63.6223%. Macro differs from GPU LP by approximately -0.019pp.
- Initial real FT intervals: roughly 0.217–0.239 s/step at batch 32, slower
  than the local GPU reference (~0.127 s/step). Synthetic warmed-step timings
  do not represent this full pipeline.

- FT epoch 2: macro **68.183684%**, micro **69.274193%**; GPU epoch 2:
  macro 68.255639%, micro 69.348121%. Macro delta **-0.071955pp**.
- Saved best and last checkpoints both contain exactly **8,364** global,
  scheduler and AdamW updates. All model tensors are finite, frozen parameters
  equal the LP parent, and NPU RNG state is saved.
- Same-epoch throughput: NPU median **0.226485 s/step**, local GPU
  **0.126565 s/step** (39 within-epoch intervals each, batch 32). NPU takes
  **1.7895×** as long per step. First audited step through epoch-2 checkpoint:
  NPU 1,994.281 s / 33.24 min; GPU 1,111.408 s / 18.52 min. These spans include
  validation and checkpoint saving, exclude initial setup. See
  [throughput evidence](../results/rematch750_npu_throughput.json).

- Checkpoint reload on NPU reproduced validation macro/micro exactly across all
  14,880 validation images.
- FP32, batch-128 test inference and remote submission validation completed in
  115.86 s, generating 37,444 predictions with zero corrupt images. Local
  validation of the downloaded package also passed. Training/validation use AMP;
  the fixed submission inference protocol uses FP32.

## Delivered artifacts

Local root: `/home/lux1/noise`; remote root: `/workspace/noise`.

- Submission: `outputs/rematch750_npu/RM_FT_NPU/seed42/submission/pred_results.csv`
  and `submission.zip` (both roots).
- Best FT checkpoint: `outputs/rematch750_npu/RM_FT_NPU/seed42/checkpoints/best.pt`
  (both roots; epoch 2, SHA256 `235d39ac2c740514cacafc2577893828f0122cab44632763b2f497b42a1dd371`).
- LP checkpoint is backed up under `outputs/rematch750_npu/RM_LP/seed42/checkpoints/best.pt`.
  Local backups match remote binding hashes. The original configs/bindings retain
  remote absolute roots; replay from the prepared remote workspace, rather than
  treating the local backup as a path-portable training environment.
- CSV SHA256: `42e7ae65de4366758c62a27c075cd516edd01763702af554848aec9614a97634`.
- ZIP SHA256: `9493d5ec7b11ed9c2a98a26fa5cb8ce53c82bfb3ceba4f8b75f36307a65701a2`.
- [Execution/audit report](../results/rematch750_npu_migration_execution.json),
  [delivery record](../results/rematch750_npu_delivery.json),
  [submission manifest](../results/rematch750_npu_submission_manifest.json).
- Shared submission registry includes `RM_FT_NPU` as ready, with no platform score.
  This is a two-epoch engineering acceptance package, not a replacement for the
  existing eight-epoch RM_FT competition baseline. Its registry commit is the
  remote snapshot; the source manifest maps runtime files back to upstream `cb64838`.

Migration correctness is accepted for this tested path. The next useful work is
profiling and throughput tuning; it was **not** performed in this segment.
Future full-training accuracy and convergence remain unmeasured on NPU.
