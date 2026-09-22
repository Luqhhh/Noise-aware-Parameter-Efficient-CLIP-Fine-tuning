# REMATCH750 NPU migration (2026-09-22)

Status: implementation and device checks in progress; full real-data validation pending.

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
  train RM_FT_NPU for 8 epochs from that NPU LP. Preserve all recipe settings,
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

The queue is bounded to prepare/verify/cache/LP/FT/infer and stops on errors or
split mismatches. Logs: `outputs/rematch750_npu/migration/`. Execution report:
`results/rematch750_npu_migration_execution.json`. Data transfer must finish first.
Existing prepared assets/runs are not overwritten or silently resumed.

## Validation so far

- CUDA/CPU regression: 654 passed, two opt-in hardware cases skipped, two known historical scope-asset failures.
- Device/rematch checks after CLI updates: 23 passed, one opt-in hardware skip.
- Actual NPU tests: full FT and LoRA official CLIP + AMP + AdamW updates,
  optimizer/scheduler restore, exact NPU RNG continuation and exact restored-model
  outputs: 2 passed (37.03 s).
- Earlier synthetic batch-32 probe: 1.88 GiB peak tensor allocation; three warm
  steps 0.083–0.127 s. Not an end-to-end training speed claim.

Formal accuracy, throughput, checkpoint audit and submission hashes are pending.
