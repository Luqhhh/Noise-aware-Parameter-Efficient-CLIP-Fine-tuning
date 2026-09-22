# REMATCH750 NPU throughput tuning (2026-09-22)

Status: completed. User selected batch 1024 after repeated measurements.
Batch-32 two-epoch correctness acceptance and validated submission are complete;
large-batch convergence remains unvalidated.

## Scope and selection

Single allocated Ascend 910B2 device 0, existing isolated environment and current-stage
LP/cache from the completed migration. No test image or test prediction is used for
performance selection. Original GPU/NPU acceptance artifacts remain intact.

“Best” means highest measured sustained training images/s within the tested range,
not a claim of globally optimal throughput or best competition accuracy. Report a
recipe-preserving batch-32 recommendation separately from the fastest larger-batch
configuration, since changing batch changes optimizer updates and convergence.

Use the real `aegis_clip.trainer.train` path, full training sampler and eight-epoch
schedule, fixed LP initialization and original losses/augmentation. Each probe stops
before validation, normally after 20 warmup + 100 measured steps. Smaller diagnostic
runs insert synchronization around sections and are not throughput candidates.
Every successful probe verifies the optimizer update count. Production acceptance
must use a regular config, checkpoint/reload checks and validated CSV/ZIP artifacts.

Investigate in order:

1. Baseline synchronized attribution: loader wait, forward/backward, gradient norms,
   clipping and optimizer.
2. Workers 4/8/16, prefetch 1/2; AdamW per-tensor/foreach/NPU fused;
   original/vector/foreach gradient norm implementation.
3. Batch 32/64/128/256/512 (extend only if throughput still improves), workers and
   CPU thread refinements near the winner. Repeat finalists with longer windows.
4. Verify numerical equivalence and checkpoint restore for the selected execution
   changes. Deliver both measured recommendations and a bounded end-to-end
   acceptance package. Do not infer accuracy improvements from throughput.

Benchmark command template:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONPATH=reproducibility/aegis_f1
/workspace/noise-npu-venv/bin/python -u scripts/benchmark_rematch_npu.py \
  --name <unique_case> --batch 32 --workers 16 --prefetch 2 \
  --optimizer fused --norm original --warmup 20 --steps 100
```

Probe configs are generated as ignored `configs/npu_probe_*.local.yaml`, logs under
`outputs/npu_tuning`, JSON results under `results/npu_tuning`. Probe monkeypatches
exist only in the benchmark process; selected settings must be integrated into the
normal trainer before final verification.

## Initial attribution

A synchronized 15-step profile after 10 warmup steps measured 0.2311 s/step:
loader wait 0.1173 s, forward 0.0191 s, backward 0.0263 s, gradient norms 0.0093 s,
clipping 0.0034 s, AdamW 0.0311 s. This is diagnostic, not an uninstrumented
performance result. A separate 100-step baseline measured 150.2 images/s.

The installed torch_npu optimizer implementation was inspected alongside the
[official fused-optimizer migration guidance](https://www.hiascend.com/doc_center/source/zh/canncommercial/63RC2/modeldevpt/ptmigr/ptmigr_0060.html).
Measured behavior and numerical/restore tests determine adoption.

## Implemented execution changes

Implementation commit: `248e741cdbfb65a9a066de0edafb2241734a3ae3`.
The new optimizer factory preserves the previous default and makes foreach AdamW
and `NpuFusedAdamW` explicit opt-ins. Fused AdamW needs persistent gradient buffers
(`zero_grad(set_to_none=False)`); the initial failed probe exposed this requirement
and remains in the raw results. Foreach gradient norms retain the nonfinite guard.
NPU pinned memory is explicit (`pin_memory_device="npu"`) and defaults off for
existing recipes. Existing CUDA configs keep their execution path.

The batch-32 config keeps LR, losses, seed, sampler, LP initialization and eight-epoch
schedule fixed. Worker count changes augmentation RNG assignment, and fused kernels
can change rounding, so this is recipe preservation, not bitwise trajectory identity.
The two-epoch acceptance covers CE warmup. Separate bounded GCE probes verify
throughput and real optimizer updates; they do not replace a full eight-epoch run.

Validation before performance acceptance:

- Local suite: **662 passed, 7 hardware skipped, 2 known historical ScopePreflightError
  failures** in `test_scope_protocol.py`, as permitted by `CLAUDE.md`.
- NPU hardware suite: **6 passed** in 50.99 s, including AdamW numerical comparison,
  exact optimizer continuation after restore, and official CLIP AMP save/resume.
- Explicit pinned/unpinned DataLoader hardware tests: **2 passed** in 9.21 s
  (one overlaps the earlier suite; seven unique hardware cases covered).
- Every successful throughput probe checks optimizer step count against warmup plus
  measured steps, as well as the trainer's finite-gradient checks.

`results/rematch750_npu_tuning_source_manifest.json` maps 181 runtime/config files to
upstream commit `248e741`; the remote snapshot commit is
`1642427f786e59c6eecf3875bf234954bf240361`. The two commit identities are intentionally
recorded separately. Early in-process probes have their earlier source mapping in
`results/npu_tuning/pre_integration_source.json`; final production probes include
runtime source hashes in each result.

## Recipe-preserving control and acceptance

Use `configs/rematch750_ft_npu_tuned.yaml`: **batch 32, workers 16, prefetch 2,
NPU pinned memory, NpuFusedAdamW, foreach norms, OMP threads 4, CPU 144–167**.
CPU IDs are specific to device 0 on `vllm-lqh-86`; recheck topology on other hosts.

```bash
cd /workspace/noise
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH=reproducibility/aegis_f1
taskset -c 144-167 /workspace/noise-npu-venv/bin/python -m aegis_clip.cli.rematch train \
  --config configs/rematch750_ft_npu_tuned.yaml
```

This is the full eight-epoch launch command, **not an eight-epoch result**. Actual
bounded acceptance used the separate `rematch750_ft_npu_tuned_acceptance.yaml`
(two epochs, same eight-epoch LR schedule):

```bash
taskset -c 144-167 /workspace/noise-npu-venv/bin/python -u \
  scripts/run_rematch_npu_tuned_acceptance.py --execute
```

The runner refuses to overwrite its completed report. Replay in a fresh workspace,
with the documented current-stage assets and LP checkpoint provisioned.

- Independent best-checkpoint evaluation: **macro 68.3530867%, micro 69.3817198%**.
  Original NPU epoch-2 macro 68.1836843%; difference **+0.1694024pp**. This single
  run supports no observed degradation, not an accuracy-improvement claim.
- Both best and last: epoch 2, global/optimizer/scheduler steps **8,364**; all weights
  finite, checked frozen visual weights unchanged, NPU RNG saved, reload metrics exact.
- Regular training log: **0.079585 s/step median**, 39 within-epoch intervals.
  Original NPU 0.226485 s/step: **2.846× speedup**. Local GPU 0.126565 s/step:
  **1.590× speedup**. These are measured systems, not a device-only comparison.
- First-step audit to epoch-2 checkpoint saved: **706.523 s** (11.78 min), versus
  original NPU 1,994.281 s and local GPU 1,111.408 s. Includes the validation/save
  inside that span; excludes startup and final re-evaluation/inference.
- Acceptance train command, including setup and final re-evaluation: **768.850 s**.
- Longer batch-32 throughput confirmations: CE **396.8 images/s** (400 measured
  steps after 20 warmup), GCE **403.8 images/s** (250 + 20).

Artifacts on both local and remote under
`outputs/npu_tuning/RM_FT_NPU_TUNED_E2/seed42/`:
`checkpoints/best.pt`, `checkpoints/best_evaluation.json`, `logs/train.log`,
`submission/pred_results.csv`, `submission/submission.zip`, validation and binding
metadata. Remote also retains `last.pt`. The 37,444-row CSV/ZIP passed the remote
CLI checker and independent local `scripts/check_submission.py` check; the shared
submission registry has the new candidate. **No platform upload or platform score.**

Machine-readable audit: `results/rematch750_npu_tuned_acceptance.json`.
Local verification log: `results/npu_tuning_checks/submission_check.txt`.


## Final user selection: batch 1024

The user explicitly selected **1024** after seeing the repeat results. Adopt
`configs/rematch750_ft_npu_throughput.yaml`: **batch 1024, workers 40, prefetch 4,
NPU pinned memory, fused AdamW, foreach norms, OMP 4, CPUs 144–191**.

| Configuration | CE images/s | Peak tensor allocation |
|---|---:|---:|
| Initial batch 32 / workers 4 | 150.2 | see raw result |
| Tuned batch 32, longer confirmation | 396.8 | 2.15 GiB |
| Batch 512 / workers 40 / pinned | 1,652.3 | 12.50 GiB |
| **Selected batch 1024 / workers 40 / pinned** | **1,831.6 / 1,832.5** | **23.52 GiB** |
| Batch 2048 / workers 40 / pinned | 1,860.7 / 1,880.0 | 45.56 GiB |

Batch 1024 averages **1,832.053 images/s** over two runs; batch 2048 averages
1,870.343 images/s, only **2.09%** more for roughly twice the tensor memory.
Tensor allocation excludes runtime/driver memory. Device 0 has 65,536 MB HBM.
This is an empirical choice within tested settings, not a global optimum claim.

```bash
cd /workspace/noise
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH=reproducibility/aegis_f1
taskset -c 144-191 /workspace/noise-npu-venv/bin/python -m aegis_clip.cli.rematch train \
  --config configs/rematch750_ft_npu_throughput.yaml
```

The command is provided for the next training run; a full large-batch run has **not**
been launched. Batch 1024 reduces updates from 4,182 to **131 per epoch**; no learning
rate scaling was introduced in the throughput probes. Its accuracy/convergence has
not been validated. The accepted CSV/ZIP belongs to the tuned **batch-32** two-epoch
run and must not be described as a batch-1024 submission.

Reproduce the selected CE confirmation with the same environment:

```bash
taskset -c 144-191 /workspace/noise-npu-venv/bin/python scripts/benchmark_rematch_npu.py \
  --production --name replay1024_ce --batch 1024 --workers 40 --prefetch 4 \
  --optimizer fused --norm foreach --threads 4 --pin-memory --warmup 20 --steps 105
```

For GCE add `--loss-phase gce` and use a fresh `--name`. The GCE probe exercises that
loss immediately for bounded timing/update checks; it is not a converged GCE model.

Selected batch-1024 GCE confirmation: **1802.7 images/s**, 105 measured steps after 20 warmup; all 125 optimizer updates completed.

There are **39 probe records: 38 passed and one initial fused-optimizer failure**
(the persistent-gradient fix is documented above). Raw step arrays, arguments and
source hashes are in `results/npu_tuning/`; compact comparison is
`results/rematch750_npu_tuning_cases.csv`. The final user choice and the separate
maximum-throughput winner are both retained in `results/npu_tuning/selection.json`.
Full summary: `results/rematch750_npu_tuning_summary.json`. Regenerate it from
collected evidence with `python3 scripts/summarize_rematch_npu_tuning.py`.
