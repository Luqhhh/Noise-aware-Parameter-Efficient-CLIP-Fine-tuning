# REMATCH750 NPU throughput tuning (2026-09-22)

Status: in progress. User explicitly requested performance tuning and the best configuration.

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
