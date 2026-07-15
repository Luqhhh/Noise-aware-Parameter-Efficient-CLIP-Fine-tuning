# C-1: GCE q=0.7 + Linear Head EMA (decay=0.99)

**Parent**: B2_GCE07 (configs/gce_q07.yaml)
**Split**: D3_STRICT (outputs/ref/seed42), train=91,375 / val=10,316
**Training time**: 2h 43m 53s (50 epochs, 714 batches/epoch)
**GPU peak memory**: ~3,400 MiB

## Results

| Metric | best_raw (epoch 42) | best_ema (epoch 43) | B2_GCE07 | Δ raw vs GCE07 | Δ ema vs GCE07 |
|--------|---------------------|---------------------|----------|----------------|----------------|
| Micro | 0.693292 | 0.692807 | 0.695909 | -0.002617 | **-0.003102** |
| Macro | 0.692779 | 0.692278 | 0.695380 | -0.002601 | **-0.003102** |
| Bottom-10% | 0.277612 | 0.276501 | 0.283653 | -0.006041 | **-0.007152** |
| Micro-Macro Gap | +0.000513 | +0.000530 | +0.000529 | — | — |

## Parameter Distance (raw vs EMA at best)

| Metric | Value |
|--------|-------|
| Head L2 distance | 2.335953 |
| Head max_abs_diff | 0.02977180 |
| Backbone max_abs_diff | 0.00e+00 (< 1e-5 ✅) |

## EMA Behaviour

- Warmup: 5 epochs (3,570 steps), raw == EMA during warmup ✅
- Post-warmup divergence confirmed epoch 6+: L2 > 0, max_abs_diff > 0 ✅
- EMA num_updates == optimizer steps throughout (no AMP overflow skips) ✅
- Backbone frozen: max diff = 0.00e+00 ✅

## Checkpoint SHAs

| File | SHA-256 |
|------|---------|
| best_raw.pt | f08addb9b045e08bcc2c6026ba5d46af4b5644a0ad012b55afe54ed39e443bce |
| best_ema.pt | cc8266f0b6debf01b9037d6933c1a8a76b64eb22a19ccca3e87faabbb7cb7e98 |

## Submission

| File | SHA-256 |
|------|---------|
| submission.zip | 9562ddc51c9025b3aca27f5fbbd654b45e5d7d3a69f01da3e45008fbabadac31 |

- Single-view, no TTA
- 24,967 predictions, 495/500 unique classes
- All 9 submission checks passed ✅

## Conclusion

**STOP.** Head EMA (decay=0.99, warmup=5) with GCE q=0.7 did not improve over B2_GCE07 baseline. Both micro and macro declined by ~0.3pp, bottom-10% dropped by ~0.7pp. EMA consistently lagged raw throughout training (best_ema epoch 43: 0.6928 vs best_raw epoch 42: 0.6933). Head EMA does not provide benefit for this task under this configuration.

**Decision:**
- Head EMA 路线 STOP — 不测试 decay=0.999。
- C-2 PEFT 独立保留，等待 A 指定最终父 checkpoint。
