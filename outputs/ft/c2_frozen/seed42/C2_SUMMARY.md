# C2: Frozen Control from W1_CE5_GCE05

**Parent**: W1_CE5_GCE05 (CE warmup + GCE q=0.5)
**Split**: outputs/ref/seed42, train=91,375 / val=10,316
**Training time**: 17m 40s (6 epochs, 714 batches/epoch, early stop)
**GPU peak memory**: ~1,700 MiB

## Results

| Metric | best (epoch 1) | parent (W1_CE5_GCE05) | Δ |
|--------|---------------|----------------------|---|
| Micro | 0.731582 | 0.731388 | **+0.000194** |
| Macro | 0.731126 | 0.730900 | **+0.000226** |
| Bottom-10% | 0.335710 | 0.336700 | **-0.000990** |
| Micro-Macro Gap | +0.000456 | +0.000488 | — |

## Epoch-0 Gate

| Metric | Value |
|--------|-------|
| Epoch-0 val acc | 0.731679 |
| Parent val acc | 0.731388 |
| Delta | 0.000291 |
| Threshold | 0.0005 |
| Result | PASSED ✅ |

## Training Summary

- Best: epoch 1 (73.16% val), early stop at epoch 6/12 (patience=5)
- Trainable params: 256,500 (linear head only, backbone frozen)
- Loss: GCE q=0.5, head_lr=1e-4, cosine schedule, warmup=2 epochs
- AMP: True, no NaN/Inf/OOM

## Checkpoint SHAs

| File | SHA-256 |
|------|---------|
| best.pt | feb03aeff430d69b0a1407ef2bb6f71a26ca48126fb2f4b84c0d8feebdc55b4b |

## Conclusion

Frozen control from W1_CE5_GCE05: best val acc 73.16% vs parent 73.14% = +0.02pp — within noise level. Just more training of the linear head adds nothing meaningful beyond the parent. This establishes the baseline for C2_LN_PROJ delta computation (paired experiment shares the same parent, split, loss, LR, schedule, seed, epochs, and early-stop rules; differs only in unfreezing ln_post + visual.proj).
