# B2 GCE07 Multi-Seed Validation

**Date:** 2026-07-14

**Commits:** `11adc87` (seed42), `1e763f7` + `7644816` (seeds 2026, 3407)

## Experiment

B2 = Generalized Cross Entropy (q=0.7) on D3_STRICT fixed split (`outputs/d3_strict/seed42`).

| Parameter | Value |
|---|---|
| Backbone | CLIP ViT-B/32, frozen |
| Head | Linear |
| Loss | GCE, q=0.7, ε=1e-7 |
| Split | D3_STRICT seed42 (fixed across all seeds) |
| Augmentation | a0 (none beyond preprocessing) |
| LR | 0.005, cosine schedule |
| Warmup | 2 epochs |
| Epochs | 50, early stop patience=10 |
| Batch size | 128, AMP enabled |

## Results

| Train Seed | Best Micro | Best Macro | Median/Bottom10 | Best Epoch | Early Stop |
|---|---|---|---|---|---|
| 42 | 69.59% | 69.54% | — | 41 | No |
| 2026 | 69.54% | 69.49% | 75.0% / 28.5% | 35 | Yes (epoch 45) |
| 3407 | 69.56% | 69.51% | 75.0% / 28.3% | 41 | No |

**Spread:** 69.54%–69.59% (range 0.05pp). GCE q=0.7 is stable across training seeds.

## Platform Performance

| Experiment | TTA | Platform Score | Δ vs D3 |
|---|---|---|---|
| B2 GCE07 seed42 | None (bare) | 58.96% | +1.62pp |
| B2 GCE07 seed42 | 2-view hflip | — | — |

Note: B2 bare platform (58.96%) was the best bare submission at the time, confirming GCE reduces local overfitting (local-platform gap 10.63pp vs D3's 13.32pp).

## Comparison: GCE vs CE

| Method | Val Micro | Platform (bare) |
|---|---|---|
| ref (CE) | 70.66% | — |
| gce_q07 | 69.59% | ~58.96% |
| base_ce | — | 57.22% |

GCE trades ~1pp local accuracy for ~1.7pp platform gain — the noise-robust loss improves generalization to the private test set.

## TTA on gce_q07

2-view horizontal-flip TTA applied to gce_q07:

| Seed | Platform (TTA) |
|---|---|
| seed42 | 59.41% |
| seed2026 | 59.49% |
| **Best** | **59.49%** |

4-view TTA (hflip + vflip) was tested on B2 but discarded: val dropped to 68.07% (−1.52pp), as vertical flip degrades fine-grained classification.

---

## B-EXP-3: CE 5-Epoch Warmup → GCE q=0.7

**Date:** 2026-07-14 | **Commit:** local (not yet pushed)

CE warmup (epochs 1–5) → GCE q=0.7 (epochs 6–50). Uses A's loss schedule infrastructure.

| Metric | B2 GCE07 (no warmup) | W1_CE5_GCE07 | Δ |
|---|---|---|---|
| Val Micro | 69.59% | **69.78%** | +0.19pp |
| Val Macro | 69.54% | **69.72%** | +0.18pp |
| Bottom-10% | — | 29.14% | — |
| Best Epoch | 41 | 38 | faster |
| Early Stop | No (50) | Yes (48) | — |

### TTA Validation

| Metric | Baseline | TTA (2-view hflip) | Δ |
|---|---|---|---|
| Micro | 69.78% | 70.14% | +0.36pp |
| Macro | 69.72% | 70.09% | +0.37pp |
| Bottom-10% | 29.14% | 29.63% | +0.49pp |
| Prediction Change | — | 5.56% | — |

### Platform Score

| Experiment | TTA | Platform | Δ vs gce_q07+TTA seed2026 |
|---|---|---|---|
| gce_q07+TTA seed2026 | 2-view hflip | 59.49% | — |
| **W1_CE5_GCE07+TTA** | 2-view hflip | **59.79%** | **+0.30pp 🥇** |

CE warmup delivers +0.19pp local and +0.30pp platform over pure GCE. **New platform high score.**

## Updated Platform Leaderboard

| Rank | Experiment | TTA | Platform |
|---|---|---|---|
| 🥇 | **W1_CE5_GCE07** | 2-view hflip | **59.79%** |
| 🥈 | gce_q07 seed2026 | 2-view hflip | 59.49% |
| 🥉 | gce_q07 seed42 | 2-view hflip | 59.41% |
| 4 | w2_ema_loss | 2-view hflip | 59.39% |
| 5 | B2 GCE07 | bare | 58.96% |
