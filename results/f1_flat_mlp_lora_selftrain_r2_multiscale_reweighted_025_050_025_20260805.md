# F1 R2 reweighted multiscale 0.25/0.50/0.25 (2026-08-05)

## Objective

The equal 128/144/160 fusion reached a new platform best of
`68.5464813553891%` (17,114 / 24,967), improving by 21 correct predictions over
the 144/160 pair. This segment retains all three beneficial scales while
increasing the 144 scale from one third to one half. The declared scale weights
are 128/144/160 = `0.25/0.50/0.25`.

## Scale decomposition cache

The historical crop160 prediction was first reproduced exactly while retaining
its pre-prior fused logits:

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /tmp/noise_r2_crop160_cache_20260805 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_crop --local-crop-size 160 \
  --local-top-k 5 --local-weight 0.4 --local-temperature 1.5 \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128 \
  --dump-logits outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_crop160_fused_logits_ep3_l040_f050_t15.pt
```

The reproduced CSV SHA-256 is
`36a92d0caa0ac9e3e9648816eafcf817c6aa053e3038647a690ddbb74edbfe0f`,
identical to the historical crop160 platform submission. The retained dump
SHA-256 is
`23585c7517c731ef70c76aa9abfacca5ff5865376372ad53d2f626bf58558595`.

For single-scale fused probabilities `P128`, `P144`, and `P160`, the equal
nested fusions give exact identities:

```text
P160 = single160
P144 = 2 * pair(144,160) - P160
P128 = 3 * triple(128,144,160) - 2 * pair(144,160)
```

All recovered probabilities were non-negative. Their maximum normalization
error was below `3e-7`.

## Exact submission command

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer_reweighted_multiscale_submission \
  --triple-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_multiscale_128_144_160_fused_logits_ep3_l040_f050_t15.pt \
  --pair-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_multiscale_144_160_fused_logits_ep3_l040_f050_t15.pt \
  --single-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_crop160_fused_logits_ep3_l040_f050_t15.pt \
  --scales 128,144,160 --weights 0.25,0.50,0.25 \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_multiscale_128_144_160_w025_050_025_fp32_ep3_l040_f050 \
  --strength 0.85 --acknowledge-balanced-test-prior
```

## Verification

- Inference mode: `attention_reweighted_multiscale_flip:crops=128-144-160:weights=0.25-0.5-0.25:balanced_prior=0.85`
- Predictions: 24,967; corrupt images: 0
- Changes versus the 68.546481% equal-weight best: 187 (0.75%)
- Aegis submission audit: PASS (24,967 rows, 500 classes)
- Repository submission checker: all checks passed
- ZIP contents: root-level `pred_results.csv` only
- Full test suite: 268 passed, 8 warnings

## Artifact hashes

- Checkpoint SHA-256: `67efab2bf954139b59df074ccf00c0113cbc6ff96163d6e8d66ffbe553b910a4`
- CSV SHA-256: `b47dd399383e662008eae413386648d53736f33d6ce419d454c6623bb636037f`
- ZIP SHA-256: `38b072f0127e89650cb8c71901c60a063e68b27d94f8d1b796d3a6808d11e95a`
- Manifest SHA-256: `8282227431736382c38ee10ec343946b39f0c1af249476c0cb493bc74699b210`

## Status

Platform score: **68.59855008611368%** (17,127 / 24,967 correct). This improves
on the equal-weight 128/144/160 platform best by **0.052069 percentage points**,
or exactly 13 additional correct predictions, and is promoted as the new
audited best. The prior equal-weight package remains recoverable.
