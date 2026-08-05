# F1 R2 reweighted multiscale 0.20/0.60/0.20 (2026-08-05)

## Objective

The 128/144/160 weighting `0.25/0.50/0.25` reached a new platform best of
`68.59855008611368%` (17,127 / 24,967), adding 13 correct predictions over
equal weights. This segment continues that verified direction by raising the
144 scale to 0.60 and symmetrically reducing the 128 and 160 scales to 0.20.
No model, checkpoint, temperature, flip weight, local weight, or prior setting
changes.

## Exact submission command

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer_reweighted_multiscale_submission \
  --triple-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_multiscale_128_144_160_fused_logits_ep3_l040_f050_t15.pt \
  --pair-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_multiscale_144_160_fused_logits_ep3_l040_f050_t15.pt \
  --single-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_crop160_fused_logits_ep3_l040_f050_t15.pt \
  --scales 128,144,160 --weights 0.20,0.60,0.20 \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_multiscale_128_144_160_w020_060_020_fp32_ep3_l040_f050 \
  --strength 0.85 --acknowledge-balanced-test-prior
```

## Verification

- Inference mode: `attention_reweighted_multiscale_flip:crops=128-144-160:weights=0.2-0.6-0.2:balanced_prior=0.85`
- Predictions: 24,967; corrupt images: 0
- Changes versus the 68.598550% `0.25/0.50/0.25` best: 104 (0.42%)
- Changes versus equal 128/144/160: 289
- Changes versus equal 144/160: 469
- Reconstructed probabilities are non-negative; maximum normalization error is below `3e-7`
- Aegis submission audit: PASS (24,967 rows, 500 classes)
- Repository submission checker: all checks passed
- ZIP contents: root-level `pred_results.csv` only
- Full test suite: 268 passed, 8 warnings

## Artifact hashes

- Checkpoint SHA-256: `67efab2bf954139b59df074ccf00c0113cbc6ff96163d6e8d66ffbe553b910a4`
- CSV SHA-256: `87a6bd69f48b4a3011cca070991e25eab0d8e7d1b48400043da72b596b40c5d8`
- ZIP SHA-256: `cd65da1caabb24f501959d648d971d5457746d2dff88b1b8f7155e21ec1556a6`
- Manifest SHA-256: `ca7c51456599f74b620c8ea4bb204f3751ae93d05d91ede52d9bb39b3b4ea2dd`

## Status

Platform score: **68.58252893819842%** (17,123 / 24,967 correct). This is
**0.016021 percentage points**, or exactly four correct predictions, below the
`0.25/0.50/0.25` best and is not promoted. The result shows that raising the
144 scale to 0.60 overshoots the useful range.
