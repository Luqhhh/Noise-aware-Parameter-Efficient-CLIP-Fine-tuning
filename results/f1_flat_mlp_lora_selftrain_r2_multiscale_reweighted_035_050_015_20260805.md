# F1 R2 reweighted multiscale 0.35/0.50/0.15 (2026-08-05)

## Objective

The current platform best uses 128/144/160 weights `0.25/0.50/0.25` and scores
`68.59855008611368%`. Raising the 144 weight to 0.60 regressed by four correct
predictions, so this segment holds the validated 144 weight at 0.50 and tests a
different single variable: transferring 0.10 probability weight from crop160
to crop128. This follows the earlier observed gain of 34 correct predictions
between weights `0.00/0.50/0.50` and `0.25/0.50/0.25`.

## Exact submission command

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer_reweighted_multiscale_submission \
  --triple-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_multiscale_128_144_160_fused_logits_ep3_l040_f050_t15.pt \
  --pair-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_multiscale_144_160_fused_logits_ep3_l040_f050_t15.pt \
  --single-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_crop160_fused_logits_ep3_l040_f050_t15.pt \
  --scales 128,144,160 --weights 0.35,0.50,0.15 \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_multiscale_128_144_160_w035_050_015_fp32_ep3_l040_f050 \
  --strength 0.85 --acknowledge-balanced-test-prior
```

## Verification

- Inference mode: `attention_reweighted_multiscale_flip:crops=128-144-160:weights=0.35-0.5-0.15:balanced_prior=0.85`
- Predictions: 24,967; corrupt images: 0
- Changes versus the 68.598550% `0.25/0.50/0.25` best: 191 (0.77%)
- Changes versus equal 128/144/160: 268
- Changes versus equal 144/160: 645
- Reconstructed probabilities are non-negative; maximum normalization error is below `3e-7`
- Aegis submission audit: PASS (24,967 rows, 500 classes)
- Repository submission checker: all checks passed
- ZIP contents: root-level `pred_results.csv` only
- Full test suite: 268 passed, 8 warnings

## Artifact hashes

- Checkpoint SHA-256: `67efab2bf954139b59df074ccf00c0113cbc6ff96163d6e8d66ffbe553b910a4`
- CSV SHA-256: `57b0b3fd61af7dbc7c373103a8935678d1dd244b2bc4cade05bc6b8e8b180128`
- ZIP SHA-256: `62d4ec351373d8dbecddb983c1f0ec8a76ba0235358235fdd0a8121f8055acdb`
- Manifest SHA-256: `8774edc6a92b372541e19c9c4be39210eb623bc7ed05ac88f34bd88bdd248858`

## Status

Platform score: **68.63860295590179%** (17,137 / 24,967 correct). This improves
on the `0.25/0.50/0.25` platform best by **0.040053 percentage points**, or
exactly 10 additional correct predictions, and is promoted as the new audited
best. Shifting weight from crop160 to crop128 while holding crop144 at 0.50 is
therefore independently platform-positive.
