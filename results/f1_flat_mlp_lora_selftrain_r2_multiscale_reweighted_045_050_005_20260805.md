# F1 R2 reweighted multiscale 0.45/0.50/0.05 (2026-08-05)

## Objective

The 128/144/160 weighting `0.35/0.50/0.15` reached a new platform best of
`68.63860295590179%` (17,137 / 24,967), adding 10 correct predictions over
`0.25/0.50/0.25`. This segment holds crop144 at its validated weight 0.50 and
continues the same single-variable transfer of 0.10 weight from crop160 to
crop128, producing weights `0.45/0.50/0.05`.

## Exact submission command

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer_reweighted_multiscale_submission \
  --triple-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_multiscale_128_144_160_fused_logits_ep3_l040_f050_t15.pt \
  --pair-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_multiscale_144_160_fused_logits_ep3_l040_f050_t15.pt \
  --single-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_crop160_fused_logits_ep3_l040_f050_t15.pt \
  --scales 128,144,160 --weights 0.45,0.50,0.05 \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_multiscale_128_144_160_w045_050_005_fp32_ep3_l040_f050 \
  --strength 0.85 --acknowledge-balanced-test-prior
```

## Verification

- Inference mode: `attention_reweighted_multiscale_flip:crops=128-144-160:weights=0.45-0.5-0.05:balanced_prior=0.85`
- Predictions: 24,967; corrupt images: 0
- Changes versus the 68.638603% `0.35/0.50/0.15` best: 188 (0.75%)
- Changes versus `0.25/0.50/0.25`: 377
- Changes versus equal 128/144/160: 420
- Reconstructed probabilities are non-negative; maximum normalization error is below `3e-7`
- Aegis submission audit: PASS (24,967 rows, 500 classes)
- Repository submission checker: all checks passed
- ZIP contents: root-level `pred_results.csv` only
- Full test suite: 268 passed, 8 warnings

## Artifact hashes

- Checkpoint SHA-256: `67efab2bf954139b59df074ccf00c0113cbc6ff96163d6e8d66ffbe553b910a4`
- CSV SHA-256: `f29aae12f87ae0a59289f4c775031f6c9270605bdd0bdcba288d0c776368ddd8`
- ZIP SHA-256: `6e6243da2d2e93c8e9951234f1441ab19e19798b456f5c8f41cba0f53cb01a16`
- Manifest SHA-256: `17710b4616b6e617243d2a1ef25fbfe3e15feccf824dd38c147b6818b056dfdc`

## Status

Candidate generated and audited; platform score pending. The promoted
68.638603% package remains recoverable for immediate rollback.
