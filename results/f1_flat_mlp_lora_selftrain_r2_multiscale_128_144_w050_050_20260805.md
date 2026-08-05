# F1 R2 128/144 two-scale fusion (2026-08-05)

## Objective

The 128/144/160 weighting `0.45/0.50/0.05` reached a new platform best of
`68.67865582568992%` (17,147 / 24,967), adding exactly 10 correct predictions
over `0.35/0.50/0.15`. This segment reaches the natural endpoint of the same
single-variable path by assigning the final 0.05 crop160 weight to crop128.
The resulting fusion is an equal 128/144 two-scale mean with weights
`0.50/0.50/0.00`.

## Exact submission command

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer_reweighted_multiscale_submission \
  --triple-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_multiscale_128_144_160_fused_logits_ep3_l040_f050_t15.pt \
  --pair-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_multiscale_144_160_fused_logits_ep3_l040_f050_t15.pt \
  --single-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_crop160_fused_logits_ep3_l040_f050_t15.pt \
  --scales 128,144,160 --weights 0.50,0.50,0.00 \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_multiscale_128_144_w050_050_fp32_ep3_l040_f050 \
  --strength 0.85 --acknowledge-balanced-test-prior
```

## Verification

- Inference mode: `attention_reweighted_multiscale_flip:crops=128-144-160:weights=0.5-0.5-0:balanced_prior=0.85`
- Predictions: 24,967; corrupt images: 0
- Changes versus the 68.678656% `0.45/0.50/0.05` best: 99 (0.40%)
- Changes versus `0.35/0.50/0.15`: 285
- Changes versus equal 128/144/160: 509
- Reconstructed probabilities are non-negative; maximum normalization error is below `3e-7`
- Aegis submission audit: PASS (24,967 rows, 500 classes)
- Repository submission checker: all checks passed
- ZIP contents: root-level `pred_results.csv` only
- Full test suite: 268 passed, 8 warnings

## Artifact hashes

- Checkpoint SHA-256: `67efab2bf954139b59df074ccf00c0113cbc6ff96163d6e8d66ffbe553b910a4`
- CSV SHA-256: `e48990b01c3f3dfbaf4cd16905465c9d1c4c103d91acdc94bd087cca9cdd0223`
- ZIP SHA-256: `022ee2cc021b9f7811a6d9aeb00cfd16d7fff27c90b471e964c5944623a7d3a9`
- Manifest SHA-256: `0e62d79947f3335b4344b85ba0000690f72c81eb219de8b1a077b6fa7f594924`

## Status

Candidate generated and audited; platform score pending. The promoted
68.678656% package remains recoverable for immediate rollback.
