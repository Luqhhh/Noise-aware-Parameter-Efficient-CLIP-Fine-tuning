# F1 R2 reweighted multiscale 0.44/0.50/0.06 (2026-08-05)

## Objective

The current platform best uses 128/144/160 weights `0.45/0.50/0.05` and scores
`68.67865582568992%`. The 128/144 endpoint `0.50/0.50/0.00` regressed by 13
correct predictions, so this segment performs a narrow one-dimensional search
immediately before the best point: `0.44/0.50/0.06`. All model and inference
settings except the scale weights remain fixed.

## Exact submission command

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer_reweighted_multiscale_submission \
  --triple-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_multiscale_128_144_160_fused_logits_ep3_l040_f050_t15.pt \
  --pair-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_multiscale_144_160_fused_logits_ep3_l040_f050_t15.pt \
  --single-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_crop160_fused_logits_ep3_l040_f050_t15.pt \
  --scales 128,144,160 --weights 0.44,0.50,0.06 \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_multiscale_128_144_160_w044_050_006_fp32_ep3_l040_f050 \
  --strength 0.85 --acknowledge-balanced-test-prior
```

## Verification

- Inference mode: `attention_reweighted_multiscale_flip:crops=128-144-160:weights=0.44-0.5-0.06:balanced_prior=0.85`
- Predictions: 24,967; corrupt images: 0
- Changes versus the 68.678656% `0.45/0.50/0.05` best: 22 (0.088%)
- Changes versus `0.35/0.50/0.15`: 166
- Changes versus the 128/144 endpoint: 121
- Reconstructed probabilities are non-negative; maximum normalization error is below `3e-7`
- Aegis submission audit: PASS (24,967 rows, 500 classes)
- Repository submission checker: all checks passed
- ZIP contents: root-level `pred_results.csv` only
- Full test suite: 268 passed, 8 warnings

## Artifact hashes

- Checkpoint SHA-256: `67efab2bf954139b59df074ccf00c0113cbc6ff96163d6e8d66ffbe553b910a4`
- CSV SHA-256: `c8b9b37e1353c31f8a5927e7c1ae9486e88600c578fe0988c70ddd09c49d6106`
- ZIP SHA-256: `7e0576e56185c70e969a7f36d4d399e78a237c8d9c014568ddc12923122063b4`
- Manifest SHA-256: `9cbb0de6185896f3914708a5aaff2bd779ae62257ed839ed67ff6a3081977173`

## Status

Platform score: `68.65462410381704%`, which is 6 correct predictions and
`0.02403172187288` percentage points below the promoted `0.45/0.50/0.05`
result. This candidate is valid but not promoted. The `68.67865582568992%`
package remains the platform best and is recoverable for immediate rollback.
