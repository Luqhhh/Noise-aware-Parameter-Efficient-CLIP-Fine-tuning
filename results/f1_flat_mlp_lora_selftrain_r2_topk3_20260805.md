# F1 R2 attention top-k=3 inference (2026-08-05)

## Objective

Test a different attention-localization geometry on the platform-best R2
checkpoint. Platform accuracy remains the only promotion criterion; local
validation is diagnostic and is not an advancement gate.

## Validation diagnostic

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.sweep_localization \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/localization_topk3_5_9_scales128_144_160_t15.json \
  --crop-sizes 128,144,160 --top-ks 3,5,9 --local-weights 0.4 \
  --include-horizontal-flip --flip-weights 0.5 --temperature 1.5 \
  --batch-size 128 --overwrite
```

- top-k=3 raw / clean-core micro accuracy: `0.73158198595` / `0.83822125196`
- top-k=5 raw / clean-core micro accuracy: `0.73090344667` / `0.83781200647`
- top-k=9 raw / clean-core micro accuracy: `0.73100036383` / `0.83767563105`

## Inference

The same R2 epoch-3 checkpoint was inferred with top-k=3 for scale sets
`128,144,160`, `144,160`, and `160`. Each inference used horizontal flip
weight `0.5`, local weight `0.4`, temperature `1.5`, and prior alignment
strength `0.85`. The final package was reconstructed with:

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer_reweighted_multiscale_submission \
  --triple-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/inference_topk3/triple_128_144_160_logits.pt \
  --pair-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/inference_topk3/pair_144_160_logits.pt \
  --single-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/inference_topk3/single_160_logits.pt \
  --scales 128,144,160 --weights 0.45,0.50,0.05 \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_multiscale_topk3_128_144_160_w045_050_005_fp32_ep3_l040_f050 \
  --strength 0.85 --acknowledge-balanced-test-prior --overwrite
```

## Verification and artifacts

- Experiment: `F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32`, epoch 3, seed 42
- Checkpoint SHA-256: `67efab2bf954139b59df074ccf00c0113cbc6ff96163d6e8d66ffbe553b910a4`
- Predictions: 24,967; changes versus the 68.67865582568992% best: 1,141
- Reconstructed probabilities: all non-negative; maximum normalization error `2.980232238769531e-07`
- Aegis submission audit: PASS
- Repository nine-criterion submission checker: PASS
- Full test suite: 269 passed, 8 warnings
- ZIP contents: root-level `pred_results.csv` only
- CSV SHA-256: `e658e8eac5c8611224648af9cbfc3216ee96a078f0ff92fca1dbb4d091e66206`
- ZIP SHA-256: `1456d246106600cd171b8f4d4daa8ad84756f1cf30d9c78ba5927cd4fc8c7daf`
- Manifest SHA-256: `719d66b63bd4a9189607a26ed54a4b55aacbb8508df10863474f485e1585fdd7`

The desktop submission was replaced and hash-verified.

## Status

Platform score: `68.63860295590179%` (17,137 / 24,967), which is 10 correct
predictions and `0.04005286978813` percentage points below the top-k=5 R2
platform best. The candidate is valid but not promoted. The diagnostic local
top-k advantage did not transfer reliably to the platform.
