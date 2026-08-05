# F1 R2 crop128 local-feature Adapter (2026-08-05)

## Objective

Retrain the successful R2 local-only Adapter on crop128, which contributes 45%
of the platform-best local-scale fusion. All other data, model, and inference
settings remain fixed; platform accuracy is the promotion criterion.

## Training

The same fixed 65,115-image high-clean split was cached at crop128/top5:

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.cache_local_adapter_features \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --split-csv artifacts/r2_local_feature_adapter/seed42/train_clean070.csv \
  --output outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP128_FP32/seed42/cache/train_clean070_crop128_bs128.pt \
  --batch-size 128 --num-workers 4 --crop-size 128 --top-patches 5
```

Center and crop128 M1 reference audits had 100% prediction agreement; the
center was bit-exact and fusion recomputation error was `3.814697265625e-6`.

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train_local_feature_adapter \
  --parent-checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --train-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP128_FP32/seed42/cache/train_clean070_crop128_bs128.pt \
  --validation-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP128_FP32/seed42/cache/validation_crop128_bs128.pt \
  --output-dir outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP128_FP32/seed42/checkpoints \
  --center-reference outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_FP32/seed42/cache/validation_center_bs128.pt \
  --m1-reference outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP128_FP32/seed42/cache/validation_m1_crop128_bs128.pt \
  --expected-train-samples 65115 --seed 42 --bottleneck-dim 32 \
  --residual-scale 0.25 --dropout 0.1 --learning-rate 5e-4 \
  --weight-decay 1e-4 --batch-size 1024 --max-epochs 20 --patience 5 \
  --gce-q 0.5 --local-loss-weight 0.25 --feature-anchor-weight 2.0 \
  --device cuda
```

- Selected epoch: 2; Adapter parameters: 34,336
- Clean-core micro delta: `+0.6001889705657959pp`
- Raw micro delta: `+0.6010055541992188pp`
- Trusted macro delta: `+0.5205214023590088pp`
- Local cosine drift: `0.0023001537807415787`
- Composite checkpoint SHA-256: `b78bd7216c7a6bdab373e6511ac9a0f090c0521a952b9fa88c9b0405bec1d09b`
- Adapter-only SHA-256: `73dea8b76eedbd7429684b6edfb3788e0412bef7a9becd3db16021e64f718755`

## Inference and verification

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP128_FP32/seed42/checkpoints/best.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_local_adapter_crop128_multiscale_128_144_160_w045_050_005_fp32_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_multiscale --local-crop-sizes 128,144,160 \
  --local-scale-weights 0.45,0.50,0.05 --local-top-k 5 \
  --local-weight 0.4 --local-temperature 1.5 --adapt-local-features \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128 \
  --dump-logits outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP128_FP32/seed42/inference/test_weighted_multiscale_logits.pt \
  --overwrite
```

- Predictions: 24,967; changes versus the 68.85889373973644% best: 270
- Aegis audit and repository nine-criterion checker: PASS
- Full test suite: 274 passed, 8 warnings
- ZIP contains root-level `pred_results.csv` only
- CSV SHA-256: `43b9898a7fecd1f208152138692f587dea0603db93b8c4aefb809f2e3f4163f3`
- ZIP SHA-256: `f779e292ad06760d4daed8f8da01a4ec29d55489c996d05c4bd0d7fac4e31685`
- Manifest SHA-256: `02b3a6c7a655fa2638ba3a945c6da3d0e11353562e3a9defb8156085344a4198`

The desktop package was replaced and hash-verified. Platform result is pending.
