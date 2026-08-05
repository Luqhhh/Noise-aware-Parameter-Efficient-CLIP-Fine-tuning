# F1 R2 crop144 local-feature Adapter (2026-08-05)

## Objective

Retrain the successful R2 local-only Adapter on crop144, the dominant local
scale in the platform-best `0.45/0.50/0.05` fusion. The global CLIP path,
shared classifier, training subset, and final inference protocol are unchanged.

## Cache and training

The fixed 65,115-image high-clean split from the promoted crop160 Adapter was
reused. Crop144/top5 validation and training features were cached at batch 128:

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.cache_local_adapter_features \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --split-csv artifacts/r2_local_feature_adapter/seed42/train_clean070.csv \
  --output outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP144_FP32/seed42/cache/train_clean070_crop144_bs128.pt \
  --batch-size 128 --num-workers 4 --crop-size 144 --top-patches 5
```

The center reference was bit-exact and the crop144 M1 reference had 100%
prediction agreement with `3.814697265625e-6` fusion-recompute error. Training:

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train_local_feature_adapter \
  --parent-checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --train-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP144_FP32/seed42/cache/train_clean070_crop144_bs128.pt \
  --validation-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP144_FP32/seed42/cache/validation_crop144_bs128.pt \
  --output-dir outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP144_FP32/seed42/checkpoints \
  --center-reference outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_FP32/seed42/cache/validation_center_bs128.pt \
  --m1-reference outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP144_FP32/seed42/cache/validation_m1_crop144_bs128.pt \
  --expected-train-samples 65115 --seed 42 --bottleneck-dim 32 \
  --residual-scale 0.25 --dropout 0.1 --learning-rate 5e-4 \
  --weight-decay 1e-4 --batch-size 1024 --max-epochs 20 --patience 5 \
  --gce-q 0.5 --local-loss-weight 0.25 --feature-anchor-weight 2.0 \
  --device cuda
```

- Selected Adapter epoch: 2; parameters: 34,336
- Clean-core micro delta: `+0.654757022857666pp`
- Raw micro delta: `+0.504070520401001pp`
- Trusted macro delta: `+0.5040287971496582pp`
- Local cosine drift: `0.0016163921069618156`
- Composite checkpoint SHA-256: `3896f1c2a493ce25042c70e60f965f51de02daaab4e5f85ae43b4852e44efe57`
- Adapter-only SHA-256: `c665743f24d17d54de99fe5415e5483954b1defa22ef4e678247003d882484cb`

## Inference and verification

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP144_FP32/seed42/checkpoints/best.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_local_adapter_crop144_multiscale_128_144_160_w045_050_005_fp32_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_multiscale --local-crop-sizes 128,144,160 \
  --local-scale-weights 0.45,0.50,0.05 --local-top-k 5 \
  --local-weight 0.4 --local-temperature 1.5 --adapt-local-features \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128 \
  --dump-logits outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP144_FP32/seed42/inference/test_weighted_multiscale_logits.pt \
  --overwrite
```

- Predictions: 24,967; changes versus the 68.82284615692714% best: 225
- Aegis audit and repository nine-criterion checker: PASS
- Full test suite: 274 passed, 8 warnings
- ZIP contents: root-level `pred_results.csv` only
- CSV SHA-256: `f25fe1e11a0a3b37aaf2a5c65d065ceb43c924d7c21ea18141c85c83c0202eb0`
- ZIP SHA-256: `abdaa32624261238af82604b1dff50243384ec0e7c6532d2f69a4ce706bcf8c3`
- Manifest SHA-256: `be86bcc9074b1bc8e171c7c695a9948c9cec07cfd169ffaeffcbe4edd7748669`

The desktop package was replaced and hash-verified.

Platform score: `68.85889373973644%` (17,192 / 24,967), a new best. This is 9
additional correct predictions and `0.03604758280931` percentage points above
the crop160-trained Adapter. The candidate is promoted. Reaching 70% still
requires 285 additional correct predictions.
