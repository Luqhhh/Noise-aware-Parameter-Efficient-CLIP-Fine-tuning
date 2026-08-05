# F1 R2 crop112 local-feature Adapter (2026-08-05)

## Objective

Continue the platform-positive Adapter training-crop sequence from 160 to 144
to 128 by training the same local-only Adapter at crop112. All other model,
data, training, and inference settings remain fixed. Platform accuracy is the
promotion criterion.

## Cache construction

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.cache_local_adapter_features \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --split-csv artifacts/r2_local_feature_adapter/seed42/train_clean070.csv \
  --output outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP112_FP32/seed42/cache/train_clean070_crop112_bs128.pt \
  --batch-size 128 --num-workers 4 --crop-size 112 --top-patches 5

PYTHONPATH=$PWD python3 -m aegis_clip.cli.cache_local_adapter_features \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --split-csv artifacts/stages/preliminary/seed42/val.csv \
  --output outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP112_FP32/seed42/cache/validation_crop112_bs128.pt \
  --batch-size 128 --num-workers 4 --crop-size 112 --top-patches 5

PYTHONPATH=$PWD python3 -m aegis_clip.cli.cache_validation_logits \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --output outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP112_FP32/seed42/cache/validation_m1_crop112_bs128.pt \
  --batch-size 128 --num-workers 4 --view-mode attention_local_global \
  --force-online-images --crop-size 112 --top-patches 5
```

- Train samples: 65,115; validation samples: 10,316; overlap: 0
- Minimum training clean probability: `0.7000356316566467`
- Center reference: bit-exact, prediction agreement 1.0
- M1 recomputation maximum difference: `3.814697265625e-6`, agreement 1.0

## Training

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train_local_feature_adapter \
  --parent-checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --train-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP112_FP32/seed42/cache/train_clean070_crop112_bs128.pt \
  --validation-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP112_FP32/seed42/cache/validation_crop112_bs128.pt \
  --output-dir outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP112_FP32/seed42/checkpoints \
  --center-reference outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_FP32/seed42/cache/validation_center_bs128.pt \
  --m1-reference outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP112_FP32/seed42/cache/validation_m1_crop112_bs128.pt \
  --expected-train-samples 65115 --seed 42 --bottleneck-dim 32 \
  --residual-scale 0.25 --dropout 0.1 --learning-rate 5e-4 \
  --weight-decay 1e-4 --batch-size 1024 --max-epochs 20 --patience 5 \
  --gce-q 0.5 --local-loss-weight 0.25 --feature-anchor-weight 2.0 \
  --device cuda
```

- Selected epoch: 1; Adapter parameters: 34,336
- Clean-core micro delta: `+0.8184432983398438pp`
- Raw micro delta: `+0.7464170455932617pp`
- Trusted macro delta: `+0.7670164108276367pp`
- Local cosine drift: `0.002732931620023564`
- Composite checkpoint SHA-256: `693235caf07c5fa94df5b9d4cf0a032dbbcbe737bb447c94cb49dd71e9a775b3`
- Adapter-only SHA-256: `2eb8dfb05c42f05fc54eaa77d797fa71d29eddaeea3620c58e7ba19ad2f13fda`

## Inference and verification

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP112_FP32/seed42/checkpoints/best.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_local_adapter_crop112_multiscale_128_144_160_w045_050_005_fp32_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_multiscale --local-crop-sizes 128,144,160 \
  --local-scale-weights 0.45,0.50,0.05 --local-top-k 5 \
  --local-weight 0.4 --local-temperature 1.5 --adapt-local-features \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128 \
  --dump-logits outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP112_FP32/seed42/inference/test_weighted_multiscale_logits.pt \
  --overwrite
```

- Predictions: 24,967; changes versus the 68.88292546160932% best: 314
- Aegis acknowledged-same-model-TTA audit: PASS
- Repository nine-criterion checker: PASS
- Full test suite: 275 passed, 8 warnings
- ZIP contains root-level `pred_results.csv` only
- CSV SHA-256: `89595eee8924b339c40f35a280d6522d121fb3505670534b77bb629dc3694c4a`
- ZIP SHA-256: `9e06af3972cb6ad1468a27c06d87e48584d33ac3cf4e2005384e7a9ac12e01e9`
- Manifest SHA-256: `26646cf0bcd98a15d9381f37938f0929801aa962b3b68933515d0c65dffdc63d`

The desktop package was replaced and hash-verified. Platform result is pending.
