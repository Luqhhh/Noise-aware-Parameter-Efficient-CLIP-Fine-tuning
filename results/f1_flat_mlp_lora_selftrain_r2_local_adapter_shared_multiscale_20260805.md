# F1 R2 shared multiscale local-feature Adapter (2026-08-05)

## Objective

Train one local-only Adapter jointly on aligned crop128, crop144, and crop160
features. The model remains one R2 CLIP ViT-B/32 checkpoint with one shared
34,336-parameter Adapter; it is not an ensemble. Platform accuracy is the
promotion criterion.

## Implementation

- Repeated `--train-cache`, `--validation-cache`, and `--m1-reference`
  arguments enable aligned multiscale training.
- Each training sample is represented at all three scales. Scale-local
  probabilities are fused with weights `0.45/0.50/0.05` before the fused and
  local GCE objectives are evaluated.
- Validation selection uses the same weighted multiscale probability fusion.
- The logits dump now uses an atomic writer that creates its parent directory.

## Training

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train_local_feature_adapter \
  --parent-checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --train-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP128_FP32/seed42/cache/train_clean070_crop128_bs128.pt \
  --train-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP144_FP32/seed42/cache/train_clean070_crop144_bs128.pt \
  --train-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_FP32/seed42/cache/train_clean070_bs128.pt \
  --validation-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP128_FP32/seed42/cache/validation_crop128_bs128.pt \
  --validation-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP144_FP32/seed42/cache/validation_crop144_bs128.pt \
  --validation-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_FP32/seed42/cache/validation_bs128.pt \
  --output-dir outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_SHARED_MULTISCALE_FP32/seed42/checkpoints \
  --center-reference outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_FP32/seed42/cache/validation_center_bs128.pt \
  --m1-reference outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP128_FP32/seed42/cache/validation_m1_crop128_bs128.pt \
  --m1-reference outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP144_FP32/seed42/cache/validation_m1_crop144_bs128.pt \
  --m1-reference outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_FP32/seed42/cache/validation_m1_bs128.pt \
  --scale-weights 0.45,0.50,0.05 --expected-train-samples 65115 \
  --seed 42 --bottleneck-dim 32 --residual-scale 0.25 --dropout 0.1 \
  --learning-rate 5e-4 --weight-decay 1e-4 --batch-size 1024 \
  --max-epochs 20 --patience 5 --gce-q 0.5 --local-loss-weight 0.25 \
  --feature-anchor-weight 2.0 --device cuda
```

- Selected epoch: 2; Adapter parameters: 34,336
- Weighted multiscale clean-core micro delta: `+0.5865514278411865pp`
- Weighted multiscale raw micro delta: `+0.4846811294555664pp`
- Weighted multiscale trusted macro delta: `+0.49123167991638184pp`
- Weighted local cosine drift: `0.0018426605946983464`
- Composite checkpoint SHA-256: `4c8faeab0168bbcef9f8cb0e5cf7a9f6ddea721bbcb895aa188731e01bd692fc`
- Adapter-only SHA-256: `5a22c38cb950534a06dbedb26fdaf0d6a1ce811e3a188673eb89991ffe52daa9`

## Inference and verification

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_SHARED_MULTISCALE_FP32/seed42/checkpoints/best.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_local_adapter_shared_multiscale_128_144_160_w045_050_005_fp32_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_multiscale --local-crop-sizes 128,144,160 \
  --local-scale-weights 0.45,0.50,0.05 --local-top-k 5 \
  --local-weight 0.4 --local-temperature 1.5 --adapt-local-features \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128 \
  --dump-logits outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_SHARED_MULTISCALE_FP32/seed42/inference/test_weighted_multiscale_logits.pt \
  --overwrite
```

- Predictions: 24,967; changes versus the 68.88292546160932% best: 194
- Aegis acknowledged-same-model-TTA audit: PASS
- Repository nine-criterion checker: PASS
- Full test suite: 275 passed, 8 warnings
- ZIP contains root-level `pred_results.csv` only
- CSV SHA-256: `df2cd5f332fd5d57d467e3cca35cff0456eaf3ff431ee5b49f7504c5aec54a7c`
- ZIP SHA-256: `be006ff199daa9c2d82edfd50b77dece79805ad49e3e5e09c224c1cd6e81c3de`
- Manifest SHA-256: `959c45968ead009d67bfa53c91cda443ae41c5e33168c2b9e8e7ddc7a4745356`

The desktop package was replaced and hash-verified. Platform result is pending.
