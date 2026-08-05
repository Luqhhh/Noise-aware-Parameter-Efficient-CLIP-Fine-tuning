# F1 R2 crop112 Part-Token residual Adapter (2026-08-05)

## Objective

Test the preregistered Part-Token residual direction on the current R2 parent
instead of continuing the diminishing local-feature crop-size sweep. The model
remains a single OpenAI CLIP ViT-B/32 checkpoint with one shared linear
classifier and one 34,336-parameter local-only Adapter. Patch tokens come from
the same local-view forward pass; there is no ensemble or external data.
Platform accuracy is the promotion criterion.

## Implementation

- Added generic Part-Token adaptation for already-cropped local views.
- Added `--adapt-part-token-features` to the audited multiscale/flip inference
  path, mutually exclusive with the local-feature Adapter.
- The residual remains anchored on the exact native local logits; a zero-output
  Adapter is bit-exact to the native local branch.
- Full test suite: 277 passed, 8 warnings.

## Cache construction

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.cache_part_token_adapter_features \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --split-csv artifacts/r2_local_feature_adapter/seed42/train_clean070.csv \
  --output outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_PART_TOKEN_ADAPTER_CROP112_FP32/seed42/cache/train_clean070_crop112_bs128.pt \
  --batch-size 128 --num-workers 4 --crop-size 112 --top-patches 5 \
  --part-top-patches 8 --part-temperature 0.07

PYTHONPATH=$PWD python3 -m aegis_clip.cli.cache_part_token_adapter_features \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --split-csv artifacts/stages/preliminary/seed42/val.csv \
  --output outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_PART_TOKEN_ADAPTER_CROP112_FP32/seed42/cache/validation_crop112_bs128.pt \
  --batch-size 128 --num-workers 4 --crop-size 112 --top-patches 5 \
  --part-top-patches 8 --part-temperature 0.07
```

- Train samples: 65,115; validation samples: 10,316; overlap: 0
- Pool: CLS-cosine top-8 patch tokens, temperature 0.07
- Train cache SHA-256: `b03b9916cb8f71a968422df333764597a8cb24ca5b0de02d1d26155c05f6cbf6`
- Validation cache SHA-256: `5230ea1f474377e33dbe1d9a7767a236df6b8a5a3f52b2fe1e3fe06752e434f1`

## Training

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train_part_token_adapter \
  --parent-checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --train-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_PART_TOKEN_ADAPTER_CROP112_FP32/seed42/cache/train_clean070_crop112_bs128.pt \
  --validation-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_PART_TOKEN_ADAPTER_CROP112_FP32/seed42/cache/validation_crop112_bs128.pt \
  --output-dir outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_PART_TOKEN_ADAPTER_CROP112_FP32/seed42 \
  --center-reference outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_FP32/seed42/cache/validation_center_bs128.pt \
  --m1-reference outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_CROP112_FP32/seed42/cache/validation_m1_crop112_bs128.pt \
  --expected-train-samples 65115 --expected-cache-batch-size 128 \
  --seed 42 --bottleneck-dim 32 --residual-scale 0.25 --dropout 0.1 \
  --learning-rate 5e-4 --weight-decay 1e-4 --batch-size 1024 \
  --max-epochs 20 --patience 5 --gce-q 0.5 --local-loss-weight 0.25 \
  --feature-anchor-weight 2.0 --device cuda
```

- Selected epoch: 13; Adapter parameters: 34,336
- Clean-core micro delta: `+0.8730053901672363pp`
- Raw micro delta: `+0.853043794631958pp`
- Trusted macro delta: `+0.8489012718200684pp`
- Clean-core corrected/harmed/net: 115 / 51 / +64
- Local feature drift: `0.003911757940652528`
- Global path bit-exact and epoch-zero M1 reproduction: PASS
- Composite checkpoint SHA-256: `26916fd3ec96311dcab7a637f416ad3455cf7c78087844d408a38958f168962a`
- Adapter-only SHA-256: `6f6c5360ac89482abfc776a736ac92b453435f4c6e5bf8b6f1d06a27d162af93`

## Inference and verification

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_PART_TOKEN_ADAPTER_CROP112_FP32/seed42/best.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_part_token_adapter_crop112_multiscale_128_144_160_w045_050_005_fp32_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_multiscale --local-crop-sizes 128,144,160 \
  --local-scale-weights 0.45,0.50,0.05 --local-top-k 5 \
  --local-weight 0.4 --local-temperature 1.5 \
  --adapt-part-token-features --acknowledge-local-view-risk \
  --prior-alignment-strength 0.85 --acknowledge-balanced-test-prior \
  --batch-size 128 \
  --dump-logits outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_PART_TOKEN_ADAPTER_CROP112_FP32/seed42/test_weighted_multiscale_logits.pt \
  --overwrite
```

- Predictions: 24,967; all 500 classes represented
- Changes versus the 68.88693074858814% best: 728
- Aegis acknowledged-same-model-TTA audit: PASS
- Repository nine-criterion checker: PASS
- ZIP contains root-level `pred_results.csv` only
- CSV SHA-256: `c6ed3e6a7f63c49a9b821f0e09222a153d926702de6f4c42505781aa7ae89fdd`
- ZIP SHA-256: `6333375eea0f0b7575b833de16daf89c897df521c9eaa3f64a71e546c5ec4dc6`
- Manifest SHA-256: `0ddff9c4e03b0bfefe3c8671d388679a4a33dbbcc547b9f88d1b440fdca1c06e`

The desktop package was replaced and hash-verified.

## Platform result

- Accuracy: `68.90295189650338%` (`17203 / 24967`)
- Improvement over the crop112 local-feature Adapter: `+0.01602114791524pp`
  and `+4` correct
- Remaining gap to 70%: `274` correct
- Status: `platform_valid_promoted`; new audited platform best
