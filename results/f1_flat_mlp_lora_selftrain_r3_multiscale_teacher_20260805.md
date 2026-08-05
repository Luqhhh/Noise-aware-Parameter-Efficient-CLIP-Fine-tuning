# F1 R3 multiscale-teacher continuation (2026-08-05)

## Objective

Continue the platform-best R2 checkpoint for three fixed epochs using the
attention-multiscale teacher trust bundle. Platform accuracy remains the only
promotion criterion; overlapping local validation metrics are diagnostic.

## Training

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train \
  --config configs/f1_flat_mlp_lora_selftrain_r3_multiscale.yaml --overwrite
```

- Experiment: `F1_FLAT_MLP_LORA_SELFTRAIN_R3_MULTISCALE_FP32`
- Selection policy: fixed `last_epoch`; selected epoch 3
- Epoch 3 raw micro/macro: `0.7246994972` / `0.7243751884`
- Epoch 3 clean-core micro/macro: `0.8356295228` / `0.8415039182`
- Flip prediction agreement: `0.9048080444`
- Mean feature drift: `0.0098809004`
- Epoch-0-to-3 raw gain: `0.0013571382`
- Epoch-0-to-3 clean-core gain: `0.0025917292`
- Checkpoint: `epoch_3.pt`
- Checkpoint SHA-256: `42aa6ba5db03e0dd742d439f1d9ebbd1b533cb7a94d7190cf694553ed2a7117a`

## Inference

The checkpoint was inferred three times with the same horizontal-flip,
attention-local, temperature, local/global, and prior-alignment protocol as the
current platform best. The successful local scale sets were `128,144,160`,
`144,160`, and `160`; their cached probabilities were reconstructed and fused
with weights `0.45/0.50/0.05` using:

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer_reweighted_multiscale_submission \
  --triple-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R3_MULTISCALE_FP32/seed42/inference/triple_128_144_160_logits.pt \
  --pair-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R3_MULTISCALE_FP32/seed42/inference/pair_144_160_logits.pt \
  --single-dump outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R3_MULTISCALE_FP32/seed42/inference/single_160_logits.pt \
  --scales 128,144,160 --weights 0.45,0.50,0.05 \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R3_MULTISCALE_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r3_multiscale.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r3_multiscale_teacher_128_144_160_w045_050_005_fp32_ep3_l040_f050 \
  --strength 0.85 --acknowledge-balanced-test-prior --overwrite
```

## Verification and artifacts

- Predictions: 24,967; corrupt images: 0
- Changes versus the `68.67865582568992%` R2 platform best: 178
- Reconstructed probabilities: all non-negative; maximum normalization error
  `2.980232238769531e-07`
- Aegis submission audit: PASS
- Repository nine-criterion submission checker: PASS
- ZIP contents: root-level `pred_results.csv` only
- CSV SHA-256: `c8a1b6f1eabb3ea87fb608eb83843d7b2e207926ec358bd0771232d60720c150`
- ZIP SHA-256: `ad9dc9f5bac9dafa9c710d469ef0d9a4407d2e5c7bb863bbdb2fe9d9b7df2f90`
- Manifest SHA-256: `59ee5752b6f4d543288dfa8ae06826820f0bf97b5e118f6908df0515c341e18f`

The desktop submission was replaced and hash-verified. Intermediate
`epoch0.pt`, `epoch_1.pt`, `epoch_2.pt`, `best.pt`, and `last.pt` were deleted
after inference, freeing 1,799,272,172 bytes (about 1.68 GiB). The exact
submission checkpoint `epoch_3.pt` and all inference caches were retained.

## Status

Platform score: `68.62658709496536%`, which is 13 correct predictions and
`0.05206873072456` percentage points below the R2 `0.45/0.50/0.05` best.
The candidate is valid but not promoted. This confirms again that the higher
overlapping local metrics do not reliably transfer to the platform; no local
promotion gate was applied.
