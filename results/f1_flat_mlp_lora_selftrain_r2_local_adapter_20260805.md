# F1 R2 local-feature Adapter continuation (2026-08-05)

## Objective

Adapt only the attention-local features of the platform-best R2 checkpoint
while keeping its global CLIP path and shared linear classifier unchanged.
Platform accuracy remains the only promotion criterion.

## Data and training

The fixed high-clean training subset was prepared with:

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.prepare_high_clean_split \
  --source-csv artifacts/stages/preliminary/seed42/train.csv \
  --validation-csv artifacts/stages/preliminary/seed42/val.csv \
  --trust-bundle artifacts/trust/selftrain_r1_teacher_v2_relaxed.pt \
  --output-csv artifacts/r2_local_feature_adapter/seed42/train_clean070.csv \
  --threshold 0.70 --expected-selected 65115 --expected-classes 500
```

- Selected samples: 65,115; classes: 500; validation overlap: 0
- Training CSV SHA-256: `df59f40fb20b622e90b54e3bc4df14baed3a91bdb2241788df0247c77a372e7d`
- Parent R2 epoch-3 SHA-256: `67efab2bf954139b59df074ccf00c0113cbc6ff96163d6e8d66ffbe553b910a4`
- Strict cache audit: center max difference `0`, agreement `1.0`; M1
  fusion-recompute max difference `3.814697265625e-6`, agreement `1.0`

After caching crop160/top5 local features at batch size 128, the Adapter was
trained with:

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train_local_feature_adapter \
  --parent-checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --train-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_FP32/seed42/cache/train_clean070_bs128.pt \
  --validation-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_FP32/seed42/cache/validation_bs128.pt \
  --output-dir outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_FP32/seed42/checkpoints \
  --center-reference outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_FP32/seed42/cache/validation_center_bs128.pt \
  --m1-reference outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_FP32/seed42/cache/validation_m1_bs128.pt \
  --expected-train-samples 65115 --seed 42 --bottleneck-dim 32 \
  --residual-scale 0.25 --dropout 0.1 --learning-rate 5e-4 \
  --weight-decay 1e-4 --batch-size 1024 --max-epochs 20 --patience 5 \
  --gce-q 0.5 --local-loss-weight 0.25 --feature-anchor-weight 2.0 \
  --device cuda
```

- Selected Adapter epoch: 2; parameters: 34,336
- Clean-core micro delta: `+0.46378374099731445pp`
- Raw micro delta: `+0.35866498947143555pp`
- Trusted macro delta: `+0.40181875228881836pp`
- Local cosine drift: `0.0010630902785485731`
- Composite checkpoint SHA-256: `d1345ee61d25aae78e74c6fb6d26ae428d2282d529048f0b8b05f75b2e540c5e`
- Adapter-only SHA-256: `1b460da5df06963c9f54c200962cb597be21525bbac3e5b938fc72ce9d1fc836`

## Inference

The Adapter was applied to every original/flipped local crop while the global
path remained native R2. Direct weighted multiscale fusion is mathematically
equivalent to the previous nested-cache reconstruction and avoids three test
passes:

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_FP32/seed42/checkpoints/best.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_local_adapter_multiscale_128_144_160_w045_050_005_fp32_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_multiscale --local-crop-sizes 128,144,160 \
  --local-scale-weights 0.45,0.50,0.05 --local-top-k 5 \
  --local-weight 0.4 --local-temperature 1.5 --adapt-local-features \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128 \
  --dump-logits outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_LOCAL_ADAPTER_FP32/seed42/inference/test_weighted_multiscale_logits.pt \
  --overwrite
```

## Verification and status

- Predictions: 24,967; changes versus the 68.67865582568992% best: 423
- Aegis submission audit: PASS
- Repository nine-criterion submission checker: PASS
- Full test suite: 274 passed, 8 warnings
- ZIP contents: root-level `pred_results.csv` only
- CSV SHA-256: `cbb1a82acc08d6cd1f2e00a1b3563614e1ee2b17df94d26a2f757a03fa7d59c3`
- ZIP SHA-256: `6b8ba1504ae19277f06b144febfae92a68ef091902188d1ae985eba6aa194e0b`
- Manifest SHA-256: `6a02f2e67c1a538b44828017cb20cf6c5bc6bf41413cf33e68383ef6a2ccad9b`

The desktop package was replaced and hash-verified.

Platform score: `68.82284615692714%` (17,183 / 24,967), a new best. This is 36
additional correct predictions and `0.14419033123723` percentage points above
the unadapted R2 best. The candidate is promoted. Reaching 70% still requires
294 additional correct predictions.
