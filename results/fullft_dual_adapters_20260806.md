# F1_FLAT_FULL_FT_R3MS 双 Adapter（2026-08-06）

## 结论

在全微调父模型（F1_FLAT_FULL_FT_R3MS epoch3）上训练 O3 局部特征 Adapter 与
Part-Token Adapter（BN64），两者均通过晋级门控（各 +0.71pp clean-core），
合成双 Adapter composite checkpoint，测试推理完成并生成 25 个唯一 TTA 候选包。
平台分数待回传。

## 缓存

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
# O3
PYTHONPATH=$PWD python3 -m aegis_clip.cli.cache_local_adapter_features \
  --checkpoint outputs/F1_FLAT_FULL_FT_R3MS/seed42/checkpoints/best.pt \
  --split-csv artifacts/r2_local_feature_adapter/seed42/train_clean070.csv \
  --output outputs/F1_FLAT_FULL_FT_R3MS/seed42/cache/train_clean070_crop112_bs128.pt \
  --batch-size 128 --num-workers 4 --crop-size 112 --top-patches 5
# + validation_crop112_bs128.pt, validation_center_bs128.pt,
#   validation_m1_crop112_bs128.pt（cache_validation_logits）
# PTA
PYTHONPATH=$PWD python3 -m aegis_clip.cli.cache_part_token_adapter_features \
  --checkpoint outputs/F1_FLAT_FULL_FT_R3MS/seed42/checkpoints/best.pt \
  --split-csv artifacts/r2_local_feature_adapter/seed42/train_clean070.csv \
  --output outputs/F1_FLAT_FULL_FT_R3MS/seed42/cache/train_clean070_crop112_pta_bs128.pt \
  --batch-size 128 --num-workers 4 --crop-size 112 --top-patches 5 \
  --part-top-patches 8 --part-temperature 0.07
```

## 训练

- O3：`train_local_feature_adapter`，bottleneck32，best epoch 4，
  clean-core +0.7093pp / raw +0.8530pp，gate PASS
- PTA：`train_part_token_adapter`，bottleneck64，best epoch 4，
  clean-core +0.7093pp / raw +0.7949pp，gate PASS
- 合成：`outputs/F1_FLAT_FULL_FT_R3MS/seed42/dual_adapters/best.pt`
  （SHA-256 `f72b0104257f49d2667fe335553a861dd1dea947753feebdc7301b8890b48765`）

## 推理与候选

- 主包：`outputs/delivery/fullft_dual_112_128_144_160_w020_030_040_010_l040_f050_t15_pa085/`
  （ZIP `d99f2b64...`，CSV `c3fe3d84...`，审计 PASS）
- 25 个唯一预测集在 `桌面/fullft_dual_sweep/`（均审计通过）
- 与当前平台最佳（R3 dual prior 0.91，69.075179%）差异 3497–3569
