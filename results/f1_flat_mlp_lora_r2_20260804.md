# F1_FLAT_MLP_LORA_R2_FP32（2026-08-04）

## 结论

从当前平台最佳 `F1_FLAT_MLP_LORA_FP32` 固定 epoch-3 checkpoint 继续进行
3 个低学习率 epoch。继承的 attention-LoRA 保持冻结，仅更新全 12 层 rank-4
MLP-LoRA 与分类头。按平台唯一裁判原则，使用最终 `epoch_3.pt` 直接生成候选，
本地指标只用于确认数值有限、500 类覆盖完整与训练轨迹正常。

平台实测 **67.86558256899107%**，比父候选 67.78147154243601% 提升
**0.084111 个百分点**，成为新的平台最佳。

一次独立 AMP 尝试在首步有限梯度审计中检测到视觉梯度 `inf`，在任何有效更新
前失败关闭。正式续训使用 FP32、batch 32，并从独立输出目录完成。

## 复现

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
python3 -m aegis_clip.cli.train \
  --config configs/f1_flat_mlp_lora_r2.yaml
python3 -m aegis_clip.cli.train \
  --config configs/f1_flat_mlp_lora_r2.yaml \
  --resume outputs/F1_FLAT_MLP_LORA_R2_FP32/seed42/checkpoints/last.pt
```

- 配置：`reproducibility/aegis_f1/configs/f1_flat_mlp_lora_r2.yaml`
- 父 checkpoint：`F1_FLAT_MLP_LORA_FP32` epoch 3，SHA-256
  `df6f385c9cc7c1c82d9a20bcf4792d408d11f07d0835f3ec56b7091b5aa2e2eb`
- 最终 checkpoint：epoch 3，SHA-256
  `fe901eb0cfed4368ce4ea68c8ccc83cba74d73fabffdd0938611b3145edbe3b5`
- 训练：seed 42、FP32、batch 32、head LR `5e-7`、MLP-LoRA LR `1e-5`、
  schedule epochs 12、固定 3 epochs
- 回归测试：`250 passed, 8 warnings`

## 训练记录

| 指标 | 起点 | epoch 1 | epoch 2 | epoch 3 |
|---|---:|---:|---:|---:|
| global raw micro | 71.4327% | 71.5006% | 71.5878% | 71.5587% |
| trusted macro | 81.2691% | 81.2894% | 81.4150% | 81.4361% |
| proxy macro | 80.1557% | 80.1649% | 80.2082% | 80.2200% |
| clean-core micro | 82.1034% | 82.1170% | 82.2534% | 82.2671% |
| flip agreement | 90.1706% | 90.2094% | 90.1609% | 90.3548% |

## 候选提交

```bash
python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_r2_fp32_ep3_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_crop --local-crop-size 160 --local-top-k 5 \
  --local-weight 0.4 --local-temperature 1.5 \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128
```

- 单模型、单 checkpoint；M1 crop160/top5/local0.40 + flip0.50，temperature
  1.5，balanced prior 0.85
- prediction count：24,967；classes：500；corrupt images：0
- 相对当前平台最佳 67.7815% 包改变预测：653（2.62%）
- CSV SHA-256：
  `9776b3b99b5295c3c2b5c326e86b2d431394d8492c421ea2f1c9d6c4e37012b9`
- ZIP SHA-256：
  `f3a2f9c8a58c0560f403ecdbfcec943c3946ac992f36a828afbea3364da0eb19`
- `aegis_clip.cli.audit_submission --allow-tta`：PASS
- 桌面副本 SHA-256 完全一致
- 平台得分：67.86558256899107%
- 平台状态：`platform_valid_promoted`（当前最佳，尚未达到 70%）
