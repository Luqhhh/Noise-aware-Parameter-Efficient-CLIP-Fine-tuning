# F1_FLAT_MLP_LORA_R3_FP32（2026-08-04）

## 结论

从平台最佳 `F1_FLAT_MLP_LORA_R2_FP32` 固定 epoch-3 checkpoint 继续训练
3 个更低学习率 epoch。attention-LoRA 继续冻结，只更新全 12 层 rank-4
MLP-LoRA 与分类头。使用最终 `epoch_3.pt` 生成平台候选，不使用本地晋级门槛。

平台实测 **67.92566187367325%**，比 R2 的 67.86558256899107% 提升
**0.060079 个百分点**，成为新的平台最佳。

## 复现

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train \
  --config configs/f1_flat_mlp_lora_r3.yaml
```

- 配置：`reproducibility/aegis_f1/configs/f1_flat_mlp_lora_r3.yaml`
- 父 checkpoint：R2 epoch 3，SHA-256
  `fe901eb0cfed4368ce4ea68c8ccc83cba74d73fabffdd0938611b3145edbe3b5`
- 最终 checkpoint：R3 epoch 3，SHA-256
  `bcb45ae4345ca17f3afc3b796edaa107a98970310c184fac727a8ffc38da8df9`
- 训练：seed 42、FP32、batch 32、head LR `2.5e-7`、MLP-LoRA LR
  `5e-6`、schedule epochs 18、固定 3 epochs
- 回归测试：`253 passed, 8 warnings`

## 训练记录

| 指标 | R2 起点 | epoch 1 | epoch 2 | epoch 3 |
|---|---:|---:|---:|---:|
| global raw micro | 71.5587% | 71.5587% | 71.5587% | 71.5684% |
| trusted macro | 81.4361% | 81.4056% | 81.4326% | 81.4588% |
| proxy macro | 80.2200% | 80.2345% | 80.2395% | 80.2153% |
| clean-core micro | 82.2671% | 82.2398% | 82.2671% | 82.2807% |
| flip agreement | 90.3548% | 90.2482% | 90.1803% | 90.2385% |

以上仅用于确认数值健康和轨迹，不决定是否生成提交包。

## 候选提交

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_R3_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_r3.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_r3_fp32_ep3_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_crop --local-crop-size 160 --local-top-k 5 \
  --local-weight 0.4 --local-temperature 1.5 \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128
```

- 单模型、单 checkpoint；推理协议与 67.8656% R2 包完全一致
- prediction count：24,967；classes：500；corrupt images：0
- 相对 R2 平台最佳包改变预测：388（1.55%）
- CSV SHA-256：
  `2897d01fabe0221cb39b73f7941314d30bad59dbed39ec202282592982e34e37`
- ZIP SHA-256：
  `a8144d9925082c72c232b3ea347d84828fdc66279ada981805a71a2f6a33db2d`
- `aegis_clip.cli.audit_submission --allow-tta`：PASS
- 桌面副本 SHA-256 完全一致
- 平台得分：67.92566187367325%
- 平台状态：`platform_valid_promoted`（当前最佳，尚未达到 70%）
