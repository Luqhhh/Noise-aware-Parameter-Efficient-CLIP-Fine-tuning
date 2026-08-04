# F1_FLAT_JOINT_LORA_R1_FP32（2026-08-04）

## 结论

从平台最佳父 checkpoint `F1_FLAT_MLP_LORA_R2_FP32` epoch 3 继续训练固定
3 epochs。新增的 `mlp_lora_train_attention` 开关只在本实验启用，使全 12 层
attention-LoRA 与全 12 层 rank-4 MLP-LoRA 联合更新；OpenAI CLIP 基础权重继续
冻结。最终固定 `epoch_3.pt` 生成平台候选，不使用本地晋级条件。

平台实测 **67.92165658669444%**，比 R3 最佳 67.92566187367325% 低
**0.004005 个百分点**。联合更新与 R3 基本持平但未晋级，attention-LoRA 联合
续训方向关闭，平台最佳仍为 R3。

## 实现与验证

- `visual_lora_mlp_lora` 默认行为不变；联合 attention-LoRA 训练为显式可选开关
- effective spec 记录开关状态，训练器验证实际可训练 attention 参数
- 真实图像 FP32 反向审计：1,067,508 个可训练参数；72 个 attention-LoRA 与
  48 个 MLP-LoRA 张量均获得有限非零梯度
- 提交生成前完整回归测试：`254 passed, 8 warnings`

## 复现

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train \
  --config configs/f1_flat_joint_lora.yaml --overwrite
```

- 配置：`reproducibility/aegis_f1/configs/f1_flat_joint_lora.yaml`
- 父 checkpoint：R2 epoch 3，SHA-256
  `fe901eb0cfed4368ce4ea68c8ccc83cba74d73fabffdd0938611b3145edbe3b5`
- 最终 checkpoint：epoch 3，SHA-256
  `f303f87f89f25e0770c43db1e565691d9e4c1931868863971ee8eec3ef776e5e`
- 训练：seed 42、FP32、batch 32、head LR `2.5e-7`、joint-LoRA LR
  `5e-6`、schedule epochs 18、固定 3 epochs

## 训练记录

| 指标 | R2 起点 | epoch 1 | epoch 2 | epoch 3 |
|---|---:|---:|---:|---:|
| global raw micro | 71.5587% | 71.5975% | 71.5587% | 71.6460% |
| trusted macro | 81.4361% | 81.4596% | 81.3921% | 81.5799% |
| proxy macro | 80.2200% | 80.2360% | 80.1610% | 80.3480% |
| clean-core micro | 82.2671% | 82.2807% | 82.1989% | 82.4035% |
| flip agreement | 90.3548% | 90.1609% | 90.3160% | 90.1512% |

以上只用于确认数值健康和训练轨迹，不决定是否生成提交包。

## 候选提交

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_JOINT_LORA_R1_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_joint_lora.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_joint_lora_r1_fp32_ep3_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_crop --local-crop-size 160 --local-top-k 5 \
  --local-weight 0.4 --local-temperature 1.5 \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128 \
  --dump-logits outputs/F1_FLAT_JOINT_LORA_R1_FP32/seed42/test_fused_logits_ep3_l040_f050_t15.pt
```

- 单模型、单 checkpoint；推理协议与 67.9257% R3 平台最佳包完全一致
- prediction count：24,967；classes：500；corrupt images：0
- 相对 R3 平台最佳包改变预测：451（1.81%）
- CSV SHA-256：
  `f9393e666c773e41376aebd96ceea081f3483ee5cd51a92de9c385d41e55ea74`
- ZIP SHA-256：
  `339040d932699e75131eaf9c5df651525b8ad54739d93c7132910cf1c9e2d8b4`
- `aegis_clip.cli.audit_submission --allow-tta`：PASS
- 桌面副本 SHA-256 完全一致
- 平台得分：67.92165658669444%
- 平台状态：`platform_valid_not_promoted`
