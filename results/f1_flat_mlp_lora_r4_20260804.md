# F1_FLAT_MLP_LORA_R4_FP32（2026-08-04）

## 结论

从平台最佳 `F1_FLAT_MLP_LORA_R3_FP32` 固定 epoch-3 checkpoint 继续训练
3 个更低学习率 epoch。attention-LoRA 显式冻结，只更新全 12 层 rank-4
MLP-LoRA 与分类头。最终固定 `epoch_3.pt` 生成平台候选，不使用本地晋级门槛。

平台实测 **67.91765129971562%**，比 R3 最佳 67.92566187367325% 低
**0.008011 个百分点**。继续降低 MLP-LoRA 学习率未带来平台增益，平台最佳仍为
R3，该续训阶梯在 R4 关闭。

## 复现

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train \
  --config configs/f1_flat_mlp_lora_r4.yaml
```

- 配置：`reproducibility/aegis_f1/configs/f1_flat_mlp_lora_r4.yaml`
- 父 checkpoint：R3 epoch 3，SHA-256
  `bcb45ae4345ca17f3afc3b796edaa107a98970310c184fac727a8ffc38da8df9`
- 最终 checkpoint：R4 epoch 3，SHA-256
  `9480bcff9b4ad471b8baa53321466c95d8c82e25f802b0e4bab7b1327b9c4d91`
- 训练：seed 42、FP32、batch 32、head LR `1.25e-7`、MLP-LoRA LR
  `2.5e-6`、schedule epochs 24、固定 3 epochs
- 回归测试：`254 passed, 8 warnings`

## 训练记录

| 指标 | R3 起点 | epoch 1 | epoch 2 | epoch 3 |
|---|---:|---:|---:|---:|
| global raw micro | 71.5684% | 71.5781% | 71.6072% | 71.5975% |
| trusted macro | 81.4588% | 81.4671% | 81.4995% | 81.5249% |
| proxy macro | 80.2153% | 80.2320% | 80.2642% | 80.3123% |
| clean-core micro | 82.2807% | 82.3080% | 82.3353% | 82.3489% |
| flip agreement | 90.2385% | 90.2579% | 90.2966% | 90.3354% |

以上只用于确认数值健康和轨迹，不决定是否生成提交包。

## 候选提交

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_R4_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_r4.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_r4_fp32_ep3_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_crop --local-crop-size 160 --local-top-k 5 \
  --local-weight 0.4 --local-temperature 1.5 \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128 \
  --dump-logits outputs/F1_FLAT_MLP_LORA_R4_FP32/seed42/test_fused_logits_ep3_l040_f050_t15.pt
```

- 单模型、单 checkpoint；推理协议与 67.9257% R3 最佳包完全一致
- prediction count：24,967；classes：500；corrupt images：0
- 相对 R3 平台最佳包改变预测：209（0.84%）
- CSV SHA-256：
  `a94e6b82eeb0430fed83ad38f8fe0156792471e9c35ed765dcab49ee536c4727`
- ZIP SHA-256：
  `93e024d26892aedea2570a9c64d8dd4a71f3be3e96d0cc582cbc20ca8443bf6c`
- `aegis_clip.cli.audit_submission --allow-tta`：PASS
- 桌面副本 SHA-256 完全一致
- 平台得分：67.91765129971562%
- 平台状态：`platform_valid_not_promoted`
