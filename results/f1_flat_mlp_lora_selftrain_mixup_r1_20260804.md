# F1_FLAT_MLP_LORA_SELFTRAIN_MIXUP_R1_FP32（2026-08-04）

## 结论

这是当前平台最佳 `F1_FLAT_MLP_LORA_SELFTRAIN_R1_FP32` 的严格配对替代：使用同一
full-fit 父 checkpoint、同一教师信任包、同一 seed、学习率和固定 3-epoch 预算，
唯一训练策略差异是加入轻量 MixUp（`alpha=0.2`、`probability=0.2`）。模型仍为
单模型、单 checkpoint，测试集仅用于推理。

原验证划分包含在完整训练集中，因此以下本地指标只用于数值健康检查，不能解释为
独立泛化提升。配置固定选择 `last_epoch`，不设置本地晋级门槛，等待平台测试判定。

## 数据、信任包与谱系

- 官方训练图：103,218；诊断验证图：10,316；类别：500
- 外部数据：无；测试集：24,967 张，仅推理
- 教师信任包 SHA-256：
  `852f795daba2f9e98faa43338248d0884abddc7167f6964dd2a5512ebab6108a`
- 信任包总纠正：5,149（原 OOF 4,257 + 教师新增 892）
- 父 checkpoint SHA-256：
  `868a6eef9aa7b4d1d1806aa20498e7d2d2d980ebd2496be5c568f7e62944209c`
- 父/子训练图集合完全一致，父/子诊断验证集完全一致，标签冲突 0
- 数据审计、显式重叠谱系审计和 checkpoint 审计：PASS

## 训练

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train \
  --config configs/f1_flat_mlp_lora_selftrain_mixup_r1.yaml --overwrite
```

- 配置：`reproducibility/aegis_f1/configs/f1_flat_mlp_lora_selftrain_mixup_r1.yaml`
- seed 42、FP32、batch 32、head LR `2.5e-7`、MLP-LoRA LR `5e-6`
- 全 12 层 rank-4 MLP-LoRA 与分类头可训练；attention-LoRA 冻结
- schedule epochs 18，固定训练 3 epochs
- MixUp：`alpha=0.2`、`probability=0.2`
- 最终 checkpoint SHA-256：
  `8bbe51d4d9099fa460fecaf9253970d2e8dcc841bdbf1858e47ae44a0126e1e7`
- 完整回归测试：`262 passed, 8 warnings`

| 指标 | 父模型 | epoch 1 | epoch 2 | epoch 3 |
|---|---:|---:|---:|---:|
| raw micro | 71.9271% | 72.0240% | 72.1307% | 72.2373% |
| raw macro | - | 71.9917% | 72.0999% | 72.2095% |
| trusted macro | - | 81.2703% | 81.3654% | 81.5306% |
| proxy macro | - | 81.0987% | 81.1748% | 81.3326% |
| clean-core micro | 82.7582% | 82.9491% | 83.0582% | 83.1947% |
| clean-core macro | - | 83.5752% | 83.6681% | 83.8241% |
| flip agreement | 90.3160% | 90.1609% | 90.5487% | 90.4517% |
| mean feature drift | - | 0.009191 | 0.009426 | 0.009478 |

相对无 MixUp 的 self-training R1 epoch 3，本候选 raw micro +0.0679、clean-core
micro +0.1092、flip agreement +0.1938 个百分点；这些仍只是重叠诊断指标。

## 候选提交

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_MIXUP_R1_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_mixup_r1.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_mixup_r1_fp32_ep3_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_crop --local-crop-size 160 --local-top-k 5 \
  --local-weight 0.4 --local-temperature 1.5 \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128 \
  --dump-logits outputs/F1_FLAT_MLP_LORA_SELFTRAIN_MIXUP_R1_FP32/seed42/test_fused_logits_ep3_l040_f050_t15.pt
```

- prediction count：24,967；classes：500；corrupt images：0
- 相对当前 68.109905% self-training 最佳包改变预测：524（2.10%）
- CSV SHA-256：
  `c102c607fb81ed4e510d8de66646a3076212b322f0584d4aa1c5b2eb4daea28b`
- ZIP SHA-256：
  `4c1e7f2e749d6f615379eaf9e1a39aeeb0ca9d0c5b8da2d39a183fb1eb64d5fc`
- manifest SHA-256：
  `ede3a92ace7d6b39c4d0861b293bedaec20e7a233aed2d983bbf75b61c1553d0`
- `aegis_clip.cli.audit_submission --allow-tta`：PASS
- ZIP 只包含根目录文件 `pred_results.csv`
- 桌面副本逐字节一致
- 平台得分：待测试
- 平台状态：`platform_pending`
