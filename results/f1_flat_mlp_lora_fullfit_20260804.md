# F1_FLAT_MLP_LORA_FULLFIT_R1_FP32（2026-08-04）

## 结论

从当前平台最佳 `F1_FLAT_MLP_LORA_R3_FP32` 的固定 epoch-3 checkpoint
继续训练 3 个 epoch。训练集是原训练划分和原验证划分的精确并集，共
103,218 张官方训练图；不使用外部数据，测试集仅用于最终推理。attention-LoRA
继续冻结，只更新全 12 层 rank-4 MLP-LoRA 与分类头。

由于原验证划分已并入训练，训练期间对该划分的所有指标都只是数值健康检查，
不具备独立验证意义，也不用于选 epoch。配置固定采用 `last_epoch`，最终由
`epoch_3.pt` 生成平台候选，不设置本地晋级门槛。平台结果待回填。

## 数据与谱系审计

- 父训练 CSV：92,902 行，SHA-256
  `a726b8a3ca8bc5857136106aca80f01d557104d3661ef92ccedfb2c0ea087875`
- 父验证 CSV：10,316 行，SHA-256
  `54a790b35f836cfba4c19cbb5fe38c4b1b37aab62cc9d477f9285496b2d5568e`
- 全训练 CSV：103,218 行、500 类、无重复，SHA-256
  `7643e120589e69d6bdc0c54abc605a7271d78c41aed1a638534ea43e4c0c4a90`
- 谱系条件：子训练集精确等于父训练集与父验证集的并集；子诊断验证集精确等于
  父验证集；标签冲突 0；`protocol_valid: true`
- 数据审计：官方训练样本 103,218，测试样本 24,967，外部数据 `false`，
  测试用途 `inference_only`
- `allow_parent_val_in_child_train` 是显式、封闭的 full-fit 权限：若不是上述精确
  并集或诊断验证集发生变化，谱系审计会失败。新增 3 个单元测试覆盖通过与拒绝路径。

全训练 CSV 是被忽略的派生制品，由仓库已有的确定性合并命令生成：

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.prepare_final_train \
  --train-csv artifacts/stages/preliminary/seed42/train.csv \
  --val-csv artifacts/stages/preliminary/seed42/val.csv \
  --output-csv artifacts/stages/preliminary/final_full_train.csv \
  --expected-samples 103218
```

该命令拒绝重复路径或行数不符，并按 `image_path` 排序；内容哈希记录在配置
审计和谱系审计输出中。

## 训练复现

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.audit \
  --config configs/f1_flat_mlp_lora_fullfit.yaml
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train \
  --config configs/f1_flat_mlp_lora_fullfit.yaml
```

- 配置：`reproducibility/aegis_f1/configs/f1_flat_mlp_lora_fullfit.yaml`
- 父 checkpoint：R3 epoch 3，SHA-256
  `bcb45ae4345ca17f3afc3b796edaa107a98970310c184fac727a8ffc38da8df9`
- 最终 checkpoint：full-fit epoch 3，SHA-256
  `868a6eef9aa7b4d1d1806aa20498e7d2d2d980ebd2496be5c568f7e62944209c`
- 训练：seed 42、FP32、batch 32、head LR `2.5e-7`、MLP-LoRA LR
  `5e-6`、schedule epochs 18、固定 3 epochs
- 回归测试：`257 passed, 8 warnings`

## 训练期健康记录

| 指标 | R3 起点 | epoch 1 | epoch 2 | epoch 3 |
|---|---:|---:|---:|---:|
| global raw micro | 71.5684% | 71.6266% | 71.8496% | 71.9271% |
| trusted macro | 81.4588% | 81.5138% | 81.7587% | 81.8744% |
| proxy macro | 80.2153% | 80.3182% | 80.5523% | 80.6244% |
| clean-core micro | 82.2807% | 82.3626% | 82.6217% | 82.7582% |
| flip agreement | 90.2385% | 90.1706% | 90.2966% | 90.3160% |
| mean feature drift | - | 0.008937 | 0.009051 | 0.009132 |

这些数值包含训练重叠，只证明训练稳定，不能解释为独立验证提升。

## 候选提交

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_FULLFIT_R1_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_fullfit.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_fullfit_r1_fp32_ep3_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_crop --local-crop-size 160 --local-top-k 5 \
  --local-weight 0.4 --local-temperature 1.5 \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128 \
  --dump-logits outputs/F1_FLAT_MLP_LORA_FULLFIT_R1_FP32/seed42/test_fused_logits_ep3_l040_f050_t15.pt
```

- 单模型、单 checkpoint；推理协议与 67.9257% R3 最佳包完全一致
- prediction count：24,967；classes：500；corrupt images：0
- 相对 R3 平台最佳包改变预测：406（1.63%）
- CSV SHA-256：
  `2ca1a0bf15106ba03d5b51b59385d90f4d89b5426b822f46d807d417bcfdb1cc`
- ZIP SHA-256：
  `43d9fc5d22cdaff6f4ebad5a1296674fbf97889ccba81d3adb8b9964fb70e7fd`
- `aegis_clip.cli.audit_submission --allow-tta`：PASS
- ZIP 只包含一个根目录文件 `pred_results.csv`
- 桌面副本逐字节一致，SHA-256 完全相同
- 平台得分：待提交
- 平台状态：`pending`
