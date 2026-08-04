# F1_FLAT_MLP_LORA_SELFTRAIN_R1_FP32（2026-08-04）

## 结论

以平台新最佳 `F1_FLAT_MLP_LORA_FULLFIT_R1_FP32`（67.957704%）为固定教师，
在 103,218 张官方训练图上做原图/水平翻转双视图预测。只从原 CVT 信任包未纠正、
基础可信度低于 0.60 的样本中，接纳双视图一致且高置信的教师冲突，再继续训练
全 12 层 rank-4 MLP-LoRA 与分类头 3 个 epoch。attention-LoRA 继续冻结。

原验证划分已包含在父/子训练集中，所以所有本地指标仅用于数值健康检查；配置固定
采用 `last_epoch`，不设置本地晋级门槛。

平台实测 **68.1099050746986%**，比 full-fit 父模型 67.95770416950374%
提升 **0.152201 个百分点**，成为新的审计完整平台最佳。严格受限的 892 个教师
纠正带来了明确平台增益，后续优先沿教师监督与训练正则化方向推进。

## 教师信任包

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.build_teacher_trust \
  --checkpoint outputs/F1_FLAT_MLP_LORA_FULLFIT_R1_FP32/seed42/checkpoints/epoch_3.pt \
  --train-csv artifacts/stages/preliminary/final_full_train.csv \
  --base-trust artifacts/trust/cvt_v1.pt \
  --output artifacts/trust/fullfit_r1_teacher_v1.pt \
  --audit-output outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R1_FP32/teacher_trust_audit.json \
  --batch-size 128 --num-workers 4 --temperature 1.0 \
  --minimum-confidence 0.90 --minimum-margin 0.75 \
  --maximum-clean-probability 0.60 --admission-clean-probability 0.65 \
  --correction-alpha 0.50 --maximum-class-fraction 0.08
```

- 教师 checkpoint SHA-256：
  `868a6eef9aa7b4d1d1806aa20498e7d2d2d980ebd2496be5c568f7e62944209c`
- 基础 CVT 信任包 SHA-256：
  `5eb1624e5fff21d74df9458e2b56543398f850a4c53e6e7656f86475ac00b979`
- 教师信任包 SHA-256：
  `852f795daba2f9e98faa43338248d0884abddc7167f6964dd2a5512ebab6108a`
- 双视图一致：95,629；教师/噪声标签冲突：21,266
- 阈值合格：901；类别双向限额后新增纠正：892
- 平均接纳置信度：0.955662；平均 margin：0.934353
- 单一源类别最多 16，单一目标类别最多 17，配置上限最多 18
- 原 OOF 纠正 4,257 个全部保留，总纠正数 5,149
- 路径集合、唯一性、样本数和 canonical path 重排均为 fail-closed

## 谱系与训练

父/子训练集均为同一 103,218 张官方训练图，父/子诊断验证集均为同一 10,316
张；诊断验证集完整包含于训练集，标签冲突 0。继续 full-fit 需要显式
`allow_parent_train_in_child_val`，并且只有训练/验证划分完全不变时才可通过。

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train \
  --config configs/f1_flat_mlp_lora_selftrain_r1.yaml --overwrite
```

- 配置：`reproducibility/aegis_f1/configs/f1_flat_mlp_lora_selftrain_r1.yaml`
- 训练：seed 42、FP32、batch 32、head LR `2.5e-7`、MLP-LoRA LR
  `5e-6`、schedule epochs 18、固定 3 epochs
- 最终 checkpoint SHA-256：
  `7d9b12140572596b9acafcf037f8a3937b100c2aafd3bf46e6d7a2b572e1d38a`
- 完整回归测试：`262 passed, 8 warnings`

| 指标 | 父模型 | epoch 1 | epoch 2 | epoch 3 |
|---|---:|---:|---:|---:|
| raw micro | 71.9271% | 71.9465% | 72.1307% | 72.1694% |
| trusted macro | 81.0773% | 81.1410% | 81.3542% | 81.3831% |
| proxy macro | 80.9301% | 81.0374% | 81.1907% | 81.1714% |
| clean-core micro | 82.7582% | 82.8264% | 83.0446% | 83.0855% |
| flip agreement | 90.3160% | 90.3257% | 90.2772% | 90.2579% |
| mean feature drift | 0.009132 | 0.009228 | 0.009382 | 0.009473 |

以上指标包含训练重叠，不能解释为独立泛化提升。

## 候选提交

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R1_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r1.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r1_fp32_ep3_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_crop --local-crop-size 160 --local-top-k 5 \
  --local-weight 0.4 --local-temperature 1.5 \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128 \
  --dump-logits outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R1_FP32/seed42/test_fused_logits_ep3_l040_f050_t15.pt
```

- 单模型、单 checkpoint；测试集只用于推理，无测试时训练
- prediction count：24,967；classes：500；corrupt images：0
- 相对 67.957704% full-fit 父包改变预测：333（1.33%）
- CSV SHA-256：
  `e17e3ce798a62529b1fd76a7e2e01f87e77458de64c2cf51387c864f1c52382e`
- ZIP SHA-256：
  `1961f6ccbbabe6565aeb553544c4f2be582ca8897b0f880e3a8cdd5aa28ab3ad`
- `aegis_clip.cli.audit_submission --allow-tta`：PASS
- ZIP 只包含根目录文件 `pred_results.csv`
- 桌面副本逐字节一致
- 平台得分：68.1099050746986%
- 相对 full-fit 父模型：+0.152201 个百分点
- 平台状态：`platform_valid_promoted`
