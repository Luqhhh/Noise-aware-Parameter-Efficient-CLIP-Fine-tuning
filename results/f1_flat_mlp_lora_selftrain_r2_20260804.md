# F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32（2026-08-04）

## 结论

以平台最佳 `F1_FLAT_MLP_LORA_SELFTRAIN_R1_FP32`（68.109905%）为教师和父模型，
在第一轮 5,149 个纠正上增补 414 个较低一档但双视图一致的教师纠正。新增纠正
采用较低 `alpha=0.35`，类别双向上限 6%，随后以更低学习率固定续训 3 个 epoch。

诊断验证集包含于完整训练集，所有本地指标只用于数值健康检查；固定提交 epoch 3，
不设置本地晋级门槛。平台结果待回传。

## 教师信任包

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.build_teacher_trust \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R1_FP32/seed42/checkpoints/epoch_3.pt \
  --train-csv artifacts/stages/preliminary/final_full_train.csv \
  --base-trust artifacts/trust/fullfit_r1_teacher_v1.pt \
  --output artifacts/trust/selftrain_r1_teacher_v2_relaxed.pt \
  --audit-output outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/teacher_trust_relaxed_audit.json \
  --batch-size 128 --num-workers 4 --temperature 1.0 \
  --minimum-confidence 0.85 --minimum-margin 0.65 \
  --maximum-clean-probability 0.60 --admission-clean-probability 0.65 \
  --correction-alpha 0.35 --maximum-class-fraction 0.06
```

- 教师 checkpoint SHA-256：
  `7d9b12140572596b9acafcf037f8a3937b100c2aafd3bf46e6d7a2b572e1d38a`
- 基础信任包 SHA-256：
  `852f795daba2f9e98faa43338248d0884abddc7167f6964dd2a5512ebab6108a`
- R2 信任包 SHA-256：
  `ff8688a818219d3737715cd7dc9d11d014e7a5e973fa68f966c0ce370a88a246`
- 双视图一致：95,655；教师/噪声标签冲突：21,196
- 阈值合格：433；类别限额后新增纠正：414；总纠正：5,563
- 平均新增置信度：0.877128；平均 margin：0.823289
- 单一源类最多 6，单一目标类最多 13，配置上限 13
- 原 5,149 个纠正逐元素保留，新增纠正 alpha 均为 0.35

## 训练与审计

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml --overwrite
```

- 配置：`reproducibility/aegis_f1/configs/f1_flat_mlp_lora_selftrain_r2.yaml`
- seed 42、FP32、batch 32、head LR `1.5e-7`、MLP-LoRA LR `3e-6`
- 固定 3 epochs；无 MixUp；attention-LoRA 冻结
- 数据、路径覆盖、显式重叠谱系、标签一致性和 checkpoint 审计：PASS
- 相关前置测试：`31 passed`；全量测试：`262 passed, 8 warnings`
- checkpoint SHA-256：
  `67efab2bf954139b59df074ccf00c0113cbc6ff96163d6e8d66ffbe553b910a4`

| 指标 | 父模型 | epoch 1 | epoch 2 | epoch 3 |
|---|---:|---:|---:|---:|
| raw micro | 72.1694% | 72.1985% | 72.2082% | 72.3342% |
| raw macro | 72.1384% | 72.1680% | 72.1788% | 72.3036% |
| clean-core micro | 83.0855% | 83.1810% | 83.2356% | 83.3038% |
| clean-core macro | 83.6784% | 83.7917% | 83.8792% | 83.9125% |
| flip agreement | 90.2579% | 90.4614% | 90.4130% | 90.5293% |
| mean feature drift | 0.009473 | 0.009512 | 0.009632 | 0.009692 |

## 候选提交

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_fp32_ep3_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_crop --local-crop-size 160 --local-top-k 5 \
  --local-weight 0.4 --local-temperature 1.5 \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128 \
  --dump-logits outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_fused_logits_ep3_l040_f050_t15.pt
```

- 单模型、单 checkpoint；测试集只用于推理
- prediction count：24,967；classes：500；corrupt images：0
- 相对 68.109905% 最佳包改变预测：218（0.87%）
- CSV SHA-256：
  `36a92d0caa0ac9e3e9648816eafcf817c6aa053e3038647a690ddbb74edbfe0f`
- ZIP SHA-256：
  `93959e195590c09d9c3fb6251d7d6d6ffdd5eceb3b62b38dfccd2ba7ed50c40e`
- manifest SHA-256：
  `7ac34fc92e8dac21a37332d99f39a712d8fb8dd65c93b0afc540f000c6bf82d4`
- 提交审计：PASS；ZIP 只含根目录 `pred_results.csv`
- 桌面副本逐字节一致
- 平台状态：`platform_pending`
