# F1_FLAT_MLP_LORA_SELFTRAIN_R3_WIDE_FP32（2026-08-05）

## 结论

以当前平台最佳 R2 checkpoint（多尺度推理 68.322185%）为父模型与教师，在原
5,563 个纠正上新增 840 个原图/翻转一致的教师纠正。相对 R2 放宽教师阈值到
confidence 0.75、margin 0.45，但将新增纠正 alpha 降到 0.20，并保持源/目标类别
双向上限。固定续训 3 epochs；验证集包含于完整训练集，不设本地晋级门槛。

候选继续使用平台已验证有效的 144/160/176px 多尺度局部概率均值，不使用平台
回归的单 crop176。提交审计通过并已替换桌面，平台结果待回传。

## 教师缓存与信任包

`build_teacher_trust` 新增可复用教师 logits 缓存：首次运行原子保存原图/翻转
logits、标签、canonical paths、checkpoint/CSV SHA；复用时对字段、哈希、形状、
有限值与路径集合 fail-closed。实际复用将同一 103K 样本的阈值审计从约 6 分钟
降至 3 秒。

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.build_teacher_trust \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --train-csv artifacts/stages/preliminary/final_full_train.csv \
  --base-trust artifacts/trust/selftrain_r1_teacher_v2_relaxed.pt \
  --output artifacts/trust/selftrain_r2_teacher_v3_wide.pt \
  --audit-output outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R3_WIDE_FP32/teacher_trust_audit.json \
  --teacher-logits-cache outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/teacher_train_center_flip_logits.pt \
  --batch-size 128 --num-workers 4 --temperature 1.0 \
  --minimum-confidence 0.75 --minimum-margin 0.45 \
  --maximum-clean-probability 0.60 --admission-clean-probability 0.65 \
  --correction-alpha 0.20 --maximum-class-fraction 0.10
```

- 教师/父 checkpoint SHA-256：
  `67efab2bf954139b59df074ccf00c0113cbc6ff96163d6e8d66ffbe553b910a4`
- 教师 logits cache SHA-256：
  `ce834c802a8e02d27e8878a7539c0239d70682750e2e1f8272a0fe00151840bc`
- R3 信任包 SHA-256：
  `28ace0cf8eb6e7d2e22a654ba8c68bd467e68dbff7f82a907ddeeebfb6125547`
- 双视图一致：95,744；教师/噪声标签冲突：21,149
- 阈值合格：851；类别限额后新增：840；总纠正：6,403
- 新增平均 confidence：0.801023；平均 margin：0.709570
- 单一源类最多 16，单一目标类最多 21，配置上限最多 22

## 训练与审计

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train \
  --config configs/f1_flat_mlp_lora_selftrain_r3_wide.yaml --overwrite
```

- seed 42、FP32、batch 32、head LR `1.5e-7`、MLP-LoRA LR `3e-6`
- 固定 3 epochs；无 MixUp；attention-LoRA 冻结
- checkpoint、谱系、路径覆盖、样本数、标签一致性审计：PASS
- 定向测试：`8 passed`；全量测试：`264 passed, 8 warnings`
- 固定 epoch 3 checkpoint SHA-256：
  `9b94429139cc76e1ab5a007a58a175d815c177f1d35664b1ac9d673508cf8c89`

| 重叠诊断指标 | R2 / epoch 0 | epoch 1 | epoch 2 | epoch 3 |
|---|---:|---:|---:|---:|
| raw micro | 72.3342% | 72.4603% | 72.4312% | 72.4893% |
| raw macro | 72.3036% | 72.4276% | 72.3999% | 72.4575% |
| proxy macro | 81.8166% | 81.9324% | 81.9257% | 82.0259% |
| clean-core micro | 83.3038% | 83.4947% | 83.4811% | 83.5902% |
| clean-core macro | 83.9125% | 84.1164% | 84.0784% | 84.1971% |
| flip agreement | 90.5293% | 90.4905% | 90.5777% | 90.5487% |

以上指标含训练重叠，只用于数值健康检查。训练后按清理策略仅保留 epoch 3，删除
epoch0/1/2、best、last 重复 checkpoint，额外释放约 1.68 GiB。

## 候选提交

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R3_WIDE_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r3_wide.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r3_wide_multiscale_fp32_ep3_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_multiscale --local-crop-sizes 144,160,176 \
  --local-top-k 5 --local-weight 0.4 --local-temperature 1.5 \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128 \
  --dump-logits outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R3_WIDE_FP32/seed42/test_multiscale_fused_logits_ep3_l040_f050_t15.pt
```

- prediction count：24,967；classes：500；corrupt images：0
- 相对 68.322185% 多尺度 R2 最佳改变预测：150（0.60%）
- CSV SHA-256：
  `50ce4e27b7dc47fef43395b960e792c1d70593d39a74354c2dcd4eaa0ade836c`
- ZIP SHA-256：
  `795b4d91229425eb13be7d403a59db328de12f4e0c5d1bba622c6d5db2cade5d`
- manifest SHA-256：
  `b8a93f20a682be37bbff7951a4dd65f0e80f5e20056226b6fefb76a24880f1a2`
- 提交审计：PASS；ZIP 只含根目录 `pred_results.csv`
- 桌面副本逐字节一致
- 平台得分：待回传
- 平台状态：`platform_pending`
