# F1_FLAT_MLP_LORA_SELFTRAIN_R2_MULTISCALE_M1（2026-08-05）

## 结论

在平台最佳 `F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32`（68.133937%）的同一单
checkpoint 上，将原单一 160px attention crop 扩展为 144/160/176px 三尺度
确定性局部视图。每个尺度仍使用最后一层 CLS→patch attention 的 top-5 加权中心；
先对三尺度局部概率取均值，再按原有 local0.4、flip0.5、temperature1.5 与全局
原图/翻转分支融合，最后使用已验证的 balanced prior 0.85。

不引入第二模型、外部数据或测试时训练。验证集包含于完整训练集，诊断指标只用于
确认数值稳定，不设本地晋级门槛。平台实测 **68.32218528457564%**，比同一
checkpoint 的单尺度 68.13393679657148% 提升 **0.188248 个百分点**，成为新的
审计完整平台最佳，证明多尺度局部概率平均能在平台上转化为增益。

## 实现与验证

- 新增 `fuse_global_multilocal_flip_probabilities`，严格按“每尺度原图/翻转概率融合
  → 三尺度局部概率均值 → 全局/局部融合”的顺序计算。
- `infer` 新增显式 `attention_multiscale` 模式与 `--local-crop-sizes`，manifest
  记录完整尺度和风险确认。
- localization sweep 同步支持多尺度 stacked flip 诊断。
- 定向测试：`25 passed`。
- 全量测试首次出现一个与本次修改无关的随机浮点 `allclose` 瞬时失败；该单项立即
  复跑通过，随后全量复跑为 `264 passed, 8 warnings`。

固定诊断命令：

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.sweep_localization \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/multiscale_sweep_t15.json \
  --crop-sizes 144,160,176 --top-ks 5 --local-weights 0.4 \
  --include-horizontal-flip --flip-weights 0.5 --temperature 1.5 \
  --batch-size 128 --overwrite
```

| 固定诊断指标 | 单尺度 160 | 多尺度 144/160/176 | 变化 |
|---|---:|---:|---:|
| raw micro | 73.1582% | 73.1097% | -0.0485pp |
| trusted macro | 81.6709% | 81.6834% | +0.0125pp |
| proxy macro | 81.7209% | 81.7526% | +0.0318pp |
| clean-core micro | 83.7539% | 83.7812% | +0.0273pp |
| clean-core macro | 84.2426% | 84.2499% | +0.0073pp |

两者均覆盖 500 类；这些指标含训练重叠，不能解释为独立泛化提升。

## 候选提交

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_multiscale_fp32_ep3_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_multiscale --local-crop-sizes 144,160,176 \
  --local-top-k 5 --local-weight 0.4 --local-temperature 1.5 \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128 \
  --dump-logits outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_multiscale_fused_logits_ep3_l040_f050_t15.pt
```

- checkpoint SHA-256：
  `67efab2bf954139b59df074ccf00c0113cbc6ff96163d6e8d66ffbe553b910a4`
- prediction count：24,967；classes：500；corrupt images：0
- 相对 68.133937% 单尺度最佳包改变预测：804（3.22%）
- CSV SHA-256：
  `355e6046b01f7f6a7cbeb9ffe50fe8e61df7161701038a1946c38707cd6a9726`
- ZIP SHA-256：
  `eaf93105260340d073ab74aa91058acffcece4a28d2e49ac2d5317297762233c`
- manifest SHA-256：
  `2f535fb3cd17f7086d31b0e7048891fb7dd39aff9d46a5c26a0fbe7077804ffe`
- 提交审计：PASS；ZIP 只含根目录 `pred_results.csv`
- 桌面副本逐字节一致
- 平台得分：68.32218528457564%
- 相对单尺度 R2：+0.188248 个百分点
- 平台状态：`platform_valid_promoted`
