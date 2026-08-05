# F1_FLAT_MLP_LORA_SELFTRAIN_R2_CROP176（2026-08-05）

## 结论

在当前平台最佳 `F1_FLAT_MLP_LORA_SELFTRAIN_R2_MULTISCALE_M1`（68.322185%）
使用的同一单 checkpoint 上，进行严格单变量消融：把 144/160/176px 三尺度局部
概率平均改为单一 176px attention crop。其余 top-5、local0.4、flip0.5、
temperature1.5、balanced prior 0.85 全部不变。

不引入第二模型、外部数据或测试时训练。验证集包含于完整训练集，诊断指标只用于
确认数值稳定，不设本地晋级门槛。平台实测 **67.5731966195378%**，比多尺度
最佳 68.32218528457564% 下降 **0.748989 个百分点**，未晋级；桌面提交包已恢复
为多尺度最佳。这也再次确认重叠本地诊断不能替代平台反馈。

## 固定诊断

同一 localization sweep 中，176px 单尺度 stacked flip 的诊断结果为：

| 固定诊断指标 | 多尺度 144/160/176 | 单尺度 176 | 变化 |
|---|---:|---:|---:|
| raw micro | 73.1097% | 73.0903% | -0.0194pp |
| trusted macro | 81.6834% | 81.7623% | +0.0789pp |
| proxy macro | 81.7526% | 81.8006% | +0.0480pp |
| clean-core micro | 83.7812% | 83.8494% | +0.0682pp |
| clean-core macro | 84.2499% | 84.3395% | +0.0896pp |

两者均覆盖 500 类；这些指标含训练重叠，不能解释为独立泛化提升。

## 候选提交

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_crop176_fp32_ep3_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_crop --local-crop-size 176 --local-top-k 5 \
  --local-weight 0.4 --local-temperature 1.5 \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128 \
  --dump-logits outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_crop176_fused_logits_ep3_l040_f050_t15.pt
```

- checkpoint SHA-256：
  `67efab2bf954139b59df074ccf00c0113cbc6ff96163d6e8d66ffbe553b910a4`
- prediction count：24,967；classes：500；corrupt images：0
- 相对 68.322185% 多尺度最佳包改变预测：1,006（4.03%）
- CSV SHA-256：
  `f638ddf77ab83de9d9750103f12c1b7181e7695c2775c9a71e3e967c9328a295`
- ZIP SHA-256：
  `75899e05c1a533f1afb0e2d42cb25b9a1917897d472b93ec2b12d1f931df245e`
- manifest SHA-256：
  `fdd425980062d4d02f524bfdd53993e85a94f977b9cb5c87b35b0d6d164d4ac6`
- 提交审计：PASS；ZIP 只含根目录 `pred_results.csv`
- 平台得分：67.5731966195378%
- 相对多尺度最佳：-0.748989 个百分点
- 平台状态：`platform_valid_not_promoted`
