# F1_FLAT_MLP_LORA_SELFTRAIN_R2_MULTISCALE_144_160（2026-08-05）

## 结论

在当前平台最佳 R2 三尺度候选（68.322185%）上做严格单变量消融：保留
144/160px 两个 attention crop，移除 176px 分支；checkpoint、top-5、local0.4、
flip0.5、temperature1.5、balanced prior 0.85 全部不变。

依据是单 crop176 平台仅 67.573197%，较三尺度下降 0.7490pp，而三尺度整体较
单 crop160 提升 0.1882pp。该候选直接检验增益是否主要来自 144/160 的互补，
不使用本地晋级条件。提交审计通过并已替换桌面，平台结果待回传。

## 候选提交

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_multiscale_144_160_fp32_ep3_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_multiscale --local-crop-sizes 144,160 \
  --local-top-k 5 --local-weight 0.4 --local-temperature 1.5 \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128 \
  --dump-logits outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_multiscale_144_160_fused_logits_ep3_l040_f050_t15.pt
```

- checkpoint SHA-256：
  `67efab2bf954139b59df074ccf00c0113cbc6ff96163d6e8d66ffbe553b910a4`
- prediction count：24,967；classes：500；corrupt images：0
- 相对三尺度 R2 平台最佳改变预测：517（2.07%）
- 相对单 crop160 R2 改变预测：759（3.04%）
- CSV SHA-256：
  `c0c09013809d479ca39dfe5f8d5cd248dab5bfb72aaea01a9278890b24f2a5be`
- ZIP SHA-256：
  `dbefd19aef90e6094b8e1c056f73cf8d20f33e15a6a8c8c0bd23255693d3b0ea`
- manifest SHA-256：
  `efb8671853689f9360863c881be647a716316c5a0fea612d09869915e0310b00`
- 提交审计：PASS；ZIP 只含根目录 `pred_results.csv`
- 桌面副本逐字节一致
- 平台得分：待回传
- 平台状态：`platform_pending`
