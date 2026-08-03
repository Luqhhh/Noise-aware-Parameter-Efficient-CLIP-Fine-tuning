# 交付摘要（2026-08-03 15:50 起草）

## 保底提交包（可上传）
`/home/lux1/noise/outputs/delivery/current_best_a12corr/submission.zip`
- 模型：A12_CORR（LoRA 全 12 block + 伪标签修正）
- 协议：M1+flip（crop160/top5/l040/f050）+ temp1.5 + prior0.85
- **平台 67.6853%**（审计完整平台最佳）
- ZIP SHA-256 `b2b1924a…e0585`

## 本轮优化点（75p_round2）
| 候选 | 结果 |
|---|---|
| clip_letterbox 推理 | ❌ 全部指标下降（raw −5pp） |
| trust-aware 父模型（丢弃版） | ❌ clean 0.7835 < 0.8088 |
| trust-aware 父模型（连续加权版） | ⚠️ clean 0.8085 ≈ 噪声盲 0.8088，持平 |
| flat-LR 子模型（schedule 16） | ⏳ 训练中（ep1 0.8126 与 A12_CORR 一致） |

## 结论（待 flat 子模型完成后定稿）
- 信任机制在父阶段无明确收益（父模型主要价值在全部数据 + GCE + MixUp）。
- flat-LR 子模型是唯一可能有上行空间的候选；若 clean > 0.8183 → 生成其包。
- **75% 目标现实评估**：本地 clean-core 0.82 → 平台 67.7%（14pp 分布 gap）。
  需 clean-core ~0.89 才能到 75%，当前增量杠杆（±0.5pp clean）不足以达成。
  本轮最多预期 67.7 → ~68%。
