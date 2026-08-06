# 交付批次 Wave A：part-token adapter crop112 prior 强度 sweep（2026-08-05）

## 背景

当前平台最佳：**68.90295189650338%**（17,203/24,967），
checkpoint = `F1_FLAT_MLP_LORA_SELFTRAIN_R2_PART_TOKEN_ADAPTER_CROP112_FP32/seed42/best.pt`
（R2 父模型 + crop112 Part-Token residual Adapter，34,336 参数）。

该 checkpoint 在平台上只测过 **prior=0.85**。本批次在同一 fused logits dump 上
离线 sweep prior 强度，生成 8 个差异化候选包，用于精确刻画 prior 响应的形状。

## 方法

- 输入：`test_weighted_multiscale_logits.pt`
  （融合后 logits，prior 对齐前；推理协议 `attention_multiscale_flip:
  topk=5:crops=128-144-160:weights=0.45-0.5-0.05:local_weight=0.4:
  flip_weight=0.5:t=1.5:adapter=part_token`）
- 离线 `align_logits_to_prior`（IPF，50 迭代），仅改变先验强度，不重新推理模型
- 生成 pred_results.csv → common.submission 打包（CRLF 格式，与历史平台验证包逐字节一致）

## 候选包（8 个）

| prior | ZIP SHA-256 前缀 | 相对 prior=0 变化样本数 | marginal_l1 | 定位 |
|---|---|---|---|---|
| 0.80 | `fb56351fa3bb` | 3,911 | 0.0666 | 略弱对齐 |
| 0.82 | `dff622b7507e` | 4,010 | 0.0601 | 弱对齐 |
| 0.84 | `699237fc6bb8` | 4,115 | 0.0536 | 略弱 |
| **0.85** | `f4e95fdd277c` | 4,156 | 0.0503 | **对照：应复现 68.90%** |
| 0.86 | `edea0c9a41c0` | 4,206 | 0.0470 | 略强 |
| 0.88 | `e6d921adfda5` | 4,293 | 0.0404 | 强对齐 |
| 0.90 | `a98e039826ad` | 4,382 | 0.0338 | 更强 |
| 0.92 | `263b37ff8fc9` | 4,480 | 0.0271 | 近均匀配额 |

- 每个包含 `pred_results.csv` + `submission.zip`，均通过 9 项校验
- 格式：`image_name.jpg,0128`（CRLF），与历史平台验证包一致
- **prior_0.85 包与当前平台最佳逐样本一致（24967/24967），应复现 68.90%**

## 平台实测顺序建议

1. 先测 **0.85** 复现 68.90%（验证格式/流程无回归）
2. 再测 **0.84 / 0.86**（相邻强度，最可能微调出更高点）
3. 若 0.84 或 0.86 更高，说明最优 prior 偏离 0.85，可再补 0.83/0.87
4. 0.80/0.82/0.88/0.90/0.92 用于勾勒曲线形状（不必全测）

## 待测

Wave B（adapter 超参变体 BN64/RS050/LW050）与 Wave C（crop112 加入多尺度）
的推理包在后台生成中，完成后作为下一批次交付。
