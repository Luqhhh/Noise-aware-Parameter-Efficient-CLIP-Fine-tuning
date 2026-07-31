# Balanced-Prior Alignment 平台结果（2026-07-31）

## 结论

在平台最佳 F1 REBUILD R1 + M1/Flip（`local0.40/flip0.50`）之上叠加
`align_logits_to_prior` 均衡先验校准（strength=0.25，IPF 类别偏置拟合），平台实测
**65.5786%**，新的审计完整平台最佳：

- 相对无校准的 M1+Flip 包（63.7802%）：**+1.7984pp**；
- 相对 F1 REBUILD R1 + M1 weight 0.35（62.9791%）：+2.5995pp；
- 相对已报告原 F1 + M1（63.3276%）：+2.2510pp；
- 距离 70 分：`4.4214pp`；
- local→platform gap 由 `8.35pp` 收窄至 `6.55pp`。

这验证了两个假设：
1. **平台测试类别均衡**（校准到 uniform 有效）；
2. **模型预测严重不均衡**（校准前测试集最差类仅 2 个预测、最好类 190 个，均衡期望
   ~50/类），校准纠正了类别偏置。

## 动机

模型在测试集上的 argmax 类别计数分布严重偏斜（`raw_argmax_count_min=2`,
`max=190`）。若平台测试 500 类均衡（官方 24,967 张 ≈ 50/类），把 soft 边际拟合到
uniform 先验应提升准确率。`align_logits_to_prior` 用迭代比例拟合（IPF）估计单一
类别偏置向量，`strength` 在原始模型（0）与完全拟合（1）之间插值；不改变模型参数，
只调整推理时的 logits。

## 包与审计

- 路径：
  `reproducibility/aegis_f1/outputs/F1_VISUAL_LORA_CLEAN_CORE_REBUILD_R1/seed42/submissions/m1_flip_l040_f050_pa025/submission.zip`
- checkpoint：F1 REBUILD R1（`805e0df7…6fb281`，与 63.7802 包同一 checkpoint）
- inference mode：
  `attention_crop_flip:topk=5:crop=160:local_weight=0.4:flip_weight=0.5:t=1:balanced_prior=0.25`
- prediction CSV SHA-256：
  `549d9b45b08353c00c5ed07f6b585ca13afcf99d4b75546d0f36c5a15ab603e0`
- submission ZIP SHA-256：
  `c0bbcee6e33b04f04411099a73ebd3f6444e81cc83a3fe47aae038dfa78af6d6`
- 相对无校准包：1,541 / 24,967 个预测改变（6.17%）
- `aegis-audit-submission --allow-tta`：passed（24,967 条、500 类）
- 平台：**65.5786%**，状态 `platform_valid_promoted`

## 对齐诊断

- `initial_marginal_l1`：0.3303（模型 soft 边际偏离 uniform 的 L1 距离）
- `final_marginal_l1`：0.2530（strength 0.25 后）
- `raw_argmax_count_min/max`：2 / 190（校准前）
- `aligned_argmax_count_min/max`：4 / 144（strength 0.25 后）
- `fitted_max_marginal_error`：5.2e-5（50 次迭代收敛）

## 下一步

- strength 0.25 已 +1.8pp，验证方向正确；扫描更高强度 `{0.5, 0.75, 1.0}` 判断
  是否进一步收窄 gap（见 `results/70p_campaign_20260731.md` P70-PA-002）。
- 此校准仅改变推理 logits，可与任何新 checkpoint 叠加，是独立于训练的信号。
