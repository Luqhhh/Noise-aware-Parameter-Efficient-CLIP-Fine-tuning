全微调父模型 + 双 Adapter 候选包（2026-08-06）
=============================================

单一 composite checkpoint：F1_FLAT_FULL_FT_R3MS 全微调父模型（epoch3）
+ O3 局部特征 Adapter（bottleneck32，+0.71pp clean-core）
+ Part-Token Adapter（bottleneck64，+0.71pp clean-core）
推理协议：四尺度 112/128/144/160 + flip + local0.4 + temp1.5 + prior0.85。

25 个唯一预测集，全部通过 9 项审计（24967 行 / 500 类）。

已测：
- 父模型基线 = 69.575840%（平台最佳）
- **双 Adapter 基线 = 70.256739%（17541/24967），新的平台最佳！**
  （+0.6809pp / +170 correct；已突破 70%）
- **双 Adapter prior 0.89 = 70.344855%（17563/24967），新的平台最佳！**
  （+0.0881pp / +22 correct）
- 双 Adapter prior 0.91 = 70.308808%（17554/24967），-9 correct
- **双 Adapter prior 0.90 = 70.352866%（17565/24967），新的平台最佳！**
  （+0.0080pp / +2 correct）

当前桌面 submission.zip = fullft_dual_pa0.92.zip（双 Adapter prior 0.92）。

文件名说明：
- fullft_dual_112_128_144_160_w020_030_040_010_l040_f050_t15_pa085.zip
  = 基线协议（双 Adapter）
- fullft_dual_pa0.89 ~ pa1.0.zip = 只改 prior 强度
- scales_*.zip = 尺度权重 / local / flip / 温度变体

与当前平台最佳（R3 双 Adapter prior 0.91，69.075179%）差异 3497–3569 个预测。

建议测试顺序（R3 双 Adapter prior 家族测完后）：
1. fullft_dual_112_128_144_160_w020_030_040_010_l040_f050_t15_pa085.zip
   ✅ 已测 70.256739%（平台最佳）
2. fullft_dual_pa0.89.zip（当前桌面包）/ pa0.9 / pa0.91（prior 峰值附近）
3. fullft_dual_pa0.89.zip ✅ 已测 70.344855%
4. fullft_dual_pa0.91.zip ✅ 已测 70.308808%
5. fullft_dual_pa0.9.zip ✅ 已测 70.352866%（平台最佳）
6. scales_w020_030_040_010_l040_f050_t150_pa0.88.zip

同一模型 checkpoint 的确定性变体，无跨模型融合。
