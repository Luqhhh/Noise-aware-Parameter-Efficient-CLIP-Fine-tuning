全微调父模型候选包（F1_FLAT_FULL_FT_R3MS，2026-08-06）
=====================================================

单一 checkpoint：reproducibility/aegis_f1/outputs/F1_FLAT_FULL_FT_R3MS/
seed42/checkpoints/best.pt（epoch 3，全视觉塔微调，conv1/pos emb 冻结，
GCE q0.5 + 特征蒸馏锚定 + trust 加权；本地 raw 79.23% / clean-core 91.94%）。

全部 24 个包来自同一次推理（四尺度 112/128/144/160 + flip + local0.4 +
temp1.5 + prior0.85），只离线改变尺度权重 / local 权重 / flip 权重 /
温度 / prior 强度。每个包都通过 9 项审计（24967 行 / 500 类）。

已测：fullft_r3ms_112_128_144_160_w020_030_040_010_l040_f050_t15_pa085.zip
= **69.575840%（17371/24967），新的平台最佳！**（+0.5007pp vs R3 dual 0.91）
当时距 70% 只差 106 个正确样本（后续双 Adapter 已突破 70%）。

文件名说明：
- fullft_r3ms_112_128_144_160_w020_030_040_010_l040_f050_t15_pa085.zip
  = 基线协议（与 R2/BN64+crop112 相同配方）
- fullft_r3ms_pa0.89 ~ pa1.0.zip = 只改 prior 强度
- scales_*.zip = 尺度权重 / local / flip / 温度变体

建议测试顺序（在 R3 双 Adapter prior 曲线测完后）：
1. fullft_r3ms_112_128_144_160_w020_030_040_010_l040_f050_t15_pa085.zip
   ✅ 已测 69.575840%（平台最佳）
2. scales_w020_030_040_010_l040_f050_t150_pa0.88.zip（prior 高侧）
3. scales_w020_030_040_010_l040_f050_t150_pa0.87.zip
4. scales_w020_030_040_010_l0.45_f050_t150_pa085.zip（local 权重微调）
5. scales_w020_030_040_010_l040_f0.6_t150_pa085.zip（flip 权重微调）
6. scales_w025_035_030_010_l040_f050_t150_pa085.zip（尺度权重）
7. scales_w020_030_040_010_l040_f050_t150_pa0.86.zip
8. scales_w020_030_040_010_l040_f050_t150_pa0.84.zip

所有包只来自一个模型 checkpoint，未做跨模型融合，符合赛规。
