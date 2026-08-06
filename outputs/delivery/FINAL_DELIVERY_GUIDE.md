# 最终交付指南（2026-08-05 23:00）

> 目标：平台 75%。当前最佳：**68.90295189650338%**（17,203/24,967）。
> 所有包格式：`image_name.jpg,0128`（CRLF），与历史平台验证包逐字节一致，通过 9 项校验。

## 快速测试顺序（最高信息价值优先）

| 优先级 | 包 | 定位 | 与 best 差异 |
|---|---|---|---|
| 1 | `parttoken_crop112_prior_0.85` | 原版对照 | 0（应复现 68.90%） |
| 2 | **`parttoken_crop112_bn64_multiscale_128_144_160_w045_050_005_fp32_l040_f050`** | 最强 adapter（bottleneck 64） | **812** |
| 3 | **`parttoken_crop112_bn64_scales112_128_144_160_w020_030_040_010_fp32_l040_f050`** | BN64 + crop112 组合 | **874** |
| 4 | `parttoken_crop112_RS050_multiscale_128_144_160_w045_050_005_fp32_l040_f050` | residual_scale 0.5 | 361 |
| 5 | `parttoken_crop112_LW050_multiscale_128_144_160_w045_050_005_fp32_l040_f050` | local_loss 0.5 | 203 |
| 6 | `parttoken_crop112_scales_112_128_144_160_w_` | crop112 加入 4 尺度 | 384 |
| 7 | `parttoken_crop112_prior_0.84` / `0.86` | prior 微调 | 48 / 54 |

## 全部候选包（25 个，含差异数）

### Wave B：adapter 超参变体（3 个）★ 最有希望
- `parttoken_crop112_bn64_multiscale_128_144_160_w045_050_005_fp32_l040_f050`（bottleneck 64）812
- `parttoken_crop112_RS050_multiscale_128_144_160_w045_050_005_fp32_l040_f050` 361
- `parttoken_crop112_LW050_multiscale_128_144_160_w045_050_005_fp32_l040_f050` 203

### Wave B3：BN64 + crop112 组合（推理中）
`parttoken_crop112_bn64_scales112_128_144_160_w020_030_040_010_fp32_l040_f050`

### Wave B2：BN64 prior sweep（5 个）
`parttoken_crop112_bn64_prior_{0.82,0.84,0.85,0.86,0.88}`（812-833）

### Wave C：crop112 4尺度 prior sweep（7 个）+ 主包（1 个）
`parttoken_crop112_scales112_128_144_160_prior_{0.8,0.82,0.84,0.85,0.86,0.88,0.9}`
`parttoken_crop112_scales_112_128_144_160_w_`（主包，384）

### Wave A：原版 crop112 prior sweep（8 个）
`parttoken_crop112_prior_{0.8,0.82,0.84,0.85,0.86,0.88,0.9,0.92}`（0-389）

## 关键洞察

1. **BN64 系最具差异性**（812-833 diffs）：adapter 容量是关键方向，平台测试首选
2. **BN64 与 crop112 互补**（887 样本互相不同）：组合包（Wave B3）值得测
3. **BN64 与 RS050 互补**（798 样本互相不同）：两个变体方向独立
4. **prior 0.85 是最优点**（0.84/0.86 仅差 48/54 样本）：prior 方向边际已尽

## 决策树

1. **BN64 平台 > 68.90%** → adapter 容量是方向，继续加大/搜索更多超参
2. **Wave C（crop112）> 原版** → crop112 尺度有效，优化权重
3. **BN64 + crop112 组合 > 单个方向** → 组合是正协同，深入组合空间
4. **所有 R2 变体 ≈ 68.90%** → R2 饱和，转 R3 基础（需重建缓存）
