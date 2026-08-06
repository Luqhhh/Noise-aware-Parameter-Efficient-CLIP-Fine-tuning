# 提交包交付状态（2026-08-06 更新）

当前平台最佳：**70.35286578283333%**（17,565/24,967）—— **全微调父模型 +
双 Adapter prior 0.90！**（pa0.89 70.344855% → 70.352866%，+0.0080pp，
+2 正确样本；距 75% 还有 1,161 正确）
目标：**75%**（需 +1487 正确样本）

## 2026-08-06 新增平台记录
| 包 | 平台分数 | 结论 |
|---|---|---|
| R3 双 Adapter（prior 0.85） | 68.963031% | 与 BN64 完全打平（17,218/24,967）；not promoted |
| R3 双 Adapter（prior 0.86） | 68.983058% | 17,223/24,967，+5 correct vs 0.85；仍低于最佳；not promoted |
| R3 双 Adapter（prior 0.84） | 68.955021% | 17,216/24,967，-7 correct vs 0.85；先验最优在 0.86 以上；not promoted |
| R3 双 Adapter（prior 0.87） | 69.003084% | 17,228/24,967，+5 correct vs 0.86；单调上升；not promoted |
| R3 双 Adapter（prior 0.88） | 69.043137% | 17,238/24,967，+10 correct vs 0.87；与当前平台最佳持平；tied_best |
| **R3 双 Adapter（prior 0.90）** | **69.067169%** | **17,244/24,967，新平台最佳；+6 correct vs 0.88** |
| **R3 双 Adapter（prior 0.91）** | **69.075179%** | **17,246/24,967，新平台最佳；+2 correct vs 0.90** |
| R3 双 Adapter（prior 0.92） | 69.047142% | 17,239/24,967，-5 correct vs 0.90；峰值在 0.91 附近 |
| R3 双 Adapter（prior 0.94） | 69.059158% | 17,242/24,967，-4 correct vs 0.91；**用户指示停止本维度搜索** |
| **FULLFT_R3MS 基线** | **69.575840%** | **17,371/24,967，新平台最佳；+125 correct vs R3 dual 0.91** |
| **FULLFT_DUAL 基线** | **70.256739%** | **17,541/24,967，新平台最佳；+170 correct vs FULLFT_R3MS** |
| **FULLFT_DUAL pa0.89** | **70.344855%** | **17,563/24,967，新平台最佳；+22 correct vs 双 Adapter 基线** |
| FULLFT_DUAL pa0.91 | 70.308808% | 17,554/24,967，-9 correct vs pa0.89；峰值 ~0.89-0.90 |
| **FULLFT_DUAL pa0.90** | **70.352866%** | **17,565/24,967，新平台最佳；+2 correct vs pa0.89** |

## 桌面当前候选（2026-08-06 更新）
- 桌面 `submission.zip` = **全微调父模型 + 双 Adapter prior 0.92**
  `fullft_dual_pa0.92`（ZIP sha256 `7994ed58...`）
- 24 个 TTA 候选在 `桌面/fullft_r3ms_sweep/`，25 个双 Adapter 候选在
  `桌面/fullft_dual_sweep/`
- R3 双 Adapter prior 维度已停止（用户指示）；该家族最佳 = prior 0.91

## 平台验证记录
| 包 | 平台分数 | 提升 |
|---|---|---|
| 原版 | 68.902952% | — |
| BN64 | 68.963031% | +0.0601pp |
| **BN64+crop112** | **69.043137%** | **+0.0801pp（新最佳）** |
| BN128 | 69.003084% | −0.0401pp（未提升） |
| RS050 | 68.999079% | −0.0441pp（未提升） |

> **BN128**：bottleneck=64 是容量甜点，128 过拟合。
> **RS050**：residual_scale=0.5 过强有害。
> **验证方向**：BN64 adapter + 多尺度（含 crop112）。

## 平台验证记录（更新）
| 包 | 平台分数 | 结论 |
|---|---|---|
| 原版 | 68.902952% | 基线 |
| BN64 | 68.963031% | ✅ adapter 有效 |
| **BN64+crop112** | **69.043137%** | ✅ 当前最佳 |
| BN128 | 69.003084% | ⚠️ 平台>BN64（本地指标低估容量变体） |
| RS050 | 68.999079% | ⚠️ 平台>BN64（本地指标低估强度变体） |
| BN64 5尺度(含176) | 68.914968% | ❌ 176 有害（-0.13pp） |

> **Workflow 关键洞察（2026-08-05）**：
> 1. 本地 clean-core 指标**系统性偏差**——低估容量/强度变体（BN128、RS050 平台都>BN64）
> 2. **4尺度权重 0.20/0.30/0.40/0.10 可能非最优**——3尺度最优形状是 128 高/160 低
> 3. LW050 是正交新 knob，最高信息价值

## 桌面替换工作流
| 顺序 | 包 | 状态 |
|---|---|---|
| 1 | BN64 | ✅ 68.963031% |
| 2 | BN64+crop112 | ✅ 69.043137%（最佳） |
| 3 | BN128 | ⚠️ 69.003084% |
| 4 | RS050 | ⚠️ 68.999079% |
| 5 | BN64 5尺度 | ❌ 68.914968% |
| 6 | LW050 | ❌ 68.914968%（用户两次回传确认） |
| 7 | BN64 权重变体 w025_035_030_010（168差异）/ w030_040_020_010（350差异） | ✅ 已生成待测 |

> **所有 adapter 微调变体（BN128/RS050/LW050/5scale）均未超越 BN64+crop112**。
> Workflow 结论：微调方向边际耗尽，真正杠杆是父模型重构 + TTA 重调。

## 战略方向（Workflow synthesis, 2026-08-05）
| 优先级 | 动作 | 预期 | 状态 |
|---|---|---|---|
| P1 | B: R3 双 adapter（O3 LFA + PTA BN64） | 直接作用 R3 父 | 待执行 |
| P2 | A: 父模型全微调 + TTA 重调 | +1~3pp（唯一多pp杠杆） | 代码待扩展 |
| P3 | C: 全量 103K adapter | 检验噪声假设 | 待执行 |
| P4 | D: 软非均匀 prior | 免费诊断 | 待执行 |

## 已就绪候选包

### Wave A：part-token crop112，prior 强度 sweep（8 个）
基于原版 checkpoint（`test_weighted_multiscale_logits.pt`，128/144/160 权重 0.45/0.50/0.05）
| prior | 目录 | 定位 |
|---|---|---|
| 0.80 | `parttoken_crop112_prior_0.8` | 弱对齐 |
| 0.82 | `parttoken_crop112_prior_0.82` | 弱对齐 |
| 0.84 | `parttoken_crop112_prior_0.84` | 略弱 |
| 0.85 | `parttoken_crop112_prior_0.85` | **对照（应复现 68.90%）** |
| 0.86 | `parttoken_crop112_prior_0.86` | 略强 |
| 0.88 | `parttoken_crop112_prior_0.88` | 强对齐 |
| 0.90 | `parttoken_crop112_prior_0.90` | 更强 |
| 0.92 | `parttoken_crop112_prior_0.92` | 近均匀 |

### Wave C：part-token crop112 + crop112 尺度，prior sweep（8 个）
基于 crop112 加入 4 尺度推理（112/128/144/160，权重 0.20/0.30/0.40/0.10）
与当前 best 有 384 个预测差异
| prior | 目录 | 定位 |
|---|---|---|
| 0.80 | `parttoken_crop112_scales112_128_144_160_prior_0.8` | 弱对齐 |
| 0.82 | `parttoken_crop112_scales112_128_144_160_prior_0.82` | 弱对齐 |
| 0.84 | `parttoken_crop112_scales112_128_144_160_prior_0.84` | 略弱 |
| 0.85 | `parttoken_crop112_scales112_128_144_160_w_` | **主包（应复现/改善 68.90%）** |
| 0.86 | `parttoken_crop112_scales112_128_144_160_prior_0.86` | 略强 |
| 0.88 | `parttoken_crop112_scales112_128_144_160_prior_0.88` | 强对齐 |
| 0.90 | `parttoken_crop112_scales112_128_144_160_prior_0.9` | 更强 |

> 注：`parttoken_crop112_scales_112_128_144_160_w_`（主包，prior 0.85，zip sha256 `087038bcac7167...`）与
> `parttoken_crop112_scales112_128_144_160_prior_0.85`（sweep 派生，zip sha256 `5685e4069eda...`）
> 预测一致，只是 zip 哈希不同（manifest 差异）。用主包即可。

## 待生成

### Wave B：adapter 超参变体（本地指标更优）
| 变体 | 本地 clean delta | 目录 | 状态 | 与 best 预测差异 |
|---|---|---|---|---|
| **BN64** (bottleneck=64) | **+1.023pp** | `parttoken_crop112_bn64_multiscale_128_144_160_w045_050_005_fp32_l040_f050` | ✅ 就绪 | **812** |
| RS050 (residual_scale=0.5) | +0.955pp | `parttoken_crop112_RS050_multiscale_128_144_160_w045_050_005_fp32_l040_f050` | ✅ 就绪 | **361** |
| LW050 (local_loss=0.5) | +0.928pp | `parttoken_crop112_LW050_multiscale_128_144_160_w045_050_005_fp32_l040_f050` | ✅ 就绪 | **203** |

### Wave B3：BN64 + crop112 组合（✅ 就绪）
| 变体 | 目录 | 状态 | 与 best 差异 |
|---|---|---|---|
| BN64 + crop112 4尺度 | `parttoken_crop112_bn64_scales112_128_144_160_w020_030_040_010_fp32_l040_f050` | ✅ 就绪 | **874** |

> 组合包与 best 有 874 差异（最多），结合了 BN64 adapter 与 crop112 尺度的独立贡献
> （vs BN64 标准 369 差异，vs WaveC 主包 758 差异）。zip sha256 `06d1972f95c69562...`

### Wave B2：BN64 prior 强度 sweep（5 个，已就绪）
基于 BN64 标准推理的 fused logits 离线派生
| prior | 目录 |
|---|---|
| 0.82 | `parttoken_crop112_bn64_prior_0.82` |
| 0.84 | `parttoken_crop112_bn64_prior_0.84` |
| 0.85 | `parttoken_crop112_bn64_prior_0.85` |
| 0.86 | `parttoken_crop112_bn64_prior_0.86` |
| 0.88 | `parttoken_crop112_bn64_prior_0.88` |

### Wave C2：BN64 + crop112 组合（待推理）

## 平台实测建议顺序

1. **先测 Wave A 的 0.85** 确认格式/流程无回归（应得 68.90%）
2. **Wave A 的 0.84 / 0.86** —— 相邻 prior 最可能微调出更高点
3. **Wave B 的 BN64**（标准协议）—— 隔离 adapter 增益，若平台更高则是最有希望的方向
4. **Wave C 的 0.85** —— 判断 crop112 尺度是否有效
