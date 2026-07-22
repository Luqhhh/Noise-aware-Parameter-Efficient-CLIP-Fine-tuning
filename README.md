# Noise-Aware Parameter-Efficient CLIP Fine-Tuning

面向噪声标签数据的细粒度图像识别（500 类，~103K 训练图）。基于 CLIP ViT-B/32 冻结 backbone + 线性分类头，系统消融 head 类型、数据增强和标签噪声的影响，并实现部分解冻基础设施用于后续视觉特征微调。

## 已完成工作

### 1. Baseline 建立与优化

- **B0 回归**：复现原始 baseline，验证基础设施正确性（61.39% vs 61.38%）
- **Head 类型对比**：Linear Head (69.86%) 在所有学习率下均显著优于 Cosine Head (63.61%)，差距 ~6pp
- **学习率搜索**：5 个 lr × 3 个 wd = 15 trials，确认 lr=5e-3 最优，weight decay 无影响（冻结 backbone 场景下预期行为）
- **训练策略**：50 epochs + Cosine LambdaLR（保持参数组 LR 比例）+ 早停（patience=10），替代原始 20 epoch 固定训练

### 2. 数据增强消融

系统测试 4 级增强预设（A0-A3），结论：**在细粒度 + 噪声标签任务上，所有数据增强均无正面收益**。

| 预设 | 内容 | 最佳结果 |
|------|------|---------|
| A0 | CLIP 标准预处理（无增强） | **69.86%** |
| A1 | RandomResizedCrop + RandomHorizontalFlip | 69.77% (lr=5e-3 对齐对照) |
| A2 | + ColorJitter | 67.36% (平台期截断) |
| A3 | + RandomErasing | 未完成（已弃用） |

A1 在匹配学习率后与 A0 几乎持平（Δ = −0.09pp），A2 的 ColorJitter 显著破坏细粒度判别信息。最终采用 A0。

### 3. 跨类别重复图像处理

扫描发现训练集中存在 1,032 组跨类别 SHA-256 完全重复，涉及 2,095 张图片（2.0%）——同一张图被放入 2-4 个不同类别目录，训练监督彼此矛盾。

实现 **CLIP 特征质心仲裁去重**：
- 从 101,123 张非冲突图片计算每类 10% trimmed 质心（排除离群错标）
- 三条件接受：全局 Top-1 必须在候选类别中 + 绝对相似度 ≥ p10 阈值 + Top1-Top2 margin ≥ 0.02
- 203/1032 组高置信度仲裁，829 组低置信度整组移除
- 最终过滤 1,892 张图片

**历史探索结果**：旧独立 split 上 D3 达到 70.53%（+0.67pp vs 旧 E0），但因验证集与 E0 不共享，该 +0.67pp 不作为严格消融证据。当前严格 paired delta 等待 E0_STRICT 干净重跑完成后计算。

### 4. 部分解冻基础设施

为后续视觉特征微调（F0-F3）实现了完整的基础设施：

- **CLIPLinearClassifier 选择性解冻**：始终先冻结全部 visual 参数，再按 `unfreeze_last_n_blocks` + `train_ln_post` + `train_visual_proj` 精确解冻
- **判别式优化器**：head 和 backbone 使用不同 LR / weight decay，通过 `get_param_groups()` 暴露
- **比例 LambdaLR Scheduler**：替代 CosineAnnealingLR，保证 backbone_lr / head_lr 比例全程恒定
- **`--init-checkpoint` CLI**：仅加载模型权重（不恢复 optimizer/scheduler/epoch），用于从冻结 baseline checkpoint 启动部分解冻实验
- **训练诊断**：每 epoch CSV 输出 head_lr、backbone_lr、head_grad_norm、backbone_grad_norm

### 5. 测试覆盖

团队根测试当前收集 405 项；Aegis 隔离实验线当前有 **201 项测试并全部通过**。主要覆盖包括：
- `test_partial_unfreeze.py`（16 tests）：参数冻结/解冻、train mode 行为
- `test_discriminative_optimizer.py`（11 tests）：参数组结构、LR/WD 正确性、覆盖率
- `test_init_checkpoint.py`（4 tests）：权重加载、跨架构兼容、requires_grad 保持
- `test_scheduler_ratio.py`（7 tests）：余弦因子边界、比例保持、缓存 guard
- `test_run_artifact_guard.py`（7 tests）：fresh-run 产物保护、resume 放行、--allow-overwrite
- `test_best_checkpoint_post_eval.py`（4 tests）：best.pt 重载、strict load 校验
- `test_metric_consistency.py`（7 tests）：micro-macro gap 一致性、bottom-10% 计算
- `test_submission_manifest.py`（18 tests）：SHA-256 哈希、ZIP vs CSV hash 区分、标签格式、预测计数、重复登记拒绝、manifest schema
- Aegis 独立套件：配置合规、LoRA/AdaptFormer/visual prompt、OOF 重建、局部推理、M1/M3、多类噪声诊断、Q1 trajectory、T0/T1 可信梯度子空间与 U0 数字 Prompt 审计；最新隔离整合回归 `201 passed`

### 6. 平台结果总览（updated 2026-07-22）

**Top TTA 分数：**

| 实验 | 平台 TTA | vs ref (D3) | 推理策略 |
|------|---------|-------------|----------|
| **AEGIS F1 + M1 attention-local/global** | **63.33%** | **+5.99pp** | center + attention-local，1:1 概率均值 |
| **A2 + M1 attention-local/global** | **62.67%** | **+5.33pp** | center + attention-local，1:1 概率均值 |
| **A2 + M3 complementary fusion** | **62.03%** | **+4.69pp** | Flip 分支 + M1 分支，1:1 概率均值 |
| **NR_CL_KNN_DROP (A2, kNN consensus drop, seed=42)** | **61.21%** | **+3.87pp** | 2-view Flip TTA |
| **A2 STRICT (A2 parent + LoRA, lineage-fixed, seed=42)** | **61.15%** | **+3.81pp** | Flip mean-prob T=0.5 |
| AEGIS F1 (visual LoRA, clean≥0.7, distill) | 61.10% | +3.76pp | Flip mean-prob T=0.5 |
| s_oof_zero_0001_ff (OOF zero p<0.001, final_fit) | 60.51% | +3.17pp | 2-view Flip TTA |
| S_MIXUP_CE5 (CE5 warmup + MixUp + GCE q=0.5) | 60.48% | +3.14pp | 2-view Flip TTA |
| w1_gce05_mixup (MixUp + GCE q=0.5) | 60.36% | +3.02pp | 2-view Flip TTA |
| **NR_CL_KNN_DROP (A2, seed=3407)** | **60.31%** | **+2.97pp** | 2-view Flip TTA |
| nr_ctrl_fixed (A0, reject_policy=drop) | 60.31% | +3.03pp | 2-view Flip TTA |
| s_oof_zero_0001 (OOF zero-weight p<0.001) | 60.28% | +2.94pp | 2-view Flip TTA |
| w1_ce5_gce05 (CE5 warmup + GCE q=0.5) | 60.25% | +2.91pp | 2-view Flip TTA |
| robust_lora (LoRA rank=8, last_block) | 60.24% | +2.90pp | 2-view Flip TTA |
| b2_gce05 (纯 GCE q=0.5) | 60.16% | +2.82pp | 2-view Flip TTA |
| s_oof_zero_001 (OOF zero-weight p<0.01) | 59.92% | +2.58pp | 2-view Flip TTA |
| **NR_CONSENSUS_RELABEL_V2 (A3, 5-signal relabel 100)** | **59.89%** | **+2.55pp** | 2-view Flip TTA |
| **NR_CL_CLASSWISE_DROP (A1, classwise drop 8680)** | **59.55%** | **+2.21pp** | 2-view Flip TTA |

**Top Bare 分数：**

| 实验 | 平台 Bare | vs ref (D3) | 推理策略 |
|------|---------|-------------|----------|
| **A2 STRICT (A2 parent + LoRA, lineage-fixed, seed=42)** | **60.65%** | **+3.31pp** | 单视图 |
| **A2 STRICT (A2 parent + LoRA, lineage-fixed, seed=3407)** | **60.64%** | **+3.30pp** | 单视图 |
| **AEGIS F1 (visual LoRA, clean≥0.7, distill)** | **60.52%** | **+3.18pp** | 单视图 |
| s_oof_zero_0001_ff (OOF zero p<0.001, final_fit) | 60.29% | +2.95pp | 单视图 |
| s_oof_zero_0001 (OOF zero-weight p<0.001) | 59.96% | +2.62pp | 单视图 |
| nr_ctrl_fixed (A0, reject_policy=drop) | 59.90% | +2.56pp | 单视图 |
| w1_gce05_mixup (MixUp + GCE q=0.5) | 59.86% | +2.52pp | 单视图 |
| s_d3_mixup (GCE q=0.5 + MixUp, d3 control) | 59.86% | +2.52pp | 单视图 |

**Noise-Robust 消融矩阵（Wave A）：**

| 实验 | 操作 | 样本 | Bare | TTA | vs A0 TTA | 判定 |
|------|------|------|------|------|-----------|------|
| A0 `NR_CTRL_FIXED` | p<0.001 零权重 | 6,354 (7%) | 59.90% | 60.31% | — | 对照基线 |
| **A2** `NR_CL_KNN_DROP` | 三方共识删除 | 991 (1.1%) | — | **61.21%** | **+0.90** | ✅ 最佳冻结 |
| A3 `NR_CONSENSUS_RELABEL` | 5-signal relabel | 100 (0.1%) | — | 59.89% | −0.42 | ❌ 关闭 |
| A1 `NR_CL_CLASSWISE_DROP` | CL classwise 删除 | 8,680 (9.5%) | — | 59.55% | −0.76 | ❌ 关闭 |

> **注意**：A1 和 A3 均在全局黑名单引入前完成训练，即 991 个三方共识确认错标样本仍以 weight=1.0 参与训练。但即使补上黑名单，relabel 100 个或低精度删除 8680 个带来的边际增益也大概率淹没在噪声里。

**多 seed 稳定性：**

| 实验 | seed | 本地 Val | 平台 Bare | 平台 TTA | Paired Delta vs s42 |
|------|------|----------|----------|----------|---------------------|
| A2 | 42 | 69.44% | — | **61.21%** | — |
| A2 | 3407 | 69.39% | 59.81% | **60.31%** | −0.07pp (p=0.457) |
| A2 STRICT | 42 | 69.64% | **60.65%** | **61.15%** | — |
| A2 STRICT | 3407 | — | **60.64%** | — | −0.01pp |

> A2 多 seed 稳定性确认：本地 paired delta 仅 7 张图差异（p=0.457），但平台 TTA 波动达 0.90pp。A2 STRICT LoRA 双 seed Bare 仅差 0.01pp（60.65% vs 60.64%），确认 LoRA 增益高度稳定。所有后续实验必须跑双 seed 验证。

**基线定义：**
- **平台 Bare 最佳**：A2 STRICT = **60.65%**（A2 parent + visual LoRA rank-8, clean filter, distill）
- **平台多视图推理最佳**：F1 + M1 = **63.3276%**（F1 visual LoRA + attention-local/global probability fusion）
- **平台普通 Flip TTA 最佳**：A2 seed=42 = **61.2128%**（frozen CLIP + GCE q=0.5 + MixUp + kNN consensus drop）
- **最佳冻结合理期望**：约 60.76% TTA（A2 两 seed 平均）
- **训练基线**：s_d3_mixup (GCE q=0.5 + MixUp, d3_strict) —— 所有 OOF 实验的配对对照

**核心发现（2026-07-22 修订）：**
- **attention-local/global 是当前最强跨模型信号**：M1 相对 F1 Flip 提升 +2.2269pp，相对 A2 Flip 提升 +1.4619pp；F1 + M1 达到 63.3276%。
- **更多视图不等于更好**：A2 + M3 平台 62.0259%，比纯 A2 + M1 低 0.6488pp。带噪本地排序不能代替平台验证，M3 只保留作消融。
- **Purification 精度 > 覆盖面**：删 991 个高精度样本 > 删 6354 个中精度 > 删 8680 个低精度。精度碾压数量。
- **删除 > 重标**：A3 五信号共识 relabel 100 个样本（0.1%）反而有害（−0.42pp）。OOF 预测准确率 ~69% 不足以支撑可靠重标。当你确定标签错了但不确定正确答案时，删除比重标更安全。
- **冻结 CLIP + GCE + MixUp 上限已触达**：A0→A2 本地 paired delta 仅 +17 张图（0.165pp, p=0.196），平台天花板 ~60.5-61% TTA。Purification 的边际增益已饱和。
- **单 seed 不可靠**：A2 seed=42 TTA 61.21% vs seed=3407 TTA 60.31% = 0.90pp 波动。所有候选必须在 seed=3407 上验证后才能宣称收益。
- **本地 val 与平台持续反相关**：A3 本地最高（69.47%）平台最差（59.89%）。本地分数不能用于模型选择。
- **表示适配与细粒度局部推理互补**：AEGIS F1 证明干净监督下的 visual LoRA 能贡献 bare 增益；M1 又在 F1 上获得比 A2 更大的平台提升，说明局部细节视图与 LoRA 表示适配存在正协同。
- **Split-lineage protocol 至关重要**：原始 A2 parent swap 因 parent (d3_strict) 与 child (AEGIS prepare) 使用不同 split，导致本地 raw_micro 从真实 69.43% 假胀至 79.22%（+8.5pp 假信号）。修复后 epoch-0 baseline 精确匹配，证实验证必须与训练用同一 split。
- **A2 parent swap 确认成立**：双 seed promotion 通过，bare +0.14pp, TTA +0.05pp vs F1 E2 parent。方向正确但收益太小，不进参数搜索。
- **数字类别不能直接继承语义 Prompt 鲁棒性**：U0 固定数字 Prompt 的 raw/clean-core 仅 0.232648%/0.229854%，500 个文本方向的 90% 能量秩为 1；direct numeric shared-context CoOp 已关闭。该结论是 train/validation-only 本地审计，不是平台成绩，也不排除另行设计视觉原型锚定 soft token。

### 7. 本地评估与待完成

**已完成 (d3_strict, seed=42, reeval from best.pt):**

| Experiment | Local Micro | Local Macro | Best Epoch | Platform Bare | Platform TTA |
|---|---|---|---|---|---|
| **A2** `NR_CL_KNN_DROP` | 69.44% | 69.45% | 48 | — | **61.21%** |
| **A2** `NR_CL_KNN_DROP` seed=3407 | 69.39% | 69.40% | 43 | **59.81%** | **60.31%** |
| **A3** `NR_CONSENSUS_RELABEL_V2` | 69.47% | 69.47% | 40 | — | **59.89%** |
| **A1** `NR_CL_CLASSWISE_DROP` | 68.61% | 68.61% | 45 | — | **59.55%** |
| **A0** `nr_ctrl_fixed` (reject_policy=drop) | 69.33% | — | 50 | 59.90% | 60.31% |
| **A2 STRICT** `F1_VISUAL_LORA_CLEAN_CORE_A2_PARENT_STRICT` | 69.71% | 69.71% | 2 (raw) / 6 (clean_core) | **60.65%** | **61.15%** |
| **A2 STRICT** seed=3407 | 69.82% | 69.83% | 3 (raw) / 5-6 (clean_core) | 待提交 | — |
| s_d3_mixup（MixUp d3 control） | 69.47% | 69.47% | 40 | 59.86% | — |
| s_oof_zero_0001_ff（OOF p<0.001, final_fit） | — | — | — | **60.29%** | **60.51%** |
| s_oof_zero_0001（OOF zero-weight p<0.001） | 69.37% | 69.37% | 44 | 59.96% | 60.28% |
| s_oof_zero_0001（OOF p<0.001） | 69.37% | 69.37% | 44 | **59.96%** | 60.28% |
| s_oof_zero_001（OOF p<0.01） | 69.02% | 69.01% | 37 | 59.38% | 59.92% |
| s_oof_discrete（OOF 3-tier） | 68.65% | — | 41 | 59.28% | 59.28% |
| robust_oof_soft（OOF soft target distillation） | 69.29% | — | 37 | — | 59.87% |
| s_elr_base（GCE+MixUp+ELR） | 68.20% | 68.21% | 19 | 58.59% | 59.14% |

> ⚠️ **Important**: 本地 val 不能预测平台表现。A3 本地最高（69.47%）平台最差（59.89%）。所有模型选择必须以平台 Bare/TTA 为准，本地分数仅作辅助诊断。**单 seed 平台结果不可靠**（A2 两 seed TTA 差 0.90pp），所有候选必须在 seed=3407 上验证。

**已关闭方向**：Dropout、ColorJitter/RandomErasing、Cosine Head、Label Smoothing、Head EMA、EMA Loss、Prototype Weighting、CE 下部分解冻、Head-only EMA Teacher + Consistency、GCE q=0.9、4-view TTA、vertical flip、OOF 3-tier discrete weight、OOF relabel/pseudo-label（A3 5-signal 共识仍有害）、Classwise CL-only drop（A1 −0.76pp）、ELR（本地 −1.2pp vs OOF）、PEFT LN-tune（freeze_clip=true 模式下无增益）、Rejected 半监督回收、NR_COMBINED_CLEAN_CORE（Layer 2/3 均为负信号）

### 下一步

平台名额优先用于已经完成训练和审计的 **F2 + M1 → O1 + M1 → N3 + M1**。完整独立实验总账、状态和合规边界见 [`docs/aegis_independent_experiments_2026-07-22.md`](docs/aegis_independent_experiments_2026-07-22.md)。O3-R1、Q1A、R1 与 T0/T1 均尚未运行；其中 T0/T1 是严格配对的可信梯度子空间实验，只有在明确授权并确认不占用团队任务后才可启动。U0 已完成，不需要继续训练；它只关闭 direct numeric shared-context CoOp。

## 项目结构

```
├── common/              # 共享代码（dataset, cache, transforms, evaluation 等）
├── experiments/
│   ├── baseline/        # Linear Head 实验（train/evaluate/infer/model）
│   └── cosine/          # Cosine Head 实验（委托 baseline）
├── configs/             # 每个实验一个 YAML
├── scripts/             # 数据准备、超参搜索、去重仲裁、提交验证
├── tests/               # 团队根测试套件（当前收集 405 项）
├── reproducibility/     # 隔离的 Aegis 独立实验线（当前 201 项测试）
├── outputs/             # 实验结果（tracked in git，*.pt 忽略）
└── docs/superpowers/    # 设计文档与实施计划
```

## 关键技术细节

- **特征缓存**：一次性 CLIP 编码全量训练集（~30min），训练时直接读 `[B,512]` 特征（~3s/epoch vs 在线 ~140s/epoch）。缓存的 cache/hit 比训练本身更快
- **类别映射**：目录名字典序 → 0..499 索引，通过 `class_to_idx.json` / `idx_to_class.json` 在所有阶段复用
- **配置优先级**：CLI 显式指定 > YAML > 硬编码默认值（`runtime_config.py` 统一解析）
- **提交格式**：`submission.zip` 内含 `pred_results.csv`（`image_name.jpg, 0001`），9 项自动校验
- **早停机制**：`train.early_stop_patience: 10`，连续 N 个 epoch 无提升自动终止
- **指纹校验**：缓存与数据集通过 SHA-256 全量指纹匹配，防止特征-图片错位


## Git 策略

- ✅ 跟踪：`.json/.csv/.log/.yaml` 结果文件
- ❌ 忽略：`.pt` 检查点、`cache/`、`train/`、`train_dedup/`、`test/`
