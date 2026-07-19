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

322 个测试全部通过，新增：
- `test_partial_unfreeze.py`（16 tests）：参数冻结/解冻、train mode 行为
- `test_discriminative_optimizer.py`（11 tests）：参数组结构、LR/WD 正确性、覆盖率
- `test_init_checkpoint.py`（4 tests）：权重加载、跨架构兼容、requires_grad 保持
- `test_scheduler_ratio.py`（7 tests）：余弦因子边界、比例保持、缓存 guard
- `test_run_artifact_guard.py`（7 tests）：fresh-run 产物保护、resume 放行、--allow-overwrite
- `test_best_checkpoint_post_eval.py`（4 tests）：best.pt 重载、strict load 校验
- `test_metric_consistency.py`（7 tests）：micro-macro gap 一致性、bottom-10% 计算
- `test_submission_manifest.py`（18 tests）：SHA-256 哈希、ZIP vs CSV hash 区分、标签格式、预测计数、重复登记拒绝、manifest schema

### 6. 平台结果总览（updated 2026-07-19 15:00）

**Top TTA 分数：**

| 实验 | 平台分数 | vs ref (D3) | 推理策略 |
|------|---------|-------------|----------|
| **nr_ctrl_fixed (A0, reject_policy=drop)** | **60.31%** | **+3.03pp** | 2-view Flip TTA |
| s_oof_zero_0001_ff (OOF zero p<0.001, final_fit) | 60.51% | +3.17pp | 2-view Flip TTA |
| S_MIXUP_CE5 (CE5 warmup + MixUp + GCE q=0.5) | 60.48% | +3.14pp | 2-view Flip TTA |
| w1_gce05_mixup (MixUp + GCE q=0.5) | 60.36% | +3.02pp | 2-view Flip TTA |
| s_oof_zero_0001 (OOF zero-weight p<0.001) | 60.28% | +2.94pp | 2-view Flip TTA |
| w1_ce5_gce05 (CE5 warmup + GCE q=0.5) | 60.25% | +2.91pp | 2-view Flip TTA |
| b2_gce05 (纯 GCE q=0.5) | 60.16% | +2.82pp | 2-view Flip TTA |
| **robust_lora (LoRA rank=8, last_block)** | **60.24%** | **+2.90pp** | 2-view Flip TTA |
| s_oof_zero_001 (OOF zero-weight p<0.01) | 59.92% | +2.58pp | 2-view Flip TTA |
| robust_oof_soft (OOF soft target distillation) | 59.87% | +2.53pp | 2-view Flip TTA |

**Top Bare 分数：**

| 实验 | 平台分数 | vs ref (D3) | 推理策略 |
|------|---------|-------------|----------|
| **s_oof_zero_0001_ff (OOF zero p<0.001, final_fit)** | **60.29%** | **+2.95pp** | 单视图 |
| s_oof_zero_0001 (OOF zero-weight p<0.001) | 59.96% | +2.62pp | 单视图 |
| w1_gce05_mixup (MixUp + GCE q=0.5) | 59.86% | +2.52pp | 单视图 |
| s_d3_mixup (GCE q=0.5 + MixUp, d3 control) | 59.86% | +2.52pp | 单视图 |
| s_mixup_ce5 (CE5 warmup + MixUp) | 59.70% | +2.36pp | 单视图 |

**基线定义：**
- **平台 Bare 最佳**：s_oof_zero_0001_ff = **60.29%**（OOF zero p<0.001 final_fit，首个突破 60%）
- **平台 TTA 最佳**：s_oof_zero_0001_ff + Flip TTA = **60.51%**（首个突破 60.5%）
- **训练基线**：s_d3_mixup (GCE q=0.5 + MixUp, d3_strict) —— 所有 OOF 实验的配对对照

**核心发现：**
- **OOF zero-weight 有效**：p<0.001 阈值排除 7% 最低置信度样本，平台 Bare 59.96% (+0.10pp vs MixUp control)，本地低但平台反超——OOF 预测本身比本地 val 更可靠
- **阈值敏感**：p<0.01 排除 12% 样本效果更差（TTA 59.92% vs 60.28%），过度排除损失有用数据
- **OOF discrete（3-tier）无效**：TTA = Bare（59.28%），zero gain，已关闭
- CE warmup 本地 +3.65pp 但平台 Bare 完全持平（59.61% vs 59.62%），本地分数无法预测平台表现
- MixUp 是唯一将本地增益传递到平台的 baseline 方法
- Horizontal-flip TTA 持续提供 +0.3-0.5pp 平台增益
- 冻结 CLIP + 线性头框架下，平台天花板约 60-61%，目前最佳 Bare 59.96%

### 7. 本地评估与待完成

**已完成 (d3_strict, seed=42, reeval from best.pt):**

| Experiment | Local Micro | Local Macro | Best Epoch | Platform Bare | Platform TTA |
|---|---|---|---|---|---|
| w1_ce5_gce05（CE5 warmup） | 73.14% | 73.09% | 50 | 59.61% | 60.25% |
| s_peft/e2_ln_5e7 | 71.21% | — | 1 | — | — |
| w1_gce05_mixup（MixUp） | 71.16% | 71.12% | 46 | 59.86% | 60.36% |
| ref（D3_STRICT） | 70.66% | 70.61% | 49 | 57.34% | 58.31% |
| s_mixup_ce5（warmup+MixUp） | 70.25% | 70.19% | 43 | 59.70% | **60.48%** |
| gce_q07（B2_GCE07） | 69.59% | 69.53% | 41 | 58.96% | 59.41% |
| b2_gce05（GCE q=0.5） | 69.49% | 69.49% | 50 | 59.62% | 60.16% |
| s_d3_mixup（MixUp d3 control） | 69.47% | 69.47% | 40 | 59.86% | — |
| s_oof_zero_0001_ff（OOF p<0.001, final_fit） | — | — | — | **60.29%** | **60.51%** |
| nr_ctrl_fixed（A0 causal control, GCE+MixUp+OOF zero） | 62.81% | — | 4 | **60.24%** | — |
| robust_lora（LoRA rank=8 last_block, freeze_clip=false） | 69.40% | — | 1 | — | **60.24%** |
| s_oof_zero_0001（OOF p<0.001） | 69.37% | 69.37% | 44 | **59.96%** | 60.28% |
| s_oof_zero_001（OOF p<0.01） | 69.02% | 69.01% | 37 | 59.38% | 59.92% |
| s_oof_discrete（OOF 3-tier） | 68.65% | — | 41 | 59.28% | 59.28% |
| robust_oof_soft（OOF soft target distillation） | 69.29% | — | 37 | — | 59.87% |
| s_elr_base（GCE+MixUp+ELR） | 68.20% | 68.21% | 19 | 58.59% | 59.14% |

> ⚠️ **Important**: 本地 val 不能预测平台表现。OOF zero-weight 本地 69.37%（低于 MixUp 71.16%）但平台 Bare 59.96% 超越 MixUp 59.86%。所有模型选择必须以平台 Bare 为准，本地分数仅作辅助诊断。

**Platform Submissions (updated 2026-07-19):**

| Submission | Platform | vs ref | 推理 |
|------------|---------|--------|------|
| **AEGIS F1 + Flip mean-prob T=0.5** | **61.10%** | **+3.76pp** | 2-view Flip TTA（NEW BEST） |
| **AEGIS F1 bare** | **60.52%** | **+3.18pp** | 单视图（BEST BARE） |
| s_oof_zero_0001_ff + Flip TTA | **60.51%** | **+3.17pp** | 2-view Flip TTA |
| S_MIXUP_CE5 + Flip TTA | **60.48%** | **+3.14pp** | 2-view Flip TTA |
| w1_gce05_mixup + Flip TTA | 60.36% | +3.02pp | 2-view Flip TTA |
| s_oof_zero_0001 + Flip TTA | 60.28% | +2.94pp | 2-view Flip TTA |
| w1_ce5_gce05 + Flip TTA | 60.25% | +2.91pp | 2-view Flip TTA |
| b2_gce05 + Flip TTA | 60.16% | +2.82pp | 2-view Flip TTA |
| robust_lora + Flip TTA | 60.24% | +2.90pp | 2-view Flip TTA |
| s_oof_zero_0001 bare | **59.96%** | **+2.62pp** | 单视图（BEST BARE） |
| s_oof_zero_001 + Flip TTA | 59.92% | +2.58pp | 2-view Flip TTA |
| s_d3_mixup bare | 59.86% | +2.52pp | 单视图 |
| s_mixup_ce5 bare | 59.70% | +2.36pp | 单视图 |
| gce_q07 + Flip TTA | 59.41% | +2.07pp | 2-view Flip TTA |
| ref（D3_STRICT） | 57.34% | — | 单视图 |

> `AEGIS F1` 指 `AEGIS_F1_VISUAL_LORA_CLEAN_CORE`，与下文因验证泄漏而废弃的旧 `F1-strict` 无关。完整配置、合规说明和哈希见 `docs/AEGIS_F1_VISUAL_LORA.md`；可复现代码快照位于 `reproducibility/aegis_f1/`。

**已关闭方向**：Dropout、ColorJitter/RandomErasing、Cosine Head、Label Smoothing、Head EMA、EMA Loss、Prototype Weighting、CE 下部分解冻、Head-only EMA Teacher + Consistency、GCE q=0.9、4-view TTA、vertical flip、OOF 3-tier discrete weight、ELR（本地 -1.2pp vs OOF）、PEFT LN-tune（freeze_clip=true 模式下无增益）

## 项目结构

```
├── common/              # 共享代码（dataset, cache, transforms, evaluation 等）
├── experiments/
│   ├── baseline/        # Linear Head 实验（train/evaluate/infer/model）
│   └── cosine/          # Cosine Head 实验（委托 baseline）
├── configs/             # 每个实验一个 YAML
├── scripts/             # 数据准备、超参搜索、去重仲裁、提交验证
├── tests/               # 322 个 pytest 测试
├── reproducibility/     # 隔离的外部实验快照（含 AEGIS F1）
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


## Strict Validation Protocol (2026-07-12)

After discovering 88% validation leakage in F1 and 3.79% content leakage in the original seed 42 split, all baselines were rebuilt on a SHA-256 dedup-enabled master split.

| Experiment | Local Micro | Local Macro | Gap | Epochs | Status |
|---|---|---|---|---|---|
| E0-strict | 70.5409% | 70.5015% | 0.0394pp | 47/50 | valid (D3−E0 = +0.1163pp) |
| D3-strict | 70.6572% | 70.6100% | 0.0473pp | 49† | valid_seed42_pending_multiseed |
| F0-strict | 70.6378% | 70.5939% | 0.0439pp | 5† | control_complete_no_gain |
| F1-strict | 70.7832% | 70.7458% | 0.0374pp | 4† | below_gain_threshold |

† Early stopped (patience=10). All metrics from `reeval_best.json` (best.pt reload).
E0_STRICT: clean rerun completed (50 epochs, best epoch 47, no early stop).

**Platform Submissions (registered in `results/submission_registry.csv`, updated 2026-07-18):**

| Submission | Platform | vs ref | 推理 |
|------------|---------|--------|------|
| **AEGIS F1 + Flip mean-prob T=0.5** | **61.10%** | **+3.76pp** | 2-view Flip TTA（NEW BEST） |
| **AEGIS F1 bare** | **60.52%** | **+3.18pp** | 单视图（BEST BARE） |
| S_MIXUP_CE5 + Flip TTA | **60.48%** | **+3.14pp** | 2-view Flip TTA |
| s_oof_zero_0001 + Flip TTA | 60.28% | +2.94pp | 2-view Flip TTA |
| s_oof_zero_0001 bare | **59.96%** | +2.62pp | 单视图（BEST BARE） |
| s_d3_mixup bare | 59.86% | +2.52pp | 单视图 |
| s_oof_zero_001 + Flip TTA | 59.92% | +2.58pp | 2-view Flip TTA |
| gce_q07 + Flip TTA | 59.41% | +2.07pp | 2-view Flip TTA |
| ref（D3_STRICT） | 57.34% | — | 单视图 |
| ref + Flip TTA | 58.31% | +0.97pp | 2-view Flip TTA |

**Pattern note (seed=42):** Both GCE and prototype-weighting showed local raw accuracy regression but positive platform gains, suggesting raw noisy-label validation may invert model ranking vs clean test performance. Multi-seed confirmation in progress (gce_q07 seeds 2026, 3407).

**Key changes from original:**
- Master split rebuilt with `duplicate_grouping_enabled: true` (SHA-256 group-aware) — **0 cross-boundary SHA-256 groups** (was 192 groups / 391 leaked images)
- All experiments share `outputs/data/master_splits/seed42/` for train/val
- D3 train-only cleaning (CLIP centroid arbitration, content-based removal list)
- Parent-child split audit + epoch-0 validation gate
- Old F1 80.13% deprecated due to 88% validation leakage; output dirs deleted
- All post-training metrics reloaded from best.pt (no in-memory contamination)
- `micro_macro_gap == micro - macro` enforced at 1e-10 precision
## Git 策略

- ✅ 跟踪：`.json/.csv/.log/.yaml` 结果文件
- ❌ 忽略：`.pt` 检查点、`cache/`、`train/`、`train_dedup/`、`test/`
