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
| A3 | + RandomErasing | 待定 |

A1 在匹配学习率后与 A0 几乎持平（Δ = −0.09pp），A2 的 ColorJitter 显著破坏细粒度判别信息。最终采用 A0。

### 3. 跨类别重复图像处理

扫描发现训练集中存在 1,032 组跨类别 SHA-256 完全重复，涉及 2,095 张图片（2.0%）——同一张图被放入 2-4 个不同类别目录，训练监督彼此矛盾。

实现 **CLIP 特征质心仲裁去重**：
- 从 101,123 张非冲突图片计算每类 10% trimmed 质心（排除离群错标）
- 三条件接受：全局 Top-1 必须在候选类别中 + 绝对相似度 ≥ p10 阈值 + Top1-Top2 margin ≥ 0.02
- 203/1032 组高置信度仲裁，829 组低置信度整组移除
- 最终过滤 1,892 张图片

**去重后从 69.86% → 70.53%（+0.67pp）**，标签矛盾的直接证据和修复收益均被验证。

### 4. 部分解冻基础设施

为后续视觉特征微调（F0-F3）实现了完整的基础设施：

- **CLIPLinearClassifier 选择性解冻**：始终先冻结全部 visual 参数，再按 `unfreeze_last_n_blocks` + `train_ln_post` + `train_visual_proj` 精确解冻
- **判别式优化器**：head 和 backbone 使用不同 LR / weight decay，通过 `get_param_groups()` 暴露
- **比例 LambdaLR Scheduler**：替代 CosineAnnealingLR，保证 backbone_lr / head_lr 比例全程恒定
- **`--init-checkpoint` CLI**：仅加载模型权重（不恢复 optimizer/scheduler/epoch），用于从冻结 baseline checkpoint 启动部分解冻实验
- **训练诊断**：每 epoch CSV 输出 head_lr、backbone_lr、head_grad_norm、backbone_grad_norm

### 5. 测试覆盖

104 个测试全部通过，新增：
- `test_partial_unfreeze.py`（16 tests）：参数冻结/解冻、train mode 行为
- `test_discriminative_optimizer.py`（11 tests）：参数组结构、LR/WD 正确性、覆盖率
- `test_init_checkpoint.py`（4 tests）：权重加载、跨架构兼容、requires_grad 保持
- `test_scheduler_ratio.py`（7 tests）：余弦因子边界、比例保持、缓存 guard

### 6. 当前状态与待完成

**已完成 (strict protocol, seed=42):**
- E0-strict (69.09%): Frozen CLIP + linear head baseline on unified master split
- D3-strict (69.32%): Train-only dedup, +0.23pp vs E0
- F0-strict (69.33%): Frozen continue control from D3-strict (no gain beyond epoch 1)
- F1-strict (69.43%): ln_post+proj unfrozen from D3-strict, +0.12pp vs D3

**已废弃 (data leakage):**
- F1 (old): 80.13% — invalid due to 88% validation overlap with parent training data
- F1b (old): 80.13% — same leakage as F1
- F2 (old): blocked by invalid parent F1

**待完成:**
- **多 seed 确认 (D3 only)**: D3-strict + E0-strict 在 seeds 3407/2026 上做 paired delta 验证（F1 未通过 +0.30pp gate，不继续多 seed）
- **E4 (A3) 增强对照**: 正在由队友执行

## 项目结构

```
├── common/              # 共享代码（dataset, cache, transforms, evaluation 等）
├── experiments/
│   ├── baseline/        # Linear Head 实验（train/evaluate/infer/model）
│   └── cosine/          # Cosine Head 实验（委托 baseline）
├── configs/             # 每个实验一个 YAML
├── scripts/             # 数据准备、超参搜索、去重仲裁、提交验证
├── tests/               # 104 个 pytest 测试
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

| Experiment | Val Acc | vs E0 | vs D3 | Epochs | Notes |
|---|---|---|---|---|---|
| E0-strict | 70.45% | — | — | 34 (manual stop) | Frozen CLIP + linear head, seed=42 |
| D3-strict | 70.66% | +0.21pp | — | 49† | Train-only dedup (1,527 removed), seed=42 |
| F0-strict | 70.64% | — | −0.02pp | 5† | Frozen continue control (D3-strict init), no gain |
| F1-strict | 70.78% | — | +0.12pp | 4† | ln_post+proj unfrozen (D3-strict init), below +0.30pp gate |

† Early stopped (patience=10).

**Key changes from original:**
- Master split rebuilt with `duplicate_grouping_enabled: true` (SHA-256 group-aware) — **0 cross-boundary SHA-256 groups** (was 192 groups / 391 leaked images)
- All experiments share `outputs/master_splits/seed42/` for train/val
- D3 train-only cleaning (CLIP centroid arbitration, content-based removal list)
- Parent-child split audit + epoch-0 validation gate
- Old F1 80.13% deprecated due to 88% validation leakage; output dirs deleted

**Per-class metrics (seed=42):** micro ≈ macro (gap < 0.01pp); bottom-10% accuracy: E0 27.8% → D3 28.9% → F1 29.8% — D3/F1 help hardest classes
## Git 策略

- ✅ 跟踪：`.json/.csv/.log/.yaml` 结果文件
- ❌ 忽略：`.pt` 检查点、`cache/`、`train/`、`train_dedup/`、`test/`
