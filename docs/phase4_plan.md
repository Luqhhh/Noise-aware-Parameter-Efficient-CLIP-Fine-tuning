# Phase 4：Visual LoRA 因果消融与参数优化

> 日期：2026-07-20  
> 父模型：A2 `NR_CL_KNN_DROP` seed=42 best.pt（epoch 48, val 69.44%, TTA 61.21%）  
> 前提：A2_AEGIS_PARENT_SWAP **结论已出——更强 parent 不提高 LoRA（-0.22pp bare, -0.23pp TTA）**  
> 原则：每次只变一个变量，禁止笛卡尔积搜索

---

## 0. 前置：A2_AEGIS_PARENT_SWAP

严格复刻 AEGIS F1，只替换 init_checkpoint：

```yaml
# 完全保持 AEGIS F1 配置，不变：
augmentation: weak_rrc_flip
lora_last_n_blocks: 4        # blocks 8–11
lora_rank: 8
lora_adapt_qv: true          # Q/V
lora_adapt_out: true         # output projection
clean_threshold: 0.70
gce_q: 0.5
feature_distillation_weight: 2.0
head_lr: 5e-5
backbone_lr: 2e-5
epochs: 6
drift_budget: 0.01           # 1%

# 唯一变化：
init_checkpoint: <A2 best.pt>  # 替换 E2 epoch44
```

**回答的问题**：更强的 A2 parent 是否能提高 AEGIS LoRA，而不引入其他混杂因素？

### 结果（2026-07-21）

| 指标 | F1 (E2 parent) | A2 swap | Δ |
|------|:---:|:---:|:---:|
| raw_micro (val) | 0.7068 | 0.7922 | +8.5pp ⚠️ 假信号 |
| clean_core_micro (val) | — | 0.8832 | — |
| best epoch | 4 | **1** | 极快过拟合 |
| drift @ best | — | 0.0021 | — |
| **Bare 平台** | **60.52%** | 60.29% | **-0.22pp** |
| **TTA 平台** | **61.10%** | 60.87% | **-0.23pp** |

**结论：更强的 A2 parent 不能提高 AEGIS LoRA。方向关闭。**

- 本地 val 暴涨 8.5pp 是假信号——LoRA 在 val set 上过拟合到 A2 特征空间，泛化到 test set 反而变差
- Epoch 1 即最佳（vs F1 epoch 4），之后持续 drift 上升
- TTA gain +0.58pp 与 F1 一致（+0.59pp），说明 TTA 收益与 parent 无关
- **教训**：本地 val metrics 不能替代平台验证；大 parent 改进不一定传递到 LoRA

---

## P3：LoRA 因果消融矩阵

> ⚠️ **2026-07-21 更新**：A2_AEGIS_PARENT_SWAP 负收益（-0.22pp bare, -0.23pp TTA）。A2 parent 不如 E2 parent。
> P3/P4 的因果消融链仍可执行，但 baseline 应该是 F1 (E2 parent)，不是 A2 swap。
> L0（从 A2 继续训练）仍有独立价值——它测试的是继续训练本身，不涉及 LoRA。

Parent swap ~~确认有效后~~ 已出负结果，但消融矩阵本身仍有价值。第一轮全部 seed=42，本地通过安全门后只提交关键端点。

### 实验矩阵

| ID | LoRA | Clean≥0.70 | Distill | 增强 | 研究问题 |
|:---|:---:|:---:|:---:|:---|:---|
| L0 | 否 | 否 | 否 | weak_rrc_flip | 从 A2 init 继续训练 6 epoch 是否有收益？ |
| L1 | 是 | 否 | 否 | weak_rrc_flip | 纯 visual LoRA，不加任何过滤 |
| L2 | 是 | 是 | 否 | weak_rrc_flip | Clean filter 的边际贡献 |
| L3 | 是 | 是 | 是 | weak_rrc_flip | Feature distillation 的边际贡献（= AEGIS F1 全配置） |
| L4 | 是 | 是 | 是 | **A0** | 弱增强是否必要？用 CLIP 标准预处理替代 weak_rrc_flip |

**Configs:**

| ID | Config file | Key diff from L3 |
|:---|:---|:---|
| L0 | `l0_frozen_continue.yaml` | `peft_mode` 非 LoRA, clean 关, distill 关 |
| L1 | `l1_lora_only.yaml` | clean 关, distill 关 |
| L2 | `l2_lora_clean.yaml` | distill 关 |
| L3 | `l3_lora_clean_distill.yaml` | (baseline) |
| L4 | `l4_lora_clean_distill_a0.yaml` | `train_augmentation: clip_center_crop` |
| A2 swap | `f1_visual_lora_clean_core_a2_parent.yaml` | `init_checkpoint: A2 best.pt`（前置验证） |

**L0 是因果基线。** 如果 L0 本身就有正收益（继续训练 6 epoch > A2 bare），那后续 LoRA 的收益需要扣除 L0 的贡献。

### 本地安全门

所有实验必须满足：

```text
protocol_audit passed
clean_core_micro 不崩塌
flip_agreement 不显著下降
drift (if applicable) < 0.01
predicted class count == 500
no class accuracy drops >25pp with prediction count collapse >50%
```

L0 额外：feature drift 应接近 0（backbone 冻结）。

### 平台提交顺序

```
L1 Bare → L3 Bare → 胜者 seed=3407 → 双 seed 有收益后 TTA
```

最多消耗 3 次 Bare + 1 次 TTA 提交。

### 判断逻辑

```
L1 Bare > A2 Bare + 0.20pp？
  ├── 是 → LoRA 本身有效，进入 L2/L3
  └── 否 → LoRA 无效，关闭 PEFT 方向

L2 Bare > L1 Bare + 0.10pp？
  ├── 是 → clean filter 有边际收益
  └── 否 → clean filter 在 A2 数据上冗余（A2 已删 991 个确认错标）

L3 Bare > L2 Bare + 0.10pp？
  ├── 是 → feature distillation 有边际收益
  └── 否 → distillation 在 A2 数据上冗余

L4 Bare vs L3 Bare？
  ├── L4 ≥ L3 → 弱增强不必要，可用 A0 简化管线
  └── L4 < L3 → weak_rrc_flip 对 LoRA 训练有帮助
```

---

## P4：参数优化（仅在 LoRA 有效后）

禁止直接大规模网格搜索。按以下顺序逐一优化，每个阶段只调一个参数。

### 1. Backbone LR

固定其他配置（L3 全配置）：

| backbone_lr | 预期 |
|:---|:---|
| 1e-5 | 更保守，drift 更低 |
| 2e-5 | AEGIS F1 默认 |
| 4e-5 | 更激进，drift 风险 |

监控指标：

```text
clean_core_micro
raw_micro
flip_agreement
mean_feature_cosine_distance
p95_feature_cosine_distance
LoRA gradient norm
```

选择标准：clean_core_micro 最高且 drift < 1%。

**Configs:**

| Config | backbone_lr | vs L3 |
|:---|:---|:---|
| `p4_lora_blr_1e5.yaml` | 1e-5 | 唯一变化 |
| L3 (baseline) | 2e-5 | — |
| `p4_lora_blr_4e5.yaml` | 4e-5 | 唯一变化 |

### 2. LoRA 位置与容量

固定最优 backbone LR：

| 配置 | rank | blocks | 参数量 | 预期 |
|:---|:---:|:---|:---|:---|
| Last-2, R8 | 8 | 10–11 | ~50% of AEGIS | 可能足够，当前数据噪声比 AEGIS 少 |
| Last-4, R8 | 8 | 8–11 | AEGIS F1 默认 | baseline |
| Last-4, R4 | 4 | 8–11 | ~50% of AEGIS | 最小有效配置 |

暂不测试 rank=16。当前数据仍有标签噪声，容量过大更容易破坏 CLIP 表征。

**Configs:**

| Config | blocks | rank | lora_alpha | vs L3 |
|:---|:---:|:---:|:---|:---|
| `p4_lora_pos2_r8.yaml` | 2 | 8 | 8.0 | 唯一变化 |
| L3 (baseline) | 4 | 8 | 8.0 | — |
| `p4_lora_pos4_r4.yaml` | 4 | 4 | 4.0 | 唯一变化 |

### 3. Distillation 强度

用特征漂移作为控制目标，不盲目固定 λ=2.0：

| λ | 目标 |
|:---|:---|
| 0.5 | 更弱的 anchor，允许更多适配 |
| 1.0 | 折中 |
| 2.0 | AEGIS F1 默认 |

约束：

```text
目标 drift：0.3%–0.8%
硬上限：1.0%
如果 λ=2.0 时 drift < 0.3% → 可以降低 λ 允许更多适配
如果 λ=0.5 时 drift > 1.0% → 必须提高 λ 加强约束
```

仓库已有根据 task loss / feature loss 比例校准 λ 的机制，利用该机制而非盲目搜索。

**Configs:**

| Config | λ | vs L3 |
|:---|:---|:---|
| `p4_distill_lam05.yaml` | 0.5 | 唯一变化 |
| `p4_distill_lam10.yaml` | 1.0 | 唯一变化 |
| L3 (baseline) | 2.0 | — |

### 4. Clean threshold

仅在 LoRA 已确认有效后测试：

| threshold | 预期 |
|:---|:---|
| 0.65 | 更多样本获得分类监督 |
| 0.70 | AEGIS F1 默认 |
| 0.75 | 更严格，更少样本 |

必须输出：

```text
总保留率
每类保留率（min / median / max）
每类最少样本数
被 global blacklist (A2) 与 clean filter 同时拒绝的重叠率
被 global blacklist 拒绝但 clean filter 接受的样本数
```

如果出现"过滤越多性能越差"（类似 A1/A3 的教训），立即固定阈值，不再微调。

**Configs:**

| Config | selection_threshold | clean_core_threshold | vs L3 |
|:---|:---:|:---:|:---|
| `p4_clean_thresh065.yaml` | 0.65 | 0.65 | 唯一变化 |
| L3 (baseline) | 0.70 | 0.70 | — |
| `p4_clean_thresh075.yaml` | 0.75 | 0.75 | 唯一变化 |

### P4 Config 总览

所有 P4 config 位于 `reproducibility/aegis_f1/configs/`，`stage: p4_ablation`，基于 L3 且每次只变一个变量：

| # | Config | 维度 | 变化 | 执行顺序 |
|:---|:---|:---|:---|:---:|
| 1 | `p4_lora_blr_1e5.yaml` | Backbone LR | 2e-5 → 1e-5 | P4.1 |
| 2 | `p4_lora_blr_4e5.yaml` | Backbone LR | 2e-5 → 4e-5 | P4.1 |
| 3 | `p4_lora_pos2_r8.yaml` | LoRA 位置 | last-4 → last-2 | P4.2 |
| 4 | `p4_lora_pos4_r4.yaml` | LoRA 容量 | rank=8 → rank=4 | P4.2 |
| 5 | `p4_distill_lam05.yaml` | Distill λ | 2.0 → 0.5 | P4.3 |
| 6 | `p4_distill_lam10.yaml` | Distill λ | 2.0 → 1.0 | P4.3 |
| 7 | `p4_clean_thresh065.yaml` | Clean thresh | 0.70 → 0.65 | P4.4 |
| 8 | `p4_clean_thresh075.yaml` | Clean thresh | 0.70 → 0.75 | P4.4 |

**执行约束**：P4.1→P4.2→P4.3→P4.4 顺序执行。每个阶段选最优值固定后再进入下一阶段。不允许跨阶段同时搜索。

---

## 全局约束

- **所有实验 seed=42 首轮，只有通过本地门+平台 Bare 正收益的候选补 seed=3407**
- **每次只变一个参数**，不跑组合网格
- **禁止**：rank 搜索 + block 搜索 + LR 搜索 + threshold 搜索同时进行
- **失败即停**：L0/L1 无收益 → 关闭 PEFT，不继续 L2-L4 和 P4
- **数据固定**：使用 AEGIS F1 的 train.csv（91,195 samples），不引入全局黑名单（AEGIS 代码不知道黑名单的存在，保持变量隔离）
- **提交纪律**：先 Bare，Bare 有正信号再 TTA；单 seed 不宣称收益
