# 后续突破性实验执行方案

**项目**：Noise-aware Parameter-Efficient CLIP Fine-tuning  
**日期**：2026-07-21  
**当前主分支**：`60fc60878e6acdf00dad1b4e990eb9086add9167`  
**目标**：停止 0.05–0.15pp 级别的低收益参数微调，优先验证可能改变平台分数分布的机制级方向。

---

## 1. 当前状态与判断

当前主要平台结果：

| 实验                        |       Bare |        TTA |
| --------------------------- | ---------: | ---------: |
| A2 `NR_CL_KNN_DROP` seed=42 |  约 60.64% | **61.21%** |
| AEGIS F1（E2 parent）       |     60.52% |     61.10% |
| A2 STRICT + AEGIS LoRA s=42 | **60.65%** |     61.15% |
| A2 STRICT + AEGIS LoRA s=3407 |   60.64% |         — |

A2 STRICT 修复 parent-child split lineage 后表明：

- epoch-0 与 A2 父模型本地准确率一致；
- 双 seed 的 LoRA 本地增益同符号；
- 平台 Bare 仅提升约 0.14pp；
- 平台 TTA 仅提升约 0.05pp；
- 当前 LoRA 配方已进入明显的边际收益区。

因此，后续不应继续把主要算力投入：

- GCE 的普通 `q` 搜索；
- CE warmup 长度；
- Label Smoothing；
- Head EMA；
- 普通 sample weighting；
- Cosine Head；
- 更激进的数据删除；
- 大规模伪标签重标；
- LoRA rank、层数、学习率、distillation weight 的笛卡尔积搜索。

后续实验必须改变至少一个关键环节：

1. 哪些样本有资格更新视觉 LoRA；
2. 如何增强细粒度类别间的视觉分离；
3. 一个类别是否需要多个视觉中心；
4. 如何减少单 epoch 和单 seed 的偶然性。

---

# 2. 总体实验路线

采用四级漏斗：

| 优先级 | 方向                            | 成本 | 目标                         |
| ------ | ------------------------------- | ---: | ---------------------------- |
| P0     | 多原型与结构化分类头            | 很低 | 利用现有特征修正决策边界     |
| P1     | 同训练轨迹 checkpoint averaging |   低 | 降低单 epoch 偶然性          |
| P2     | Clean-Routed LoRA               |   中 | 控制哪些样本能够更新视觉主干 |
| P3     | Trusted Prototype-Contrastive   | 中高 | 增强细粒度特征分离           |
| P4     | 动态样本分组                    |   高 | 用训练动态修正静态 trust     |

推荐资源分配：

```text
20%：P0 低成本筛选
15%：P1 checkpoint averaging
45%：P2 Clean-Routed LoRA
15%：P3 prototype-contrastive
5%：P4 动态 trust 原型验证
```

---

# 3. P0：低成本结构化分类头实验

## 3.1 多原型分类头

### 研究问题

当前 Linear Head 每个类别只有一个权重向量。细粒度类别可能包含多个视角、姿态、背景和亚型，因此单中心分类边界可能不足。

多原型分类头使用每类多个视觉中心：

\[
s*c(x)=\operatorname{Agg}*{j=1}^{K}\left(\operatorname{sim}(f(x),p\_{c,j})\right)
\]

其中：

- \(K\) 为每类 prototype 数量；
- `Agg` 可选 `max` 或 `logmeanexp`；
- 最终与原 Linear Head logits 小比例融合。

### 现有入口

```text
reproducibility/aegis_f1/aegis_clip/cli/sweep_multiprototype_head.py
```

### 第一轮实验矩阵

| ID   | prototypes/class | trust power | aggregation | alpha |
| ---- | ---------------: | ----------: | ----------- | ----: |
| MP-1 |                2 |           1 | logmeanexp  |  0.10 |
| MP-2 |                2 |           1 | logmeanexp  |  0.25 |
| MP-3 |                2 |           2 | logmeanexp  |  0.10 |
| MP-4 |                2 |           2 | logmeanexp  |  0.25 |
| MP-5 |                4 |           1 | logmeanexp  |  0.10 |
| MP-6 |                4 |           2 | max         |  0.10 |

第一轮不要尝试：

```text
alpha >= 0.5
prototypes/class > 4
```

多原型只能作为原分类头的小幅结构修正，不能直接替代已训练 Linear Head。

### 基线

```text
F1_VISUAL_LORA_CLEAN_CORE_A2_PARENT_STRICT
seed=42
best.pt
```

### 本地晋级条件

候选必须同时满足：

```text
predicted_class_count == 500
proxy_macro >= base + 0.15pp
trusted_macro >= base + 0.15pp
raw_macro >= base - 0.10pp
raw_fixed > raw_broken
changed_predictions_ratio in [0.5%, 8%]
```

第一轮最多保留一个候选。

---

## 3.2 LDA / Ridge 结构化分类头

### 研究问题

普通 Linear Head 通过 SGD 隐式学习决策边界，但没有直接利用：

- 类内协方差；
- 类间判别统计；
- trust-weighted 全局统计；
- corrected target。

结构化 Head 可利用训练特征拟合：

- Shrinkage LDA；
- Ridge Regression；
- Corrected-target Ridge。

### 现有入口

```text
reproducibility/aegis_f1/aegis_clip/cli/sweep_structural_head.py
```

### 第一轮矩阵

| ID   | 方法            | trust power |           参数 | alpha |
| ---- | --------------- | ----------: | -------------: | ----: |
| SH-1 | LDA             |           1 | shrinkage=0.75 |  0.10 |
| SH-2 | LDA             |           1 | shrinkage=0.75 |  0.25 |
| SH-3 | LDA             |           2 | shrinkage=0.75 |  0.10 |
| SH-4 | Ridge corrected |           1 |      lambda=10 |  0.10 |
| SH-5 | Ridge corrected |           1 |      lambda=10 |  0.25 |
| SH-6 | Ridge corrected |           2 |      lambda=10 |  0.10 |

### 晋级门槛

```text
predicted_class_count == 500
proxy_macro >= base + 0.15pp
trusted_macro >= base + 0.15pp
raw_macro >= base - 0.10pp
```

P0 中只允许多原型和结构化 Head 合计保留一个平台候选。

---

# 4. P1：同轨迹 Checkpoint Averaging

## 4.1 动机

A2 STRICT 中出现：

- raw 指标最佳 epoch 较早；
- clean-core 指标最佳 epoch 较晚；
- 不同 epoch 在不同样本子集之间存在折中。

单独选择一个 epoch 可能放大随机优化噪声和模型选择偏差。Checkpoint averaging 试图在不增加推理模型数量的前提下，保留多个 epoch 的共同有效方向。

## 4.2 必须保存的 checkpoint

重新运行 A2 STRICT，保存：

```text
epoch_1.pt
epoch_2.pt
epoch_3.pt
epoch_4.pt
epoch_5.pt
epoch_6.pt
```

同时保留：

```text
epoch0.pt
best.pt
metrics.csv
epoch0_evaluation.json
promotion.json
```

## 4.3 第一轮平均方案

| ID    | 平均范围   | 方式        |
| ----- | ---------- | ----------- |
| SWA-1 | epoch 2–6  | 等权平均    |
| SWA-2 | epoch 2–4  | 等权平均    |
| SWA-3 | epoch 3–6  | 等权平均    |
| SWA-4 | epoch 2 起 | greedy soup |

只平均：

```text
LoRA 参数
classifier.weight
classifier.bias
```

不平均 frozen CLIP base 权重。

## 4.4 禁止操作

不要直接平均不同训练 seed 的 LoRA `A/B` 因子。

因为：

\[
\Delta W = BA
\]

不同 seed 的低秩分解存在等价变换。跨 seed 平均必须：

1. 重建每层有效增量 \(\Delta W\)；
2. 平均多个 \(\Delta W\)；
3. 对平均结果做 truncated SVD；
4. 重新压回 rank 8。

第一阶段只做同一训练轨迹、不同 epoch 的平均。

## 4.5 晋级条件

满足以下一种：

### 条件 A

```text
raw_micro >= best_raw_epoch
clean_core_micro >= best_clean_core_epoch
```

### 条件 B

```text
raw_micro >= best_raw_epoch - 0.05pp
clean_core_micro >= best_clean_core_epoch + 0.10pp
```

同时：

```text
predicted_class_count == 500
mean_feature_drift <= 1%
```

---

# 5. P2：Clean-Routed LoRA

## 5.1 核心问题

当前 AEGIS 主要用 `clean_probability` 调整分类损失权重，但没有完全回答：

> 哪些样本有资格改变视觉编码器？

低可信样本即使分类 loss 较小，仍可能经过 LoRA 路径、参与蒸馏并影响视觉梯度。

Clean Routing 将：

```text
样本是否可信
```

从普通 loss weight 升级为：

```text
样本是否拥有更新 LoRA 的权限
```

## 5.2 数学形式

\[
f*i=f*{\mathrm{CLIP}}(x*i)+g_i\Delta f*{\mathrm{LoRA}}(x_i)
\]

其中：

- \(f\_{\mathrm{CLIP}}\) 为 frozen base；
- \(\Delta f\_{\mathrm{LoRA}}\) 为低秩视觉增量；
- \(g_i\) 为样本级 routing gate。

## 5.3 第一轮实验矩阵

| ID   | Head 训练    | LoRA 更新         | Gate       |
| ---- | ------------ | ----------------- | ---------- |
| CR-0 | 当前配置     | 当前配置          | 无 routing |
| CR-1 | 所有样本 GCE | clean≥0.70        | hard gate  |
| CR-2 | 所有样本 GCE | clean probability | soft gate  |
| CR-3 | clean+hard   | clean≥0.80        | hard gate  |

最高优先级：`CR-1`。

## 5.4 CR-1 具体行为

### 高可信样本

```text
clean_probability >= 0.70
```

执行：

```text
更新 classifier
更新 LoRA
计算 classification loss
计算 feature distillation
```

### 低可信样本

```text
clean_probability < 0.70
```

执行：

```text
更新 classifier
不允许产生 LoRA gradient
使用 frozen CLIP feature
不参与 LoRA feature distillation
```

## 5.5 推荐实现方式

不要对同一个 batch 逐样本开关整个模块参数。

推荐拆分视觉输出：

```python
base_feature = frozen_visual_forward(images)
lora_delta = visual_lora_delta(images)
feature = base_feature + gate[:, None] * lora_delta
```

其中：

```python
gate = (clean_probability >= threshold).float()
```

这样：

- gate=0 的样本不会给 LoRA 增量产生梯度；
- classifier 仍可利用全部样本；
- batch 内不需要动态修改 `requires_grad`；
- 便于 soft gate 扩展。

## 5.6 CR-2 Soft Gate

建议：

\[
g_i=\operatorname{clip}\left(\frac{p_i-0.5}{0.5},0,1\right)
\]

对应：

```python
gate = ((clean_probability - 0.5) / 0.5).clamp(0.0, 1.0)
```

第一轮不要加入可学习 gate，避免引入新的确认偏差。

## 5.7 固定变量

所有 CR 实验固定：

```text
parent = A2 STRICT parent
lora_rank = 8
lora_last_n_blocks = 4
adapt_qv = true
adapt_out = true
gce_q = 0.5
feature_distillation_weight = 2.0
augmentation = weak_rrc_flip
epochs = 6
head_lr = 5e-5
backbone_lr = 2e-5
drift_budget = 0.01
```

只改变 routing。

## 5.8 实验顺序

```text
Step 1：跑 CR-0 seed42，重新确认基线
Step 2：跑 CR-1 seed42
Step 3：跑 CR-2 seed42
Step 4：CR-1/CR-2 中最多保留一个
Step 5：补 seed3407
Step 6：两个 seed 同方向才提交平台 Bare
```

## 5.9 晋级条件

两个 seed 必须同时满足：

```text
selector_gain >= +0.20pp
clean_core_micro_gain >= +0.20pp
raw_micro_gain >= -0.15pp
predicted_class_count == 500
mean_feature_drift <= 1%
LoRA gain same-sign across seeds
```

平台晋级：

```text
Bare >= 60.85%
```

低于 60.85% 视为与当前 60.65% 持平，不继续调参。

---

# 6. P3：Trusted Prototype-Contrastive Learning

## 6.1 动机

当前分类损失主要优化“样本是否被正确分类”，但细粒度识别还需要优化：

- 同类特征是否紧凑；
- 近邻类别是否真正分离。

普通 batch SupCon 不适合当前 500 类场景，因为 batch 内同类样本过少。推荐采用 class prototype 作为稳定正样本锚点。

## 6.2 Trusted EMA Prototype

对每个类别维护：

\[
c_k\leftarrow mc_k+(1-m)\operatorname{mean}\{f_i:y_i=k,p_i\ge0.8\}
\]

推荐：

```text
prototype_momentum = 0.99
clean_threshold = 0.80
```

## 6.3 Prototype Contrastive Loss

对高可信样本：

\[
L*{\mathrm{proto}}=-\log\frac{\exp(\operatorname{sim}(f_i,c*{y_i})/\tau)}{\sum_k\exp(\operatorname{sim}(f_i,c_k)/\tau)}
\]

第一轮固定：

```text
temperature = 0.10
loss_weight = 0.05
only_clean_samples = true
```

总损失：

\[
L=L*{\mathrm{classification}}+\lambda*{\mathrm{distill}}L*{\mathrm{distill}}+0.05L*{\mathrm{proto}}
\]

## 6.4 第一轮只跑一个配置

```text
最佳 Clean Routing
+ Trusted Prototype Contrastive
lambda=0.05
temperature=0.10
momentum=0.99
threshold=0.80
```

只有 seed42 出现明显正收益，才补：

```text
lambda=0.10
```

不要一开始做温度、momentum、threshold 网格。

## 6.5 晋级条件

相对最佳 Clean Routing：

```text
clean_core_micro >= +0.20pp
proxy_macro >= +0.20pp
raw_micro >= -0.10pp
predicted_class_count == 500
mean_feature_drift <= 1%
```

---

# 7. P4：动态样本分组

## 7.1 当前问题

现有 trust bundle 在训练开始前生成，之后保持静态。随着 LoRA 表征变化：

- hard 样本可能逐渐变得可信；
- 部分初始 clean 样本可能出现两视图不一致；
- 静态阈值无法适应训练动态。

## 7.2 最小可行实验

不要直接实现完整 DivideMix 或双网络 Co-teaching。只做一次中途刷新：

```text
epoch 0–1：
    使用静态 CVT trust

epoch 2：
    用 EMA model 对 original + flip 重新推理
    更新 clean / hard / reject 分组

epoch 2–6：
    clean：更新 head + LoRA
    hard：只更新 head，使用 GCE
    reject：不参与分类，或只做一致性约束
```

第一轮只刷新一次。

## 7.3 动态分组建议

### Clean

```text
mean confidence >= 0.80
flip agreement = true
predicted label == original label
```

### Hard

```text
mean confidence in [0.50, 0.80)
or
flip agreement = false
but original label remains top-3
```

### Reject

```text
mean confidence < 0.50
and
predicted label != original label
and
two-view prediction agreement = true
```

---

# 8. 不建议继续投入的方向

## 8.1 普通 LoRA 参数网格

暂停：

```text
rank 4/8/16
last 2/4/6 blocks
lr 1e-5/2e-5/4e-5
distill 0.5/1/2
threshold 0.65/0.70/0.75
```

原因：

- 当前 A2 parent swap 平台收益只有 0.05–0.14pp；
- 小于已观察到的 seed 波动；
- 易产生偶然最优；
- 不能解释机制。

## 8.2 大规模删除

后续只允许：

```text
极高置信度 blacklist
```

不允许继续扩大删除比例。

## 8.3 大规模伪标签重标

后续伪标签只能用于：

```text
辅助 loss
prototype 更新
dynamic routing
```

不建议直接永久替换原标签。

## 8.4 双网络 Co-teaching / DivideMix

暂不优先：

- 训练成本高；
- 工程复杂；
- 难以归因；
- 与当前 CLIP PEFT 主线不一致；
- 当前最重要问题是视觉梯度污染，而不是缺少更复杂的噪声框架。

---

# 9. 实验执行顺序

## 第一阶段：不重新训练或低成本

```text
P0-1 多原型 Head
P0-2 LDA / Ridge Head
P1 同轨迹 Checkpoint Averaging
```

目标：

```text
选出最多一个平台候选
```

## 第二阶段：Clean Routing

```text
CR-0 seed42
CR-1 seed42
CR-2 seed42
最佳候选 seed3407
平台 Bare
```

目标：

```text
验证视觉更新权限控制是否有效
```

## 第三阶段：Trusted Prototype Contrastive

```text
最佳 Clean Routing
+ prototype contrastive lambda=0.05
```

目标：

```text
验证细粒度特征分离是否可带来进一步增益
```

## 第四阶段：动态 trust

```text
epoch-2 单次 trust refresh
```

目标：

```text
验证静态 trust 是否已经成为主要瓶颈
```

---

# 10. 平台提交策略

## 提交优先级

1. P0 中最强的一个候选 Bare；
2. Clean Routing 最强候选 Bare；
3. Prototype Contrastive 最强候选 Bare；
4. Bare 明显提升后再补 TTA。

## 不允许的行为

```text
不要一次提交多个 alpha
不要用 TTA 掩盖 Bare 无提升
不要只凭单 seed 本地指标提交
不要用 raw_micro 单独选模型
不要把平台 +0.05pp 宣称为突破
```

---

# 11. 统一晋级标准

## 本地晋级

两个训练 seed 必须满足：

```text
提升方向相同
clean_core/proxy_macro 至少 +0.20pp
raw_micro 不下降超过 0.15pp
predicted_class_count == 500
mean_feature_drift <= 1%
```

## 平台晋级

当前 Bare 基线：

```text
60.65%
```

新方法至少达到：

```text
60.85%
```

才定义为有效正收益。

达到：

```text
61.10% Bare
```

才定义为明显突破。

TTA 目标：

```text
>= 61.40%
```

才可视为超过当前平台波动范围的有效进展。

---

# 12. 结果产物要求

每个实验必须保存：

```text
resolved_config.yaml/json
git_commit.txt
split_lineage_audit.json
epoch0_evaluation.json
metrics.csv
promotion.json
effective_model_spec.json
checkpoint_sha256.txt
prediction_records.csv
submission_manifest.json
```

每条平台结果必须登记：

```text
experiment_id
train_seed
split_seed
checkpoint_sha256
prediction_csv_sha256
submission_zip_sha256
local metrics
online score
Bare/TTA mode
parent checkpoint
```

---

# 13. 推荐分工

## A：公共机制与审计

负责：

```text
Clean Routing 公共接口
LoRA delta 分离
gate 单元测试
checkpoint averaging 工具
统一 promotion gate
结果审计
```

## B：低成本结构化实验

负责：

```text
Multiprototype Head
LDA / Ridge Head
paired prediction analysis
候选筛选
```

## C：表示学习实验

负责：

```text
Trusted EMA prototype
Prototype contrastive loss
dynamic trust refresh
双 seed 训练
```

---

# 14. 最终决策树

```text
P0 有 >=0.20pp 本地稳定增益？
    是 → 提交一个 Bare
    否 → 关闭结构化 Head

P1 averaging 同时改善 raw 和 clean-core？
    是 → 提交 Bare
    否 → 关闭 checkpoint averaging

CR-1/CR-2 双 seed 同方向？
    是 → 提交 Bare
    否 → 关闭 Clean Routing

Clean Routing 平台 Bare >=60.85%？
    是 → 做 Prototype Contrastive
    否 → 不继续调 routing threshold

Prototype Contrastive 双 seed 明显提升？
    是 → 提交 Bare/TTA
    否 → 做一次动态 trust MVP

动态 trust 仍无增益？
    是 → 当前 CLIP ViT-B/32 方法族基本到顶
    下一阶段考虑更强 backbone 或模型级 ensemble
```

---

# 15. 核心结论

当前最值得做的不是继续调：

```text
q
rank
lr
threshold
distill weight
```

而是优先验证：

1. 多原型或结构化分类边界能否改善细粒度多模态问题；
2. 同轨迹权重平均能否降低模型选择偶然性；
3. Clean Routing 能否阻止低可信样本污染 LoRA；
4. Trusted Prototype Contrastive 能否增强类别间特征分离；
5. 动态 trust 是否比静态 trust 更适合当前任务。

最有可能形成真正突破的主线是：

```text
A2 parent
→ Clean-Routed Visual LoRA
→ Trusted Prototype Contrastive
→ Dynamic Trust Refresh
```

这条路线改变的是视觉参数的更新机制，具有明确因果问题和可解释性，优先级高于普通参数搜索。
