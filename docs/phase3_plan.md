# Phase 3 Robust Fine-Tuning Experiment Plan

> 项目：Noise-aware-Parameter-Efficient-CLIP-Fine-tuning  
> 日期：2026-07-13  
> 当前骨干：OpenAI CLIP ViT-B/32  
> 当前主模型：冻结 CLIP visual backbone，仅训练 Linear Head  
>
> **训练基线**：b2_gce05（GCE q=0.5）单视图 = 58.9578%（待平台确认单视图）  
> **提交基线**：b2_gce05 + 2-view horizontal-flip TTA = 60.1594%  
> **距 70% 目标**：9.8406pp  
>
> **2026-07-15 更新**：q=0.5 + TTA 平台 60.1594%，超越 q=0.7 + TTA (59.4064%) +0.75pp。  
> q=0.5 替代 q=0.7 成为新的训练和提交基线。  
> 后续新训练方法应先与单视图 b2_gce05 比较；只有训练方法本身有效，才叠加 Flip TTA。

---

# 1. Phase 3 目标

Phase 3 不再继续大范围尝试普通分类头、通用增强或无约束解冻，而是集中解决三个核心问题：

1. 如何进一步降低错误标签对梯度的影响；
2. 如何自动识别并削弱高风险训练样本；
3. 如何在鲁棒监督保护下对 CLIP 表征进行受控适配。

技术主线：

```text
GCE 精调
    ↓
训练动态可信度：EMA Loss / Prototype
    ↓
OOF 标签质量估计
    ↓
软重加权 / 保守自动重标注
    ↓
鲁棒 PEFT
    ↓
EMA Teacher / 一致性训练
```

---

# 2. 当前已知结论

## 2.1 基线定义

- **训练基线**：b2_gce05（GCE q=0.5）单视图——所有新训练方法与此比较
- **提交基线**：b2_gce05 + 2-view horizontal-flip TTA，平台 60.1594%——最终提交叠加 TTA
- **旧基线（已替换）**：gce_q07（GCE q=0.7）+ TTA = 59.4064%，因 q=0.5 平台反超 +0.75pp 而降级
- TTA 收益已确认（q=0.7 +0.45pp，q=0.5 预估 +0.75pp），后续除非 TTA 与新方法冲突，默认叠加

## 2.2 保留方向

- b2_gce05（GCE q=0.5）是当前主训练基线，2026-07-15 替换 gce_q07；
- gce_q07（GCE q=0.7）仍保留作为对比参考；
- horizontal-flip TTA 已确认有效（q=0.7 +0.45pp，q=0.5 平台 60.16%）；
- Prototype Weighting 平台有正收益，可作为可信度信号；
- 轻量部分解冻需要在 GCE 保护下重新测试；
- 自动样本可信度和 OOF 标签质量是下一阶段主线。

## 2.3 关闭或降低优先级的方向

以下方向不再作为 Phase 3 主线：

- Label Smoothing；
- 现有 Cosine Head；
- ColorJitter；
- GaussianBlur；
- RandomGrayscale；
- 4-view TTA；
- vertical flip；
- 强 RandomResizedCrop；
- 强 RandAugment；
- 大面积 Random Erasing；
- 多 checkpoint ensemble；
- 全参数微调；
- 无审计的直接删样本；
- 无 OOF 保护的直接伪标签替换。

## 2.4 本计划新增或修正的内容

相对旧版计划，本版新增：

- Linear Head EMA 独立实验；
- GCE + EMA Loss 独立实验；
- GCE + Prototype 与 GCE + EMA Loss 的完整 2×2 消融；
- 一个低强度 MixUp 诊断实验；
- 更明确的 OOF 样本质量协议；
- 更明确的 Wave 进入条件、停止条件和平台提交规则。

本版删除：

- gce_q07 + horizontal-flip TTA 平台测试任务（已完成：59.4064%，+0.45pp vs 裸模型）；
- 已完成或正在执行的 TTA 打包与提交步骤。

---

# 3. 统一实验契约

所有新增训练实验默认使用：

```yaml
backbone: OpenAI CLIP ViT-B/32
head: linear
num_classes: 500

train_split: strict train split
val_split: fixed master validation split
split_seed: 42
train_seed: 42

batch_size: 128
num_workers: 8
epochs: 50
early_stop_patience: 10

optimizer: AdamW
scheduler: cosine
base_head_lr: 5.0e-3
weight_decay: 1.0e-4

augmentation: A0 or explicitly specified
mixed_precision: same as B2_GCE07
```

除实验明确指定的变量外，其他条件不得变化。

每个实验必须产出：

```text
resolved_config.yaml
best.pt
last.pt
train_log.csv
reeval_best.json
per_class_metrics.csv
prediction_records.csv
protocol_audit.json
artifact_manifest.json
```

`artifact_manifest.json` 至少记录：

```text
experiment_id
parent_experiment
commit_sha
config_sha256
checkpoint_sha256
train_split_sha256
val_split_sha256
prediction_sha256
```

---

# 4. 统一评价标准

不能只依据 raw validation micro accuracy 选模。

每个实验必须报告：

```text
raw_micro
raw_macro
raw_bottom10
trusted_micro
trusted_macro
trusted_class_balanced
trust_weighted_accuracy
rejected_micro
prediction_change_vs_parent
best_epoch
train_val_gap
```

模型保留条件：

1. protocol audit 通过；
2. trusted class-balanced 或 trust-weighted accuracy 有正信号；
3. raw macro 不出现明显下降；
4. bottom-10% 不发生灾难性退化；
5. 不存在类别预测塌缩；
6. 至少一个额外 seed 方向一致后，才定义为稳定候选。

## 两阶段评估管线

所有新模型必须严格按以下顺序推进，禁止跳过步骤：

```text
Step A: 新训练方法单视图
    │
    ▼
与 b2_gce05 单视图比较（或平台单视图 TBD）
    │
    ├── 无正收益 → 关闭分支，不进入 TTA
    │
    └── 确认训练策略有效
            │
            ▼
        Step B: 叠加 2-view horizontal-flip TTA
            │
            ▼
        与当前提交基线 59.4064% 比较
```

- **Step A** 证明训练方法本身有效——TTA 不能掩盖训练退化
- **Step B** 确认 TTA 与训练方法兼容——如果 TTA 收益消失或反转，需诊断原因
- 只有 **Step A 和 Step B 都通过** 的候选才能成为新的提交基线

平台提交原则：

- 每个 Wave 最多提交 1 个主候选；
- 只有候选显著优于当前提交基线时才提交；
- 不利用平台结果反向搜索大量超参数；
- 提交结果必须写入 `results/submission_registry.csv`。

---

# 5. Wave 1：GCE 精调与训练稳定器

Wave 1 目标：

1. 确认 q=0.7 是否接近最优；
2. 判断 CE warmup 是否改善困难干净样本学习；
3. 判断 Head EMA 是否稳定分类头；
4. 用一个低强度 MixUp 单点实验判断输入混合是否值得保留。

---

## W1-1：GCE q=0.5

```yaml
experiment_id: W1_GCE05
parent: B2_GCE07

loss:
  name: gce
  q: 0.5
  probability_epsilon: 1.0e-7
```

假设：

- 相比 q=0.7 更接近 CE；
- 可能减少对困难样本的过度抑制；
- 可能提高 raw validation，但噪声鲁棒性可能减弱。

---

## W1-2：GCE q=0.9

```yaml
experiment_id: W1_GCE09
parent: B2_GCE07

loss:
  name: gce
  q: 0.9
  probability_epsilon: 1.0e-7
```

假设：

- 更强抑制高损失样本；
- 如果错误标签比例高，可能优于 q=0.7；
- 风险是压制困难但正确标注的样本。

---

## W1-3：CE 5 Epoch Warmup → GCE q=0.7

```yaml
experiment_id: W1_CE5_GCE07
parent: B2_GCE07

loss_schedule:
  - epochs: [1, 5]
    name: cross_entropy
  - epochs: [6, 50]
    name: gce
    q: 0.7
```

实现要求：

- loss phase 写入日志；
- resume 必须恢复正确 phase；
- epoch 5/6 边界必须有单元测试；
- 不修改学习率、数据和增强。

---

## W1-4：GCE q=0.7 + Linear Head EMA

```yaml
experiment_id: W1_GCE07_HEAD_EMA099
parent: B2_GCE07

head_ema:
  enabled: true
  decay: 0.99
  warmup_epochs: 5
  use_ema_for_validation: true
  save_raw_and_ema: true
```

每个 epoch 同时记录：

```text
raw_head_val_metrics
ema_head_val_metrics
parameter_distance_raw_vs_ema
```

默认只运行 decay=0.99。

只有在 0.99 有稳定正收益时，才补：

```text
W1_GCE07_HEAD_EMA0999
```

---

## W1-5：低强度 MixUp 诊断

```yaml
experiment_id: W1_GCE07_MIXUP
parent: B2_GCE07

loss:
  name: gce
  q: 0.7

mixup:
  enabled: true
  alpha: 0.2
  probability: 0.2
```

限制：

- 使用 A0；
- 不叠加 Random Erasing；
- 不叠加 Label Smoothing；
- 不叠加 EMA Loss；
- 不搜索更多 alpha 和 probability；
- 若本实验无明确正信号，关闭 MixUp 分支。

---

## Wave 1 保留规则

保留最多两个训练候选。

优先顺序：

```text
1. trusted class-balanced
2. trust-weighted accuracy
3. raw macro
4. bottom-10%
5. 实现复杂度
```

若两个候选差异小于 0.10pp，保留实现更简单者。

---

# 6. Wave 2：动态样本可信度与完整消融

Wave 2 是 Phase 3 的核心训练阶段。

目标：

- 判断 EMA Loss 是否能独立识别高风险样本；
- 判断 Prototype Weight 是否能独立工作；
- 判断两种信号是否互补。

完整消融矩阵：

| 实验 | GCE | Prototype | EMA Loss |
|---|---:|---:|---:|
| B2_GCE07 | 是 | 否 | 否 |
| W2-1 | 是 | 否 | 是 |
| W2-2 | 是 | 是 | 否 |
| W2-3 | 是 | 是 | 是 |

---

## W2-1：GCE + EMA Loss Soft Weighting

```yaml
experiment_id: W2_GCE07_EMA_LOSS
parent: B2_GCE07

loss:
  name: gce
  q: 0.7

ema_loss:
  enabled: true
  momentum: 0.9
  warmup_epochs: 5
  ranking: classwise

sample_weight:
  min_weight: 0.4
  max_weight: 1.0
```

建议 schedule：

```text
Epoch 1–5:
  all weights = 1.0

Epoch 6–15:
  classwise highest EMA-loss 10% → weight 0.6
  others → weight 1.0

Epoch 16–50:
  classwise highest EMA-loss 20% → weight 0.4
  others → weight 1.0
```

必须：

- 使用稳定 sample_id；
- 保存 EMA loss 状态；
- resume 后正确恢复；
- 类别内排名，禁止全局排名；
- 输出样本 loss history；
- 输出每类被降权比例。

---

## W2-2：GCE + Conservative Prototype Weight

```yaml
experiment_id: W2_GCE07_PROTO_MIN04
parent: B2_GCE07

prototype_weight:
  enabled: true
  classwise_percentile: true
  min_weight: 0.4
  max_weight: 1.0
```

权重：

\[
w_i=0.4+0.6c_i^{prototype}
\]

要求：

- prototype 仅用 strict train split 构建；
- validation/test 不参与；
- 不删除样本；
- 缺失权重必须 fail-closed；
- 输出 prototype margin 与样本损失、预测正确性的关系。

---

## W2-3：GCE + Prototype + EMA Loss Hybrid

只有 W2-1 或 W2-2 至少一项独立有效时执行。

```yaml
experiment_id: W2_GCE07_PROTO_EMA
parent: best_of_W2_1_W2_2

sample_confidence:
  prototype_ratio: 0.7
  ema_loss_ratio: 0.3

sample_weight:
  min_weight: 0.4
  max_weight: 1.0
```

置信度：

\[
c_i =
0.7c_i^{prototype}
+
0.3c_i^{ema-loss}
\]

权重：

\[
w_i=0.4+0.6c_i
\]

禁止在同一实验继续加入：

- Head EMA；
- MixUp；
- CE warmup；
- OOF weight；
- relabel。

---

## W2-4：最佳 Sample Weighting + Head EMA

只有 W1-4 与 W2-1/W2-2/W2-3 都独立有效时执行。

```yaml
experiment_id: W2_BEST_WEIGHT_HEAD_EMA
parent: best_W2_candidate

head_ema:
  enabled: true
  decay: 0.99
```

该实验用于验证训练稳定器是否能与样本可信度策略组合。

---

# 7. Wave 3：三折 OOF 标签质量与保守重标注

Wave 3 目标：

> 为每个训练样本生成没有“见过该样本”的模型预测，并据此估计标签可信度。

---

## W3-0：Duplicate-Group-Aware 3-Fold Split

划分要求：

- strict train split 内做 3-fold；
- 相同 SHA-256 或 duplicate group 必须在同一 fold；
- 每类尽量保持比例；
- validation 不进入任何 fold 训练；
- 输出 fold leakage audit。

输出：

```text
outputs/phase3/oof/
├── fold_assignments.csv
├── fold_audit.json
├── fold_0/
├── fold_1/
├── fold_2/
└── manifest.json
```

---

## W3-1：生成 OOF Logits 与 Sample Quality

每折：

1. 使用另外两折训练；
2. 对 holdout fold 推理；
3. 只保存 holdout 预测；
4. 合并为完整 OOF prediction。

`sample_quality.csv` 至少包含：

```text
sample_id
image_path
original_label
oof_top1
p_original_label
p_top1
top1_margin
oof_cross_entropy
prototype_own_similarity
prototype_margin
knn_agreement
flip_consistency
duplicate_conflict_flag
class_frequency
```

---

## W3-2：OOF Soft Weighting

```yaml
experiment_id: W3_OOF_SOFT_WEIGHT
parent: best_Wave2_candidate
```

建议质量分数：

```text
quality =
  0.35 * classwise_percentile(p_original_label)
+ 0.25 * classwise_percentile(prototype_margin)
+ 0.25 * knn_agreement
+ 0.15 * flip_consistency
```

权重：

\[
w_i=0.3+0.7quality_i
\]

要求：

- 所有 percentile 按类别计算；
- 权重范围 `[0.3, 1.0]`；
- 不删除样本；
- 输出每类权重分布；
- 若某类超过 30% 样本权重低于 0.5，必须告警；
- 告警不允许人工修改训练集。

---

## W3-3：OOF Weight-Only 对照

为了判断“自动重标注”是否必要，先运行：

```yaml
experiment_id: W3_OOF_WEIGHT_ONLY
parent: best_Wave2_candidate
```

对高风险样本只降权，不改标签。

推荐：

```text
high-confidence clean → 1.0
medium-confidence → 0.6
low-confidence → 0.3
```

---

## W3-4：Conservative Automatic Relabel

只有 W3-2/W3-3 有明确正收益，且 OOF 诊断显示高置信错标簇时执行。

样本必须同时满足：

1. `oof_top1 != original_label`；
2. `p_top1 >= 0.90`；
3. `top1_margin >= 0.30`；
4. kNN 多数标签等于 `oof_top1`；
5. prototype Top-1 等于 `oof_top1`；
6. horizontal flip 前后预测一致；
7. 不属于未决 duplicate conflict；
8. 每个原类别最多重标注 5%；
9. 全局重标注比例最多 3%。

实验：

```text
W3_RELABEL_WEIGHT_ONLY
W3_RELABEL_HARD
```

必须同时跑 weight-only 对照，禁止只跑 hard relabel。

输出：

```text
relabel_manifest.csv
class_transition_matrix.csv
relabel_summary.json
relabel_audit.json
```

禁止人工挑选或删除自动重标注结果。

---

# 8. Wave 4：鲁棒监督下的参数高效视觉适配

Wave 4 不使用普通 CE 重新做 F1，而是从当前最佳鲁棒冻结模型继续训练。

---

## W4-1：ln_post + visual_proj Continue Training

```yaml
experiment_id: W4_ROBUST_LNPROJ
init_checkpoint: best_frozen_robust_checkpoint

trainable:
  linear_head: true
  visual_ln_post: true
  visual_projection: true
  transformer_blocks: 0

epochs: 12
early_stop_patience: 5

optimizer:
  head_lr: 1.0e-4
  backbone_lr: 1.0e-6
```

要求：

- 初始化 epoch-0 预测必须与父 checkpoint 一致；
- 每 epoch 输出 head/backbone grad norm；
- trusted accuracy 下降超过 0.5pp 时停止；
- 不同时解冻 Transformer block。

---

## W4-2：Visual LayerNorm-Only Tuning

若 W4-1 无收益或过拟合：

```yaml
experiment_id: W4_ROBUST_VISUAL_LN_ONLY

trainable:
  visual_layernorms: true
  ln_post: true
  visual_projection: false
  attention: false
  mlp: false
```

---

## W4-3：Last-Block LoRA Rank 4

只有 W4-1 或 W4-2 有明确正收益时执行。

```yaml
experiment_id: W4_ROBUST_LASTBLOCK_LORA_R4

lora:
  target_block: 11
  target_modules:
    - attention_in_projection
    - attention_out_projection
  rank: 4
  alpha: 8
  dropout: 0.0

optimizer:
  head_lr: 1.0e-4
  lora_lr: 1.0e-5

epochs: 15
early_stop_patience: 5
```

第一轮不搜索 rank、alpha、dropout。

---

# 9. Wave 5：EMA Teacher 与一致性训练

只有以下任一条件满足后进入：

- Wave 3 的 OOF weighting/relabel 明确有效；
- Wave 4 的 robust PEFT 明确有效；
- 当前最好平台成绩达到阶段性目标；
- 当前训练基线在多 seed 下稳定。

---

## W5-1：EMA Teacher + Flip Consistency

```yaml
experiment_id: W5_EMA_TEACHER_FLIP

teacher:
  type: ema
  decay: 0.999

student_view:
  horizontal_flip_probability: 0.5

teacher_view:
  standard_or_weak_flip: true

consistency:
  confidence_threshold: 0.90
  max_weight: 0.5
  ramp_epochs: 10
```

总损失：

\[
L =
L_{robust-supervised}
+
\lambda(t)L_{consistency}
\]

限制：

- 不使用 ColorJitter；
- 不使用 RandomGrayscale；
- 不使用 vertical flip；
- 不使用强 RandAugment；
- teacher 低置信预测不参与一致性损失；
- 最终推理只使用单个 EMA Teacher checkpoint。

---

## W5-2：GCE + ELR

如果 EMA Teacher 实现成本过高，可先测试 ELR：

```yaml
experiment_id: W5_GCE_ELR

warmup_epochs: 5
prediction_history_momentum: 0.9
```

必须保存和恢复每样本历史预测状态。

---

# 10. 多随机种子策略

不需要所有实验都立即跑三 seed。

建议：

## 单 seed 探索

所有新方向先运行：

```text
seed = 42
```

## 双 seed 验证

候选通过本地 gate 后补：

```text
seed = 3407
```

## 三 seed 确认

准备平台主提交或最终模型时补：

```text
seed = 2026
```

稳定候选要求：

- 多 seed 方向一致；
- 不因单个 seed 出现完全相反结论；
- 平均 trusted/macro 有正收益；
- 方差可接受。

---

# 11. Wave 进入条件与停止条件

| Wave | 进入条件 | 停止条件 |
|---|---|---|
| Wave 1 | 评估基础设施就绪 | 最多保留两个候选 |
| Wave 2 | gce_q07 基线稳定 | EMA Loss/Prototype 均无收益则停止 hybrid |
| Wave 3 | Wave 2 至少一个可信度策略有效，或当前仍明显受噪声限制 | OOF 信号不能区分高低风险样本则停止 relabel |
| Wave 4 | 当前最佳冻结鲁棒模型确定 | 受控解冻连续退化则关闭 PEFT |
| Wave 5 | OOF 或 robust PEFT 至少一个有效 | teacher confirmation bias 或一致性无收益则停止 |

---

# 12. 平台提交优先级

本计划不包含正在进行的 B2+TTA 平台测试。

后续平台提交顺序：

```text
1. Wave 1 最佳 GCE / warmup / Head EMA 候选
2. Wave 2 最佳动态样本重加权候选
3. Wave 3 最佳 OOF weighting / conservative relabel 候选
4. Wave 4 最佳 robust PEFT 候选
5. 最终最佳单 checkpoint 的合法推理配置
```

每个 Wave 最多提交一个主候选，避免用平台做大规模超参数搜索。

---

# 13. 实验优先级

## P0：必须完成

```text
W1-1 GCE q=0.5
W1-2 GCE q=0.9
W1-3 CE5 → GCE07
W2-1 GCE + EMA Loss
W2-2 GCE + Prototype
W3-0/W3-1 OOF 基础设施与 sample quality
W3-2 OOF soft weighting
```

## P1：高价值

```text
W1-4 Head EMA
W2-3 Prototype + EMA Loss
W3-3 OOF weight-only
W3-4 Conservative relabel
W4-1 Robust ln_post + visual_proj
W4-2 Visual LayerNorm-only
```

## P2：条件实验

```text
W1-5 Low-strength MixUp
W2-4 Best weighting + Head EMA
W4-3 Last-block LoRA
W5-1 EMA Teacher
W5-2 ELR
```

---

# 14. 不建议直接组合的实验

以下组合禁止在没有独立消融前执行：

```text
GCE + MixUp + EMA Loss
GCE + Prototype + EMA Loss + Head EMA
OOF Weight + Relabel + MixUp
Robust PEFT + Teacher + Relabel 同时启用
```

组合原则：

> 一个新实验最多引入一个未经独立验证的新变量。

---

# 15. 结果登记

新增：

```text
results/phase3_experiments.csv
```

字段：

```text
experiment_id
parent_experiment
wave
priority
commit_sha
config_path
output_dir
split_seed
train_seed
loss_name
loss_parameters
sample_weighting
augmentation
head_ema
trainable_parameters
best_epoch
checkpoint_sha256
train_split_sha256
val_split_sha256
raw_micro
raw_macro
raw_bottom10
trusted_micro
trusted_macro
trusted_class_balanced
trust_weighted_accuracy
rejected_micro
prediction_change_vs_parent
platform_score
platform_delta_vs_b2
status
notes
```

状态枚举：

```text
planned
running
failed
protocol_invalid
local_rejected
candidate
seed_validation
submitted
platform_rejected
platform_best
closed
```

---

# 16. 推荐执行顺序

```text
Step 1
W1-1 / W1-2 / W1-3
同时实现 W1-4 Head EMA 基础设施

Step 2
选择 Wave 1 最佳候选
运行 W1-4
可选运行 W1-5

Step 3
W2-1 GCE + EMA Loss
W2-2 GCE + Prototype

Step 4
若 W2-1/W2-2 有正收益：
W2-3 Hybrid
可选 W2-4 Head EMA Combination

Step 5
实现 W3-0 3-fold OOF
生成 W3-1 sample quality

Step 6
W3-2 OOF soft weighting
W3-3 OOF weight-only

Step 7
若 OOF 证据充分：
W3-4 conservative relabel

Step 8
确定最佳冻结鲁棒模型
W4-1 → W4-2 → 条件执行 W4-3

Step 9
只在前述主线有效后：
W5-1 EMA Teacher
或 W5-2 ELR
```

---

# 17. 70% 目标判断节点

当前平台最好结果距离 70% 仍有较大差距，因此设置以下判断节点：

| 阶段 | 期望状态 | 决策 |
|---|---|---|
| Wave 1 完成 | GCE 稳定提升，平台接近或超过 60% | 继续样本可信度 |
| Wave 2 完成 | 动态重加权产生独立增益 | 进入 OOF |
| Wave 3 完成 | 平台达到约 63%–65% 或出现明确方法级增益 | 继续 robust PEFT |
| Wave 4 完成 | 平台达到约 65%–67% | 继续 Teacher/ELR 冲刺 |
| Wave 5 完成 | 平台接近 68% 以上 | 70% 才具有较强现实性 |

若 Wave 3 完成后仍低于 62%，应将 70% 降为低概率冲刺目标，并优先保证稳定晋级方案。

---

# 18. 最终主线

Phase 3 的核心不是继续增加普通增强，而是建立以下闭环：

```text
鲁棒 loss 降低错误标签梯度
    ↓
EMA Loss / Prototype 动态估计样本可信度
    ↓
OOF 避免训练样本自我评估偏差
    ↓
保守软重加权或少量自动重标注
    ↓
鲁棒监督保护下的 CLIP 参数高效适配
    ↓
EMA Teacher 或 ELR 稳定决策边界
```

最优候选应始终保持：

- 单一 CLIP ViT-B/32；
- 单一 checkpoint；
- 自动、可复现的数据处理；
- 不使用外部数据；
- 不使用测试集训练；
- 不使用多模型集成；
- 完整实验审计和产物哈希。