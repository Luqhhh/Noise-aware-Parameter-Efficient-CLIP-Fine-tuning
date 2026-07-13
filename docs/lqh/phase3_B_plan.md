# B 负责人 Phase 3 执行计划

> 项目：Noise-aware-Parameter-Efficient-CLIP-Fine-tuning  
> 日期：2026-07-13  
> 负责人：B  
> 当前训练基线：B2_GCE07，平台单视图 58.9578%  
> 当前提交基线：B2_GCE07 + 2-view horizontal-flip TTA，平台 59.4064%  
> 当前已完成基础：Full Forward / Feature Bank 0 mismatch；Trusted Validation v2 已完成并通过 28 项测试。  
> 说明：公共训练入口、Loss Schedule、Sample Weight Provider、Manifest Loader 和 Registry 由 A 维护；B 不直接修改这些公共接口。

---

# 1. B 的职责范围

B 是 Phase 3 的 GCE 参数精调、OOF 标签质量和实验审计负责人，主要负责：

1. 运行 GCE q 参数精调实验；
2. 运行 CE warmup → GCE 实验；
3. 运行一个低强度 MixUp 诊断实验；
4. 构建 duplicate-group-aware 三折 OOF；
5. 生成完整 OOF logits 和训练样本质量表；
6. 生成 OOF soft-weight、离散 weight-only 和 conservative relabel manifest；
7. 维护 trusted validation、sample quality 和 protocol audit；
8. 对 A、C 的平台候选进行独立审计。

B 不负责：

- 修改 `train.py` 主循环；
- 实现公共 Loss Registry；
- 实现公共 Loss Schedule；
- 实现 Sample Weight Provider；
- 运行 EMA Loss、Prototype、Hybrid 主线实验；
- 实现 Head EMA、PEFT、LoRA 或 Teacher；
- 最终平台提交与 Registry 主维护。

---

# 2. 当前基础状态

以下任务已完成，不需要重复：

```text
W0-1：
Full Forward 与 Feature Bank 一致性
0 / 10316 mismatch
max error = 0

W0-2：
Trusted Validation v2
28 tests passed

B2_GCE07：
平台单视图 58.9578%

B2_GCE07 + TTA：
平台 59.4064%
```

当前进行中：

```text
B2_GCE07 多 seed 稳定性验证
```

B 在开始 GCE 新实验前，应读取并登记多 seed 结果，但不重复运行现有 B2 seed。

---

# 3. B 与 A 的接口契约

A 交付给 B：

```text
统一配置 schema
Loss Registry
Loss Schedule
统一训练命令
统一结果输出
统一 Registry
统一 checkpoint schema
```

B 的 GCE 实验只能通过配置启用：

```yaml
loss:
  name: gce
  q: 0.5
```

或：

```yaml
loss:
  schedule:
    - start_epoch: 1
      end_epoch: 5
      name: cross_entropy
    - start_epoch: 6
      end_epoch: 50
      name: gce
      q: 0.7
```

B 不应为单个实验复制 loss 实现或修改训练循环。

B 生成给 A 的 manifest 必须遵循 A 定义的统一 schema：

```text
manifest_version
sample_id
original_label
training_label
sample_weight
quality_score
source
```

---

# 4. Wave 1：GCE 参数精调

## B-EXP-1：GCE q=0.5

实验 ID：

```text
W1_GCE05
```

父实验：

```text
B2_GCE07
```

配置：

```yaml
loss:
  name: gce
  q: 0.5
  probability_epsilon: 1.0e-7
```

目的：

- 判断更接近 CE 的 GCE 是否能减少困难干净样本被抑制；
- 判断 raw validation、trusted validation 和平台表现之间的关系；
- 判断 q=0.7 是否过强。

必须输出：

```text
raw_micro
raw_macro
raw_bottom10
trusted_class_balanced
trust_weighted_accuracy
rejected_micro
best_epoch
train_val_gap
prediction_change_vs_B2
```

停止条件：

- 训练不收敛；
- trusted 和 macro 同时下降；
- 类别预测塌缩；
- protocol audit 失败。

---

## B-EXP-2：GCE q=0.9

实验 ID：

```text
W1_GCE09
```

配置：

```yaml
loss:
  name: gce
  q: 0.9
  probability_epsilon: 1.0e-7
```

目的：

- 测试更强高损失样本抑制；
- 判断当前数据噪声是否需要更强鲁棒性；
- 观察困难类别和低频类别是否被过度压制。

额外分析：

```text
高损失样本梯度变化
bottom-10% 类别变化
per-class accuracy delta
prediction entropy
```

---

## B-EXP-3：CE 5 Epoch Warmup → GCE q=0.7

实验 ID：

```text
W1_CE5_GCE07
```

配置：

```yaml
loss:
  schedule:
    - start_epoch: 1
      end_epoch: 5
      name: cross_entropy
    - start_epoch: 6
      end_epoch: 50
      name: gce
      q: 0.7
```

B 的职责：

- 使用 A 已完成的 Loss Schedule；
- 核对 epoch 5 日志为 CE；
- 核对 epoch 6 日志为 GCE；
- 检查 resume 后 phase 是否正确；
- 比较 warmup 前后 best epoch、收敛速度和样本损失分布。

B 不负责重新实现 schedule。

---

## B-EXP-4：低强度 MixUp 诊断

实验 ID：

```text
W1_GCE07_MIXUP
```

配置固定：

```yaml
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
- 不叠加 Head EMA；
- 不搜索第二组 alpha/probability；
- seed=42 无正信号即关闭。

MixUp 必须额外分析：

```text
细粒度相邻类别混淆
prediction entropy
macro accuracy
bottom-10%
trusted accuracy
```

---

# 5. Wave 1 选模与多 Seed

seed=42 初筛后，从以下实验中最多保留两个：

```text
W1_GCE05
W1_GCE09
W1_CE5_GCE07
```

MixUp 不占主 GCE 候选名额，只作为独立诊断。

保留优先级：

```text
1. trusted class-balanced
2. trust-weighted accuracy
3. raw macro
4. bottom-10%
5. 实现复杂度
```

若候选差异小于 0.10pp，保留实现更简单者。

对最佳 GCE 候选补：

```text
seed=3407
```

只有准备成为平台主候选时，再建议补：

```text
seed=2026
```

B 负责 GCE 精调候选的 seed 扩展，不要求 A 重跑。

---

# 6. Wave 3：Duplicate-Group-Aware OOF

## B-OOF-1：三折划分

任务 ID：

```text
W3_DUPLICATE_GROUP_3FOLD
```

要求：

- 在 strict train split 内划分 3 folds；
- 相同 SHA-256 或 duplicate group 必须处于同一 fold；
- 类别比例尽量平衡；
- validation 不参与；
- fold seed 固定；
- 划分产物带 SHA-256；
- 不允许任何 duplicate group 跨 fold。

输出：

```text
fold_assignments.csv
fold_audit.json
class_distribution.csv
duplicate_group_distribution.csv
manifest.json
```

`fold_audit.json` 至少包含：

```text
num_samples
num_groups
fold_sizes
per_class_max_deviation
duplicate_group_leakage_count
validation_overlap_count
```

验收：

```text
duplicate_group_leakage_count = 0
validation_overlap_count = 0
```

---

## B-OOF-2：三折训练与 Holdout 推理

每折流程：

```text
使用另外两折训练
对 holdout fold 推理
只保存 holdout 样本 logits
合并三折结果
```

必须保证：

- 每个训练样本只有一条 OOF prediction；
- OOF 模型没有训练过该样本；
- 使用相同 backbone、head、数据预处理；
- 使用当前已确认的最佳鲁棒 loss；
- 每折输出 checkpoint 和配置 hash。

输出：

```text
fold_0/oof_logits.pt
fold_1/oof_logits.pt
fold_2/oof_logits.pt
oof_logits.pt
oof_predictions.csv
oof_coverage.json
```

验收：

```text
OOF coverage = 100%
duplicate sample_id = 0
missing sample_id = 0
```

---

# 7. Sample Quality 构建

## B-QUALITY-1：基础 OOF 信号

生成：

```text
sample_quality.csv
class_quality_summary.csv
```

`sample_quality.csv` 必须包含：

```text
sample_id
image_path
original_label
oof_top1
p_original_label
p_top1
top1_margin
oof_cross_entropy
prediction_entropy
class_frequency
```

---

## B-QUALITY-2：Prototype 信号

加入：

```text
prototype_own_similarity
prototype_top1
prototype_margin
prototype_agrees_with_label
prototype_agrees_with_oof
```

要求：

- prototype 只使用 strict train；
- 不使用 validation/test；
- 类别内 percentile 单独保存；
- 记录 prototype 文件 hash。

---

## B-QUALITY-3：kNN 信号

加入：

```text
knn_majority_label
knn_agreement
knn_margin
```

要求：

- 明确 k；
- duplicate/self neighbor 处理清楚；
- 不允许样本自己作为最近邻；
- 使用固定 feature bank；
- 输出低一致性样本清单。

---

## B-QUALITY-4：Flip Stability

加入：

```text
original_top1
flip_top1
flip_consistency
flip_jsd
```

只使用：

```text
original
horizontal flip
```

不使用 vertical flip。

---

## B-QUALITY-5：Duplicate Conflict

加入：

```text
duplicate_group_id
duplicate_conflict_flag
duplicate_labels
```

未决 duplicate conflict 样本后续禁止 hard relabel。

---

# 8. OOF Weight Manifest

## B-MANIFEST-1：连续 Soft Weight

输出：

```text
oof_soft_weight_manifest.csv
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

- 所有 percentile 按类别内部计算；
- quality、weight 都限制在 [0,1]；
- 最终 weight 范围 [0.3,1.0]；
- 每类权重均值和分位数写入报告；
- 若某类超过 30% 样本低于 0.5，输出告警；
- 告警不能人工修改训练数据。

输出：

```text
oof_soft_weight_manifest.csv
oof_soft_weight_summary.json
oof_soft_weight_class_stats.csv
manifest.sha256
```

---

## B-MANIFEST-2：离散 Weight-Only

输出：

```text
oof_discrete_weight_manifest.csv
```

建议：

```text
high-confidence clean → 1.0
medium-confidence → 0.6
low-confidence → 0.3
```

阈值必须在本地训练集/可信验证协议中确定，不能根据平台反向调节。

---

# 9. Conservative Relabel

只有以下条件满足后进入：

- OOF soft weighting 或 weight-only 明确有效；
- OOF 信号能稳定区分高低风险样本；
- A 的 weight-only 训练已完成；
- protocol audit 通过。

## B-RELABEL-1：候选生成

样本必须同时满足：

```text
oof_top1 != original_label
p_top1 >= 0.90
top1_margin >= 0.30
kNN 多数标签 = oof_top1
prototype_top1 = oof_top1
horizontal flip 预测一致
不属于 duplicate conflict
```

约束：

```text
每个原类别最多修改 5%
全局最多修改 3%
```

---

## B-RELABEL-2：Manifest 输出

输出：

```text
relabel_manifest.csv
relabel_summary.json
class_transition_matrix.csv
relabel_audit.json
relabel_examples_index.csv
manifest.sha256
```

`relabel_manifest.csv` 必须包含：

```text
sample_id
original_label
training_label
sample_weight
quality_score
oof_top1
p_top1
top1_margin
knn_agreement
prototype_margin
flip_consistency
source
```

禁止人工删改 manifest。

---

# 10. Trusted Validation 与结果分析

B 负责对 A、C 所有候选统一计算：

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
```

必须输出：

```text
trusted_validation.json
per_class_delta.csv
prediction_disagreement.csv
low_margin_cases.csv
```

对冻结 backbone 的实验可使用 Feature Bank；最终权威结果仍使用 full forward re-evaluation。

---

# 11. Protocol Audit

B 对平台候选执行独立审计。

检查：

```text
config diff
train split hash
val split hash
checkpoint hash
trainable parameter list
sample manifest hash
prediction count
submission file ownership
parent experiment
commit SHA
```

审计状态：

```text
pass
fail
warning
```

规则：

```text
protocol_invalid 的实验不得提交平台
```

B 对候选具有审计否决权，但不负责最终提交。

---

# 12. B 的执行顺序

## Stage B0：读取公共接口

确认 A 已完成：

```text
Loss Registry
Loss Schedule
统一实验配置
统一输出目录
```

---

## Stage B1：GCE 参数精调

按顺序：

```text
W1_GCE05
W1_GCE09
W1_CE5_GCE07
W1_GCE07_MIXUP
```

完成 seed=42 初筛，最多保留两个 GCE 候选。

---

## Stage B2：最佳 GCE 补 Seed

```text
最佳候选 seed=3407
```

生成 GCE 精调总结：

```text
gce_sweep_summary.md
gce_sweep_results.csv
```

将最佳 GCE 配置交给 A。

---

## Stage B3：OOF 基础设施

```text
3-fold split
fold audit
OOF training
OOF logits
sample quality
```

---

## Stage B4：Manifest

```text
soft weight
discrete weight-only
```

交付 A 训练。

---

## Stage B5：Relabel

只有 OOF weighting 有效后：

```text
relabel candidate
relabel audit
manifest
```

---

## Stage B6：最终审计

对 A、C 的平台候选：

```text
trusted validation
protocol audit
submission ownership audit
```

---

# 13. B 的 Git 责任边界

B 默认维护：

```text
configs/phase3/gce/
analysis/trusted_validation/
analysis/oof/
analysis/relabel/
tools/protocol_audit/
outputs/analysis/
tests/test_oof_*.py
tests/test_relabel_*.py
```

B 不直接修改：

```text
train.py
common/loss_registry.py
common/loss_schedule.py
common/sample_weighting.py
checkpoint schema
results registry
```

建议分支：

```text
phase3/b-gce-sweep
phase3/b-oof
phase3/b-relabel
phase3/b-audit
```

---

# 14. B 的交付物

B 最终必须交付：

```text
GCE q sweep configs/results
CE5→GCE07 result
MixUp diagnostic
3-fold assignments
fold audit
OOF logits
sample_quality.csv
soft-weight manifest
weight-only manifest
relabel manifest
trusted validation outputs
protocol audit reports
GCE sweep summary
OOF/relabel method说明
```

---

# 15. B 的优先级

## P0

```text
W1_GCE05
W1_GCE09
W1_CE5_GCE07
3-fold split
OOF logits
sample_quality.csv
```

## P1

```text
OOF soft-weight manifest
OOF weight-only manifest
trusted validation
protocol audit
```

## P2

```text
MixUp diagnostic
Conservative relabel
额外 seed=2026
```

---

# 16. 完成标准

B 的工作完成需满足：

- GCE 精调最多保留两个候选；
- 最佳 GCE 至少双 seed 方向一致；
- OOF coverage 100%；
- duplicate-group leakage 为 0；
- validation overlap 为 0；
- sample quality 可完整复现；
- manifest 覆盖率 100%；
- relabel 比例满足上限；
- 所有候选有可信验证和 protocol audit；
- B 不绕过 A 的公共接口修改训练循环。
