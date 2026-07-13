# A 负责人 Phase 3 执行计划

> 项目：Noise-aware-Parameter-Efficient-CLIP-Fine-tuning  
> 日期：2026-07-13  
> 负责人：A  
> 当前训练基线：B2_GCE07，平台单视图 58.9578%  
> 当前提交基线：B2_GCE07 + 2-view horizontal-flip TTA，平台 59.4064%  
> 说明：GCE 参数精调 `W1_GCE05 / W1_GCE09 / W1_CE5_GCE07` 由 B 负责，A 不运行这些实验。  
> 说明：所有新策略先做单视图评估；只有通过本地 gate 的候选才补 2-view horizontal-flip TTA。

---

# 1. A 的职责范围

A 是 Phase 3 的公共基础设施和主线实验负责人，主要负责：

1. 提前完成所有公共训练接口；
2. 维护训练入口、配置协议和 checkpoint schema；
3. 实现并运行 EMA Loss、Prototype、Hybrid 等主线鲁棒实验；
4. 接收 B 生成的 OOF/Relabel manifest 并完成训练；
5. 接收 C 提供的 Head EMA/PEFT/Teacher 模块并完成组合实验；
6. 负责主线候选多随机种子验证；
7. 负责最终模型整合、提交文件生成和结果登记。

A 不负责：

- GCE q=0.5/q=0.9 参数精调；
- CE5→GCE07 实验运行；
- OOF fold、sample quality、relabel manifest 生成；
- Head EMA、LoRA、EMA Teacher 算法模块本体实现；
- protocol audit 的最终独立审查。

---

# 2. 当前已完成基础

以下任务已经完成，不需要重复：

```text
R0-1 / W0-1：
Full Forward 与 Feature Bank 一致性
0 / 10316 mismatch
max error = 0

R0-3 / W0-2：
Trusted Validation v2
28 tests passed

B2_GCE07：
当前训练基线
单视图平台 58.9578%

B2_GCE07 + TTA：
当前提交基线
平台 59.4064%
```

当前进行中：

```text
B2_GCE07 多 seed 稳定性验证
```

---

# 3. A 的公共基础设施任务

公共接口必须优先于主线组合实验完成。B、C 后续应通过这些接口工作，不直接修改公共训练循环。

---

## A-INFRA-1：统一配置 Schema

统一配置入口至少支持：

```yaml
experiment:
  id:
  parent:
  wave:
  seed:
  output_dir:

loss:
  name:
  q:
  probability_epsilon:
  schedule:

sample_weighting:
  type:
  manifest_path:
  momentum:
  warmup_epochs:
  ranking:
  min_weight:
  max_weight:

head_ema:
  enabled:
  decay:
  warmup_epochs:
  use_ema_for_validation:

peft:
  type:
  train_ln_post:
  train_visual_proj:
  train_visual_layernorm:
  lora_rank:
  lora_alpha:
  lora_dropout:

teacher:
  enabled:
  ema_decay:
  confidence_threshold:
  consistency_weight:
  ramp_epochs:
```

验收要求：

- 未知字段 fail-closed；
- 所有默认值显式写入 resolved config；
- 实验输出目录中保存 `resolved_config.yaml`；
- 配置 hash 写入 artifact manifest；
- 相同 experiment ID 不允许覆盖已有目录。

---

## A-INFRA-2：Loss Registry

实现统一入口：

```python
criterion = build_loss(loss_config)
```

支持：

```text
cross_entropy
gce
scheduled_loss
```

接口要求：

- 支持 per-sample loss；
- 支持 reduction=`none`；
- sample weighting 在训练循环统一处理；
- GCE 的 q 和 epsilon 显式记录；
- 不允许各实验自行复制 loss 实现。

---

## A-INFRA-3：Loss Schedule

A 提前实现：

```text
CE → GCE
按 epoch 切换 loss
resume 后恢复正确 phase
日志记录当前 loss phase
```

配置示例：

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

必须测试：

```text
epoch 5 使用 CE
epoch 6 使用 GCE
resume from epoch 4
resume from epoch 6
checkpoint 保存当前 phase
```

B 负责运行 `W1_CE5_GCE07`，但不修改该接口。

---

## A-INFRA-4：统一 Sample Weight Provider

实现：

```python
weights = sample_weight_provider.get_weights(
    sample_ids=sample_ids,
    labels=labels,
    epoch=epoch,
    per_sample_loss=per_sample_loss,
)
```

必须支持：

```text
none
static_manifest
prototype
ema_loss
prototype_ema_hybrid
oof_manifest
relabel_manifest
```

共同要求：

- sample ID 稳定且唯一；
- 缺失样本 fail-closed；
- 重复 sample ID fail-closed；
- manifest 覆盖率必须为 100%；
- 输出每类权重均值、分位数和低权重比例；
- 权重范围强校验；
- stateful provider 支持 checkpoint save/load；
- 所有 weighting 策略不修改训练主循环。

---

## A-INFRA-5：Head EMA Hook

A 提供训练入口和 checkpoint 接口：

```python
ema_controller.update(model)
ema_controller.state_dict()
ema_controller.load_state_dict()
ema_controller.swap_to_ema()
ema_controller.restore_raw()
```

A 负责接入：

- config parser；
- train loop update hook；
- validation raw/EMA 双路径；
- checkpoint schema；
- resume；
- metric logging。

C 负责具体 EMA 模块实现和单元测试。

---

## A-INFRA-6：PEFT Hook

实现统一入口：

```python
configure_trainable_parameters(model, peft_config)
```

支持：

```text
linear_head_only
ln_post_and_projection
visual_layernorm_only
last_block_lora
```

共同要求：

- 输出 trainable parameter names；
- 输出 trainable parameter count；
- 输出每组学习率；
- checkpoint 记录 PEFT 配置；
- epoch-0 父 checkpoint 等价性测试；
- 不允许 silent unfreeze。

C 负责 PEFT/LoRA 模块本体。

---

## A-INFRA-7：Teacher–Student Hook

A 提供：

```text
student forward
teacher forward
teacher EMA update
confidence mask
consistency loss hook
checkpoint save/load
```

要求：

- teacher 不参与反向传播；
- teacher 参数和状态单独保存；
- resume 后完全恢复；
- 置信度 mask 可审计；
- teacher/student 视图配置写入日志。

C 负责 Teacher 算法逻辑。

---

## A-INFRA-8：OOF / Relabel Manifest Loader

统一 manifest schema：

```text
manifest_version
sample_id
original_label
training_label
sample_weight
quality_score
source
```

支持：

```text
weight only
relabel only
weight + relabel
```

必须审计：

- manifest hash；
- 原标签与数据集标签一致；
- training_label 合法；
- sample_weight 范围合法；
- 每类修改比例；
- 全局重标注比例；
- 未匹配样本；
- 重复样本。

B 负责 manifest 生成，A 负责训练端消费。

---

## A-INFRA-9：统一实验产物

所有训练实验必须生成：

```text
resolved_config.yaml
best.pt
last.pt
train_log.csv
reeval_best.json
per_class_metrics.csv
prediction_records.csv
artifact_manifest.json
```

B 的审计流程额外生成：

```text
protocol_audit.json
trusted_validation.json
```

`artifact_manifest.json` 至少包含：

```text
experiment_id
parent_experiment
commit_sha
config_sha256
checkpoint_sha256
train_split_sha256
val_split_sha256
prediction_sha256
manifest_sha256
```

---

## A-INFRA-10：统一结果登记

维护：

```text
results/phase3_experiments.csv
results/submission_registry.csv
```

提供命令：

```bash
python tools/register_experiment.py \
  --experiment-dir outputs/... \
  --results-csv results/phase3_experiments.csv
```

实验状态：

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

# 4. A 的主线实验

---

## A-EXP-1：GCE + EMA Loss Soft Weighting

实验 ID：

```text
W2_GCE07_EMA_LOSS
```

父实验：

```text
B2_GCE07
```

配置：

```yaml
loss:
  name: gce
  q: 0.7

sample_weighting:
  type: ema_loss
  momentum: 0.9
  warmup_epochs: 5
  ranking: classwise
  min_weight: 0.4
  max_weight: 1.0
```

权重计划：

```text
Epoch 1–5：
所有样本权重 = 1.0

Epoch 6–15：
类别内 EMA-loss 最高 10% → 0.6
其余 → 1.0

Epoch 16–50：
类别内 EMA-loss 最高 20% → 0.4
其余 → 1.0
```

实现要求：

- 使用稳定 sample ID；
- EMA loss 状态写入 checkpoint；
- resume 后继续正确更新；
- 每类独立排名；
- 不做全局排名；
- 输出每类被降权比例；
- 输出 EMA loss 与 prediction correctness 的关系；
- 输出高损失样本的类别分布。

本地 gate：

- protocol audit 通过；
- trusted class-balanced 或 trust-weighted 有正收益；
- raw macro 不明显下降；
- bottom-10% 不灾难性下降；
- 无类别塌缩。

通过后：

```text
seed=3407
必要时 seed=2026
```

只有成为 Wave 2 最佳候选后才补 TTA。

---

## A-EXP-2：GCE + Conservative Prototype Weight

实验 ID：

```text
W2_GCE07_PROTO_MIN04
```

父实验：

```text
B2_GCE07
```

权重：

\[
w_i=0.4+0.6c_i^{prototype}
\]

配置：

```yaml
loss:
  name: gce
  q: 0.7

sample_weighting:
  type: prototype
  min_weight: 0.4
  max_weight: 1.0
  classwise_percentile: true
```

要求：

- prototype 只使用 strict train split；
- validation/test 不参与；
- 不删除样本；
- 缺失权重 fail-closed；
- 输出 prototype margin 与 loss、正确率的关系；
- 特别分析 bottom-10% 类别；
- 与旧 B3_STATIC 做差异说明。

本地 gate 同 A-EXP-1。

---

## A-EXP-3：Prototype + EMA Loss Hybrid

进入条件：

```text
A-EXP-1 或 A-EXP-2 至少一个独立有效
```

实验 ID：

```text
W2_GCE07_PROTO_EMA
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

限制：

- 不加 Head EMA；
- 不加 MixUp；
- 不加 CE warmup；
- 不加 OOF；
- 不加 relabel；
- 每次只新增 hybrid 这一变量。

若两种独立策略都无效，则不运行。

---

## A-EXP-4：最佳 Weighting + Head EMA

进入条件：

1. C 的 `W1_GCE07_HEAD_EMA099` 独立有效；
2. A-EXP-1/2/3 至少一个独立有效。

实验 ID：

```text
W2_BEST_WEIGHT_HEAD_EMA
```

要求：

- 使用最佳 weighting checkpoint 配置；
- 仅新增 Head EMA；
- raw head 和 EMA head 都评估；
- 不同时加入其他策略。

---

## A-EXP-5：OOF Soft Weight 训练

B 交付：

```text
oof_soft_weight_manifest.csv
manifest.json
manifest.sha256
```

A 负责：

```text
W3_OOF_SOFT_WEIGHT
```

配置：

```yaml
sample_weighting:
  type: oof_manifest
  manifest_path: ...
  min_weight: 0.3
  max_weight: 1.0
```

训练前必须完成：

- manifest coverage audit；
- label consistency audit；
- weight range audit；
- classwise weight distribution audit。

训练后交由 B 完成质量分析。

---

## A-EXP-6：OOF Weight-Only 对照

实验 ID：

```text
W3_OOF_WEIGHT_ONLY
```

离散权重建议：

```text
high-confidence clean → 1.0
medium-confidence → 0.6
low-confidence → 0.3
```

目的：

- 判断连续权重是否必要；
- 建立 relabel 前置对照；
- 不修改标签。

---

## A-EXP-7：Conservative Relabel Weight-Only

B 交付：

```text
relabel_manifest.csv
```

A 先运行：

```text
W3_RELABEL_WEIGHT_ONLY
```

行为：

- 可疑样本降低权重；
- 保留原始标签；
- 作为 Hard Relabel 的必要对照。

---

## A-EXP-8：Conservative Hard Relabel

只有以下条件满足后执行：

- W3 OOF weighting 有明确正收益；
- B 的 relabel audit 通过；
- Weight-only 对照已完成；
- 高置信错标簇证据充分。

实验 ID：

```text
W3_RELABEL_HARD
```

限制：

```text
每个原类别最多重标注 5%
全局最多重标注 3%
```

禁止人工挑选 relabel 样本。

---

## A-EXP-9：主线多随机种子

A 负责以下候选的多 seed：

```text
最佳 EMA Loss
最佳 Prototype
最佳 Hybrid
最佳 OOF Weight
最佳 Relabel
最终组合模型
```

策略：

```text
探索：seed=42
通过 gate：seed=3407
平台主候选：seed=2026
```

A 不负责 B 的 GCE 精调 seed 扩展。

---

## A-EXP-10：最终组合

A 接收：

```text
B：最佳 GCE、OOF、Relabel 建议
C：最佳 Head EMA、PEFT、Teacher 模块
```

组合顺序：

```text
最佳鲁棒 loss
    ↓
最佳 sample weighting
    ↓
最佳 OOF/relabel
    ↓
最佳 PEFT
    ↓
可选 Head EMA / Teacher
    ↓
2-view horizontal-flip TTA
```

组合原则：

```text
每次只增加一个此前未组合验证的新变量
```

禁止直接一次性组合：

```text
GCE + EMA Loss + OOF Relabel + PEFT + Teacher
```

---

# 5. A 与 B 的接口边界

B 负责：

```text
W1_GCE05
W1_GCE09
W1_CE5_GCE07
W1_GCE07_MIXUP
OOF fold
OOF logits
sample_quality.csv
OOF weight manifest
relabel manifest
protocol audit
```

A 负责：

```text
公共 loss/schedule 接口
公共 sample weighting 接口
OOF/relabel manifest loader
使用 B 的 manifest 完成训练
最终结果登记
```

B 不直接修改：

```text
train.py
loss registry
sample weighting provider
checkpoint schema
results registry
```

---

# 6. A 与 C 的接口边界

C 负责：

```text
Head EMA 模块
PEFT 模块
LoRA 模块
EMA Teacher
ELR
```

A 负责：

```text
公共 hook
配置解析
checkpoint schema
训练循环接入
组合实验
最终平台提交
```

C 不直接修改 A 维护的公共训练入口。

---

# 7. A 的执行顺序

---

## Stage A0：公共基础设施

按顺序完成：

```text
1. Config Schema
2. Loss Registry
3. Loss Schedule
4. Sample Weight Provider
5. Head EMA Hook
6. PEFT Hook
7. Teacher Hook
8. OOF/Relabel Manifest Loader
9. Checkpoint Schema
10. Results Registry
```

验收：

- 所有现有测试通过；
- 新增接口均有单元测试；
- B/C 可只通过配置运行各自实验；
- 不允许实验逻辑散落在训练脚本中。

---

## Stage A1：Wave 2 独立实验

并行或依次运行：

```text
W2_GCE07_EMA_LOSS
W2_GCE07_PROTO_MIN04
```

完成后：

- 做单视图评估；
- 做 V2 trusted validation；
- 做 protocol audit；
- 最多保留两个候选。

---

## Stage A2：Wave 2 组合

若独立实验有效：

```text
W2_GCE07_PROTO_EMA
```

若 C 的 Head EMA 也有效：

```text
W2_BEST_WEIGHT_HEAD_EMA
```

---

## Stage A3：Wave 3 训练

接收 B 的 manifest 后：

```text
W3_OOF_SOFT_WEIGHT
W3_OOF_WEIGHT_ONLY
W3_RELABEL_WEIGHT_ONLY
条件执行 W3_RELABEL_HARD
```

---

## Stage A4：多 seed 与平台候选

对最佳主线候选：

```text
seed=3407
seed=2026
```

评估顺序：

```text
单视图本地
单视图平台候选判断
2-view TTA 本地
最终平台提交
```

不是所有模型都默认跑 TTA。

---

## Stage A5：最终组合

按独立贡献逐项叠加：

```text
最佳 loss
+ 最佳 weighting
+ 最佳 OOF/relabel
+ 最佳 PEFT
+ 可选 Head EMA/Teacher
+ 最终 TTA
```

---

# 8. A 的本地选模 Gate

所有实验先单视图评估。

必须同时检查：

```text
raw_micro
raw_macro
raw_bottom10
trusted_micro
trusted_class_balanced
trust_weighted_accuracy
rejected_micro
prediction_change_vs_parent
```

候选保留条件：

1. protocol audit 通过；
2. trusted class-balanced 或 trust-weighted 提升；
3. raw macro 无明显下降；
4. bottom-10% 无灾难性退化；
5. 无类别塌缩；
6. 多 seed 方向一致；
7. 方法复杂度与收益匹配。

TTA Gate：

只有满足以下任一条件才补 2-view TTA：

- 成为当前 Wave 最佳训练候选；
- 多 seed 方向一致；
- 本地 trusted/macro 有明显正信号；
- 准备平台提交。

---

# 9. A 的提交规则

当前基线：

```text
训练基线：
B2_GCE07 单视图 = 58.9578%

提交基线：
B2_GCE07 + horizontal-flip TTA = 59.4064%
```

后续流程：

```text
新训练模型单视图
    ↓
与 B2 单视图比较
    ↓
确认训练策略有效
    ↓
补 2-view TTA
    ↓
与 59.4064% 提交基线比较
```

每个 Wave 最多提交一个主候选。

---

# 10. A 的 Git 责任边界

A 默认维护：

```text
train.py
config schema
common/loss_registry.py
common/loss_schedule.py
common/sample_weighting.py
common/hooks.py
checkpoint schema
results/phase3_experiments.csv
results/submission_registry.csv
```

分支建议：

```text
phase3/a-main-infra
phase3/a-weighting
phase3/a-oof-training
phase3/a-final-combination
```

合并规则：

1. 公共接口先提交；
2. B/C 从公共接口 commit rebase；
3. B/C 不直接修改公共训练入口；
4. 每个实验配置单独 commit；
5. 每个运行产物绑定 commit SHA；
6. 合并前必须跑完整测试。

---

# 11. A 的交付物

A 最终必须交付：

```text
公共配置与训练接口
公共 Sample Weight Provider
Loss Schedule
EMA/PEFT/Teacher Hook
OOF/Relabel Manifest Loader
Wave 2 主线实验
Wave 3 训练实验
主线多 seed
最终组合模型
提交文件
实验 Registry
复现命令
最终方法说明
```

---

# 12. A 的优先级

## P0

```text
公共配置与训练接口
Sample Weight Provider
Loss Schedule
OOF/Relabel Manifest Loader
W2_GCE07_EMA_LOSS
W2_GCE07_PROTO_MIN04
```

## P1

```text
W2_GCE07_PROTO_EMA
W3_OOF_SOFT_WEIGHT
W3_OOF_WEIGHT_ONLY
多 seed 验证
```

## P2

```text
W2_BEST_WEIGHT_HEAD_EMA
W3_RELABEL_WEIGHT_ONLY
W3_RELABEL_HARD
最终复杂组合
```

---

# 13. 完成标准

A 的工作完成需满足：

- B/C 可在不修改公共训练入口的情况下运行实验；
- 所有状态型模块支持 checkpoint resume；
- 所有 manifest 加载 fail-closed；
- 所有实验有 resolved config 和 artifact hash；
- 所有候选有单视图结果；
- 只有主候选补 TTA；
- 所有平台提交登记完整；
- 最终模型仍是单一 CLIP ViT-B/32、单 checkpoint、单一推理流程。