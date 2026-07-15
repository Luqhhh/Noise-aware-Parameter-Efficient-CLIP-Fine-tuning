# C 负责人 Phase 3 执行计划

> 项目：Noise-aware-Parameter-Efficient-CLIP-Fine-tuning  
> 日期：2026-07-13  
> 负责人：C  
> 当前训练基线：W1_CE5_GCE05（CE warmup + GCE q=0.5），平台 bare 59.61%, TTA 60.25%  
> 当前提交基线：W1_CE5_GCE05 + 2-view horizontal-flip TTA，平台 60.25%  
> （2026-07-15 更新：已从 B2_GCE07 切换至 W1_CE5_GCE05）  
> 当前模型结构：冻结 OpenAI CLIP ViT-B/32 visual backbone，仅训练 Linear Head  
> 说明：公共训练入口、配置、checkpoint schema 和 hook 接入由 A 维护；C 负责 EMA、PEFT、LoRA、Teacher/ELR 模块本体及对应实验。

---

# 1. C 的职责范围

C 是 Phase 3 的训练稳定、参数高效视觉适配和一致性训练负责人，主要负责：

1. 实现 Linear Head EMA；
2. 运行 Head EMA 独立实验；
3. 实现并运行鲁棒 `ln_post + visual_proj` 继续训练；
4. 实现并运行 Visual LayerNorm-only tuning；
5. 实现最后一个视觉 Transformer block 的 LoRA；
6. 实现 EMA Teacher + consistency；
7. 在 Teacher 不稳定时实现 ELR 备选；
8. 向 A 提供可组合模块和明确接口。

C 不负责：

- GCE q 参数精调；
- EMA Loss、Prototype 或 OOF weighting；
- OOF、relabel manifest 生成；
- 修改公共训练入口；
- 维护最终提交 Registry；
- 最终平台提交。

---

# 2. 当前状态与进入条件

已完成：

```text
Full Forward / Feature Bank 0 mismatch
Trusted Validation v2
C-EXP-1 (Head EMA 0.99) 本地评估 — 负收益，已关闭
  - bare: 69.28% (vs gce_q07 69.59%, -0.31pp)
  - TTA:  69.61% (vs gce_q07 TTA local -0.39pp)
  - 结论: Head EMA STOP, 不进入 0.999
```

C 的任务分为三层：

```text
Layer 1：Head EMA
Layer 2：Robust PEFT
Layer 3：EMA Teacher / ELR
```

执行依赖：

- Head EMA 可立即开始；
- Robust PEFT 等待 A/B 确定最佳冻结鲁棒 checkpoint；
- Teacher 只有 OOF 或 PEFT 至少一个明确有效后进入。

---

# 3. C 与 A 的接口契约

A 提供：

```text
Head EMA hook
PEFT configure hook
Teacher–Student hook
配置 schema
checkpoint schema
统一日志与结果输出
```

C 提供：

```text
EMAController
PEFT parameter selector
LoRA modules
TeacherUpdater
ConsistencyLoss
ELR state manager
单元测试
模块说明
```

C 不直接修改：

```text
train.py
common/loss_registry.py
common/sample_weighting.py
results registry
```

如果公共接口不足，C 先提交接口需求，由 A 修改公共层。

---

# 4. Head EMA

## C-MOD-1：EMAController

实现接口：

```python
class EMAController:
    def update(self, model): ...
    def state_dict(self): ...
    def load_state_dict(self, state): ...
    def copy_to(self, model): ...
    def store(self, model): ...
    def restore(self, model): ...
```

要求：

- 只跟踪 Linear Head 参数；
- 不跟踪冻结 backbone；
- 不参与反向传播；
- FP32 master copy；
- 支持 CPU/GPU checkpoint；
- resume 后数值一致；
- 不覆盖 raw head。

单元测试：

```text
首次更新
多次更新
decay=0
decay=0.99
state_dict round trip
resume equivalence
swap/restore equivalence
```

---

## C-EXP-1：GCE q=0.7 + Head EMA 0.99

实验 ID：

```text
W1_GCE07_HEAD_EMA099
```

父实验：

```text
B2_GCE07
```

配置：

```yaml
head_ema:
  enabled: true
  decay: 0.99
  warmup_epochs: 5
  use_ema_for_validation: true
  save_raw_and_ema: true
```

每 epoch 输出：

```text
raw_head_raw_micro
ema_head_raw_micro
raw_head_trusted
ema_head_trusted
parameter_distance_raw_vs_ema
```

必须分别保存：

```text
best_raw_head.pt
best_ema_head.pt
last.pt
```

选模原则：

- EMA 不得只看单个 epoch；
- trusted/macro/bottom-10% 至少一项有稳定正信号；
- raw 与 EMA 差距需要可解释；
- 若 0.99 无收益，不测试 0.999。

**执行状态（2026-07-15）**：已完成本地评估，bare 69.28%（vs gce_q07 69.59%，-0.31pp），TTA 69.61%（-0.39pp vs gce_q07 TTA local）。**结论：STOP——Head EMA 无独立收益，C-EXP-2 (0.999) 不进入。**

---

## C-EXP-2：可选 Head EMA 0.999

进入条件：

```text
W1_GCE07_HEAD_EMA099 有稳定正收益
```

实验 ID：

```text
W1_GCE07_HEAD_EMA0999
```

目的：

- 判断更平滑的参数平均是否进一步稳定；
- 不作为默认必做实验。

若训练只有 50 epochs，需重点检查 0.999 是否更新过慢。

---

# 5. Robust PEFT 共同协议

PEFT 实验必须从已确认的最佳冻结鲁棒 checkpoint 初始化，而不是从普通 CE baseline 初始化。

共同要求：

- epoch-0 predictions 与父 checkpoint 一致；
- 输出 trainable parameter names；
- 输出 trainable parameter count；
- 输出各参数组学习率；
- 每 epoch 输出 gradient norm；
- 每 epoch 输出 parameter update norm；
- frozen 参数最大变化应为 0；
- 不允许 silent unfreeze；
- 不同时新增多个 PEFT 机制。

PEFT 评价：

```text
raw_micro
raw_macro
bottom-10%
trusted_class_balanced
trust_weighted_accuracy
feature_drift
prediction_change_vs_parent
```

---

# 6. Frozen Control (PEFT Baseline)

## C-EXP-3：W4_FROZEN_CONTROL

与 C-EXP-4 (LN+Projection) 配对，使用同一父权重、同一 split、同一超参，唯一差异是 `freeze_clip: true`。目的是隔离「继续训练」和「解冻 LN+Projection」各自的效果。

初始化：

```text
best_frozen_robust_checkpoint (W1_CE5_GCE05)
```

可训练：

```text
Linear Head（仅此）
```

冻结：

```text
全部 CLIP visual backbone（包括 ln_post、visual.proj、所有 Transformer blocks）
```

配置：

```yaml
experiment_id: C2_FROZEN (configs/c2_frozen.yaml)

epochs: 12
early_stop_patience: 5

optimizer:
  head_lr: 1.0e-4
  weight_decay: 1.0e-4

loss:
  name: gce
  q: 0.5
```

进入前检查：

```text
父 checkpoint 完整加载
epoch-0 prediction mismatch = 0
```

---

# 7. ln_post + visual_proj Continue Training

## C-EXP-4：W4_ROBUST_LNPROJ

初始化：

```text
best_frozen_robust_checkpoint
```

可训练：

```text
Linear Head
visual.ln_post
visual.proj / visual_projection
```

冻结：

```text
所有 Transformer blocks
patch embedding
positional embedding
class embedding
```

配置：

```yaml
experiment_id: W4_ROBUST_LNPROJ

epochs: 12
early_stop_patience: 5

optimizer:
  head_lr: 1.0e-4
  backbone_lr: 1.0e-6
  weight_decay: 1.0e-4
```

进入前检查：

```text
父 checkpoint 完整加载
epoch-0 prediction mismatch = 0
trainable names 精确匹配
```

停止条件：

- trusted accuracy 单 epoch 下降超过 0.5pp；
- backbone grad norm 爆炸；
- feature drift 明显但无指标收益；
- 连续 5 epoch 无改善。

---

# 8. Visual LayerNorm-Only Tuning

## C-EXP-5：W4_ROBUST_VISUAL_LN_ONLY

进入条件：

```text
C-EXP-4 (W4_ROBUST_LNPROJ) 无收益或出现过拟合
```

可训练：

```text
visual Transformer 各 block LayerNorm scale/bias
visual ln_post
Linear Head
```

冻结：

```text
attention
MLP
visual projection
embeddings
```

建议配置：

```yaml
epochs: 12
early_stop_patience: 5

optimizer:
  head_lr: 1.0e-4
  layernorm_lr: 5.0e-6
```

需要输出：

```text
每层 LayerNorm 参数更新幅度
各层 gradient norm
feature drift
```

目的：

- 允许表征重新标定；
- 降低对噪声标签的过度适配；
- 比直接解冻 attention/MLP 更保守。

---

# 9. Last-Block LoRA

## C-MOD-2：LoRA 模块

实现：

```text
last visual block
attention in projection
attention out projection
```

第一版固定：

```yaml
rank: 4
alpha: 8
dropout: 0.0
target_block: 11
```

要求：

- 原始权重冻结；
- 仅 LoRA 参数可训练；
- merge/unmerge 数值一致；
- checkpoint 可单独保存和恢复；
- train/eval 一致；
- 不同时搜索 rank/alpha/dropout。

单元测试：

```text
LoRA disabled 等价原模型
zero-init 等价原模型
merge/unmerge
state_dict round trip
trainable parameter count
```

---

## C-EXP-6：W4_ROBUST_LASTBLOCK_LORA_R4

进入条件：

```text
C-EXP-4 或 C-EXP-5 至少一个有正信号
```

配置：

```yaml
optimizer:
  head_lr: 1.0e-4
  lora_lr: 1.0e-5

epochs: 15
early_stop_patience: 5
```

评价重点：

```text
trusted improvement
macro improvement
bottom-10%
feature drift
overfitting speed
```

如果 seed=42 无正信号，不搜索其他 rank。

---

# 10. EMA Teacher

## C-MOD-3：TeacherUpdater

实现：

```python
teacher_param = decay * teacher_param + (1 - decay) * student_param
```

要求：

- teacher 不参与反向传播；
- teacher eval mode；
- BN 不适用，但状态管理仍需完整；
- teacher checkpoint 单独保存；
- resume 后 teacher/student 均恢复；
- EMA 更新位置固定在 optimizer step 后。

---

## C-MOD-4：Consistency Loss

支持：

```text
KL divergence
soft pseudo-label CE
confidence mask
ramp-up weight
```

第一版推荐：

```text
teacher confidence >= 0.90
才参与 consistency
```

总损失：

\[
L =
L\_{robust-supervised}

- \lambda(t)L\_{consistency}
  \]

其中：

```text
lambda 从 0 线性增长到 0.5
ramp epochs = 10
```

---

## C-EXP-7：W5_EMA_TEACHER_FLIP

进入条件：

- OOF weighting/relabel 明确有效，或
- Robust PEFT 明确有效。

Teacher view：

```text
standard preprocessing
或低概率 horizontal flip
```

Student view：

```text
horizontal flip probability = 0.5
```

禁止：

```text
ColorJitter
RandomGrayscale
GaussianBlur
vertical flip
强 RandAugment
大面积 Random Erasing
```

配置：

```yaml
teacher:
  ema_decay: 0.999
  confidence_threshold: 0.90

consistency:
  max_weight: 0.5
  ramp_epochs: 10
```

必须输出：

```text
teacher confidence coverage
consistency active sample ratio
teacher/student disagreement
pseudo-label class distribution
confirmation-bias indicators
```

停止条件：

- teacher 错误高置信预测持续增加；
- 伪标签集中到少数类别；
- trusted/macro 下降；
- consistency active ratio 异常低或异常高；
- 无独立收益。

最终推理只使用：

```text
单个 EMA Teacher checkpoint
```

---

# 11. ELR 备选

## C-MOD-5：ELR State Manager

如果 EMA Teacher 实现成本或稳定性问题较高，先实现 ELR。

需要为每个样本维护历史预测 target：

```text
prediction_history[sample_id]
```

要求：

- 使用稳定 sample ID；
- checkpoint 保存；
- resume 恢复；
- 不允许样本错位；
- 状态覆盖率 100%。

---

## C-EXP-8：W5_GCE_ELR

配置建议：

```yaml
warmup_epochs: 5
prediction_history_momentum: 0.9
```

实验只测试一个正则强度，避免大网格。

ELR 与 EMA Teacher 二选一优先：

```text
Teacher 复杂或不稳定 → 先 ELR
Teacher 基础设施成熟 → 优先 Teacher
```

---

# 12. C 的实验选模规则

所有实验先做单视图。

通过本地 gate 后，才交给 A 作为组合或 TTA 候选。

Gate：

1. protocol audit 通过；
2. trusted class-balanced 或 trust-weighted 提升；
3. macro 不明显下降；
4. bottom-10% 不灾难性退化；
5. trainable 参数符合预期；
6. frozen 参数无变化；
7. 多 seed 方向一致；
8. 方法复杂度与收益匹配。

TTA 不由 C 对所有实验默认执行。

只有：

```text
当前 Wave 最佳候选
准备平台提交
```

才补 2-view horizontal-flip TTA。

---

# 13. C 的执行顺序

## Stage C0：接口确认

确认 A 已提供：

```text
Head EMA hook
PEFT hook
Teacher hook
checkpoint schema
统一日志
```

---

## Stage C1：Head EMA

```text
实现 EMAController
运行 W1_GCE07_HEAD_EMA099
条件运行 0.999
```

交付 A：

```text
模块代码
单元测试
最佳配置
raw/EMA 对比
```

---

## Stage C2：PEFT 准备

等待 A/B 确定最佳冻结鲁棒 checkpoint，同时完成：

```text
trainable parameter audit
epoch-0 equivalence test
LayerNorm selector
LoRA 单元测试
feature drift logger
```

---

## Stage C3：Robust PEFT

顺序：

```text
C-EXP-3 (Frozen Control) 与 C-EXP-4 (LN+Proj) 并行
若无收益 → C-EXP-5 (W4_ROBUST_VISUAL_LN_ONLY)
若有正信号 → C-EXP-6 (W4_ROBUST_LASTBLOCK_LORA_R4)
```

---

## Stage C4：Teacher / ELR

只有 OOF 或 PEFT 有效后：

```text
W5_EMA_TEACHER_FLIP
或
W5_GCE_ELR
```

---

## Stage C5：组合交付

将有效模块交给 A：

```text
Head EMA
最佳 PEFT
Teacher/ELR
```

A 负责最终组合实验和提交。

---

# 14. C 的 Git 责任边界

C 默认维护：

```text
models/ema/
models/peft/
models/lora/
training/teacher_student/
training/elr/
tests/test_ema.py
tests/test_peft.py
tests/test_lora.py
tests/test_teacher.py
tests/test_elr.py
configs/phase3/ema/
configs/phase3/peft/
configs/phase3/teacher/
```

C 不直接修改：

```text
train.py
common/loss_registry.py
common/sample_weighting.py
checkpoint schema
results registry
```

建议分支：

```text
phase3/c-head-ema
phase3/c-peft
phase3/c-lora
phase3/c-teacher
```

---

# 15. C 的交付物

C 最终必须交付：

```text
EMAController
Head EMA 实验结果
PEFT parameter selector
ln_post + projection 实验
LayerNorm-only 实验
LoRA 模块与实验
TeacherUpdater
ConsistencyLoss
EMA Teacher 实验
ELR 备选
所有单元测试
模块使用说明
与 A 的接口说明
```

---

# 16. C 的优先级

## P0

```text
C-EXP-3 (Frozen Control)
C-EXP-4 (ln_post + projection)
PEFT parameter audit
```

## P1

```text
C-EXP-5 (Visual LayerNorm-only)
C-EXP-6 (Last-block LoRA)
feature drift logging
```

## P2

```text
C-EXP-7 (EMA Teacher)
C-EXP-8 (ELR)
C-EXP-2 (Head EMA 0.999)
```

---

# 17. 完成标准

C 的工作完成需满足：

- Head EMA 支持完整 save/resume；
- raw 与 EMA head 可同时评估；
- PEFT trainable 参数精确可审计；
- epoch-0 与父 checkpoint 等价；
- frozen 参数变化为 0；
- LoRA merge/unmerge 正确；
- Teacher 不参与反向传播；
- Teacher/ELR 状态可恢复；
- 所有有效模块可被 A 通过配置组合；
- C 不绕过 A 的公共 hook 修改训练主循环。
