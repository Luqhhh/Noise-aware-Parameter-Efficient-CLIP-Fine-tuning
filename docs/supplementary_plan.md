Supplementary Experiments — 完整修订方案

> 本文档用于规划 Phase 3 主线之外、但可能突破当前平台性能瓶颈的补充实验。  
> 核心原则：减少低价值参数网格，优先完成能够回答因果问题、改变性能上限的实验。  
> 所有最终方案必须满足赛事约束：仅使用官方当前阶段训练数据、CLIP ViT-B/32、单模型或单一推理流程，并保证训练过程自动化、可复现。

---

0. 当前状态（2026-07-18）
   0.1 当前最佳结果

   **平台最佳（截至 2026-07-18）：**

   | 方法 | 平台 Bare | 平台 TTA | 本地 Val | Split |
   |:--|:--:|:--:|:--:|:--:|
   | **S_OOF_ZERO_0001** (binary zero p<0.001) | **59.96%** | **60.28%** | 69.37% | d3 |
   | S_D3_MIXUP (d3_strict 无权重控制) | 59.86% | — | 69.47% | d3 |
   | **W1_GCE05_MIXUP** (MixUp + GCE q=0.5) | **59.86%** | **60.36%** | 71.16% | ref |
   | S_MIXUP_CE5 (CE5 warmup + MixUp) | 59.70% | 60.48% | 70.25% | ref |
   | CE 5 epoch → GCE q=0.5 | 59.61% | 60.25% | 73.14% | ref |
   | 纯 GCE q=0.5 | 59.62% | 60.16% | 69.49% | ref |
   | S_OOF_DISCRETE (3-tier OOF weight) | 59.28% | 59.28% | 68.65% | d3 |
   | S_OOF_ZERO_001 (binary zero p<0.01) | — | — | 69.02% | d3 |

   当前 Bare 最佳：**S_OOF_ZERO_0001 (59.96%)** ← 首个 Bare 显著超过 MixUp 基线的 OOF 方法
   当前 TTA 最佳：**S_MIXUP_CE5 (60.48%)** — 但 Bare 未通过 gate，不视为训练策略有效

   **d3_strict 控制对比（2026-07-18）：**

   | 实验 (d3_strict) | 排除 | Bare | vs D3_MIXUP | 判定 |
   |:--|:--:|:--:|:--:|:--|
   | S_D3_MIXUP | 0% | 59.86% | — | 控制基线 |
   | **S_OOF_ZERO_0001** | 7% (p<0.001) | **59.96%** | **+0.10pp** | **confirmed — 首个有效 OOF 改进** |
   | S_OOF_ZERO_001 | 12% (p<0.01) | — | — | pending |
   | S_OOF_DISCRETE | 3-tier | 59.28% | −0.58pp | eliminated |

   OOF binary hard-zero (p<0.001, 7% 排除) 在同 split 配对控制下验证有效。

   **Batch 1 — MixUp 参数消融（全部完成, 2026-07-17）：**

   | 实验 | α | p | 本地 Val | vs 父基线 | 判定 |
   |:--|:--:|:--:|:--:|:--:|:--|
   | W1_GCE05_MIXUP (父基线) | 0.2 | 0.2 | **71.16%** | — | 最优 |
   | S_MIXUP_A01 | 0.1 | 0.2 | 70.29% | −0.87pp | eliminated |
   | S_MIXUP_A04 | 0.4 | 0.2 | 70.42% | −0.74pp | eliminated |
   | S_MIXUP_P04 | 0.2 | 0.4 | 70.41% | −0.75pp | eliminated |
   | S_MIXUP_CE5 | 0.2 | 0.2 | 70.25% | −0.91pp | eliminated |

   结论：α=0.2, p=0.2 确认为冻结 MixUp 最优配置。按 §13.1 关闭 MixUp 参数搜索。

   **Batch 2 — PEFT E0–E3（父模型: W1_GCE05_MIXUP, 71.16%; 2026-07-17）：**

   | 实验 | 配置 | Gate | Best Val | vs E0 | 判定 |
   |:--|:--|:--:|:--:|:--:|:--|
   | E0 (Frozen Control) | 冻结 CLIP, 仅训练 head | ✓ Δ=0 | 71.17% | — | baseline |
   | E1 (LN, bb_lr=1e-6) | LN-only, head+backbone | ✓ Δ=0 | 71.18% | +0.01pp | no gain |
   | E2 (LN, bb_lr=5e-7) | LN-only, 更低 backbone LR | ✓ Δ=0 | 71.21% | +0.04pp | no gain |
   | E3 (BB-only LN) | LN-only, classifier 冻结 | ✓ Δ=0 | 71.16% | −0.01pp | neutral |

   结论：四组全部在 ±0.05pp 内，无可测差异。E3 诊断：LN-only backbone 训练
   既不破坏也不改善表征，feature drift 极小不足以改变分类决策。
   按 §13.3 关闭普通 PEFT；E4 (FeatDistill) / E5 (seed 3407) / LoRA (§9) 均不执行。

   **P2 — OOF 路线（2026-07-18）：**

   | 实验 | 平台 Bare | 平台 TTA | 本地 Val | 判定 |
   |:--|:--:|:--:|:--:|:--|
   | **S_OOF_ZERO_0001** (binary, p<0.001) | **59.96%** | **60.28%** | 69.37% | **confirmed — +0.10pp over control** |
   | S_OOF_ZERO_001 (binary, p<0.01) | — | — | 69.02% | pending |
   | S_OOF_DISCRETE (3-tier tertile) | 59.28% | 59.28% | 68.65% | eliminated |
   | S_OOF_ZERO_005 (binary, p<0.05) | — | — | — | config ready |
   | S_OOF_ZERO_010 (binary, p<0.10) | — | — | — | config ready |
   | S_OOF_ZERO_0001_FF (final_fit) | — | — | — | config ready |

   OOF binary zero p<0.001 是同 split 控制下首个验证有效的改进：
   - Bare +0.10pp over D3_MIXUP paired control
   - Bare-TTA gap 仅 0.32pp（MixUp 为 0.50pp），模型更稳定
   - 3-tier 软降权无效；binary 硬排除方向正确


0.2 已确认的核心现象
本地原始验证准确率不能可靠预测平台表现。
CE warmup 使本地验证分数提高约 3.65pp；平台 Bare 与纯 GCE q=0.5 基本持平；
MixUp 本地分数低于 CE warmup，却取得当前平台最优。
当前瓶颈主要是泛化评价不一致，而不是训练集拟合不足。
约 11–13pp 的 local-platform gap 表明本地验证标签质量、评价目标和平台人工精标
测试集之间存在显著不一致。

**MixUp 参数消融（Batch 1 完成）：** α=0.2, p=0.2 确认为最优。α=0.1（弱混合）、
α=0.4（强混合）、p=0.4（高频混合）、CE5 warmup 四种变体本地 Val 均显著低于父基线。
强混合会破坏细粒度结构（A04 本地 −0.74pp），弱混合正则化不足（A01 本地 −0.87pp）。

**PEFT LN-only 确认无效（Batch 2 完成）：** 在正确的 MixUp 父模型上重跑 E0–E3，
四组全部在 ±0.05pp 内，所有 epoch-0 gate 通过（Δ=0）。E3（backbone-only LN,
classifier 冻结）既不退化也不改善——说明 LN 的 feature drift 极小（cos_dist < 1e-4），
不足以改变分类决策。按计划规则关闭普通 PEFT，不执行 E4/E5/LoRA。

**OOF trust-weighted 训练无平台增益（Batch 3 完成）：** S_OOF_DISCRETE
Bare = TTA = 59.28%，低于父基线 MixUp bare 59.86%。OOF 信号作为训练干预手段
暂未证明有效。但 Trusted Validation 分析发现 rejected accuracy 是有效诊断指标：
模型在低 OOF 可信样本上 accuracy 越低 → 越抗拒拟合噪声 → 平台 Bare 越高。
已将 rejected accuracy 集成到 evaluate_candidate 输出中。

冻结 CLIP + 线性分类头已接近当前特征空间的局部上限（平台 Bare ~60%）。
Loss、普通增强、Head EMA、Cosine Head、继续训练、MixUp 参数搜索、普通 PEFT、
OOF 降权训练等方向均未能突破该上限。
---

1. 总体目标与优先级
   1.1 总目标
   围绕三个结构性问题开展实验：
   冻结特征条件下，MixUp 是否还有低成本可复现增益；
   时序预测或 OOF 可信度能否识别并抑制错误标签；
   在受控漂移和可信监督下，PEFT 是否能突破冻结 CLIP 的特征瓶颈。
   1.2 修订后的优先级

```text
P0：实验可信性与统一契约
    ├── 修复实验登记和平台 Bare/TTA 对应关系
    ├── 统一 checkpoint 重评估
    └── 建立 feature drift 与 prediction-change 诊断

P0：低成本冻结模型候选
    ├── S-MIXUP-1：CE5 warmup + MixUp
    ├── S-MIXUP-A01：alpha=0.1
    ├── S-MIXUP-A04：alpha=0.4
    └── S-MIXUP-P04：probability=0.4

P1：S-ELR
    └── 用样本历史预测抑制后期噪声记忆

P1：受控 PEFT E0–E5
    ├── Paired Frozen Control
    ├── LN-only 双学习率
    ├── Backbone-only 因果对照
    ├── Frozen Feature Distillation
    └── 最佳候选第二 seed

P2：S-OOF 样本可信度
    ├── 3-fold group-aware OOF
    ├── trusted validation
    ├── trust-weighted GCE/MixUp
    └── Frozen/PEFT × 普通/可信加权 2×2 实验

P3：条件式 Last-block LoRA
    └── 仅在最小 LN-only PEFT 已被证明有效时执行

P4：GCE q 补充搜索和轻增强
    └── 仅在算力空闲时执行
```

1.3 明确降级的实验
以下实验不进入第一批：
GCE q=0.3、0.4、0.6 全部并行搜索；
MixUp probability=0.5；
大规模 alpha/probability 二维网格；
LoRA rank、alpha、block、dropout 搜索；
多层 Transformer 解冻；
全参数微调。
原因：这些实验无法直接解决当前约 9.6pp 的平台缺口，并会在本地选择器失真的情况下制造大量无法排序的候选。

---

2. 全局实验契约
   以下约束适用于本文档中的所有实验。
   2.1 数据与模型约束

```yaml
data:
  official_stage_data_only: true
  split_seed: 42
  split_dir: outputs/d3_strict/seed42  # mandatory for all new experiments
  duplicate_group_aware_split: true

model:
  backbone: CLIP ViT-B/32
  classifier: linear
  ensemble: false
```

要求：
不引入外部数据；
测试集不得参与训练、自监督或超参数选择；
重复图片或冲突图片必须按 group 划分，禁止跨 train/val 泄漏；
最终提交必须来自单模型和单一推理流程。
   **2026-07-17 决议：所有新实验统一使用 `split_dir: outputs/d3_strict/seed42`。**
   `outputs/ref/seed42` 已弃用。原因：d3_strict 的 group-aware 分法防止同图跨 train/val 泄漏，
   且所有 OOF weight manifest 绑定 d3_strict 的 train 集。
2.2 Seed 协议

```text
seed 42   ：探索和初筛
seed 3407 ：最佳候选复验
seed 2026 ：仅在前两个 seed 方向一致、且接近最终提交时补充
```

不对明显失败配置补多 seed。
2.3 统一指标
所有训练实验至少输出：

```text
raw_micro_accuracy
macro_accuracy
bottom10_accuracy
trusted_accuracy（S-OOF 完成后）
train_loss
validation_loss
prediction_change_rate_vs_parent
```

涉及可训练 backbone 的实验额外输出：

```text
head_grad_norm
backbone_grad_norm
mean_feature_cosine_distance
p95_feature_cosine_distance
logit_kl_vs_parent
trainable_parameter_relative_delta
```

2.4 Checkpoint 复现要求
每个候选必须产生：

```text
best.pt
last.pt
eval_best.json
eval_last.json
reeval_best.json
prediction_records_best.csv
artifact_manifest.json
```

PEFT 实验还需要：

```text
feature_drift_best.json
gradient_diagnostics.jsonl
epoch0_gate.json
```

`reeval_best.json` 必须在训练进程退出后，重新加载 `best.pt` 独立生成。
2.5 平台提交规则
平台比较必须遵守：

```text
第一步：Bare single-view
第二步：只有 Bare 通过 Gate，才生成 Flip TTA
```

TTA 不能单独证明训练方法有效。
平台结果登记必须绑定：

```text
experiment_id
git_commit
config_sha256
checkpoint_sha256
seed
bare_submission_sha256
tta_submission_sha256
platform_bare
platform_tta
```

---

3. P0：实验登记与证据闭环
   3.1 统一事实源
   以 `submission_registry.csv` 或等价结构作为平台结果唯一事实源。README、阶段计划和实验总结均由该登记表生成，不再人工维护重复分数。
   每行必须保持固定 schema，CI 检查：
   列数一致；
   必填字段非空；
   Bare/TTA checkpoint 一致；
   TTA 必须存在对应 Bare；
   分数范围合法；
   同一 submission SHA 不得对应多个实验。
   3.2 当前结果补齐
   在启动新实验前完成：
   确认 MixUp Bare 59.86% 与 TTA 60.36% 使用同一 checkpoint；
   补齐 MixUp checkpoint SHA、config SHA 和提交文件 SHA；
   补齐 C-EXP-4、C-EXP-5 的最终 `reeval_best`；
   将失败 PEFT 的诊断结果归档，避免重复实验；
   修复所有 CSV 列错位或字段漂移。

---

4. P0：冻结模型的低成本候选
   第一批只运行 4 个具有明确比较价值的候选。
   统一父配置：

```yaml
loss:
  name: gce
  q: 0.5
  probability_epsilon: 1.0e-7

model:
  freeze_clip: true
  classifier: linear

train:
  lr: 5.0e-3
  weight_decay: 1.0e-4
  scheduler: cosine
  warmup_epochs: 2
  batch_size: 128
  epochs: 50
  early_stop_patience: 10

augmentation:
  preset: A0
```

---

4.1 S-MIXUP-1：CE5 warmup + MixUp
假设
CE warmup 可能更快形成可分决策边界，MixUp 可能防止该边界继续拟合错误标签。二者作用不同，可能产生协同。
配置

```yaml
experiment_id: S_MIXUP_CE5

loss:
  schedule:
    - start_epoch: 1
      end_epoch: 5
      name: cross_entropy
    - start_epoch: 6
      end_epoch: 50
      name: gce
      q: 0.5
      probability_epsilon: 1.0e-7

mixup:
  enabled: true
  alpha: 0.2
  probability: 0.2
```

成功标准

```text
Bare > 59.86%：证明 warmup 与 MixUp 存在平台协同
TTA  > 60.36%：成为新平台基线
```

失败解释
本地升、平台不升：warmup 仍主要改善含噪验证拟合；
本地和平台均不升：MixUp 已覆盖 warmup 的主要作用；
Bare 下降但 TTA 上升：不视为训练策略有效，只作为推理现象记录。

---

4.2 S-MIXUP-A01：弱 MixUp

```yaml
experiment_id: S_MIXUP_A01

loss:
  name: gce
  q: 0.5

mixup:
  enabled: true
  alpha: 0.1
  probability: 0.2
```

## 假设：更弱的混合保留更多细粒度纹理和局部形态。

4.3 S-MIXUP-A04：强 MixUp

```yaml
experiment_id: S_MIXUP_A04

loss:
  name: gce
  q: 0.5

mixup:
  enabled: true
  alpha: 0.4
  probability: 0.2
```

## 假设：更强的标签和输入平滑可能进一步抑制噪声记忆，但存在破坏细粒度结构的风险。

4.4 S-MIXUP-P04：提高混合频率

```yaml
experiment_id: S_MIXUP_P04

loss:
  name: gce
  q: 0.5

mixup:
  enabled: true
  alpha: 0.2
  probability: 0.4
```

## 第一轮不运行 `p=0.5`。只有 `p=0.4` 明显优于 `p=0.2` 时，才考虑补充更高概率。

4.5 第一批决策规则
本地候选选择不能只按 raw micro 排序。平台提交优先级：
不同配置带来的预测变化率足够大；
macro 或 Bottom-10% 没有明显退化；
本地结果不属于明显崩塌；
优先提交能够回答独立假设的候选；
每个候选先 Bare，Bare 有正信号后再 TTA。
第一批结束后：

```text
如果 S_MIXUP_CE5 成为新 Bare 最优：
    将其设为 ELR 和 PEFT 的候选父模型
否则：
    保留原 MixUp q=0.5 为父模型

如果 A01/A04/P04 有明确正收益：
    固定单一最佳 MixUp 配置
否则：
    不继续扩大 MixUp 网格
```

---

5. P1：S-ELR — Early Learning Regularization
   5.1 研究问题
   模型是否在训练后期逐步记忆错误标签？历史预测约束能否保留 early-learning 阶段形成的较可靠决策？
   5.2 基本形式
   对每个原始训练样本 (i) 维护历史预测：
   [
   t_i \leftarrow \beta t_i + (1-\beta)p_i
   ]
   总损失：
   [
   L = L_{\mathrm{GCE}} + \lambda_{\mathrm{ELR}}
   \log(1-p_i^\top t_i)
   ]
   注意：实现时需要根据所采用 ELR 公式确认符号和数值稳定处理，并用单元测试验证“预测与历史 target 越一致，惩罚越小”。
   5.3 第一轮配置

```yaml
experiment_id: S_ELR_BASE
parent: <CURRENT_BEST_FROZEN_MODEL>

loss:
  name: gce
  q: 0.5

elr:
  enabled: true
  momentum: 0.9
  target_weight: 1.0
  warmup_epochs: 10
  ramp_epochs: 10
  update_mode: per_sample_visit
  storage_dtype: float32
  checkpoint_memory: true
```

关键实现约束
每个样本必须有稳定的 dataset index；
running target 按样本访问时更新，不在 epoch 末统一覆盖；
checkpoint 保存并恢复：
running predictions；
当前 ELR ramp 状态；
样本索引映射版本；
resume 后同一 batch 的 loss 应与不中断训练一致；
记录：
GCE loss；
ELR loss；
当前有效 (\lambda)；
running target entropy；
高惩罚样本比例。
5.4 ELR 与 MixUp 的交互
第一轮采用最容易归因的规则：

```text
未执行 MixUp 的 batch：
    计算 GCE + ELR

执行 MixUp 的 batch：
    只计算 MixUp 分类损失
    不更新原始样本的 ELR running target
```

原因：
混合样本不对应单一真实样本；
直接混合历史 target 会引入额外实现变量；
第一轮需要先判断 ELR 本身是否有效。
只有基础 ELR 有正信号后，再测试：
[
t_{\mathrm{mix}}=\lambda t_i+(1-\lambda)t_j
]
5.5 ELR 的测试要求
至少包含：
历史 target EMA 更新正确；
相同 dataset index 更新同一 memory slot；
不同样本不会串位；
warmup 期间 ELR 权重为 0；
ramp 后达到目标值；
resume 前后 loss 一致；
MixUp batch 不错误更新 memory；
float32 memory 的归一化和数值范围合法。
5.6 ELR 成功标准
本地：
macro/trusted 指标改善；
Bottom-10% 不显著下降；
后期验证退化减弱；
多 seed 方向一致。
平台：

```text
Bare >= 当前父模型 Bare + 0.30pp：明确正信号
Bare 提升 0.10–0.30pp：待第二 seed 验证
Bare 变化在 ±0.10pp：视为持平
Bare 下降：关闭该配置
```

## 只有 Bare 有正信号才叠加 TTA。

6. P1：S-PEFT — 受控轻量 Backbone 适配
   6.1 核心假设
   冻结 CLIP 的 512 维视觉特征可能不足以区分 500 个细粒度类别，但现有实验没有证明普通 PEFT 有效。
   本阶段不直接解冻更大范围参数，也不立刻执行 LoRA。首先通过：
   paired frozen control；
   双学习率 LN-only；
   backbone-only 因果对照；
   feature drift 审计；
   frozen-parent feature distillation；
   判断 PEFT 退化的具体来源。
   6.2 父模型选择
   父模型不提前固定为某个目录，而按平台 Bare 选择：

```text
如果 S_MIXUP_CE5 Bare > 当前 MixUp Bare：
    CURRENT_BEST_BARE_PARENT = S_MIXUP_CE5 best.pt
否则：
    CURRENT_BEST_BARE_PARENT = 当前 MixUp q=0.5 best.pt
```

父模型一旦确定，E0–E5 期间不得更换。
必须记录：

```text
parent_experiment_id
parent_checkpoint_sha256
parent_config_sha256
```

6.3 E0–E4 统一契约
所有 E0–E4：
`split_seed=42`；
`train_seed=42`；
同一父 checkpoint；
同一数据顺序；
同一 GCE/MixUp 配置；
同一 epoch 和 scheduler；
同一 classifier LR；
唯一区别为可训练参数、backbone LR 或 feature distillation。
建议统一：

```yaml
loss:
  name: gce
  q: 0.5

mixup:
  enabled: true
  alpha: <PARENT_MIXUP_ALPHA>
  probability: <PARENT_MIXUP_PROBABILITY>

train:
  epochs: 15
  early_stop_patience: 5
  lr: 1.0e-4
  scheduler: cosine
  warmup_epochs: 2
```

## 如果父模型是 CE5+MixUp，继续训练阶段不再重复 CE warmup，统一使用 GCE q=0.5，避免不同 E 实验引入额外 loss schedule 变量。

6.4 Epoch-0 Gate
训练开始前，所有子实验必须重新加载父 checkpoint 并验证：

```text
prediction_mismatch = 0
val_accuracy_diff <= 0.02pp
classifier_weight_diff = 0
parent_feature_diff <= numerical_tolerance
```

参数和梯度检查：
实验 Visual grad Classifier grad
E0 0 非零
E1 目标 LN 非零 非零
E2 目标 LN 非零 非零
E3 目标 LN 非零 0
E4 目标 LN 非零 非零
未通过 Epoch-0 Gate 的实验作废，不允许进入比较。

---

6.5 E0：Paired Frozen Control

```yaml
experiment_id: S_PEFT_E0_FROZEN
init_checkpoint: <CURRENT_BEST_BARE_PARENT>

model:
  freeze_clip: true
  train_classifier: true

train:
  epochs: 15
  early_stop_patience: 5
  lr: 1.0e-4
```

目的
测量在相同继续训练条件下：
classifier 自然变化；
early stopping 波动；
继续训练是否本身产生收益；
E1–E4 的改变量是否超过随机和训练噪声。
E0 不能用旧 Frozen Control 直接替代，除非旧实验与 E1–E4 的父模型、loss、MixUp、epoch、LR、scheduler 完全一致。

---

6.6 E1：LN-only，Backbone LR = 1e-6

```yaml
experiment_id: S_PEFT_E1_LN_1E6
init_checkpoint: <CURRENT_BEST_BARE_PARENT>

model:
  freeze_clip: false
  train_layernorm_only: true
  train_visual_proj: false
  unfreeze_last_n_blocks: 0
  train_classifier: true

train:
  epochs: 15
  early_stop_patience: 5
  lr: 1.0e-4
  backbone_lr: 1.0e-6
  backbone_weight_decay: 0.01
```

目的
判断最小规模视觉参数适配是否优于 paired frozen control。

---

6.7 E2：LN-only，Backbone LR = 5e-7

```yaml
experiment_id: S_PEFT_E2_LN_5E7
init_checkpoint: <CURRENT_BEST_BARE_PARENT>

model:
  freeze_clip: false
  train_layernorm_only: true
  train_visual_proj: false
  unfreeze_last_n_blocks: 0
  train_classifier: true

train:
  epochs: 15
  early_stop_patience: 5
  lr: 1.0e-4
  backbone_lr: 5.0e-7
  backbone_weight_decay: 0.01
```

目的
排除 E1 退化只是 backbone LR 偏大。
第一轮不继续搜索 `2e-7、3e-7、8e-7`。若 E1、E2 均呈现相同退化机制，应停止普通 LR 网格。

---

6.8 E3：Backbone-only LayerNorm

```yaml
experiment_id: S_PEFT_E3_BACKBONE_ONLY
init_checkpoint: <CURRENT_BEST_BARE_PARENT>

model:
  freeze_clip: false
  train_layernorm_only: true
  train_visual_proj: false
  unfreeze_last_n_blocks: 0
  train_classifier: false

train:
  epochs: 5
  early_stop_patience: 3
  backbone_lr: 1.0e-6
  backbone_weight_decay: 0.01
```

目的与解释
结果 解释
E3 与 E1 同样退化 噪声监督直接破坏视觉表征
E3 稳定、E1 退化 classifier/backbone 联合优化或 LR 耦合存在问题
E3 提升 视觉适配可能有效，classifier 更新掩盖了收益
E3 几乎不变且 drift 极小 学习率或梯度过小，PEFT 未真正发生
E3 是诊断实验，不因本地 raw micro 小幅上升就直接提交平台。

---

6.9 E4：LN-only + Frozen Feature Distillation
E4 使用 E1/E2 中诊断更好的 backbone LR。

```yaml
experiment_id: S_PEFT_E4_LN_FEATDISTILL
init_checkpoint: <CURRENT_BEST_BARE_PARENT>

model:
  freeze_clip: false
  train_layernorm_only: true
  train_visual_proj: false
  unfreeze_last_n_blocks: 0
  train_classifier: true

feature_distillation:
  enabled: true
  teacher: frozen_parent
  compare_after_visual_projection: true
  normalize_features: true
  target_loss_ratio: 0.10-0.20

train:
  epochs: 15
  early_stop_patience: 5
  lr: 1.0e-4
  backbone_lr: <BEST_OF_E1_E2>
  backbone_weight_decay: 0.01
```

总损失：
[
L =
L\_{\mathrm{GCE/MixUp}}

- \lambda*{\mathrm{feat}}
  \left[
  1-\cos(f*\theta(x),f\_{\theta_0}(x))
  \right]
  ]
  实现约束：
  parent CLIP 全程冻结并处于 eval 模式；
  parent/student 使用同一批输入和同一增强视图；
  parent feature 必须 `detach()`；
  在 CLIP visual projection 后比较；
  两侧 feature 均进行 L2 normalization；
  feature loss 不直接约束 classifier logits；
  前 200 step 统计两个 loss 的平均量级；
  选择使 feature 项约占总 loss 10%–20% 的系数；
  第一轮只运行一个校准后的系数，不展开大网格。
  ***
  6.10 E5：最佳候选补 Seed 3407
  只从 E1–E4 中选择一个候选。
  进入 E5 的最低条件：

```text
相对 E0：
trusted accuracy（可用后） >= +0.20pp
或 macro accuracy >= +0.20pp

并且：
Bottom-10% 下降不超过 0.30pp
feature drift 未持续失控
best checkpoint 重评估可复现
训练曲线没有明显后期崩塌
```

如果 E1–E4 均不满足条件：

```text
关闭普通 PEFT
不执行 E5
不执行 LoRA
优先转入 S-OOF
```

---

6.11 PEFT Feature Drift 审计
每个 epoch 对固定验证子集计算：
[
d_i = 1-\cos(f_\theta(x_i),f_{\theta_0}(x_i))
]
输出：

```text
mean_feature_cosine_distance
p50_feature_cosine_distance
p95_feature_cosine_distance
max_feature_cosine_distance
logit_kl_vs_parent
prediction_change_rate
changed_prediction_accuracy_gain
changed_prediction_accuracy_loss
```

S-OOF 完成后，分别在：
high-trust；
medium-trust；
low-trust；
Bottom-10% 类别；
上计算上述指标。
关键诊断：

```text
drift 增大 + trusted accuracy 下降：
    噪声驱动的表征破坏

drift 极小 + 指标不变：
    PEFT 更新不足

drift 受控 + macro/trusted 提升：
    PEFT 可能有效

prediction change 很大但正确率不升：
    模型改变了决策，但方向无效
```

---

6.12 PEFT 平台 Gate
平台必须先提交 Bare。
因果有效

```text
PEFT Bare >= paired frozen control Bare + 0.30pp
```

竞争有效

```text
PEFT Bare > 当前最佳 Bare
```

结果解释
Bare 改变量 判断
≥ +0.30pp 明确正信号
+0.10～+0.30pp 暂不确定，需第二 seed
±0.10pp 基本持平
< 0 负收益
只有 Bare 通过因果有效或竞争有效之一，才生成 Flip TTA。

---

7. P2：S-OOF — 样本可信度与 Trusted Validation
   7.1 研究问题
   当前最大的决策风险是 raw validation 与平台排序不一致。S-OOF 的目标不是首先追求平台涨分，而是建立一个更接近人工精标测试目标的本地选择器。
   7.2 OOF 划分
   采用 3-fold group-aware OOF：

```text
fold 0：train folds 1+2，预测 fold 0
fold 1：train folds 0+2，预测 fold 1
fold 2：train folds 0+1，预测 fold 2
```

分组约束：
相同 SHA-256 图片必须位于同一 fold；
冲突重复组必须整体分配；
不允许同一视觉内容出现在训练和 OOF 预测两侧；
每个样本只由没有训练过它的模型生成 OOF 预测。
7.3 样本可信度特征
每个样本输出：

```text
sample_id
original_label
oof_top1
oof_original_label_probability
oof_top1_probability
oof_margin
oof_label_agreement
prediction_entropy
prototype_similarity_to_label
prototype_margin
duplicate_conflict_flag
loss_ema（可选）
forgetting_count（可选）
trust_score
trust_bucket
```

第一版 trust score 优先采用可解释组合，不直接训练复杂元模型：
[
s_i =
w_1\cdot p_{\mathrm{OOF}}(y_i)
+w_2\cdot \mathrm{margin}_{\mathrm{OOF}}
+w_3\cdot \mathrm{prototype\ margin}
-w_4\cdot \mathrm{entropy}
-w_5\cdot \mathrm{conflict}
]
权重先按归一化后的等权或少量人工设定，避免在含噪验证集上再次过拟合。
7.4 Trusted Validation
定义三档：

```text
High trust   ：top 50%
Medium trust ：middle 30%
Low trust    ：bottom 20%
```

同时保留连续 trust score。
所有历史候选重新评估：
CE；
GCE q=0.7；
GCE q=0.5；
CE5→GCE；
MixUp；
ELR；
PEFT 候选。
检查：
trusted accuracy 排序是否更接近平台排序；
high-trust 上的增益是否能解释 MixUp 的平台优势；
CE warmup 是否只改善低可信或含噪样本；
PEFT 的 prediction change 是否集中在低可信样本；
macro/trusted 指标是否比 raw micro 更稳定。
如果 trusted metric 仍不能解释平台排序，则不能将其直接作为唯一 early-stopping 指标，需要继续保留多指标 Gate。
7.5 Trust-weighted 训练
第一版不直接重标注，采用三档降权：
[
w_i=
\begin{cases}
1.0,& high\
0.6,& medium\
0.3,& low
\end{cases}
]
训练损失：
[
L=\frac{\sum_i w_i L_i}{\sum_i w_i}
]
先运行：

```text
S_OOF_FROZEN_WEIGHTED：
    Frozen CLIP + GCE q=0.5 + MixUp + trust weighting
```

若有正信号，再测试保守软标签：
[
\tilde y_i=(1-\beta_i)y_i+\beta_i p_i^{OOF}
]
建议：

```text
High trust：β=0
Medium trust：β=0.1
Low trust 且 OOF/prototype 一致：β=0.3
Low trust 且信号冲突：不改标签，只降权
```

## 不执行全量硬重标注。

8. P2：OOF × PEFT 的 2×2 因子实验
   当 S-OOF 完成后，运行最小矩阵：
   ID Backbone 样本监督
   O0 Frozen 普通 GCE/MixUp
   O1 最佳 LN-only PEFT 普通 GCE/MixUp
   O2 Frozen trust-weighted GCE/MixUp
   O3 最佳 LN-only PEFT trust-weighted GCE/MixUp + feature distillation
   解释：
   [
   \text{PEFT 主效应}=O1-O0
   ]
   [
   \text{Trust 主效应}=O2-O0
   ]
   [
   \text{可信监督下 PEFT 增量}=O3-O2
   ]
   只有当：

```text
O3 明显优于 O2
```

才能证明 PEFT 本身在可信监督下提供额外收益。
如果：

```text
O2 提升，O3 不提升：
```

## 则主要贡献来自样本可信度，而不是 PEFT，应保持冻结 backbone。

9. P3：条件式 Last-block LoRA R4
   9.1 LoRA Gate
   只有同时满足以下条件时执行：
   E4 或 O3 在两个 seed 上均不退化；
   至少一个 PEFT 候选 Bare 超过 paired frozen control；
   feature drift 受控；
   收益不能由 classifier continue-training 解释；
   trusted/macro 指标有一致正信号。
   否则：

```text
关闭 LoRA
不搜索 rank、alpha、dropout 和 block
```

9.2 第一轮唯一 LoRA 配置

```yaml
experiment_id: S_PEFT_LORA_R4
init_checkpoint: <BEST_VALIDATED_PEFT_OR_PARENT>

model:
  train_ln_post: <FOLLOW_E4_CONCLUSION>
  train_visual_proj: false

lora:
  target_block: 11
  target_modules:
    - attention_in_projection
    - attention_out_projection
  rank: 4
  alpha: 8
  dropout: 0.0

train:
  epochs: 15
  early_stop_patience: 5
  classifier_lr: 1.0e-4
  layernorm_lr: 5.0e-7
  lora_lr: 1.0e-5
  backbone_weight_decay: 0.01

feature_distillation:
  enabled: true
  teacher: frozen_parent
```

第一轮不同时训练 visual projection，避免变量过多。
9.3 LoRA 继续条件
只有 LoRA R4 Bare 明确优于对应 LN-only 对照，才考虑：
rank 8；
block 10+11；
LoRA dropout；
alpha 调整。

---

10. P4：低优先级补充实验
    10.1 GCE q 补充搜索
    仅在主线实验等待或算力空闲时执行：

```text
q=0.4
q=0.6
```

第一轮不运行 q=0.3。
原因：
q=0.5 已优于 q=0.7；
q=0.9 已崩塌；
0.4/0.6 足以判断最优区域是否需要细化；
q 调参预期只能产生小幅收益。
若某个 q 的 Bare 明显优于 q=0.5，再将其带入 MixUp，不做全组合网格。
10.2 轻增强
仅在所有主线完成后考虑：

```text
RandAugment：num_ops=2, magnitude=3
RandomErasing：scale=0.02–0.05
```

## 必须单变量对照，不与新 loss 或 PEFT 同时引入。

11. 已关闭方向
    已有充分证据，不重复投入：
    强 ColorJitter；
    强 RandomErasing；
    强 RandAugment；
    Dropout p=0.3/0.5/0.7；
    Cosine Head；
    Head EMA 0.99/0.999；
    EMA loss weighting；
    standalone prototype weighting；
    GCE + prototype weighting；
    Label Smoothing；
    Frozen continue-training；
    CE 下部分解冻；
    head-only EMA Teacher + Consistency；
    GCE q=0.9；
    vertical flip；
    4-view TTA；
    多模型投票或 ensemble。
    如果未来重新打开其中任一方向，必须明确说明出现了什么新证据，不能仅因为“还有算力”。

---

12. 执行批次与并行安排
    Batch 0：证据修复

```text
修复 registry schema
补齐 MixUp Bare/TTA 映射
补齐 C-EXP-4/C-EXP-5 重评估
实现统一 artifact manifest
```

Batch 0 完成前，不再批量生成平台候选。
Batch 1：冻结模型低成本候选
可并行：

```text
S_MIXUP_CE5
S_MIXUP_A01
S_MIXUP_A04
S_MIXUP_P04
```

完成后固定新的 frozen parent。
Batch 2：机制实验
可并行开发、不可无条件全部平台提交：

```text
S_ELR_BASE
S_PEFT_E0
S_PEFT_E1
S_PEFT_E2
S_PEFT_E3
```

执行依赖：
E0/E1/E2 可并行；
E3 可与其并行；
E4 必须等待 E1/E2 的 drift 结果；
E5 必须等待 E1–E4 选择最佳候选。
Batch 3：可信度主线

```text
3-fold OOF
trusted validation
S_OOF_FROZEN_WEIGHTED
保守 soft correction（仅在 weighting 有效后）
```

Batch 4：可信 PEFT 与 LoRA Gate

```text
O0/O1/O2/O3 2×2
最佳候选 seed 3407
符合 Gate 后才执行 Last-block LoRA R4
```

---

13. 停止规则
    为了避免无限小调参，设置明确停止条件。
    13.1 MixUp 停止规则
    如果 A01、A04、P04、CE5+MixUp 均未在 Bare 上超过当前基线：

```text
关闭 MixUp 参数搜索
保留 alpha=0.2, p=0.2
转入 ELR/OOF/PEFT
```

13.2 ELR 停止规则
如果：
两个合理配置均无 Bare 正收益；
或后期预测熵下降但 trusted/macro 不升；
或 running target 明显产生 confirmation bias；
则关闭 ELR，不继续做 EMA target network 和大范围 lambda 搜索。
13.3 普通 PEFT 停止规则
如果 E1、E2、E4 均：
不优于 E0；
feature drift 与指标退化一致；
第二 seed 无法复现；
则正式关闭普通 PEFT，转入 OOF trust weighting，不执行 LoRA。
13.4 OOF 停止规则
如果 OOF trust score：
无法区分人工抽查中的高低噪声风险；
trusted metric 无法解释已有平台排序；
trust weighting 多 seed 无正收益；
则不继续复杂重标注，只保留其作为诊断工具。

---

14. 70% 目标的条件判断
    不能将不同模块的“期望增益”线性相加，因为 MixUp、ELR、OOF 和 PEFT 的作用高度相关，收益可能重叠。
    更合理的条件区间：
    条件 合理平台判断
    当前冻结模型小调优 60%–61.5%
    ELR 或 OOF 可信度出现有效信号 61.5%–64%
    受控 PEFT 明确有效 63%–66%
    可信监督与 PEFT 有协同 65%–68%
    接近或达到 70% 需要尚未观察到的结构性突破
    阶段性判断节点：

```text
完成 Batch 1 后仍 <61%：
    确认冻结 head 小调参基本饱和

完成 ELR/OOF 后仍 <62%：
    70% 定位为低概率目标

OOF/PEFT 后达到 64%–66%：
    继续探索可信 PEFT 和 LoRA

达到 67%–68%：
    70% 才进入现实冲刺区间
```

---

15. 推荐的实验登记表（更新至 2026-07-17）

   **Batch 1 — MixUp 参数消融：**
   | experiment_id | parent | seed | status | raw_micro | platform_bare | platform_tta | decision |
   |:--|:--|:--:|:--:|:--:|:--:|:--:|:--|
   | S_MIXUP_CE5 | frozen baseline | 42 | done | 70.25% | 59.70% | 60.48% | eliminated |
   | S_MIXUP_A01 | frozen baseline | 42 | done | 70.29% | — | — | eliminated |
   | S_MIXUP_A04 | frozen baseline | 42 | done | 70.42% | — | — | eliminated |
   | S_MIXUP_P04 | frozen baseline | 42 | done | 70.41% | — | — | eliminated |

   **Batch 2 — PEFT E0–E3（父模型: W1_GCE05_MIXUP）：**
   | experiment_id | parent | seed | status | raw_micro | vs E0 | decision |
   |:--|:--|:--:|:--:|:--:|:--:|:--|
   | S_PEFT_E0_FROZEN | W1_GCE05_MIXUP | 42 | done | 71.17% | — | baseline |
   | S_PEFT_E1_LN_1E6 | W1_GCE05_MIXUP | 42 | done | 71.18% | +0.01pp | no gain |
   | S_PEFT_E2_LN_5E7 | W1_GCE05_MIXUP | 42 | done | 71.21% | +0.04pp | no gain |
   | S_PEFT_E3_BACKBONE_ONLY | W1_GCE05_MIXUP | 42 | done | 71.16% | −0.01pp | neutral |
   | S_PEFT_E4_LN_FEATDISTILL | — | — | closed | — | — | PEFT ineffective |
   | S_PEFT_E5_SEED3407 | — | — | closed | — | — | no candidate |
   | S_PEFT_LORA_R4 | — | — | closed | — | — | LoRA gate failed |

   **P2 — OOF 路线：**
   | experiment_id | parent | seed | status | raw_micro | platform_bare | platform_tta | decision |
   |:--|:--|:--:|:--:|:--:|:--:|:--:|:--|
   | S_OOF_DISCRETE | W1_GCE05_MIXUP | 42 | done | 68.65% | 59.28% | 59.28% | eliminated |

   **未执行 / 已关闭：**
   | experiment_id | status | reason |
   |:--|:--|:--|
   | S_ELR_BASE | planned | 未执行，待决策 |
   | S_OOF × PEFT 2×2 (O0–O3) | blocked | PEFT 无效，OOF 训练无效 |
   | GCE q=0.4/0.6 补充搜索 | not started | P4 优先级，算力空闲时 |
---

16. 最终推荐
    当前最合理的执行策略不是同时铺开所有方法，而是按证据逐层升级：

```text
第一层：
    固定最强 Frozen + MixUp 基线

第二层：
    用 ELR/OOF 判断能否抑制错误标签

第三层：
    用 E0–E5 判断最小 PEFT 是否真的有效

第四层：
    用 OOF × PEFT 2×2 判断可信监督是否使 PEFT 获益

第五层：
    只有最小 PEFT 已被证明有效，才进入 Last-block LoRA
```

核心判定原则：

> 一个方法只有在相同父模型、相同训练条件、Bare 平台结果和独立 checkpoint 重评估下超过 paired control，才能被认定为有效。
> 这套方案的目的不是增加实验数量，而是尽快识别：
> 冻结模型是否已经饱和；
> 噪声抑制是否能产生方法级收益；
> PEFT 失败究竟来自学习率、联合优化还是表征漂移；
> 70% 是否仍存在可验证的增长路径。
