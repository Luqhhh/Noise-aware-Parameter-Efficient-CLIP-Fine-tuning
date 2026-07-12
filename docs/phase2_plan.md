# Phase 2：平台导向优化执行计划

> 目标：以 `D3_STRICT` 为当前操作基线，尽快验证能够提升平台 Accuracy 的方向。  
> 当前锚点：本地 micro 70.6572%，macro 70.6100%，平台 57.3397%。  
> 原则：E0_STRICT 继续并行收尾，但不阻塞本阶段。

---

## 0. 本阶段不做什么

暂不继续：

- F1b dropout 搜索
- 旧 F2 配置
- 直接扩大 backbone 解冻范围
- 强数据增强
- 大规模超参数网格
- 多个新策略一次性叠加

原因：F1_STRICT 相对 D3 仅 +0.1260pp，没有达到 +0.30pp 的最低收益门槛。

---

## 1. 总体执行顺序

```
Track A：无训练平台导向优化
    TA1 2-view TTA
    TA2 class-prior logit adjustment
    TA3 TTA + prior adjustment

Track B：噪声鲁棒监督
    B1 Label Smoothing（ε=0.05，可扩展 0.03/0.10）
    B2 GCE（q=0.7，可扩展 0.5/0.9/warmup）
    B3 Prototype Confidence 样本加权（静态，可扩展 warmup/EMA hybrid）

第一轮结果
    ↓
只保留最多 2 个候选
    ↓
平台提交 1 个最有信息量的候选
    ↓
根据平台反馈决定下一阶段
```

Track A 和 Track B 可并行。

---

## 2. 统一实验约束

所有训练实验固定：

```yaml
base_experiment: D3_STRICT
backbone: OpenAI CLIP ViT-B/32
head: linear
train_split: outputs/d3_strict/seed42/train.csv
val_split: outputs/d3_strict/seed42/val.csv
augmentation: a0
batch_size: 128
epochs: 50
early_stop_patience: 10
optimizer: AdamW
lr: 5.0e-3
weight_decay: 1.0e-4
scheduler: cosine
warmup_epochs: 2
split_seed: 42
train_seed: 42
```

除指定变量外，不允许修改其他参数。

所有实验必须：

- 使用独立输出目录
- fresh run fail-closed
- best.pt 重载后复评
- 输出 micro / macro / bottom-10%
- 记录 checkpoint、config、val.csv SHA-256

当前基线指标：

```
Local micro：70.6572%
Local macro：70.6100%
Platform：57.3397%
```

所有候选都必须与 D3_STRICT 做 paired comparison。

---

## 3. Track A：无需重训的快速平台导向实验

### TA0：准备统一 logits 缓存

新增 `scripts/cache_val_test_logits.py`。

作用：加载 D3_STRICT best.pt，对 master-val 和 test 分别缓存原始 logits、标签、文件名和 class mapping。

输出：

```
outputs/phase2/d3_logits/
├── val_logits.pt
├── val_labels.pt
├── val_paths.json
├── test_logits.pt
├── test_names.json
└── manifest.json
```

`manifest.json` 至少包含：

```json
{
  "checkpoint_path": "",
  "checkpoint_sha256": "",
  "val_csv_sha256": "",
  "num_val": 10316,
  "num_test": 24967,
  "num_classes": 500
}
```

测试集必须输出 24967 条预测。图片读取失败时不能跳过。

### TA1：2-view horizontal-flip TTA

新增推理模式：

- view 1：CLIP 标准预处理
- view 2：原图水平翻转后使用同一 CLIP 标准预处理

融合：

$$\text{logits}_{tta} = \frac{\text{logits}_{original} + \text{logits}_{flip}}{2}$$

新增 `scripts/evaluate_tta.py`。

输出：

- baseline_micro / tta_micro
- baseline_macro / tta_macro
- baseline_bottom10 / tta_bottom10
- prediction_change_rate
- per_class_delta.csv

**保留条件**（满足任一）：

- micro >= D3 + 0.20pp
- macro >= D3 + 0.20pp
- bottom-10% >= D3 + 1.00pp，且 micro 不下降超过 0.05pp

若均不满足，关闭 TTA 分支，不测试更多 crop 数。

### TA2：类别先验 logit adjustment

**动机**：官方测试集类别均衡，而训练数据和模型预测可能存在类别偏置。

对分类 logits 做：

$$\tilde{z}_c = z_c - \tau \log(\pi_c + \epsilon)$$

其中 $\pi_c$ 为 D3 train.csv 中类别 c 的样本比例，$\epsilon = 10^{-12}$。

新增 `common/logit_adjustment.py` 和 `scripts/sweep_logit_adjustment.py`。

参数网格固定为：$\tau \in \{0, 0.25, 0.5, 0.75, 1.0\}$。

**选择标准**：第一排序 macro，第二 micro，第三 bottom-10%。

**保留条件**：macro 提升 >= 0.20pp 且 micro 下降 <= 0.10pp。

### TA3：组合最佳 TTA 和先验调整

只有 TA1、TA2 都通过各自 gate 时执行。

固定组合：best TTA logits + best tau adjustment。禁止重新搜索更多参数。

若组合比两个单独方法都差，则保留单独最优方法。

---

## 4. Track B：噪声鲁棒监督

### 4.0 基础设施

新增 `common/losses.py`，统一 loss 接口：

```python
def build_loss(config: dict) -> nn.Module:
    ...
```

支持 `cross_entropy`、`label_smoothing`、`gce`，均支持 `reduction="none"`。

**B0 CE 回归测试**：确认 `loss.name=cross_entropy` 与现有 CE 计算完全一致。容差：同一 batch loss 误差 <= 1e-8，同一 logits gradient 误差 <= 1e-8。只有 B0 通过后，才能启动 B1/B2。

---

### 4.1 B1：Label Smoothing 分支

#### B1-1：Label Smoothing 0.05

标签从 one-hot $q_y=1, q_{c \ne y}=0$ 变为：

$$q_y = 1 - \epsilon, \qquad q_{c \ne y} = \frac{\epsilon}{C-1}$$

其中 $\epsilon = 0.05$，$C = 500$。

配置：

```yaml
experiment:
  id: B1_LS005

loss:
  name: label_smoothing
  epsilon: 0.05
```

**作用**：减弱噪声标签导致的过度自信，缓解分类头训练后期置信度膨胀，软化细粒度相邻类别决策边界。

**重点观察**：micro / macro / validation loss / train accuracy / train-val gap / 平均最大 softmax probability。如果训练准确率下降但验证提升、validation loss 降低，说明平滑起正则化作用。

#### B1-2：Label Smoothing 0.03

仅在 B1-1 基本持平或小幅正收益、但训练准确率下降明显或困难类别准确率下降时执行。

```yaml
loss:
  name: label_smoothing
  epsilon: 0.03
```

#### B1-3：Label Smoothing 0.10

仅在 B1-1 的 micro、macro 均提升、train-val gap 仍然较大、训练后期仍有明显过拟合时执行。

```yaml
loss:
  name: label_smoothing
  epsilon: 0.10
```

不建议测试大于 0.10：500 类细粒度分类中，过强平滑容易削弱细微类别差异。

#### B1 分支停止规则

- B1-1 相对 D3 micro 下降 > 0.20pp → 停止整个 Label Smoothing 分支
- B1-1 提升 < 0.10pp → 不进行 ε sweep
- B1-1 提升 >= 0.30pp → 允许增加一个 ε 参数点
- B1 最佳结果提升 >= 0.50pp → 进入多 seed 或平台验证

---

### 4.2 B2：GCE 鲁棒损失分支

#### B2-1：GCE q=0.7

定义：

$$L_{\mathrm{GCE}} = \frac{1 - p_y^q}{q}$$

普通 CE 对低 $p_y$ 样本产生很强梯度。若标签本身错误，这些样本会强迫模型记忆错误监督。GCE 降低极低标签概率样本的影响，更适合处理严重错标和长期高损失异常样本。

配置：

```yaml
experiment:
  id: B2_GCE07

loss:
  name: gce
  q: 0.7
  probability_epsilon: 1.0e-7
```

**与 Label Smoothing 的区别**：LS 统一降低所有标签置信度；GCE 根据模型当前对标签的认可程度动态限制低可信标签的梯度。B1 偏整体正则化，B2 偏异常标签鲁棒性。

#### B2-2：GCE q=0.5

仅在 q=0.7 出现轻度欠拟合时执行（训练准确率明显低于 D3，验证接近 D3）。

```yaml
loss:
  name: gce
  q: 0.5
```

更接近 CE，学习能力更强，鲁棒性稍弱。

#### B2-3：GCE q=0.9

仅在 q=0.7 有明确正收益、但训练后期仍有噪声记忆迹象时执行。

```yaml
loss:
  name: gce
  q: 0.9
```

更鲁棒，但更容易忽略困难干净样本。必须重点观察 bottom-10% 类别准确率和低样本类别准确率。

#### B2-4：CE Warmup → GCE

如果从 epoch 1 直接使用 GCE 出现前几轮收敛明显慢、训练准确率长期过低，则执行：

```text
Epoch 1–5：普通 CE
Epoch 6–50：GCE q=0.7
```

利用 early learning：先用 CE 快速学习主要类别结构，再用 GCE 降低后期记忆噪声的速度。

#### B2 分支停止规则

- GCE q=0.7 的 micro、macro 同时下降 > 0.20pp → 停止 GCE 分支
- 训练准确率很低、验证也下降 → 最多测试 warmup 版本
- micro 提升但 bottom-10% 明显下降 → 不进入平台提交
- q=0.7 提升 >= 0.30pp → 允许测试一个相邻 q
- 最佳 GCE 提升 >= 0.50pp → 进入平台验证

---

### 4.3 B3：Prototype Confidence 样本加权分支

B3 不是只处理 D3 已发现的重复冲突，而是为全部训练样本估计标签可信度。

#### 4.3.1 构建权重

新增 `scripts/build_prototype_weights.py`。

**特征提取**：使用冻结的 CLIP 图像编码器 $z_i = f_{\mathrm{CLIP}}(x_i)$，L2 normalize。必须只使用 D3 clean train，不允许使用 validation。

**构建类别原型**：对类别 $c$，10% trimmed centroid：

$$p_c = \text{Normalize}\left(\text{TrimmedMean}\{z_i : y_i = c\}\right)$$

**样本可信度**：

自身类别相似度：$s_i = z_i^\top p_{y_i}$

类别 margin：$m_i = z_i^\top p_{y_i} - \max_{c \ne y_i} z_i^\top p_c$

在每个类别内部计算 $r_{sim}$（自身相似度百分位）和 $r_{margin}$（margin 百分位）。

综合置信度：$c_i = 0.5 \cdot r_{sim} + 0.5 \cdot r_{margin}$

样本权重：$w_i = 0.2 + 0.8 \cdot c_i$，因此 $0.2 \le w_i \le 1.0$。

**关键要求**：必须按类别内部排序，不得使用全局 percentile，不得删除样本，不得使用 validation 特征构建 centroid。

输出：

```
outputs/phase2/prototype_weights/
├── sample_weights.json
├── sample_weights.csv
├── class_statistics.csv
├── weight_distribution.json
└── manifest.json
```

`sample_weights.csv` 列：image_path, label, own_similarity, best_other_similarity, margin, similarity_percentile, margin_percentile, weight

#### 4.3.2 加权 CE

训练损失：

$$L = \frac{\sum_i w_i \cdot CE_i}{\sum_i w_i + \epsilon}$$

必须按权重和归一化，避免平均权重小于 1 时等效降低学习率。

配置：

```yaml
experiment:
  id: B3_PROTO_STATIC

loss:
  name: cross_entropy
  reduction: none

sample_weighting:
  enabled: true
  method: prototype_static
  weights_path: outputs/phase2/prototype_weights/sample_weights.json
  minimum_weight: 0.2
  normalize_by_weight_sum: true
  missing_weight_policy: error
```

`missing_weight_policy` 必须为 `error`，不能静默使用 1.0。

#### B3-2：保守最低权重 0.4

如果 B3-1 出现训练准确率明显下降、bottom-10% 类别下降、大量困难样本受到过强抑制，则把最低权重从 0.2 提高到 0.4：

$$w_i = 0.4 + 0.6 \cdot c_i$$

#### B3-3：5 Epoch Warmup 后启用权重

如果 B3-1 在训练早期收敛不稳：

```text
Epoch 1–5：所有样本权重 = 1
Epoch 6–50：启用 prototype weight
```

#### B3-4：Prototype + EMA Loss 联合权重

仅在 B3-1 明确有效后执行。

为每个样本维护 EMA loss：$\ell_i^{EMA}(t) = \beta \ell_i^{EMA}(t-1) + (1-\beta)\ell_i(t)$，推荐 $\beta = 0.9$，warmup 5 epochs，每个 epoch 更新一次。

在类别内部计算 EMA loss 百分位 $r_{loss}(i)$（loss 从低到高），loss 可信度 $c_i^{loss} = 1 - r_i^{loss}$。

联合置信度：

$$c_i = 0.7 \cdot c_i^{prototype} + 0.3 \cdot c_i^{loss}$$

最终权重：$w_i = 0.2 + 0.8 \cdot c_i$。

Prototype confidence 来自冻结 CLIP 先验（相对稳定），EMA loss 来自当前分类头（动态但容易把困难干净样本误认为噪声）。因此 prototype 占主要权重（0.7），EMA loss 作为补充（0.3）。

#### B3 状态保存要求

checkpoint 必须保存：sample weighting 配置、sample weight 文件 SHA-256、EMA loss 数组、EMA momentum、当前 epoch、每类权重统计。resume 时必须严格恢复，不能重新初始化 EMA loss。

#### B3 诊断指标

除常规准确率外，必须记录：权重均值/标准差、权重 p10/p50/p90、每类平均权重、最低权重样本数量、权重与最终 loss 的相关性、权重与正确预测的相关性、bottom-10% 类别平均权重。

重点排查：某些困难类别是否整体被赋低权重、高权重样本是否确实更容易预测正确、低权重样本是否集中在跨类冲突或异常图片中。

#### B3 分支停止规则

- B3-1 micro 下降 > 0.20pp → 不立即降低 minimum weight，先分析类别级权重分布
- B3-1 macro 或 bottom-10% 明显下降 → 优先测试 minimum_weight=0.4
- B3-1 提升 >= 0.30pp → 允许测试 warmup 或 EMA hybrid 中的一个
- B3-1 提升 >= 0.50pp → 优先平台提交
- B3-4 未超过静态 B3-1 → 停止动态 EMA 权重路线

---

## 5. 组合实验

独立实验完成前不组合。

### C1：最佳 Label Smoothing + 最佳 Prototype Weight

最优先组合，两者机制互补：

- Label Smoothing：降低标签目标的绝对置信度
- Prototype Weight：降低可疑样本的整体梯度贡献

损失：$L = \frac{\sum_i w_i L_{LS,i}}{\sum_i w_i}$

触发条件：B1 独立提升 >= 0.20pp，B3 独立提升 >= 0.30pp，二者 macro 均不下降。

### C2：最佳 GCE + Prototype Weight

需谨慎：二者都会降低疑似噪声样本影响，可能造成重复抑制。

只有满足 GCE 独立提升 >= 0.30pp、Prototype Weight 独立提升 >= 0.30pp、bottom-10% 类别均未下降时才运行。建议提高最低权重到 0.4。

### C3：Linear Head EMA

不作为第一轮独立主实验，而作为最佳方案的稳定器。

推荐 `ema_decay = 0.99`。验证和推理同时评估普通 best head 和 EMA best head。只有 EMA 在多个 epoch 上稳定优于普通 head，才用于平台提交。

---

## 6. 暂缓的策略

以下内容保留在后续阶段，不与 B1–B3 第一轮混合：

- MixUp / CutMix / RandAugment
- 强 RandomResizedCrop / ColorJitter
- EMA Teacher
- 高置信度伪标签替换
- 扩大 backbone 解冻范围

原因：当前首先要回答「哪一种标签噪声鲁棒机制真正有效」。同时加入增强、MixUp 或解冻后，即使结果提升也无法清楚归因。

---

## 7. 实验结果判定

基线固定为 D3_STRICT（micro = 70.6572%, macro = 70.6100%）。

每个候选输出：best micro / best macro / bottom-10% / best epoch / train accuracy / validation loss / train-val gap / per-class delta / prediction disagreement vs D3。

### 7.1 seed42 初筛 gate

**强保留**：micro >= D3 + 0.50pp 且 macro 不下降。

**条件保留**：

- micro >= D3 + 0.30pp 且 macro >= D3 + 0.20pp 且 bottom-10% 不下降

或：

- bottom-10% >= D3 + 1.50pp 且 micro 下降不超过 0.05pp

**淘汰**（任一成立）：

- micro 下降 > 0.20pp
- macro 下降 > 0.20pp
- 出现类别预测塌缩
- validation loss 明显恶化且 accuracy 无收益

本阶段最多保留两个训练候选。

### 7.2 第一轮结果决策表

| 结果                 | 决策                                               |
| -------------------- | -------------------------------------------------- |
| B1 正、B2 负、B3 负  | 沿 Label Smoothing 小范围搜索                      |
| B1 负、B2 正、B3 负  | 沿 GCE 或 CE→GCE warmup                            |
| B1/B2 负、B3 正      | 重点发展样本可信度建模                             |
| B1 与 B3 均正        | 运行 C1                                            |
| B2 与 B3 均正        | 谨慎运行 C2                                        |
| 三者本地均无收益     | 停止 loss/weighting 小修，转向模型适配或验证域问题 |
| 本地提升但平台不提升 | 停止围绕当前 validation 微调                       |
| 平台提升 >= 1pp      | 对该方向做多 seed 和参数精调                       |

---

## 8. 平台提交策略

平台提交不是等所有实验做完再进行。

### 8.1 第一提交候选

优先顺序：

1. 无训练方法中，本地 macro 明显提升的方法
2. prototype weighting
3. GCE
4. Label Smoothing

选择一个信息量最大的候选提交，不同时提交多个近似方法。

### 8.2 提交前硬检查

- 预测数必须为 24967
- 无缺失文件名
- 无重复文件名
- 标签范围 0000-0499
- ZIP 实际文件 SHA-256 已记录
- checkpoint / CSV / ZIP lineage 完整

### 8.3 平台决策

| 平台提升       | 行动                                                   |
| -------------- | ------------------------------------------------------ |
| >= 1.0pp       | 方向成立，立即做 3407/2026 多 seed，开始小规模参数精调 |
| 0.3-1.0pp      | 方向可能有效，仅做一个额外 seed 再决定                 |
| < 0.3pp 或下降 | 停止该方向，即使本地提升明显也不继续深挖               |

平台成绩优先于 seed42 本地微小变化。

---

## 9. 测试要求

新增：

- `tests/test_losses.py`
- `tests/test_sample_weighting.py`
- `tests/test_logit_adjustment.py`
- `tests/test_tta.py`
- `tests/test_prototype_weights.py`

至少覆盖：

- CE 回归一致性
- GCE 手算一致性
- label smoothing 参数有效
- 加权 loss 按 sum(weight) 归一化
- 缺失样本权重硬失败
- prototype 权重范围 [0.2, 1.0]
- 类别内 percentile 正确
- validation 不参与 centroid
- tau=0 与原 logits 完全一致
- TTA 融合维度和文件名顺序一致
- 测试预测数严格等于 24967

全部测试通过后再训练：

```bash
pytest -q
```

---

## 10. Stop / Go 决策

### Gate 1：Track A

- TA1/TA2 都无本地收益 → 停止推理校准路线
- 至少一个通过 → 生成完整提交并平台验证

### Gate 2：Loss（B1 / B2）

- LS/GCE 都低于 D3 → 不继续 loss 参数 sweep
- 任一通过 → 保留一个，最多补一个参数点

### Gate 3：Prototype weighting（B3）

- B3 无收益 → 检查权重与 per-class retention 的相关性，不立即尝试更低 min_weight
- B3 明显提升 → 优先平台提交

### Gate 4：下一阶段

只有以下条件之一成立，才进入组合实验：

- 某训练方法 seed42 micro >= +0.50pp
- 或平台提升 >= +1.0pp

组合实验最多一个：best robust loss + prototype weighting。禁止同时加入部分解冻。
