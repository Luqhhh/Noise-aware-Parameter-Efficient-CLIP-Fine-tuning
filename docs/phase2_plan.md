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
    A1 2-view TTA
    A2 class-prior logit adjustment
    A3 TTA + prior adjustment

Track B：噪声鲁棒监督
    B1 Label Smoothing 0.05
    B2 GCE q=0.7
    B3 类别内 prototype confidence 连续加权

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

---

## 3. Track A：无需重训的快速平台导向实验

### A0：准备统一 logits 缓存

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

### A1：2-view horizontal-flip TTA

新增推理模式：

- view 1：CLIP 标准预处理
- view 2：原图水平翻转后使用同一 CLIP 标准预处理

融合：

$$\text{logits}_{tta} = \frac{\text{logits}_{original} + \text{logits}_{flip}}{2}$$

新增 `scripts/evaluate_tta.py`。

命令：

```bash
python scripts/evaluate_tta.py \
  --config configs/d3_strict.yaml \
  --checkpoint outputs/d3_strict/seed42/checkpoints/best.pt \
  --tta horizontal_flip \
  --output-dir outputs/phase2/a1_tta_flip
```

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

### A2：类别先验 logit adjustment

**动机**：官方测试集类别均衡，而训练数据和模型预测可能存在类别偏置。

对分类 logits 做：

$$\tilde{z}_c = z_c - \tau \log(\pi_c + \epsilon)$$

其中 $\pi_c$ 为 D3 train.csv 中类别 c 的样本比例，$\epsilon = 10^{-12}$。

新增 `common/logit_adjustment.py` 和 `scripts/sweep_logit_adjustment.py`。

参数网格固定为：$\tau \in \{0, 0.25, 0.5, 0.75, 1.0\}$。

命令：

```bash
python scripts/sweep_logit_adjustment.py \
  --val-logits outputs/phase2/d3_logits/val_logits.pt \
  --val-labels outputs/phase2/d3_logits/val_labels.pt \
  --train-csv outputs/d3_strict/seed42/train.csv \
  --taus 0 0.25 0.5 0.75 1.0 \
  --output-dir outputs/phase2/a2_logit_adjustment
```

**选择标准**：第一排序 macro，第二 micro，第三 bottom-10%。

**保留条件**：macro 提升 >= 0.20pp 且 micro 下降 <= 0.10pp。如果训练类别分布接近完全均衡，脚本仍要运行，但预计改动较小。

### A3：组合最佳 TTA 和先验调整

只有 A1、A2 都通过各自 gate 时执行。

固定组合：best TTA logits + best tau adjustment。禁止重新搜索更多参数。

若组合比两个单独方法都差，则保留单独最优方法。

---

## 4. Track B：噪声鲁棒监督筛选

### 4.1 新增统一 loss 接口

新增 `common/losses.py`。

支持：

```yaml
loss:
  name: cross_entropy
  reduction: mean
```

可选：`cross_entropy`、`label_smoothing`、`gce`。

统一接口：

```python
def build_loss(config: dict) -> nn.Module:
    ...
```

所有 loss 必须支持 `reduction="none"` 以便后续样本加权。

训练日志和 checkpoint metadata 增加 `loss_name` 和 `loss_parameters`。

### B0：CE 回归测试

不需要完整重跑 50 epochs。新增单元测试和 1 epoch smoke test，确认 `loss.name=cross_entropy` 与现有 CE 计算完全一致。

容差：同一 batch loss 误差 <= 1e-8，同一 logits gradient 误差 <= 1e-8。

只有 B0 通过后，才能启动 B1/B2。

### B1：Label Smoothing 0.05

配置 `configs/r1_d3_ls005.yaml`：

```yaml
experiment:
  id: R1_D3_LS005

loss:
  name: label_smoothing
  epsilon: 0.05
```

其余全部继承 D3_STRICT。

命令：

```bash
python -m experiments.baseline.train \
  --config configs/r1_d3_ls005.yaml \
  --experiment-id R1_D3_LS005 \
  --mode dev
```

本阶段不运行 $\epsilon = 0.10$。只有 0.05 明确为正，下一轮才允许测试 0.10。

### B2：Generalized Cross Entropy

定义：

$$L_{GCE} = \frac{1 - p_y^q}{q}$$

配置 `configs/r2_d3_gce07.yaml`：

```yaml
experiment:
  id: R2_D3_GCE07

loss:
  name: gce
  q: 0.7
  probability_epsilon: 1.0e-7
```

实现要求：

```python
probs = torch.softmax(logits, dim=1)
py = probs.gather(1, labels[:, None]).squeeze(1)
py = py.clamp_min(probability_epsilon)
loss = (1.0 - py.pow(q)) / q
```

第一轮只测试 $q = 0.7$，不得立即做 q sweep。

### B3：类别内 prototype confidence 连续加权

这是本阶段最重要的数据侧实验。

#### 4.3.1 构建权重

新增 `scripts/build_prototype_weights.py`。

输入：D3 clean train.csv + 冻结 OpenAI CLIP ViT-B/32 图像特征。

全部特征做 L2 normalize。

对每类构建 10% trimmed centroid：

$$p_c = \text{Normalize}\left(\text{TrimmedMean}\{z_i : y_i = c\}\right)$$

每个样本计算：

$$s_i = z_i^\top p_{y_i}$$

$$m_i = z_i^\top p_{y_i} - \max_{c \ne y_i} z_i^\top p_c$$

在每个类别内部计算 percentile rank：

- $r_{sim}(i)$：$s_i$ 的类别内百分位
- $r_{margin}(i)$：$m_i$ 的类别内百分位

置信度：

$$c_i = 0.5 \cdot r_{sim}(i) + 0.5 \cdot r_{margin}(i)$$

样本权重：

$$w_i = 0.2 + 0.8 \cdot c_i$$

因此 $0.2 \le w_i \le 1.0$。

**关键要求**：

- 必须按类别内部排序
- 不得使用全局 percentile
- 不得删除样本
- 不得使用 validation 特征构建 centroid

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

修改 batch 解包，保留 `_paths`。

训练损失：

$$L = \frac{\sum_i w_i \cdot CE_i}{\sum_i w_i + \epsilon}$$

配置 `configs/r3_d3_proto_weight.yaml`：

```yaml
experiment:
  id: R3_D3_PROTO_WEIGHT

loss:
  name: cross_entropy
  reduction: none

sample_weighting:
  enabled: true
  weights_path: outputs/phase2/prototype_weights/sample_weights.json
  normalize_by_weight_sum: true
  missing_weight_policy: error
```

`missing_weight_policy` 必须为 `error`，不能静默使用 1.0。

---

## 5. 实验结果判定

基线固定为：

```
D3_STRICT
micro = 70.6572%
macro = 70.6100%
```

每个候选输出：

- best micro / best macro / bottom-10%
- best epoch / train accuracy / validation loss
- train-val gap
- per-class delta
- prediction disagreement vs D3

### 5.1 seed42 初筛 gate

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

---

## 6. 平台提交策略

平台提交不是等所有实验做完再进行。

### 6.1 第一提交候选

优先顺序：

1. 无训练方法中，本地 macro 明显提升的方法
2. prototype weighting
3. GCE
4. Label Smoothing

选择一个信息量最大的候选提交，不同时提交多个近似方法。

### 6.2 提交前硬检查

- 预测数必须为 24967
- 无缺失文件名
- 无重复文件名
- 标签范围 0000-0499
- ZIP 实际文件 SHA-256 已记录
- checkpoint / CSV / ZIP lineage 完整

### 6.3 平台决策

| 平台提升 | 行动 |
|---|---|
| >= 1.0pp | 方向成立，立即做 3407/2026 多 seed，开始小规模参数精调 |
| 0.3-1.0pp | 方向可能有效，仅做一个额外 seed 再决定 |
| < 0.3pp 或下降 | 停止该方向，即使本地提升明显也不继续深挖 |

平台成绩优先于 seed42 本地微小变化。

---

## 7. 测试要求

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

## 8. 执行命令顺序

```bash
# 1. 开发与测试
pytest -q

# 2. 缓存 D3 logits
python scripts/cache_val_test_logits.py \
  --config configs/d3_strict.yaml \
  --checkpoint outputs/d3_strict/seed42/checkpoints/best.pt \
  --output-dir outputs/phase2/d3_logits

# 3. Track A
python scripts/evaluate_tta.py \
  --config configs/d3_strict.yaml \
  --checkpoint outputs/d3_strict/seed42/checkpoints/best.pt \
  --tta horizontal_flip \
  --output-dir outputs/phase2/a1_tta_flip

python scripts/sweep_logit_adjustment.py \
  --val-logits outputs/phase2/d3_logits/val_logits.pt \
  --val-labels outputs/phase2/d3_logits/val_labels.pt \
  --train-csv outputs/d3_strict/seed42/train.csv \
  --taus 0 0.25 0.5 0.75 1.0 \
  --output-dir outputs/phase2/a2_logit_adjustment

# 4. 构建 prototype weights
python scripts/build_prototype_weights.py \
  --config configs/d3_strict.yaml \
  --train-csv outputs/d3_strict/seed42/train.csv \
  --output-dir outputs/phase2/prototype_weights

# 5. Track B 并行训练
python -m experiments.baseline.train \
  --config configs/r1_d3_ls005.yaml \
  --experiment-id R1_D3_LS005 \
  --mode dev

python -m experiments.baseline.train \
  --config configs/r2_d3_gce07.yaml \
  --experiment-id R2_D3_GCE07 \
  --mode dev

python -m experiments.baseline.train \
  --config configs/r3_d3_proto_weight.yaml \
  --experiment-id R3_D3_PROTO_WEIGHT \
  --mode dev
```

---

## 9. Stop / Go 决策

### Gate 1：Track A

- A1/A2 都无本地收益 → 停止推理校准路线
- 至少一个通过 → 生成完整提交并平台验证

### Gate 2：Loss

- LS/GCE 都低于 D3 → 不继续 loss 参数 sweep
- 任一通过 → 保留一个，最多补一个参数点

### Gate 3：Prototype weighting

- R3 无收益 → 检查权重与 per-class retention 的相关性，不立即尝试更低 min_weight
- R3 明显提升 → 优先平台提交

### Gate 4：下一阶段

只有以下条件之一成立，才进入组合实验：

- 某训练方法 seed42 micro >= +0.50pp
- 或平台提升 >= +1.0pp

组合实验最多一个：best robust loss + prototype weighting。禁止同时加入部分解冻。

---

## 10. 本阶段完成标准

- [ ] Track A 完成
- [ ] LS 0.05 完成
- [ ] GCE 0.7 完成
- [ ] prototype weighting 完成
- [ ] 所有结果从 best.pt 复评
- [ ] 至少完成一次新平台提交
- [ ] 根据平台反馈选出下一条主线

本阶段成功不要求立即达到 70%。

**阶段目标**：第一目标平台突破 60%；第二目标找到本地提升能够转化为平台提升的方法。
