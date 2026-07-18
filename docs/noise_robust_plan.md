# 高优先级噪声鲁棒实验实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务逐项执行。所有步骤使用复选框追踪。不得跳过测试、审计或实验 gate。

**Goal:** 在当前 `S_OOF_ZERO_0001` 正信号基础上，构建"类条件噪声识别 → 多信号高精度筛选/重标 → 被拒样本半监督回收 → 条件性 PEFT"的可复现单模型训练管线，优先提升平台 Bare Accuracy，而不是继续细调通用 loss。

**Architecture:** 复用现有 3-fold OOF logits、prototype、kNN 和 flip consistency 信号，先生成统一的 `purification_manifest.csv`。训练端通过同一个 manifest 同时控制监督权重、可选训练标签和样本角色；低可信样本的原标签监督被严格切断，仅在通过 gate 后用于无标签一致性学习。PEFT 只在净化后的 frozen control 获得稳定收益后启动，并使用冻结 CLIP 特征蒸馏限制表征漂移。

**Tech Stack:** Python 3、PyTorch、OpenAI CLIP ViT-B/32、NumPy、Pandas、scikit-learn、pytest、YAML、现有 `analysis/oof` 与 `experiments/baseline` 管线。

## Global Constraints

- 仓库 HEAD 基于 `bfcb537` 及之后的合法更新（代码不变，仅新增输出文件）；不得在旧代码上执行实验。
- 只使用当前阶段官方训练数据；测试图像不得参与训练、阈值选择、聚类、原型构建或参数更新。
- 骨干固定为 OpenAI 官方 CLIP ViT-B/32；不得引入其他视觉基础模型或额外数据训练的教师。
- 最终提交必须来自单一 student checkpoint；OOF fold 模型只用于训练数据质量估计，禁止在测试集上集成预测。
- 所有新实验统一使用 `outputs/data/d3_strict/seed42`；不得使用已废弃的 `outputs/baselines/ref/seed42`。
- 首轮统一使用 `split_seed=42, train_seed=42`；只有通过 gate 的候选补 `train_seed=3407`。
- 基础训练配方固定为：Linear Head、GCE `q=0.5`、MixUp `alpha=0.2, probability=0.2`、50 epochs、AdamW、head LR `5e-3`、weight decay `1e-4`、horizontal-flip TTA 仅在 Bare 通过后执行。
- 新实验的 `missing_weight_policy` 必须为 `error`；manifest coverage 必须为 100%。
- 不允许人工查看图片后维护黑名单或改标签；所有选择必须由脚本自动生成并可复现。
- 不 push；每个独立任务通过测试后允许本地 commit，提交信息按本计划给出。

---

## 0. 当前代码状态与必须先修的阻断问题

基于提供的仓库快照，执行前必须认识到以下事实：

1. `analysis/oof/quality.py` 已生成 `oof_top1`、`p_original_label`、`p_top1`、`top1_margin`、`prototype_top1`、`prototype_margin`、`knn_top1`、`knn_agreement`、`flip_consistency`，无需重新设计底层信号。
2. `RelabelManifestProvider.get_training_label()` 当前未被训练循环调用，`hard_relabel: true` 实际不会改变 batch label。
3. `common/mixup.py` 不返回配对索引；`experiments/baseline/train.py` 在 MixUp batch 上直接 `reduce_loss()`，绕过 sample weights。因而 `sample_weight=0` 的样本仍会在约 20% 的 MixUp batch 中用原错误标签产生监督梯度。
4. `OOFManifestProvider` 对路径做 `Path.resolve()`，`RelabelManifestProvider` 没有统一路径规范化，可能造成 manifest lookup 不一致。
5. 当前 OOF protocol audit 显示 91,195 个 strict-train 样本均有 OOF 预测，OOF accuracy 约 69.45%；本计划直接复用这些 out-of-sample 预测。

**结论：Task 1 是硬前置。Task 1 未通过时，不得运行任何新筛选、重标或半监督实验。**

---

## 1. 实验矩阵与执行顺序

| Wave | Experiment ID | 核心变量 | 首轮 seed | 是否直接平台提交 |
|---|---|---|---:|---|
| Control | `NR_CTRL_OOF_ZERO_0001_FIXED` | 修复 MixUp 权重后重跑现有 p<0.001 hard-zero | 42 | 是，作为新因果基线 |
| Wave A | `NR_CL_CLASSWISE_DROP` | class-conditional confident-joint prune，按类限额删除 | 42 | 通过本地安全 gate 后 Bare 提交 |
| Wave A | `NR_CL_KNN_DROP` | confident issue ∩ OOF/kNN 共识删除 | 42 | 通过本地安全 gate 后 Bare 提交 |
| Wave A | `NR_CONSENSUS_RELABEL_1P` | 三信号共识、高置信、全局最多 1% hard relabel | 42 | 通过本地安全 gate 后 Bare 提交 |
| Wave B | `NR_REJECT_CONSISTENCY` | rejected 样本仅做 teacher/student flip consistency | 42 | 仅当最佳 drop 候选有平台正收益 |
| Wave B | `NR_REJECT_PSEUDO_1P` | clean supervised + 1% pseudo-label supervised，rejected 其余为 0 权重 | 42 | 仅当 relabel1p 或 drop 有正收益 |
| Wave B | `NR_REJECT_SSL_COMBINED` | consistency + 1% pseudo-label | 42 | 仅当前两项至少一项正收益 |
| Wave C | `NR_PURE_LN_DISTILL` | 净化监督 + visual LayerNorm-only + frozen CLIP feature distillation | 42 | 仅当 Wave A/B Bare 稳定提升 |
| Confirm | `<winner>_S3407` | 只改变 train seed | 3407 | Bare 方向一致后再 TTA |

### 平台提交预算

- Control 固定占用 1 次 Bare 提交，用来量化"修复 MixUp 权重绕过"本身的影响。
- Wave A 最多提交 2 个候选：优先 `NR_CL_KNN_DROP` 和 `NR_CONSENSUS_RELABEL_1P`；`NR_CL_CLASSWISE_DROP` 主要用于机制对照，只有审计明显优于二者时才提交。
- Wave B 每次只提交一个最强候选，不进行 consistency weight 大网格。
- 所有 TTA 必须等对应 Bare 已达到当前有效基线后再生成/提交。

---

## 2. 统一输出契约

每个 manifest 构建任务必须生成：

```text
outputs/phase4/purification/<experiment_id>/
├── purification_manifest.csv
├── confident_joint.npy
├── class_selection_summary.csv
├── selected_samples.csv
├── protocol_audit.json
└── artifact_manifest.json
```

`purification_manifest.csv` 使用以下固定 schema：

```text
sample_id
image_path
original_label
training_label
sample_weight
quality_score
training_role
selection_reason
suggested_label
oof_top1
p_original_label
p_top1
top1_margin
prototype_top1
prototype_margin
knn_top1
knn_agreement
flip_consistency
```

`training_role` 只能取：

```text
clean       # 原标签监督，sample_weight=1
rejected    # 不使用原标签监督，sample_weight=0
pseudo      # 使用 training_label 监督，sample_weight=1
```

每个训练实验必须生成现有标准产物，并额外生成：

```text
partition_metrics.json
manifest_runtime_audit.json
mixup_weight_audit.json
```

`partition_metrics.json` 至少包含：

```json
{
  "clean_count": 0,
  "rejected_count": 0,
  "pseudo_count": 0,
  "global_reject_rate": 0.0,
  "global_relabel_rate": 0.0,
  "max_class_reject_rate": 0.0,
  "max_class_relabel_rate": 0.0,
  "classes_with_zero_clean_samples": [],
  "manifest_coverage": 1.0
}
```

---

# Task 0: 冻结执行环境与输入产物

**Files:**
- Read: `CLAUDE.md`
- Read: `COMPETITION_RULES_AGENT.md`
- Read: `outputs/phase/phase3/oof/oof_manifest.json`
- Read: `outputs/phase/phase3/oof/protocol_audit.json`
- Verify: `outputs/phase/phase3/oof/oof_logits.pt`
- Verify: `outputs/phase/phase3/oof/sample_quality.csv`
- Verify: `outputs/data/d3_strict/seed42/train.csv`

**Interfaces:**
- Consumes: 用户指定 commit `bfcb537`、现有 OOF 产物。
- Produces: 可执行环境检查结果；任何一项失败均停止。

- [ ] **Step 1: 校验 commit 和工作区**

```bash
git rev-parse --short HEAD
git status --short
```

Expected:

```text
bfcb537
```

允许存在用户已知的实验输出文件；不允许源代码处于无法解释的 dirty 状态。HEAD 不等于 `bfcb537` 时停止。

- [ ] **Step 2: 校验关键 OOF 文件存在且 hash 与 manifest 一致**

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path

manifest = json.loads(Path('outputs/phase/phase3/oof/oof_manifest.json').read_text())
checks = {
    'outputs/phase/phase3/oof/oof_logits.pt': manifest['oof_logits_sha256'],
    'outputs/phase/phase3/oof/sample_quality.csv': manifest['sample_quality_sha256'],
}
for name, expected in checks.items():
    p = Path(name)
    assert p.exists(), f'missing: {p}'
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == expected, (name, h, expected)
print('OOF_ARTIFACTS_OK')
PY
```

Expected: `OOF_ARTIFACTS_OK`。

- [ ] **Step 3: 校验 OOF 与 strict-train 一一覆盖**

```bash
python3 - <<'PY'
import pandas as pd
q = pd.read_csv('outputs/phase/phase3/oof/sample_quality.csv')
t = pd.read_csv('outputs/data/d3_strict/seed42/train.csv')
assert len(q) == len(t) == 91195
assert q.sample_id.is_unique
assert q.image_path.is_unique
assert set(q.image_path) == set(t.image_path)
assert dict(zip(q.image_path, q.original_label)) == dict(zip(t.image_path, t.label))
print('OOF_COVERAGE_OK')
PY
```

Expected: `OOF_COVERAGE_OK`。

- [ ] **Step 4: 运行当前测试作为基线**

```bash
pytest -q
```

Expected: 全部通过。记录测试数量和耗时到 `outputs/phase4/preflight.json`。

---

# Task 1: 修复重标、路径和 MixUp 权重基础设施

**Files:**
- Modify: `common/sample_weighting.py`
- Modify: `common/manifest_loader.py`
- Modify: `common/mixup.py`
- Modify: `experiments/baseline/train.py`
- Modify: `common/config_schema.py`
- Test: `tests/test_relabel_training.py`
- Test: `tests/test_weighted_mixup.py`
- Test: `tests/test_manifest_path_resolution.py`

**Interfaces:**
- Consumes: manifest required columns和 batch `paths`。
- Produces:
  - `BaseWeightProvider.get_training_labels(sample_paths, original_labels) -> torch.LongTensor`
  - `BaseWeightProvider.get_roles(sample_paths) -> list[str]`
  - `mixup_batch(...) -> (mixed_images, labels_a, labels_b, lam, permutation)`
  - `_reduce_weighted_mixup(...) -> scalar Tensor`

## 1.1 统一 canonical path

- [ ] **Step 1: 在 `common/manifest_loader.py` 新增路径函数和失败测试**

```python
from pathlib import Path


def canonical_image_path(value: str) -> str:
    return str(Path(value).expanduser().resolve())
```

测试必须证明：相对路径、绝对路径和包含 `..` 的路径得到同一 key。

Run:

```bash
pytest tests/test_manifest_path_resolution.py -v
```

Expected: 修改实现前 FAIL，加入函数并让所有 CSV provider 使用后 PASS。

## 1.2 让 hard relabel 真正进入 batch label

- [ ] **Step 2: 扩展 `BaseWeightProvider` 默认接口**

```python
def get_training_labels(
    self,
    sample_paths: list[str],
    original_labels: torch.Tensor,
) -> torch.Tensor:
    return original_labels


def get_roles(self, sample_paths: list[str]) -> list[str]:
    return ["clean"] * len(sample_paths)
```

`RelabelManifestProvider` 必须读取 `training_label` 和可选 `training_role`，使用 canonical path 保存，并实现：

```python
def get_training_labels(self, sample_paths, original_labels):
    values = []
    for path, original in zip(sample_paths, original_labels.tolist()):
        key = canonical_image_path(path)
        if key not in self._training_labels:
            if self._missing == "error":
                raise KeyError(f"Relabel label missing for: {path}")
            values.append(int(original))
        else:
            values.append(self._training_labels[key])
    return torch.tensor(values, device=original_labels.device, dtype=torch.long)
```

- [ ] **Step 3: 在 `train_one_epoch()` 中先改写训练标签，再做 MixUp**

在 `_unpack_batch()` 后、MixUp 前执行：

```python
if paths is not None and weight_provider is not None:
    labels = weight_provider.get_training_labels(list(paths), labels)
```

测试必须构造两张图片：一张 `training_label != original_label`，断言 criterion 收到新 label；`hard_relabel=False` 时仍收到 original label。

Run:

```bash
pytest tests/test_relabel_training.py -v
```

Expected: PASS。

## 1.3 修复 MixUp 绕过 sample weight

- [ ] **Step 4: 修改 `mixup_batch()` 返回 permutation**

固定返回五元组：

```python
return mixed_images, labels, labels[index], lam, index
```

未应用 MixUp 时返回 identity permutation：

```python
identity = torch.arange(images.size(0), device=images.device)
return images, labels, labels, 1.0, identity
```

- [ ] **Step 5: 在 `experiments/baseline/train.py` 新增加权 MixUp reducer**

```python
def _reduce_weighted_mixup(
    loss_a: torch.Tensor,
    loss_b: torch.Tensor,
    weights: torch.Tensor,
    permutation: torch.Tensor,
    lam: float,
    normalize_by_weight_sum: bool,
) -> torch.Tensor:
    wa = weights
    wb = weights[permutation]
    numerator = lam * wa * loss_a + (1.0 - lam) * wb * loss_b
    if normalize_by_weight_sum:
        denominator = lam * wa + (1.0 - lam) * wb
        return numerator.sum() / denominator.sum().clamp_min(1e-8)
    return numerator.mean()
```

训练循环在 MixUp 时必须先调用 provider 获取 batch weights，再调用该 reducer；不得再使用无权重 `reduce_loss(loss_per_sample)`。

- [ ] **Step 6: 添加四个加权 MixUp 测试**

必须覆盖：

1. 全 1 权重时结果等于原 MixUp loss；
2. `w_i=0` 时原标签 `y_i` 不产生梯度贡献；
3. 配对样本 `w_j=0` 时 `y_j` 不产生梯度贡献；
4. 全 0 权重时返回有限的 0，不出现 NaN。

Run:

```bash
pytest tests/test_weighted_mixup.py -v
```

Expected: 4+ tests PASS。

## 1.4 严格配置字段

- [ ] **Step 7: 统一使用 `missing_weight_policy`**

所有新 config 只能写：

```yaml
sample_weighting:
  missing_weight_policy: error
```

`common/config_schema.py` 应在发现旧字段 `missing_policy` 时抛出明确错误，避免配置看似生效、实际使用默认值。

- [ ] **Step 8: 全量回归测试并 commit**

```bash
pytest -q

git add common/sample_weighting.py common/manifest_loader.py common/mixup.py \
  common/config_schema.py experiments/baseline/train.py \
  tests/test_relabel_training.py tests/test_weighted_mixup.py \
  tests/test_manifest_path_resolution.py
git commit -m "fix: enforce manifest labels and weighted mixup"
```

---

# Task 2: 实现类条件 Confident Joint 与风险排序

**Files:**
- Create: `analysis/noisy_labels/__init__.py`
- Create: `analysis/noisy_labels/confident_joint.py`
- Create: `analysis/noisy_labels/build_purification_manifest.py`
- Test: `tests/test_confident_joint.py`
- Test: `tests/test_purification_manifest.py`

**Interfaces:**
- Consumes:
  - `sample_quality.csv`
  - `oof_logits.pt`
  - strict-train CSV
- Produces:
  - `estimate_class_thresholds(probabilities, noisy_labels, num_classes) -> np.ndarray`
  - `build_confident_joint(probabilities, noisy_labels, thresholds) -> np.ndarray`
  - `rank_label_issues(...) -> pd.DataFrame`
  - canonical `purification_manifest.csv`

## 2.1 算法定义

本项目不新增 `cleanlab` 依赖，使用可审计的最小 Confident Learning 实现。

对每个类别 `c` 计算 self-confidence threshold：

\[
t_c = \operatorname{mean}_{i:y_i=c} p_i(c)
\]

对样本 `i`，构造超过类别阈值的候选集合：

\[
S_i=\{c:p_i(c)\ge t_c\}
\]

- 若 `S_i` 非空，`suggested_label = argmax_{c in S_i} p_i(c)`；
- 否则 `suggested_label = argmax_c p_i(c)`。

confident joint：

```text
CJ[observed_label, suggested_label] += 1
```

每个 observed class 的预计问题数：

```text
estimated_issues[c] = row_sum[c] - CJ[c, c]
```

首轮 classwise drop 使用双重限额：

```text
class_drop_count[c] = min(
    estimated_issues[c],
    floor(0.10 * class_count[c])
)
```

全局 reject rate 超过 10% 时 fail closed。

issue score 固定为：

```python
issue_score = (
    0.50 * (1.0 - p_original_label)
    + 0.25 * top1_margin
    + 0.15 * (1.0 - knn_agreement)
    + 0.10 * (1.0 - flip_consistency)
)
```

仅在 `suggested_label != original_label` 的样本中按 observed class 降序选择。

## 2.2 TDD 实现

- [ ] **Step 1: 写阈值测试**

测试一个 2 类手工概率矩阵，明确断言 `t_0`、`t_1` 的数值。

- [ ] **Step 2: 写 confident joint 测试**

断言：

- joint shape 为 `(C, C)`；
- 所有元素为非负整数；
- `joint.sum() == N`；
- 手工样本进入预期 cell。

- [ ] **Step 3: 写 class cap 测试**

构造一个 estimated issue rate 为 30% 的类，断言实际最多选择 10%。

- [ ] **Step 4: 写 manifest schema 测试**

断言：

- row count 与 strict-train 完全一致；
- image_path 唯一；
- clean 为 `(training_label=original_label, sample_weight=1)`；
- rejected 为 `(training_label=original_label, sample_weight=0)`；
- 不存在其他 weight；
- 每类至少保留 90%；
- coverage=1.0。

Run:

```bash
pytest tests/test_confident_joint.py tests/test_purification_manifest.py -v
```

Expected: PASS。

## 2.3 CLI

- [ ] **Step 5: 实现统一 builder CLI**

命令：

```bash
python3 -m analysis.noisy_labels.build_purification_manifest \
  --mode cl_classwise_drop \
  --sample-quality outputs/phase/phase3/oof/sample_quality.csv \
  --oof-logits outputs/phase/phase3/oof/oof_logits.pt \
  --strict-train outputs/data/d3_strict/seed42/train.csv \
  --output-dir outputs/phase4/purification/nr_cl_classwise_drop \
  --max-class-reject-rate 0.10 \
  --max-global-reject-rate 0.10
```

Expected：输出 6 个标准文件，`protocol_audit.json` 中 `training_allowed=true`。

- [ ] **Step 6: commit**

```bash
git add analysis/noisy_labels tests/test_confident_joint.py tests/test_purification_manifest.py
git commit -m "feat: add class conditional label issue estimation"
```

---

# Task 3: 实现 OOF/kNN/prototype 高精度共识筛选与重标

**Files:**
- Modify: `analysis/noisy_labels/build_purification_manifest.py`
- Create: `analysis/noisy_labels/consensus.py`
- Test: `tests/test_consensus_selection.py`

**Interfaces:**
- Consumes: Task 2 的 issue table 与现有质量信号。
- Produces:
  - `select_consensus_drop(frame) -> Boolean Series`
  - `select_consensus_relabel(frame, target_fraction) -> Boolean Series`

## 3.1 `NR_CL_KNN_DROP` 固定规则

样本只有同时满足以下条件才变为 `rejected`：

```text
confident_joint 判断为 label issue
oof_top1 != original_label
knn_top1 != original_label
oof_top1 == knn_top1
top1_margin >= 该 observed class 的 75% 分位
knn_agreement >= 0.60
duplicate_conflict_flag == false
```

额外 gate：

```text
每类 reject <= 10%
全局 reject <= 8%
每类至少保留 50 个 clean 样本
```

`prototype_top1` 只作为审计列，不作为该实验硬条件，避免把已失败的全局质心重新变成唯一裁决者。

## 3.2 `NR_CONSENSUS_RELABEL_1P` 固定规则

候选必须满足：

```text
confident_joint 判断为 label issue
oof_top1 == knn_top1 == prototype_top1
oof_top1 != original_label
p_top1 >= 0.90
top1_margin >= 0.50
knn_agreement >= 0.70
flip_consistency == 1
duplicate_conflict_flag == false
```

在候选中按以下 score 降序：

```python
relabel_score = (
    0.40 * p_top1
    + 0.25 * top1_margin
    + 0.20 * knn_agreement
    + 0.15 * flip_consistency
)
```

选择上限：

```text
全局最多 floor(0.01 * N)
每个 original class 最多 3%
每个 target class 接收的 pseudo 样本最多为其 clean count 的 5%
```

被选样本：

```text
training_role=pseudo
training_label=oof_top1
sample_weight=1.0
```

未选中的 label issue 不在本实验中删除，保持 clean/original，用于隔离"少量高精度重标"本身的效果。

## 3.3 测试与生成

- [ ] **Step 1: 写 consensus drop 测试**

必须证明任一硬条件不满足时样本不被 drop；所有条件满足时被 drop。

- [ ] **Step 2: 写 relabel cap 测试**

必须证明全局 1%、source class 3%、target class 5% 三个上限都生效。

- [ ] **Step 3: 写 duplicate conflict 排除测试**

`duplicate_conflict_flag=true` 的样本不得自动重标。

Run:

```bash
pytest tests/test_consensus_selection.py -v
```

Expected: PASS。

- [ ] **Step 4: 生成两个 manifest**

```bash
python3 -m analysis.noisy_labels.build_purification_manifest \
  --mode cl_knn_drop \
  --sample-quality outputs/phase/phase3/oof/sample_quality.csv \
  --oof-logits outputs/phase/phase3/oof/oof_logits.pt \
  --strict-train outputs/data/d3_strict/seed42/train.csv \
  --output-dir outputs/phase4/purification/nr_cl_knn_drop

python3 -m analysis.noisy_labels.build_purification_manifest \
  --mode consensus_relabel \
  --target-relabel-fraction 0.01 \
  --sample-quality outputs/phase/phase3/oof/sample_quality.csv \
  --oof-logits outputs/phase/phase3/oof/oof_logits.pt \
  --strict-train outputs/data/d3_strict/seed42/train.csv \
  --output-dir outputs/phase4/purification/nr_consensus_relabel_1p
```

- [ ] **Step 5: commit**

```bash
git add analysis/noisy_labels tests/test_consensus_selection.py
git commit -m "feat: add multi signal drop and relabel selection"
```

---

# Task 4: 增加 manifest 审计和训练启动前 fail-closed 检查

**Files:**
- Modify: `common/manifest_loader.py`
- Modify: `experiments/baseline/train.py`
- Create: `scripts/audit_purification_manifest.py`
- Test: `tests/test_purification_audit.py`

**Interfaces:**
- Consumes: canonical purification manifest + actual `train_loader.dataset`。
- Produces: `manifest_runtime_audit.json`；任何 error 阻止 epoch 1。

- [ ] **Step 1: 扩展 optional columns**

在 `ManifestLoader.OPTIONAL_COLUMNS` 加入：

```python
"training_role",
"selection_reason",
"suggested_label",
"prototype_top1",
"knn_top1",
"top1_margin",
"duplicate_conflict_flag",
```

- [ ] **Step 2: 增加 partition audit**

审计规则：

```text
coverage == 1.0
no duplicate sample_id
no duplicate image_path
all sample_weight in [0, 1]
training_role in {clean, rejected, pseudo}
clean: training_label == original_label and weight == 1
rejected: training_label == original_label and weight == 0
pseudo: training_label != original_label and weight == 1
pseudo global <= 3%
pseudo per source class <= 5%
reject per class <= configured limit
all 500 classes retain clean samples
```

- [ ] **Step 3: 训练启动前按 actual dataset paths 重新审计**

不能只审计 CSV 对 CSV。`train.py` 构造完 dataset 后，应使用 `ds.samples` 和 `ds.labels` 与 manifest 重新比对；任何 missing/extra/label mismatch 直接抛错。

- [ ] **Step 4: 测试 fail-closed cases**

覆盖：missing row、extra row、错误 original label、非法 role、rejected 但 weight=1、pseudo 但 training_label 未变、某类全部被删。

Run:

```bash
pytest tests/test_purification_audit.py -v
```

Expected: PASS。

- [ ] **Step 5: commit**

```bash
git add common/manifest_loader.py experiments/baseline/train.py \
  scripts/audit_purification_manifest.py tests/test_purification_audit.py
git commit -m "feat: fail closed on purification manifest mismatch"
```

---

# Task 5: 创建 Wave A 配置和统一执行脚本

**Files:**
- Create: `configs/nr_ctrl_oof_zero_0001_fixed.yaml`
- Create: `configs/nr_cl_classwise_drop.yaml`
- Create: `configs/nr_cl_knn_drop.yaml`
- Create: `configs/nr_consensus_relabel_1p.yaml`
- Create: `scripts/run_noise_robust_wave_a.sh`
- Create: `results/noise_robust_wave.csv`
- Test: `tests/test_noise_robust_configs.py`

## 5.1 配置公共块

四个 config 都必须使用：

```yaml
experiment:
  mode: dev
  head_type: linear
  augmentation_preset: a0

data:
  split_seed: 42
  train_seed: 42
  split_dir: outputs/data/d3_strict/seed42
  train_dir: train
  test_dir: test
  expected_num_classes: 500
  class_mapping_path: outputs/data/master_splits/seed42
  use_full_training_set: false

model:
  clip_model_name: ViT-B/32
  freeze_clip: true
  num_classes: 500
  use_cached_features: false
  unfreeze_last_n_blocks: 0
  train_ln_post: false
  train_visual_proj: false

loss:
  name: gce
  q: 0.5
  probability_epsilon: 1.0e-7

mixup:
  enabled: true
  alpha: 0.2
  probability: 0.2

train:
  epochs: 50
  batch_size: 128
  lr: 0.005
  weight_decay: 0.0001
  warmup_epochs: 2
  scheduler: cosine
  min_lr_ratio: 0.01
  early_stop_patience: 10
  max_grad_norm: 1.0
  amp: true
  device: cuda
  num_workers: 8
```

Control 使用修复后的现有 `oof_zero_weight_manifest_thresh0.001.csv`；其他三个指向 Task 2/3 生成的 manifest。所有 config：

```yaml
sample_weighting:
  type: relabel_manifest
  min_weight: 0.0
  max_weight: 1.0
  hard_relabel: true
  normalize_by_weight_sum: true
  missing_weight_policy: error
```

即使 drop 实验没有 pseudo label，也统一用 `relabel_manifest` schema，减少训练路径差异。

## 5.2 config 测试

- [ ] **Step 1: 验证唯一变量**

测试加载四个 resolved config，除以下字段外必须完全一致：

```text
experiment.id
sample_weighting.manifest_path
output.*
train.save_dir
```

- [ ] **Step 2: 验证 split、backbone 和 policy**

断言：

```text
split_dir == outputs/data/d3_strict/seed42
clip_model_name == ViT-B/32
missing_weight_policy == error
min_weight == 0.0
```

Run:

```bash
pytest tests/test_noise_robust_configs.py -v
```

Expected: PASS。

## 5.3 统一 runner

- [ ] **Step 3: `run_noise_robust_wave_a.sh` 先审计、后训练**

脚本顺序固定：

```bash
set -euo pipefail

pytest -q
python3 scripts/audit_purification_manifest.py --config "$CONFIG"
python3 -m experiments.baseline.train --config "$CONFIG"
python3 -m experiments.baseline.evaluate --config "$CONFIG" \
  --ckpt "$CKPT" --output-json "$REEVAL"
python3 tools/evaluate_dual_validation.py --name "$EXP_ID" \
  --config "$CONFIG" --ckpt "$CKPT" --device cuda
python3 -m experiments.baseline.infer --config "$CONFIG" --ckpt "$CKPT"
python3 -m common.submission --raw "$PRED_RAW" --out_dir "$SUBMISSION_DIR"
python3 scripts/check_submission.py --test_dir test \
  --csv "$SUBMISSION_DIR/pred_results.csv" \
  --zip "$SUBMISSION_DIR/submission.zip"
```

脚本不得自动提交平台，也不得自动运行 TTA。

- [ ] **Step 4: commit**

```bash
git add configs/nr_*.yaml scripts/run_noise_robust_wave_a.sh \
  tests/test_noise_robust_configs.py results/noise_robust_wave.csv
git commit -m "exp: define noise purification wave a"
```

---

# Task 6: 运行修复后的 Control 与 Wave A

**Files:**
- Write outputs only under `outputs/nr_*`
- Update: `results/noise_robust_wave.csv`
- Update: `results/submission_registry.csv` only after actual platform score is received

**Interfaces:**
- Consumes: Tasks 1–5。
- Produces: 四个 seed42 Bare 模型及可审计比较。

## 6.1 运行顺序

- [ ] **Step 1: 重跑修复后的因果 Control**

```bash
bash scripts/run_noise_robust_wave_a.sh configs/nr_ctrl_oof_zero_0001_fixed.yaml
```

目的：量化旧 `S_OOF_ZERO_0001` 中 MixUp weight bypass 的影响。不得直接拿旧 checkpoint 当新方法 control。

- [ ] **Step 2: 运行 classwise drop**

```bash
bash scripts/run_noise_robust_wave_a.sh configs/nr_cl_classwise_drop.yaml
```

- [ ] **Step 3: 运行 CL+kNN drop**

```bash
bash scripts/run_noise_robust_wave_a.sh configs/nr_cl_knn_drop.yaml
```

- [ ] **Step 4: 运行 1% consensus relabel**

```bash
bash scripts/run_noise_robust_wave_a.sh configs/nr_consensus_relabel_1p.yaml
```

## 6.2 本地安全 gate

本地指标不能用于精确预测平台，但必须阻止明显失败模型。候选只有全部满足才可生成平台 Bare 包：

```text
protocol_audit passed
manifest_runtime_audit passed
finite train loss and gradients
predicted class count == 500
raw macro >= fixed control raw macro - 0.30pp
bottom10 accuracy >= fixed control bottom10 - 1.00pp
no class validation accuracy drops by >25pp while its prediction count collapses >50%
no class has clean_count == 0
```

本地 micro 只记录，不作为主要淘汰条件。

## 6.3 平台 gate

以修复后的 Control 平台 Bare 为新基准 `B_fixed`：

```text
candidate Bare < B_fixed - 0.10pp  -> 关闭
|candidate Bare - B_fixed| < 0.10pp -> 视为不确定，不补 TTA，不补 seed
candidate Bare >= B_fixed + 0.10pp -> 正信号，补 seed3407
candidate Bare >= B_fixed + 0.20pp -> 强正信号，可进入 Wave B
```

0.10pp 约对应 25 张测试图，只能视为弱正信号，必须补 seed。

- [ ] **Step 5: 写入结果表**

`results/noise_robust_wave.csv` 固定列：

```text
experiment_id,parent,train_seed,manifest_sha256,reject_rate,relabel_rate,
raw_micro,raw_macro,bottom10,trusted_class_balanced,bare_online,tta_online,
status,decision_reason
```

---

# Task 7: 实现 rejected-only 半监督一致性

**Gate:** 只有 Wave A 至少一个候选 Bare `>= B_fixed + 0.20pp` 时执行。

**Files:**
- Modify: `common/hooks.py`
- Modify: `experiments/baseline/train.py`
- Modify: `common/config_schema.py`
- Create: `common/partition_consistency.py`
- Create: `configs/nr_reject_consistency.yaml`
- Create: `configs/nr_reject_pseudo_1p.yaml`
- Create: `configs/nr_reject_ssl_combined.yaml`
- Test: `tests/test_partition_consistency.py`

**Interfaces:**
- Consumes: 最佳 Wave A drop manifest、父模型 checkpoint。
- Produces: rejected-only consistency loss；最终仍为单一 student。

## 7.1 Teacher 定义

使用最佳 Wave A frozen checkpoint 作为固定 teacher，不使用多个 fold 模型对测试集推理。Teacher：

```text
加载一次
requires_grad=False
eval mode
训练过程中不更新
只对 official train rejected 样本产生目标
```

新增 config：

```yaml
partition_consistency:
  enabled: true
  teacher_checkpoint: outputs/<best_wave_a>/seed42/checkpoints/best.pt
  rejected_only: true
  confidence_threshold: 0.90
  consistency_weight: 0.5
  ramp_epochs: 10
  temperature: 1.0
  view: horizontal_flip
```

## 7.2 Loss 定义

clean/pseudo 样本继续使用监督损失；rejected 样本监督 weight 为 0。对 rejected 且 teacher confidence `>=0.90` 的样本：

\[
L_u = \operatorname{KL}\left(
\operatorname{softmax}(z_t/T)\;||\;
\operatorname{softmax}(z_s^{flip}/T)
\right)
\]

总损失：

\[
L=L_{sup}+r(epoch)\lambda_uL_u
\]

其中 `r(epoch)` 使用现有 sigmoid ramp-up，`lambda_u=0.5` 固定首轮，不做网格。

## 7.3 关键实现要求

- [ ] **Step 1: provider 暴露 batch role mask**

```python
roles = weight_provider.get_roles(list(paths))
rejected_mask = torch.tensor(
    [role == "rejected" for role in roles],
    device=device,
    dtype=torch.bool,
)
```

- [ ] **Step 2: consistency 仅作用于 rejected**

最终 mask：

```python
mask = rejected_mask & teacher_confidence_mask
```

任何 clean 样本不得进入 `L_u`，以保持因果解释。

- [ ] **Step 3: MixUp batch 不执行 rejected consistency**

首版为避免 mixed image 无法对应单一样本角色，`mixup_applied=True` 时跳过 `L_u`，但监督部分仍使用 Task 1 的 weighted MixUp。

- [ ] **Step 4: 测试**

必须覆盖：

1. batch 无 rejected 时 consistency=0；
2. rejected 但 teacher confidence 低时 consistency=0；
3. clean 高置信样本不进入 consistency；
4. rejected 高置信样本产生有限正 loss；
5. teacher 参数无梯度；
6. student 参数有梯度。

Run:

```bash
pytest tests/test_partition_consistency.py -v
```

Expected: PASS。

## 7.4 三个实验

### `NR_REJECT_CONSISTENCY`

- manifest：最佳 drop manifest；
- rejected weight=0；
- 不包含 pseudo；
- 启用 consistency。

### `NR_REJECT_PSEUDO_1P`

- manifest：把最佳 drop manifest 与 1% consensus pseudo 合并；
- 不启用 consistency；
- 用来隔离 pseudo-label 监督收益。

### `NR_REJECT_SSL_COMBINED`

- 同时使用 rejected consistency 和 1% pseudo label。

- [ ] **Step 5: 运行顺序**

```bash
python3 -m experiments.baseline.train --config configs/nr_reject_consistency.yaml
python3 -m experiments.baseline.train --config configs/nr_reject_pseudo_1p.yaml
```

只有二者至少一个 Bare 高于最佳 Wave A `+0.10pp`，才运行：

```bash
python3 -m experiments.baseline.train --config configs/nr_reject_ssl_combined.yaml
```

- [ ] **Step 6: commit**

```bash
git add common/hooks.py common/partition_consistency.py common/config_schema.py \
  experiments/baseline/train.py configs/nr_reject_*.yaml \
  tests/test_partition_consistency.py
git commit -m "feat: recover rejected samples with consistency training"
```

---

# Task 8: 条件性净化监督 PEFT + 特征蒸馏

**Gate:** 只有 Wave A/B 最佳候选满足以下全部条件才执行：

```text
seed42 Bare >= B_fixed + 0.20pp
seed3407 本地方向一致且无类别塌缩
同一 manifest 的 frozen control 已完成
```

**Files:**
- Modify: `experiments/baseline/model.py`
- Modify: `experiments/baseline/train.py`
- Modify: `common/feature_distillation.py`
- Modify: `common/config_schema.py`
- Create: `configs/nr_pure_frozen_control.yaml`
- Create: `configs/nr_pure_ln_distill.yaml`
- Test: `tests/test_feature_distillation_integration.py`

## 8.1 配对设计

两个实验必须使用：

```text
同一 parent checkpoint
同一 purification manifest
同一 train seed
同一 epoch=15
同一 head lr=1e-4
同一 scheduler
```

唯一差异：

- `NR_PURE_FROZEN_CONTROL`：backbone 全冻结；
- `NR_PURE_LN_DISTILL`：visual LayerNorm-only，backbone LR `1e-6`，feature distillation。

不得把旧 `S_PEFT_E0/E1` 当配对 control，因为其监督集合不同。

## 8.2 特征蒸馏

新增 config：

```yaml
feature_distillation:
  enabled: true
  parent_checkpoint: outputs/<best_purified>/seed42/checkpoints/best.pt
  weight: 1.0
  target_ratio: 0.15
  normalize_features: true
  compare_after_projection: true
  calibrate_on_first_batch: true
```

训练首个非 MixUp batch 自动校准一次 lambda，之后冻结该值并写入 `resolved_config.yaml` 与 checkpoint metadata。

Loss：

\[
L=L_{purified}+\lambda_{feat}(1-\cos(f_{student},f_{frozen\ parent}))
\]

## 8.3 测试和 gate

- [ ] **Step 1: 测试 parent 完全冻结**
- [ ] **Step 2: 测试 student LN 有梯度、非 LN backbone 无梯度**
- [ ] **Step 3: 测试 feature loss 有限且初始接近 0**
- [ ] **Step 4: 测试 checkpoint 恢复 lambda**

Run:

```bash
pytest tests/test_feature_distillation_integration.py -v
```

- [ ] **Step 5: 先跑 frozen paired control，再跑 LN distill**

```bash
python3 -m experiments.baseline.train \
  --config configs/nr_pure_frozen_control.yaml \
  --init-checkpoint outputs/<best_purified>/seed42/checkpoints/best.pt

python3 -m experiments.baseline.train \
  --config configs/nr_pure_ln_distill.yaml \
  --init-checkpoint outputs/<best_purified>/seed42/checkpoints/best.pt
```

保留条件：

```text
LN-distill raw macro >= paired frozen raw macro - 0.20pp
LN-distill bottom10 >= paired frozen bottom10 - 0.50pp
feature drift mean cosine distance <= 0.05
Bare platform >= paired frozen Bare + 0.10pp
```

任一失败则关闭 PEFT，不跑 LoRA。

- [ ] **Step 6: commit**

```bash
git add experiments/baseline/model.py experiments/baseline/train.py \
  common/feature_distillation.py common/config_schema.py \
  configs/nr_pure_*.yaml tests/test_feature_distillation_integration.py
git commit -m "exp: add purified supervision layernorm distillation pair"
```

---

# Task 9: 多 seed 确认与最终候选选择

**Gate:** 仅对平台 Bare 正收益候选执行。

**Files:**
- Create: winner seed3407 config
- Update: `results/noise_robust_wave.csv`
- Create: `results/noise_robust_final_report.md`

- [ ] **Step 1: 复制 winner config，只改 train seed 和输出路径**

```yaml
data:
  split_seed: 42
  train_seed: 3407
```

manifest 必须保持同一个 SHA-256；不得为 seed3407 重新用 validation 或平台结果调整阈值。

- [ ] **Step 2: pair audit**

```bash
python3 tools/audit_experiment_pair.py \
  --reference-config configs/<winner_seed42>.yaml \
  --candidate-config configs/<winner_seed3407>.yaml \
  --reference-ckpt outputs/<winner>/seed42/checkpoints/best.pt \
  --candidate-ckpt outputs/<winner>/seed3407/checkpoints/best.pt \
  --output outputs/<winner>/pair_seed42_seed3407.json
```

Expected：除 train seed/output path/checkpoint 外无混杂变量。

- [ ] **Step 3: 稳定候选定义**

winner 只有满足以下条件才能进入最终 full-clean/final-fit：

```text
两个 seed 均通过 protocol audit
两个 seed raw macro 均不低于各自 paired control -0.30pp
seed42 Bare 有正收益
seed3407 预测分布无类别塌缩
两个 seed 的 test prediction disagreement 被记录
```

由于比赛禁止多模型集成，两个 seed 只能用于稳定性验证，最终只选择一个 checkpoint。

- [ ] **Step 4: TTA**

仅对最终选中的单一 checkpoint 执行 horizontal-flip TTA：

```bash
python3 scripts/infer_tta.py \
  --config configs/<winner>.yaml \
  --checkpoint outputs/<winner>/seed42/checkpoints/best.pt \
  --tta horizontal_flip \
  --output-dir outputs/<winner>/seed42/submissions_tta

python3 scripts/check_submission.py \
  --test_dir test \
  --csv outputs/<winner>/seed42/submissions_tta/pred_results.csv \
  --zip outputs/<winner>/seed42/submissions_tta/submission.zip
```

- [ ] **Step 5: 最终报告**

`results/noise_robust_final_report.md` 必须包含：

```text
1. 代码 commit 与环境
2. OOF 输入 hash
3. 每个 manifest 的样本数、reject/relabel 率、class caps
4. fixed control 与所有候选的 local/platform Bare/TTA
5. seed42/3407 稳定性
6. 失败实验及停止原因
7. 最终单 checkpoint 和提交 ZIP hash
```

---

## 3. 明确停止条件

出现以下任一情况立即停止对应分支：

- manifest coverage < 100%；
- OOF/strict-train hash 不一致；
- 某类 clean 样本归零；
- global reject > 10%；
- global relabel > 3%；
- per-class relabel > 5%；
- 训练 loss、gradient norm、logits 出现非有限值；
- predicted class count < 500；
- Wave A 三个候选 Bare 均未高于 fixed control 0.10pp：关闭重标/半监督分支，转而重新审视验证与数据异常，不继续调阈值；
- rejected consistency 无增益：不调整 `lambda_u` 大网格；
- LN-distill 不优于 paired frozen control：关闭 LoRA 和更深解冻。

---

## 4. Agent 执行纪律

1. 每次只实现并验证一个 Task；不得边改基础设施边启动长训练。
2. 每个 Task 先写 failing test，再写最小实现，再运行相关测试和全量测试。
3. 不修改用户无关的 dirty 文件；不删除现有 outputs。
4. 长训练前打印 resolved config、manifest SHA-256、dataset coverage 和 role counts。
5. 训练命令启动后，只基于日志和明确产物判断状态；不得人工修改中间 manifest。
6. 平台成绩只登记，不反向进行密集阈值搜索。
7. 所有失败都写入 `results/noise_robust_wave.csv`，不得只保留成功结果。

---

## 5. 最小成功标准

本计划不是以"所有实验跑完"为成功，而是以下证据链完整：

```text
修复后的 hard-zero control
    ↓
类条件 / 多信号筛选至少一个 Bare 稳定正收益
    ↓
第二 seed 方向一致
    ↓
被拒样本回收或净化后 PEFT 至少一个额外正收益
    ↓
最终仍为单一 CLIP ViT-B/32 student checkpoint
```

若 Wave A 没有平台正收益，计划应在 Wave A 结束，不继续 Wave B/C。该停止结论本身是有效实验结果。
