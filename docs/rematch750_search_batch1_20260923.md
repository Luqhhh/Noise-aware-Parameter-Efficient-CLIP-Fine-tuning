# 复赛提分搜索 第一批（2026-09-23）

**状态：已启动。** 第一批 4 个点在本机 GPU 上串行执行。
方案由 **GPT-6** 制定（四条轴、首轮 11 点、24 次训练总上限、+2pp 本地标记定义、平台提交编排、止损规则）。
执行方：clairvoyanttt 本机 agent。分支 `clairvoyant/search-mixup-anchor`。

**背景指令（用户原话）：「现在关键是要找提分策略，现在先做搜索。」**
即：目标是提分，动作是先跑再判断；探索期的试错是预期成本。搜前材料见 [strategy_brief_20260923.md](strategy_brief_20260923.md)。

---

## 1. 第一批 4 个点（已启动）

共同固定：本阶段同一 LP 初始化、同一划分、batch 32、原 CE→GCE 配方（CE 前 2 轮、GCE q=0.5）、单模型推理、224 center crop、无 TTA/prior。**除指定变量外不动其他设置。**

| # | 实验 ID | 轴 | 相对基线的唯一变量 | 轮数 | 配置 |
|---:|---|---|---|---:|---|
| 1 | `SEARCH_A1_MIXUP_A02` | A | `mixup_alpha` 0.0→**0.2**，`mixup_probability` 0.0→**1.0** | 8 | `configs/search_a1_mixup_a02.yaml` |
| 2 | `SEARCH_A2_MIXUP_A04` | A | `mixup_alpha` 0.0→**0.4**，`mixup_probability` 0.0→**1.0** | 8 | `configs/search_a2_mixup_a04.yaml` |
| 3 | `SEARCH_B1_ANCHOR05_LR1E5` | B | `feature_distillation_weight` 2.0→**0.5**，`backbone_lr` 3e-6→**1e-5** | 16 | `configs/search_b1_anchor05_lr1e5.yaml` |
| 4 | `SEARCH_B2_ANCHOR00_LR3E6` | B | `feature_distillation_weight` 2.0→**0.0**（关闭特征蒸馏） | 16 | `configs/search_b2_anchor00_lr3e6.yaml` |

B 轴的 16 轮伴随 `schedule_epochs` 8→16（余弦 horizon 同步），head LR 固定 1e-4，CE 仍为前 2 轮。

**基线（唯一对照）**：`configs/rematch750_ft.yaml` —— 本机 macro 71.8054% / micro 72.8696%，平台 60.9657%。

### 1.1 非科学变量改动（唯一一处，必须与科学变量分开读）

四个点的 `evaluation.selection_policy` 从 `best_selector` 改为 **`last_epoch`**：按 GPT-6 要求使用**预定训练终点**，不按本地最高分挑 checkpoint。这与基线不同，因此**对比时比的是「终点 vs 终点」，不是「终点 vs 历史最高分」** —— 基线的 72.8696% 是 `best_selector` 选的，若基线改用 last_epoch，其数值可能略低。这一点在解读任何差值时必须记住。

### 1.2 两处与方案文本的偏离（如实记录）

| 方案原文 | 实际执行 | 原因 |
|---|---|---|
| 「不用本地 GPU」，用空闲且获准的 NPU | **全部落在本机 RTX 4060 Laptop 8GB** | 用户 2026-09-23 明确告知：**NPU 暂时不能用** |
| 「有两张 NPU 就开两个独立任务」 | **串行单卡** | 本机只有一张 GPU，且显存 8GB 只够一个 full-FT（实测 4.7GB） |

时间预算因此从方案的「约 40～50 NPU·h」变为**本机约 11 小时串行**（8 轮约 1h34m，16 轮约 3h10m，按实测 0.169 s/step 推）。

### 1.3 今天的提交额度（2026-09-23）

用户要求今天用掉剩余的一个额度（每天 2 次，今天的探针已用掉 1 次），否则额度作废。
**该额度发给 A 轴的 MixUp 点**，理由：额度免费（作废即零收益），而 A1/A2 是批次里唯一已产出且机制未被平台验证过的候选。

**已核查并排除的替代项**：`RM_OOFW`（OOF 加权 full-FT，9-22 已训完）本地 **micro 72.7487% / macro 71.6743%**，比基线 FT 低 0.12pp —— 本质是「FT + OOF 加权，无变化」（0.12pp 远在单种子 0.90pp 变异内），且从未生成提交包。拿它占额度等于花在一个已知约等于 60.9% 的点上。

**规则**：A1 本地不比基线差太多 → 今天发 A1（这样 α=0.2 与 α=0.4 两个强度都能取得平台读数）；A1 本地明显崩 → 等 A2 发 A2。**无论哪种，今天必有一个包出去。**

出包由 `infer` 子命令一次完成：推理 → `scripts/check_submission.py`（`check=True`，校验不过即失败）→ `register()`。包落在 `outputs/search_20260923/<EXP_ID>/seed42/submission/`；登记表写入**工作树自己的** `results/rematch_submission_registry.csv`，不碰主仓库。

**上传前必须核对**：`selected_report.json` 里的被选轮次应等于总轮数（`last_epoch` 语义），否则包不是预定终点的模型。

---

## 2. 完整第一批范围（已登记，尚未启动）

首轮共 11 个点。除上表 4 点外还有 7 点，按 GPT-6 排序在后续补齐：

| 轴 | 剩余点位 | 说明 |
|---|---|---|
| **B（补 3 点）** | 原参数控制点（anchor=2.0, backbone_lr=3e-6, 16 轮）；anchor=0.5 × backbone_lr=3e-6；anchor=0 × backbone_lr=1e-5 | 补齐 B 的 2×2 + 控制点 |
| **C（2 点）** | 320px、384px，全参 8 轮，单尺度单前向 | 需先解决 `rematch_assets.py:31` 的「参考预处理要求 224」硬约束，并确认 8GB 显存下的 batch |
| **D（2 点）** | 训练侧成像扰动：轻档（JPEG 70–95 / 模糊 σ0.1–0.5 / 噪声 σ0–0.01）、中档（JPEG 40–90 / 模糊 σ0.1–1.0 / 噪声 σ0–0.03），每图 p=0.5 随机选一种，8 轮 | 推理保持确定性 |

**总上限 24 次训练**：11 首轮 + ≤4 组合 + ≤4 配对复验 + ≤5 邻近点。无证据不花完。

---

## 3. 判据（GPT-6 定义，已登记）

以本机可复现基线 **micro 72.8696% / macro 71.8054%** 为参照：

| 标记 | 数值定义 | 用途 |
|---|---|---|
| 显著本地跃升候选 | micro ≥ 74.8696% 且 macro ≥ 73.3054% | 追加一个 seed 检查 |
| 强本地跃升候选 | micro ≥ 76.8696% 且 macro ≥ 74.8054% | 优先取得平台证据 |
| 平台有效候选 | 实际提交优于 60.9657% | 按收益大小与配对结果决定投入 |

**这不是统计显著性声明。** 关键调整是**撤销探索期逐点必须 +0.30pp 才能继续的要求**：本地未涨也可按预定机制覆盖顺序提交平台；本地大涨只增加关注度，不自动晋级。

### 止损规则

- NaN、无有效更新、数据或映射错误 → 立即停止修复。
- 原 **−2pp 本地止损降级为警报**，不在前两轮自动杀掉强正则 / 长 horizon 候选。
- 跑过一半预算后，连续两次验证都比基线低 **≥5pp** 且无恢复趋势 → 停止该点。
- 一个轴的预定点和平台代表点均无支持 → 关闭该轴，不无限扩展。
- **达到 24 次训练或 12 次提交先结算。** 若最好的可重复平台增益仍不足约 1pp，承认本批未发现跨量级突破，**不得自动续成 80 次 FT 微调**；届时结论应是「换到新的数据/监督信息空间，或下调本轮目标」。

### 平台提交编排（预排，避免变成本地选秀）

第一对 `MixUp α=0.4` 与 `anchor=0.5、visual_lr=1e-5`；第二对 `320px` 与 `中档成像扰动`；随后 4 个名额覆盖剩余不同强度与机制的代表点；再留 4 个名额做候选—控制 seed 配对复验。**首阶段最多 12 次提交，约六天**，不是先耗尽训练预算再上平台。

---

## 4. V3

**Kill V3 作为当前提分搜索任务**：不继续扩展 batch/LR 吞吐扫描，**不删除分支或产物**（`origin/codex/rematch750_v3_scan` 及其 6 个提交保留）。已有结果可用于估时。
理由：它优化的是耗时不是平台成绩，且研究对象全在 FT 家族内部 —— FT 已降为锚点。

---

## 5. 复现

工作树 `/home/clairvoyant/code/worktrees/search-mixup-anchor`，分支 `clairvoyant/search-mixup-anchor`。
`train` / `test` / `artifacts` 为指向主仓库的**只读共享符号链接**；`init_checkpoint` 用绝对路径指向主仓库的 RM_LP 权重。

```bash
WT=/home/clairvoyant/code/worktrees/search-mixup-anchor
MAIN=/home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning
cd $WT
export PYTHONPATH=$WT/reproducibility/aegis_f1
bash _run_search_batch.sh search_a1_mixup_a02 search_a2_mixup_a04 \
     search_b1_anchor05_lr1e5 search_b2_anchor00_lr3e6
```

日志：`outputs/search_20260923/_logs/<config>.log` 与 `_driver.log`。
产物：`outputs/search_20260923/<EXPERIMENT_ID>/seed42/`。

**没有平台提交。** 本批全部为本地训练。

---

## 6. 已完成的启动前核查（供复用）

- 四个配置均通过 `aegis_clip.config.load_config` 解析，`feature_distillation_weight: 0.0` 未被守卫拒绝。
- MixUp 路径确认有效：`trainer.py:756` 无条件调用 `mixup(...)`，直接读 `loss.mixup_alpha` / `loss.mixup_probability`；且 `mixed_reference = λ·reference + (1−λ)·reference[perm]`，**参考特征按同一系数混合**，与特征蒸馏一致。
- A1 首步审计通过；LP 初始化加载成功（epoch=20）；full_finetune 85,836,270 可训练参数；200/200 有效更新。
