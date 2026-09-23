# 复赛提分搜索 第一批（2026-09-23）

**状态：已启动。** 第一批 4 个点在本机 GPU 上串行执行。

> **2026-09-23 用户指令变更算力分工：本机只出策略，训练全部交给队友的 NPU。** 见 [当前执行入口](../docs/current_execution_plan.md) 与 [交接 brief](../docs/handover_brief_20260923.md) §0.1。
> 对本批的后果：**B2 重跑（16:35 启动）是本批、也是本机最后一批长训练**；**B1 不在本机重跑**（它 15:03 崩溃后零数据），改为并入交给 NPU 的点位 —— 在 NPU 上跑一次约 34 分钟，比在本机赌 2.5 小时划算。见 §7。
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

**决策时点：今晚 22:00（用户可在该时点上传），不是训练一结束就定。**

到 22:00 时手上有三个已出包的候选（A1 约 13:58、A2 约 15:35、B1 约 18:50），足以在**完整的本地读数**下挑选，而不是在只跑完一个点时拍板：

| 包 | 本地 micro / macro | 22:00 前是否可用 |
|---|---|---|
| `SEARCH_A1_MIXUP_A02` | **72.5000 / 71.4466**（−0.37pp） | 是 |
| `SEARCH_A2_MIXUP_A04` | **71.9288 / 70.8878**（−0.94pp） | 是 |
| `SEARCH_B1_ANCHOR05_LR1E5` | **无 —— 训练崩溃，见 §1.4** | 否 |
| `SEARCH_B2_ANCHOR00_LR3E6` | **无有效值 —— 训练崩溃于第 5 轮，包是第 4 轮模型，见 §1.4** | 否 |

**选择规则（22:00 执行）**：在可用包里选本地 micro 最优、且 macro 不与之矛盾者。若与基线的差异均落在单种子 0.90pp 变异之内（即都不显著超过 72.8696%），如实记录「本批暂未显示本地跃升」。
**A1/A2 现均有实测值且均低于基线**（见 §1.4），原先「优先方案提名的 A2 与 B1」已不成立 —— A2 有实测的 −0.94pp，B1 没有结果。

> **本节已作废（2026-09-23 17:2x 关闭）**：算力分工变更后本批收尾，**今日不使用该额度**（A1/A2 均低于基线 → 按「本地提升不显著就不浪费名额」不投）。下面的选择规则与三个候选包一并作废，保留原文仅为记录当时的决策依据。最终处置见 §7。

三个包**全部预先产出**（推理约 10 分钟/个，不占提交额度），因为额度只有一个，而预先出包使 22:00 的决策不受训练进度制约。

出包由 `infer` 子命令一次完成：推理 → `scripts/check_submission.py`（`check=True`，校验不过即失败）→ `register()`。包落在 `outputs/search_20260923/<EXP_ID>/seed42/submission/`；登记表写入**工作树自己的** `results/rematch_submission_registry.csv`，不碰主仓库。

**上传前必须核对**：`selected_report.json` 里的被选轮次应等于总轮数（`last_epoch` 语义），否则包不是预定终点的模型。

### 1.4 执行结果（截至 2026-09-23 16:20）

基线 `RM_FT` 已核实 `selected_epoch: 8` = `schedule_epochs: 8`，即 `best_selector` 选中的就是最后一个 epoch —— **`last_epoch` 与 `best_selector` 在本例重合**，逐 epoch 对比是干净的 apples-to-apples。

| epoch | 基线 (α=0) micro | A1 (α=0.2) | Δ | A2 (α=0.4) | Δ |
|---:|---:|---:|---:|---:|---:|
| 2 | 69.2944 | 69.1129 | −0.18 | 68.2728 | −1.02 |
| 4 | 71.6599 | 71.2097 | −0.45 | 70.6048 | −1.06 |
| 6 | 72.7083 | 72.2782 | −0.43 | 71.6599 | −1.05 |
| 8 | **72.8696** | 72.5000 | −0.37 | 71.9288 | −0.94 |

macro 同向（ep8：基线 71.8054 / A1 71.4466 / A2 70.8878）。两点各 4 个检查点共 8 个，**全部为负**。

**三个量随 α 单调，A 轴判死**：

| ep8 | α=0 | α=0.2 | α=0.4 |
|---|---:|---:|---:|
| train_accuracy | 86.83% | 79.11% | 75.01% |
| train→val 落差 | 13.96pp | 6.61pp | 3.08pp |
| 相对锚点漂移 | 0.0585 | 0.0656 | 0.0689 |
| **val micro** | **72.87** | **72.50** | **71.93** |

过拟合压得越狠、val 越差，三点单调无例外 → **本任务这一档的瓶颈不是过拟合，把正则化调强是净损失。**

**此结论对 B 轴是反向的有利证据，不要读反**：特征锚定（`feature_distillation_weight=2.0`）本身就是一种正则化，B1（降到 0.5）与 B2（归零）都是**减弱约束**。A 轴既然证明「约束越松 val 越好」，B 轴的先验应**上调**。注意 A1/A2 只改了 mixup、锚点固定 2.0，因此它们证明的是「在锚点 2.0 的前提下加正则有害」。

#### B1 训练崩溃（rc=134）—— 结果不可用

```
TRAIN START search_b1_anchor05_lr1e5  14:49:09
TRAIN END   search_b1_anchor05_lr1e5  rc=134  15:03:35   ← SIGABRT
INFER END   search_b1_anchor05_lr1e5  rc=1    15:03:45
PACKAGE MISSING SEARCH_B1_ANCHOR05_LR1E5
```

`torch.AcceleratorError: CUDA error: unknown error`（`cudaErrorUnknown`），抛出点在 `Py_FinalizeEx`，`terminate()` → SIGABRT。

**判断：环境级故障，不是配置缺陷。** 依据：(a) 崩溃前最后一条 Progress（step 6000）完全健康 —— loss 0.949、grad_norm 33.7、`amp_scale 128` 稳定、无 NaN/inf；(b) `cudaErrorUnknown` 属运行时/驱动错误类，不是数值爆炸（那会表现为 inf/nan 被 grad scaler 跳过）；(c) 同期 `dmesg` 出现连续的 `WSL … Relay ERROR: UtilAcceptVsock: Waiting for abnormally long accept(11)`，WSL↔Windows 边界当时确有异常；(d) A1/A2 各跑满 8 轮无此问题。**未获证明**，判别性实验是重跑 B1 —— 若重跑正常即坐实环境说。

#### B1 重跑失败（未测到环境假设）

15:03 的崩溃在磁盘上留下 `outputs/search_20260923/SEARCH_B1_ANCHOR05_LR1E5/seed42`，15:03:40 启动的重跑 **18 秒即退出**：

```
=== B1 RERUN TRAIN END rc=1 2026-09-23T16:03:58+08:00 ===
FileExistsError: Run directory exists: .../SEARCH_B1_ANCHOR05_LR1E5/seed42. Use --overwrite or --resume.
```

**因此「环境级故障」这个判断至今仍未被检验** —— 重跑没有跑到任何一个梯度步就死了。

#### B2 训练崩溃（与 B1 同一签名）—— 包已产出但不可用

```
=== TRAIN END search_b2_anchor00_lr3e6 rc=134 2026-09-23T16:00:08+08:00 ===
```

与 B1 **完全相同的签名**：`trainer.py:1168 _gradient_norm` → `optim.py:25` → `torch.AcceleratorError: CUDA error: unknown error`，同样抛在 `Py_FinalizeEx` → `terminate()` → SIGABRT。崩溃前最后一条健康 Progress：15:57:03，epoch 5 / step 19200。两次崩溃配置不同（anchor 0.5 / lr 1e-5 @14 min vs anchor 0.0 / lr 3e-6 @56 min），共有的是机器与时间窗。

**同时记录反证**：`dmesg` 全程无 Xid、无 GPU reset、无 nvrm 报错，只有本会话一直存在的 `dxgkio_query_adapter_info: Ioctl failed: -22` 与 WSL Relay 噪声；单独的 GPU 健康探针（`_gpu_smoke.py`，反复执行崩溃现场那个算子并每轮做 233MB 分配/释放，跑满 90 s）结果 `SMOKE OK iters=48594` 退出 0。**结论仍是未定**：瞬态 WSL/驱动故障与「16 轮档共有的某物」无法区分。相关性可疑但样本只有 2：**16 轮档 2/2 崩溃，8 轮档 2/2 跑完**。

B2 被中断前已完成的两个评测点：

| epoch | 基线（8 轮档）micro | B2（16 轮档）micro | Δ |
|---:|---:|---:|---:|
| 2 | 69.2944 | 68.6022 | −0.69 |
| 4 | 71.6599 | 71.4113 | −0.25 |

macro 同向（ep2 67.5788、ep4 70.5030）。**缺口在收窄**，但**不能读成「B2 快追上了」**：B2 是 16 轮余弦，ep4/16 处的学习率远高于基线 ep4/8 处，同一 epoch 序号在两条 schedule 上不是同一训练进度。这只是「未被证伪」。

**B2 的包无效。** `infer` 从 `best.pt` 推理，而 B2 的 `best.binding.json` 写明 `"epoch": 4`；训练在第 5 轮被中断，`selected_report.json` **从未写出**（A1/A2 都有）。该包通过平台全部 9 项结构校验（单 checkpoint、确定性推理，**合规**），但它是 **16 轮计划的第 4 轮模型**，违反本批预登记的「被选轮次应等于总轮数」。**22:00 决策不得使用。** `_package_b2.sh` 当时只断言了 submission 目录存在，故误报 `PACKAGE` —— 已改为必须断言 `selected_report.json` 存在。

#### 三处对早期估计的更正

1. **速率**：实测 **0.117–0.125 s/step**，非我先前用的 0.169。据此 B1 应在约 17:05 训完、B2 约 19:30 结束、`BATCH1 DONE` 约 19:35 —— 不是 §1.2 写的「约 11 小时串行」。
2. ~~**B2 赶得上 22:00**~~ **已被上一条崩溃作废**。原先写「赶不上」，后依实测速率改为「赶得上」，并为此另起 `_package_b2.sh` 补出包；B2 于 16:00 崩溃，这个补丁产出的包是第 4 轮模型，不可用。两次估计都错了，且错在同一个地方 —— 拿健康的运行时长外推一个已经出过故障的批次。
3. **`--overwrite` / `--resume` 在 CLI 上不存在**，`FileExistsError` 的提示语（「Use --overwrite or --resume」）会误导人去找一个没有的开关。`train` 子命令只注册了 `--config`（`cli/rematch.py:136-143`），第 108 行以 `train(config)` 单参数调用；`trainer.py:116/118` 的 `resume` / `overwrite` 是 Python 形参，配置里也没有对应键。当崩溃残留的 `run_dir` 挡路时（`trainer.py:144`），**唯一出路是删除该目录**（`--overwrite` 内部做的就是 `shutil.rmtree(run_dir)`）或写 Python 驱动直调 `train(cfg, overwrite=True)`。

#### 看门程序曾漏报这次崩溃（已修）

`_watch_batch1.sh` v1 在 15:11 报 OK，而 B1 已于 15:03 崩溃。根因：v1 只检查「当前活跃的 run」，B1 死后编排立刻转去 B2，活跃 run 是健康的 B2 —— B1 的日志与 `_driver.log` 里的 `rc=134` / `PACKAGE MISSING` **从未被读过**。

v2 改为扫描 `_driver.log` 与全部 run 日志的异常行（非零 rc、PACKAGE MISSING、CUDA error、OOM、NaN、traceback），按「文件:行号」去重只报一次，并永久记入 `_INCIDENTS.log`（不因后续轮次健康而被覆盖）。上线首轮即复现并报出了本次漏掉的崩溃，共 7 条。

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

---

## 7. 本批收尾（2026-09-23 用户变更算力分工）

用户当日指令：**本机只负责思考策略，长训练全部由队友的 NPU 执行**（队友实测 ≈34.3 分钟/点；本机 8 轮 ≈80 分钟、16 轮 ≈2.5 小时）。据此本批收尾如下。

### 7.1 四个点的最终处置

| # | 实验 ID | 本地实测 | 处置 |
|---:|---|---|---|
| 1 | `SEARCH_A1_MIXUP_A02` | micro 72.5000 / macro 71.4466（−0.37pp） | **已完成**，包有效（`selected_epoch 8 = schedule_epochs 8`）。A 轴本地判死（α 单调、8/8 检查点全负） |
| 2 | `SEARCH_A2_MIXUP_A04` | micro 71.9288 / macro 70.8878（−0.94pp） | **已完成**，包有效 |
| 3 | `SEARCH_B1_ANCHOR05_LR1E5` | **无** —— 15:03 崩溃于第 14 分钟，重跑 16:03:58 被残留 run_dir 挡死（`FileExistsError`，18 秒） | **不在本机重跑**，并入交给 NPU 的点位 |
| 4 | `SEARCH_B2_ANCHOR00_LR3E6` | **无有效值** —— 16:00 崩溃于第 5 轮；中断前 ep2 68.6022 / ep4 71.4113（缺口 −0.69 → −0.25，但受 schedule 混淆，见 §1.4） | 16:35:13 启动的重跑**于 17:15:51 由用户裁定主动终止**（**非崩溃**；停在 epoch 2 / global_step 7200，`_driver.log` 有 `ABORTED BY USER DECISION` 记录）。本机不再出包 |

B1/B2 两次崩溃签名相同（`trainer.py:1168` → `optim.py:25` → `cudaErrorUnknown`），机制**至今未定**；「16 轮档 2/2 崩、8 轮档 2/2 跑完」这一相关性样本只有 2，不作结论。**把这两个点交给 NPU 同时也是对「是否本机环境问题」的一次判别**：若 NPU 上同样配置跑通，环境说成立。

### 7.2 与队友 V4 搜索的关系（已直接读 V4 分支核对，不再等答复）

核对方法：读 `origin/codex/rematch750_search_v4` 的 **82 份 config**（A01–H24），逐点比对配方，不依赖任何转述。

**结论：本批 4 个点与 V4 全部 82 个点没有一个相同，因此不能替代队友的任何一次运行。** 队友仍须按 V4 预注册跑完 A01/A02/D01/D02；本批结果对他是**先验**，不是**结果**。

| 我的点 | 最接近的 V4 点 | 除共同项外的差异 |
|---|---|---|
| A1 `α=0.2` | A04 `soft_CE_mixup_a0.2` | 我是 **CE**，V4 是 `cross_entropy`（同族）；但 `mixup_probability` 我 **1.0** / V4 **0.5**；轮数 8 / 16；batch 32 / 1024；`head_lr` 1e-4 / 4e-4；`backbone_lr` 3e-6 / 1.2e-5；`selection_policy` last_epoch / best_selector |
| A2 `α=0.4` | A05 `soft_CE_mixup_a0.4` | 同上 |
| B1 锚点 0.5 | D02 `feature_anchor_0.5` | 我是 **CE**，V4 是 **GCE**(q=0.5)；且我同时改了 `backbone_lr`→1e-5（与 V4 base 的 1.2e-5 不同）→ **双变量** |
| B2 锚点 0.0 | D01 `feature_anchor_0` | 同上；B2 的 `backbone_lr` 为 3e-6 |

V4 全部 82 点的取值域是统一的：`epochs ∈ {16, 20}`、`batch_size ∈ {1024, 512, 256}`、`head_lr 4e-4`、`backbone_lr 1.2e-5`（仅 H 轴为 0.0）、`loss ∈ {gce, cross_entropy, sce}`，父模型统一为 `RM_LP best.pt`。**本批 4 点的取值无一落入其中**，故不存在替代或合并关系。

**可迁移的只有方向性先验，且强度有限**：

1. **「在锚点 2.0 的前提下再加正则有害」** —— A1/A2 在 8 个检查点上**全部低于配对基线**，且按 α 单调（baseline 72.8696 → α0.2 72.5000 → α0.4 71.9288，micro）。V4 的 A 轴正是「锚点 2.0 + GCE + mixup」，这**压低** A 轴的先验。
2. 同一结果的另一读法：**约束越松、val 越好** → V4 的 **D 轴**（下调锚点）与 **E 轴**（长尾重加权）先验应**上调**。
3. **局限（必须一起读）**：我的 `mixup_probability=1.0`（每批必 mixup）是比 V4 的 0.5 强一倍的正则，**负结果可能正是这个强度造成的**。因此只能外推**方向**（α 越小越好），**不能**外推绝对值、不能断言「A04/A05 也会掉」。

**本批处置（按用户 2026-09-23 裁定）**：作为**本方案自己的策略结果**登记与推送，不冒充 V4 的点、不并入其编号。结果为负，本方案关闭，不派生参数扫描。**不使用平台名额**（A1/A2 均低于基线，按用户既有规则「本地得分提升不显著就不浪费名额」）；若用户另有裁定，以裁定为准。

### 7.3 本机保留的能力

工具与脚本保留可复用：`_package_run.sh`（严格出包，断言 `selected_report.json` 且 `selected_epoch == schedule_epochs`）、`_rerun_point.sh`（崩溃残留改名保全而非删除）、`_watch_rerun.sh`（盯单个 run、历史异常基线化）、`_watch_batch1.sh`、`_gpu_smoke.py`。这些与机器无关，NPU 侧出包若走 `aegis_clip` 同一 CLI，判据可照搬。
