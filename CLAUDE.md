# CLAUDE.md

本文件为 Codex、Claude Code 等 agent 提供在本仓库工作的统一指引。

## 项目概述

面向噪声标签数据的细粒度图像分类。骨干固定为 CLIP ViT-B/32，权重限用 OpenAI 官方版本。这是比赛参赛项目，规则要求**单模型**交付：禁止任何形式的集成，禁止测试时训练或自适应。

**本文件只写阶段之间不变的东西。** 类别数、数据规模、当前最优成绩、下一步做什么，一律不写在这里 —— 见下方「当前阶段」。

## 当前阶段 ← 唯一随阶段更新的区块

**阶段切换时只改这一节。**

- 当前阶段、权威状态与下一步：[docs/current_execution_plan.md](docs/current_execution_plan.md)
- 当前计算资源：**后续仅用本机 CPU/CUDA，NPU 已停用**；旧 NPU 交接、服务器恢复等待与用卡指令均已撤回，资源边界以当前执行入口为准
- 开始搜索前先读该入口指向的**当前搜索纪律与误差预算**；旧执行记录、已实现候选和历史待跑配置都不构成自动启动任务的依据
- 本阶段详细的执行记录与固定配方：见该文件顶部「当前执行入口」指向的当轮文档

代码与超参数经验可以跨阶段迁移；**数据、checkpoint、特征缓存、伪标签、拟合出的 prior 一律不可跨阶段复用**。

## 实验搜索与投入原则

**优化目标是单位时间找到足以缩小平台差距的提升。** 可复现性、血缘和提交校验是必要条件；实验数量、实现数量、审计次数和普通本地 accuracy 的微涨都不能当作提分进展。

- **先误差、后方法**：从当前阶段的错误出发，区分标签噪声、细粒度混淆、长尾、来源漂移和灾难性遗忘；记录群体规模、重叠、证据、未知项及可恢复收益，再选择训练范式。低置信度不等于错标，训练样本数量不等于可修复的预测错误数量。
- **区分有限探索与完整训练**：有限探索允许机制和收益未知，但须有可观察问题、明确可证伪假设、成本上限、固定评估和能改变下一步的停止条件；不要求干预前已经证明能净修正数千张平台错误。完整训练须由有限探索中的全量修正/退化及预先指定问题的改善支持，并说明合理规模、预计收益风险和平台迁移假设；未知量明确保留。训练损失下降、置信度上升或小切片微涨不构成完整训练依据。
- **先廉价证伪、后完整训练**：优先复用当前阶段已审计缓存、逐样本预测与日志；诊断和短探针预先限定点数、时长与停止条件。证据不足就交付未知项，不自动补做完整 OOF、多教师、多 seed、参数网格或十小时训练。
- **普通本地 accuracy 不能替代平台收益**：本地分用于配对诊断、复现和止损；优先寻找与主要误差相关的分组指标和独立来源留出证据，外推明确写出假设。平台提升只能依据平台实测声明。
- **取消疑似错误监督的评价**：全量原标签macro/micro及逐张修正/退化必须保留，同时用候选结果产生前冻结的内容代理分组解释变化。全量含噪accuracy下降不自动否决：若目标图像候选部分净改善而下降集中在疑似无目标部分，可保留为待验证候选；代理组不是干净真值，仅降低疑似无目标图置信度不证明识别收益。不能将“缺证据”自动转换为撤销原标签监督，也不能把所有非自然照片视作无效目标。
- **冻结项不自动复活**：当前冻结清单及在途例外只在当前执行入口维护。已启动的固定实验按原协议收尾，不因结果微涨派生强度、变体、组合或续训搜索；未启动的旧计划必须重新满足当前机制与成本门禁。
- **方法变化可大，结论必须可归因**：机制证据支持换训练范式时可以提出整体方案，不把搜索永远限制为现役模型的单变量邻域。保留必要对照并明示改变的因素，不为了形式上的严谨无限追加消融；比赛硬约束始终有效。

## 代码布局

```
common/                      稳定骨架：数据集、配置/随机种子/日志、提交生成、PEFT、噪声鲁棒工具
experiments/                 早期「一个方法一个子目录」的实验骨架（现多为薄封装）
configs/                     实验 YAML
scripts/                     数据准备与提交校验，实验无关
reproducibility/aegis_f1/    当前主线实验框架（Aegis），绝大多数近期工作在这里
results/                     每轮实验的结果记录（json/csv/md）与提交登记表
docs/                        方案、预注册、执行记录
outputs/                     产物
```

**注意**：`reproducibility/aegis_f1/` 是本仓库现在真正的前沿，`common/` + `experiments/` 是上一阶段的骨架。读代码时先确认自己看的是哪一层。

Aegis 自带独立的 `pyproject.toml`、测试（`reproducibility/aegis_f1/tests/`）与 CLI（`reproducibility/aegis_f1/aegis_clip/cli/`）。所有入口都需要 `PYTHONPATH=reproducibility/aegis_f1`。

## 常用命令

全部在仓库根目录执行。

### 当前主线（Aegis）

```bash
# CLI 总入口；子命令：prepare / verify / cache / train / infer / run / record / prepare-full
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.rematch --help

# 完整命令序列以当轮执行文档为准：见 docs/current_execution_plan.md 顶部的「可重放命令」
```

### 提交校验

```bash
# 注意：--num-classes 或 --class-mapping 二者必给其一，否则脚本 sys.exit(2)
python3 scripts/check_submission.py \
  --test_dir <测试集目录> \
  --class-mapping <class_to_idx.json> \
  --csv <pred_results.csv> \
  --zip <submission.zip>
```

校验 9 项：文件名、行数、字段数、图片名存在、重复、覆盖、标签格式、标签范围、zip 内容。

### Aegis 测试

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -m pytest reproducibility/aegis_f1/tests -q
```

**已知失败**：`test_scope_protocol.py` 中依赖上一阶段冻结资产的用例会抛 `ScopePreflightError`（如 `train/test image root is missing`、资产摘要不匹配）。那些资产在上一阶段结束时已从磁盘移除，且部分从未入库（`artifacts/` 在 `reproducibility/aegis_f1/.gitignore` 内），因此**在全新 clone 上必然失败且无法修复**。

判定「是否通过」时**不要数失败条数** —— 条数随本机残留资产的多寡而变。只要求：失败项全部落在 `test_scope_protocol.py` 且报错为 `ScopePreflightError`，其余用例全绿。2026-09-22 本机实测 **2 failed / 632 passed**（两个失败均在该文件内；CLAUDE.md 早前记的「1 failed / 4 passed」是只跑了该文件时的旧测量，口径不同）。

### 早期骨架（历史，仅在需要读旧代码时使用）

```bash
python3 scripts/check_data.py --config configs/<name>.yaml
python3 scripts/split_data.py --config configs/<name>.yaml
python3 -m experiments.baseline.train --config configs/<name>.yaml
python3 -m common.submission --raw <pred_raw.csv> --out_dir <目录>
```

这些命令依赖已删除的上一阶段数据资产，通常无法直接重放。

## 比赛硬约束

完整规则见 [COMPETITION_RULES_AGENT.md](COMPETITION_RULES_AGENT.md)。要点：

- 骨干固定 **CLIP ViT-B/32**，权重限 **OpenAI 官方**（代码内硬校验：`common/clip_utils.py` 的 `ALLOWED_MODEL_NAME` / `ALLOWED_PRETRAINED_SOURCE`）
- **禁止跨阶段复用**：数据、checkpoint、特征缓存、伪标签、原型、拟合的 prior 均不可带入新阶段
  - ⚠️ **上一阶段的部分拟合产物仍在磁盘上且被 git 跟踪**（如 `outputs/phase/phase3/oof/` 的 `sample_quality.csv`、`fold_assignments.csv`、`outputs/phase4/purification/*`、`outputs/phase4/global_rejected_paths.txt`、`outputs/data/d3_strict/`）。它们按 500 类 / `train_dedup/` 路径键，**新阶段一律不得读取**。危险点在于路径存在，误引用不会干脆报 `FileNotFoundError`，而是静默接错数据或跑到一半才 `KeyError`，失败会伪装成「方法无效」。可迁移的只有代码与公式。
- **测试集只读**：禁止测试图入训、无监督自适应、TTT/梯度更新、用测试集预测分布调参
- **禁止集成**：多模型、多 checkpoint 投票、多 seed 平均、骨干融合、多头融合、logits/概率加权组合均不允许。最终结果 = 单个训练好的 checkpoint + 单份确定性推理脚本 + 一份 `pred_results.csv`
- 多尺度 + Flip TTA 已裁定合规（前提：单 checkpoint + 单确定性流程）
- 提交格式：`文件名, 0001`（逗号 + 空格 + 4 位补零），ZIP 内只含 `pred_results.csv` 且与外部字节一致
- 平台**每日提交次数有上限**，具体按当轮文档

历史局部实验采用 **+0.30pp** 迭代投资回报门、**−2pp** 本地止损、**+0.20pp** 筛选门；已在途实验按原预注册收尾，不事后改判据。**这些门槛不是新高成本训练的启动许可，也不证明平台晋级。** 新主线先满足上文机制、收益与成本门禁；未达原门槛的轮次关闭，不自动派生参数扫描。

## 关键设计细节

**CLIP dtype**：`conv1` 在 CUDA 上以 fp16 加载而其余参数为 fp32。`build_model()` 用 `.float()` 把视觉编码器统一为 float32，否则前向报 "Input type and weight type should be the same"。

**类别映射**：按目录名字典序排序 → 索引 `0..N-1`。`class_to_idx.json` / `idx_to_class.json` 由划分脚本生成，训练与推理共用。**永远不要硬编码类别索引或类别数。**

**提交格式**：见上文命令一节。

**样本权重接口**：`TrainImageDataset` 返回 `(image, label, image_path)`，并提供 `get_sample_weights()`。这是有意的 —— 噪声鲁棒方法可借 `image_path` 赋逐样本权重，无需改数据集代码。

**坏图处理**：`_safe_load_image()` 失败返回 `None`，调用方以零张量兜底，训练不中断。注意兜底张量尺寸是写死的，若改动输入分辨率需同步检查。

## 配置系统

YAML 由 `common/utils.py:load_config()` 加载。

主 schema 顶层键：`experiment` / `data` / `model` / `train` / `eval` / `output`（`common/config_schema.py` 对未知顶层键**fail-closed**）。

另有 `train` / `loss` / `mixup` / `sample_weighting` / `peft` / `head_ema` / `posthoc` / `cache` / `soft_targets` / `hyper_search` 等可选段。

**注意**：`configs/` 下同时存在**多套互不兼容的 schema**。部分文件（早期 `prelim75_v*.yaml`、`a2_lora_*.yaml`、`oof_cache.yaml`）使用扁平的 plan schema，路径带 `../` 前缀、假定不同的 cwd，且会被主 schema 校验拒绝。使用前先确认属于哪一套。

## 协作约定

本项目多人协作，以下约定用于避免重复劳动和丢失经验。**每个实验段都必须遵守。**

**计算资源、工作目录与环境：**

- 使用当前执行入口指定的本机资源，诊断优先 CPU/已有缓存；训练和推理前只读核对本机 GPU 与已有任务，不抢占，不设用卡认领流程。不要按历史文档连接、探测、排队或恢复已停用的远端资源。
- **每个队员的每个方案都开新分支，并使用独立工作目录**，分支命名为 `成员名/方案名`；推荐用 `git worktree` 从最新 `origin/main` 创建。不要在他人正在使用的目录切换分支或修改代码。
- **输出、日志、checkpoint 使用各自独立目录**，按成员、方案、运行标识区分，禁止覆盖他人产物。原始数据共享只读；共享 Python 环境的依赖变更先协调，特殊依赖使用独立环境。

**开始新实验段之前：**

1. **每次会话/实验段一开始就先同步远端**：`git fetch origin`，确认当前分支、工作区状态及其与 `origin/main` 的关系。新方案从最新 `origin/main` 创建独立分支和 worktree；已有方案按下文 Git 细则同步。不要凭上次会话的记忆判断当前状态 —— 协作者随时可能已经推了新结果。
2. 检查全部本地与远端分支、以及近期 main 历史，看是否已有**重叠的实验名、算法、配置、结果报告或在途工作**。若有相似方向，先比对 / 协调 / 关闭，再决定是否动手，不要重复实现。

**实验段结束之后：**

3. **立刻提交并推送**已验证的改动到对应方案的远端分支；完成校验后，在用于集成的 main 工作目录中同步最新 `origin/main`，合并方案分支、处理冲突并重新校验，随后推送到 `origin/main`。报告：实验标识、确切命令/配置、结果指标、改动文件、分支名、commit SHA、push 与合并状态。**在该检查点暂停**，让协作者能复用结论或做淘汰比较。
4. 报告必须可复现，并**区分「实现改动」与「实测结果」**。没有验证输出前不得声称该段已完成。
5. 实际训练段暂停前必须交付到**可提交产物**：生成预测 CSV/ZIP、跑完提交校验、记录路径与校验结果。只有代码或只有单测不能算训练段完成。**纯文档或误差诊断段**交付已核对的文档/诊断报告，并记录现役可提交包路径及既有校验来源；不为凑新包启动训练，不把诊断或实现准备称作新候选完成。

**Git 操作细则：**

- 方案开发必须使用独立分支和工作目录；`main` 用于集成已验证方案及维护公共文档，不在其中直接开展新方案实验。
- 未推送的方案分支可 rebase 到最新 `origin/main`；已推送的方案分支通过 merge 同步 `origin/main`，避免改写已共享历史。推送方案时使用 `git push -u origin <成员名/方案名>`，不能用 `git push origin main` 代替。
- 以下同步流程用于 **main 集成目录**。混合工作区两种模式**不可混用**，下指令时必须说明用的是哪种；方案合并及校验须在 pull 之后、push 之前完成：
  - 手动：`git stash push -u -m "..."` → `git pull --rebase origin main` → `git push origin main` → `git stash pop`
  - 自动：`git pull --rebase --autostash origin main`（Git 自动恢复临时 stash，**不要再执行一次 `git stash pop`**）
- **禁止 `git push --force`** 解决 non-fast-forward
- 已验证的推送不得推迟到后续实验段

**冲突处理：能整合就整合，语义互斥才以远端为准**

默认动作是**手工整合**，不是二选一。git 报冲突只说明两侧改到了相邻行，不代表内容矛盾 —— 打开冲突标记，写出同时包含两侧信息的那一版，再 `git add`。

- **可整合且应当整合**：双方改的是同一段里的不同事实（一方补「完成了什么 / 指标」，另一方改「数据规模 / 链接」）→ 全部保留。
- **语义互斥 → 以远端为准**：同一字段给出两个互斥声明的，例如「当前平台最佳成绩」这种单值字段、一方删文件另一方改该文件、同一函数的两套不兼容实现。远端是已推送、已对他人可见的一侧，让位成本更低。
- **不要用 `-X ours` / `-X theirs`**：那是整文件的静默取舍，会无声丢掉另一侧的改动，比手工取舍更危险 —— 你看不到丢了什么。`git checkout --ours/--theirs` 同理，只在确认整个文件都该弃用时才用。
- rebase 中方向容易搞反：`git checkout --ours <file>` 取的是**远端**（rebase 时 HEAD 是 origin/main），`--theirs` 取的是你正在应用的本地提交。
- **README 保持最简通用说明**：不写具体赛题、赛事名称、数据规模、比赛规则、实验标识、成绩或相关文档链接。每次交付的状态与指标更新到 `docs/current_execution_plan.md` 及对应执行记录，不再写回 README。
- **整合完成后等于做了一次新编辑，必须重新核对**：数字有没有自相矛盾（状态段写着 RM-FT 71.83%、下面表格还停在 RM-LP 62.64% 就是错的）、链接是否仍有效、是否与执行记录一致。

## 历史

各阶段的完整过程记录见 [docs/history/README.md](docs/history/README.md) 与 `results/`；当前执行入口只维护现行状态与下一步。阶段经验提炼见 [docs/lessons_learned.md](docs/lessons_learned.md)。

上一阶段的权重与二进制缓存已删除，不可重放。
