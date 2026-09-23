# 交接 brief（2026-09-23）

**读者：Codex。** 你已在本项目里（V3 方案 `codex/rematch750_v3_tradeoff` 出自你），所以这份不是从零介绍，而是**接力用**：哪些已完成、哪些在谁手上、今天必须先做什么、以及**哪些坑我踩过不要重踩**。

**它不是权威状态源。** 权威依次是：[README 开头状态段](../README.md)（协作约定的状态落点）→ [docs/current_execution_plan.md](current_execution_plan.md)（入口 + 在途登记）→ [AGENTS.md](../AGENTS.md) / [CLAUDE.md](../CLAUDE.md)（工作与 Git 约定）。**本文件只写那些不在 repo 里、或散落多处容易漏的东西。**

编写者：clairvoyanttt。下面每条都标了**来源**（我实测 / 他人文档报告），请按此区分可信度。

---

## 1. 60 秒现状

750 类复赛，145K 训练 / 37,444 测试，骨干固定 OpenAI CLIP ViT-B/32，单模型交付。

- **平台最高分 60.9657%（RM-FT，22,828 / 37,444）**，第二 60.8856%（RM-LT），只差 0.0801pp / 30 张。**09-23 尚未上传任何包。**
- 本地最强的是 **NPU batch32 八轮 FT（macro 71.9047%）**，比 GPU RM-FT 的 71.8362% 高 0.0685pp —— **未过 +0.20pp 晋级门，无提交包、未上传**。
- **核心战略结论：full-FT 这一族已贴自身天花板，没有任何在产候选有理由超过 60.97%。** 剩下两三周要买的不是分数，是**信息**。
- 因此 09-22 预注册了唯一还在等的测量：**用今天的槽位发 RM-LP（冻结特征），按「迁移率」而非绝对分判定**。这是今天的第一件事（§5）。
- 计算已迁到**昇腾 910B2 NPU**（远端 `vllm-lqh-86`），NPU batch32 调优后 0.0796 秒/步，比本机 GPU 快 1.59 倍。V3（精度/耗时权衡）**仅方案、暂停执行**。

---

## 2. 硬约束（先读，再动手）

**比赛规则**（全文 [COMPETITION_RULES_AGENT.md](../COMPETITION_RULES_AGENT.md)）：

- 骨干固定 CLIP ViT-B/32，权重限 OpenAI 官方（代码内硬校验 `common/clip_utils.py`）
- **禁止集成**：多模型 / 多 checkpoint / 多 seed / 骨干融合 / logits 或概率加权，全不允许
- **测试集只读**：禁止测试图入训、TTT、无监督自适应、用测试分布调参
- **禁止跨阶段复用**数据 / checkpoint / 特征缓存 / 伪标签 / 拟合 prior
- 多尺度 + Flip TTA 已裁定合规（前提：单 checkpoint + 单确定性流程）
- 提交格式 `文件名, 0001`；ZIP 内只含 `pred_results.csv` 且内外字节一致
- **平台每日提交次数有上限**（按当轮文档）

**内部门槛**：+0.30pp 迭代投资回报 / −2pp 本地止损 / +0.20pp 筛选晋级。

**团队约定 —— 2026-09-22 刚改过，与你 V3 时一致，但和更早的会话不同**（[AGENTS.md](../AGENTS.md) / [CLAUDE.md § 协作约定](../CLAUDE.md)）：

- **新方案必须开 `成员名/方案名` 分支 + 独立 worktree**（推荐 `git worktree`）。`main` 只用于集成已验证方案与维护公共文档，**不再在 main 上直接做方案实验**。你已按此做了 V3，保持。
- **不许占卡**：只选当时空闲的 NPU，不设认领流程。
- 输出/日志/checkpoint 各用独立目录，禁止覆盖他人产物；原始数据共享只读。
- 已推送的分支用 merge 同步 `origin/main`，未推送的可 rebase；`git push -u origin <成员名/方案名>`；**禁止 `git push --force`**。
- main 集成目录的两种 stash 模式**不可混用**，下令时必须说明用哪种；用 `--autostash` 后**不要再 `git stash pop`**。

---

## 3. 指标总表

**平台（唯一权威分数，来源：平台回填）**

| 候选 | 平台 | 本地 macro / micro | 迁移率（平台/本地 micro） |
|---|---|---|---|
| **RM-FT**（GPU） | **60.9657%** | 71.8054 / 72.8696 | 0.8366 |
| RM-LT（GPU） | 60.8856% | 71.8655 / 72.6949 | 0.8375 |

**本地（独立验证，来源：各自执行记录）**

| 候选 | 后端 | 选中 epoch | macro | micro | 状态 |
|---|---|---|---|---|---|
| RM_FT_NPU_TUNED（b32） | NPU | 8 | **71.9047%** | 72.9839% | 效率配置，未晋级，无包 |
| RM-LT | GPU | 8 | 71.8655% | 72.6949% | 采样对照 |
| RM-FT | GPU | 8 | 71.8362% | 72.9032% | **正式 baseline** |
| RM_FT_NPU_THROUGHPUT（b1024） | NPU | 8 | 67.3704% | 68.4207% | 大 batch 掉精度，关闭 |
| RM_LORA（NPU） | NPU | 6 | 64.1064% | 65.1478% | 关闭 |
| **RM_LORA（本机 GPU）** | GPU | 8 | **64.1239%** | **65.1747%** | 我跑的，**未入库**（见 §8） |
| RM-LP | GPU | 20 | 62.6319% | 63.6358% | 冻结基线，**今天的槽位** |

> **两处 RM_LORA 差 0.0175pp macro、0.0269pp micro —— 跨后端（NPU vs GPU）+ 跨硬件独立复现，几乎完全一致。** 这条我可以背书：本机那次是我亲自跑的。
>
> ⚠️ **注意选模口径**：NPU LoRA 按 macro 选 epoch 6，本机 LoRA 选 epoch 8。同配方不同后端选到了不同 epoch —— 单 seed 选模本身就在噪声量级内游走，别对 <1pp 的排序做过度解读（[lessons §一.2](lessons_learned.md) 初赛实测单种子变异 0.90pp）。

---

## 4. 已经确定的事实 —— 别再重测

以下四条都已闭环并有文档。**重复做它们不会有新信息，只会烧掉 GPU/NPU 时间。**

1. **本地对平台无信息，且不存在本地跨来源代理。** val 是训练池的**忠实 iid 抽样**（连「长边 ≤500 且 portrait」这档测试集长相都按同比例存在：train 8.50% / val 8.58%）。11.90pp 本地/平台差**对 val 上任何指标结构性不可见**。所以「留一来源 / 域泛化代理」这条路在复赛数据上**不成立**，已正式关闭。→ [推理侧探针](rematch750_inference_side_probe_20260922.md)
2. **推理侧不是答案。** Flip TTA 全融合规则扫描本地只值 +0.155~+0.242pp，收益全在 head/medium、尾部 ≈0；把验证集压到测试集几何只掉 0.17~0.31pp —— **11.90pp 不是输入几何造成的**，改预处理关不掉。
3. **FT 相对 LP 的 +9.23pp 池内增益是均匀的**，且**更像真实表征改善而非池内域适配**：`drift ⊥ train_samples`（+0.033）而 `drift vs LP_recall` 强负（−0.603），即 FT 移动最多、提升最多的正是**冻结 CLIP 本来就分不开的类**。→ [特征漂移分析](rematch750_feature_drift_20260922.md)
   - **工具已入库**：`PYTHONPATH=reproducibility/aegis_f1 .venv/bin/python -m aegis_clip.cli.analyze_feature_drift`，带 val-micro 锚点，**可复用于任何 checkpoint**，不必重写。
4. **LoRA 低秩档位改变不了局面**：+1.54pp 本地，被 FT 主导 7.7pp，且已关闭。（注意其 config 头部写明：**它不是 FT 的单变量对照** —— LoRA 参数与 `visual` 共用 `backbone_lr`，故 LR 用了 2.0e-05 而非 FT 的 3.0e-06，warmup 1 vs 0。任何"低秩约束本身如何"的归因都不成立。）

---

## 5. 今天（09-23）的第一件事：花掉提交槽位

**动作：上传 RM-LP 包，回填平台分，按迁移率判据读。**

预注册与三档判定：→ [LP–FT 迁移对照预注册](rematch750_lp_ft_transfer_prereg_20260922.md)（**判据本身没变，别改**）。摘要：

- 判据：**RM-LP 平台分 > 53.30%** ⇒ 冻结特征比微调更抗来源漂移（迁移率超过 FT/LT 两臂 0.8366 / 0.8375）。
- 三档：**≥60%** = 真实提分，可直接留作新 baseline；**53.3–60%** = 冻结迁移更好，考虑部分微调；**<53.3%** = FT 优势按比例迁移，FT 家族仍对。
- **绝对分不判定策略，迁移率才判定。** 只有「≥60%」那档是分数直接定策略 —— 因为没有任何在产候选有理由超过 60.97%，LP 拿 55% 比 FT 拿 60% 信息量更大。
- **噪声**：单种子变异 0.90pp 量级，<1pp 的差异不要过度解读。
- **预期读数**：按 §4.3 的漂移证据，FT 的优势**更可能迁移**，故 **LP 更可能落在低档（<53.30%）**。若真如此，那是**符合预期**，不是实验失败。

**包的位置与哈希**（我 09-22 本机重建并跑过 9/9 校验）：

```
outputs/rematch750/RM_LP/seed42/submission/submission.zip
  zip sha256 ae654d8f8e1d828c813b45cfbad03debd2a4537109fb452664dbe420fc0d671a
  csv sha256 5d564b6d…      （37,444 行）
```

⚠️ **上传前必须先解决登记表账目问题（未决，见 §9.1）**：`results/rematch_submission_registry.csv` 里 RM_LP 行的 `zip_sha256` 是 `db30ff7a…`，与本机重建的 `ae654d8f…` 不同 —— 那一行描述的是**另一台机器**（`/home/lux1/noise`）上、现已删除的包。**今天的分数不能直接记到那一行**，否则谱系是假的。

**另外**：`RM-LP` 的 `pred_results.csv` 是 1,647,536 字节 / 37,444 行；上传前用 `python3 scripts/check_submission.py` 复核一遍（9 项）。

---

## 6. 待办队列

| 优先级 | 事项 | 状态 / 边界 |
|---|---|---|
| **P0** | 09-23 上传 RM-LP 探针 + 回填 + 按迁移率读 | §5。前置：解决登记表账目 |
| **P1** | **V3（精度/耗时权衡）恢复执行** | **仅方案、暂停中。用户明确要求暂不运行、不占 NPU0。没有用户新的明确指令不得自行恢复**（这是 V3 文档里你自己写的，继续保持）。恢复前需先做审计脚本适配（`scripts/audit_rematch750_v2.py` 写死 8 轮与 2/4/6/8 验证点，不能直接用于 16 轮配置）；配置里的 `npu:UNASSIGNED` 是占位符不是可用设备标识，必须显式绑定非 0 号空闲卡及其 CPU 拓扑。 |
| **P2** | 登记表账目整理（§9.1） | 需要人决定：改行 = 谱系变更，或开新 candidate ID |
| **P3** | 本机 RM_LORA 结果入库（§8） | 纯记录，无计算 |
| **—** | RM-FULL（全 148,695 张含当前闲置 val 10%） | **已取消，且文档写明不自动恢复**。我先前提出过一个新论据（既然 val 已无法裁决平台，留 val 做选模的价值下降），但**用户当时取消了，不要自行启动**。 |

---

## 7. 运行手册（环境）

**两套环境，别混。**

**A. 本机 GPU（RTX 4060 Laptop 8 GB，WSL）**

```bash
# 所有训练/推理都必须走 WSL venv（Windows 侧 python3 是另一套，torch 版本不同）
wsl.exe -e bash -lc 'cd /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning && \
  PYTHONPATH=reproducibility/aegis_f1 .venv/bin/python -m aegis_clip.cli.rematch --help'
```
WSL venv：python 3.12.13 / torch 2.13.0+cu130。子命令：`prepare / verify / cache / train / infer / run / record / prepare-full`。

**B. NPU（昇腾 910B2，ARM64，远端 `vllm-lqh-86`）**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=<非0号空闲卡>
export OMP_NUM_THREADS=8
export PYTHONPATH=reproducibility/aegis_f1
/workspace/noise-npu-venv/bin/python ...        # 独立环境，勿装 CUDA pyproject
# 依赖：reproducibility/aegis_f1/requirements-npu.txt
# 硬件自检：AEGIS_TEST_NPU=1 ... -m pytest reproducibility/aegis_f1/tests/test_npu_hardware.py -q
```
远端根 `/workspace/noise`（对应本机 `/home/lux1/noise`）。CANN 9.0.0 / Python 3.11.6 / torch 2.7.1 + torch_npu 2.7.1.post4。详见 [迁移记录](rematch750_npu_migration_20260922.md) 与 [调优记录](rematch750_npu_tuning_20260922.md)。

**测试**（CLAUDE.md 有完整口径）：
```bash
PYTHONPATH=reproducibility/aegis_f1 .venv/bin/python -m pytest reproducibility/aegis_f1/tests -q
```
**已知失败**：`test_scope_protocol.py` 中依赖上一阶段冻结资产的用例会抛 `ScopePreflightError`（那些资产已从磁盘移除、部分从未入库）。**判定时不要数失败条数** —— 只要求失败全落在该文件且均为 `ScopePreflightError`，其余全绿。2026-09-22 本机实测 2 failed / 632 passed。

---

## 8. 坑清单（我踩过的）

1. **`git -c core.fileMode=false`**：从 Windows 侧看有 ~170 个"已修改"文件，**全是 mode-only 假脏**。判断工作区前先加这个开关。
2. **`git push` 必须在 Windows 侧执行**：WSL 里 commit 正常但 push 会卡在缺失的凭据提示。**401 在一秒内返回 = 凭据问题**（而不是代理问题）。
3. **GitHub 走本机代理**：`127.0.0.1:7892`（2026-09-21 由 7897 改过来）。报 "failed to connect via 127.0.0.1" 是**代理**挂了，不是你的凭据。
4. **`outputs/rematch750/` 被 gitignore** —— 结果产物要落到 `results/` 才可提交。同理 `*.pt`、`cache/`、`train/`。
5. **`register()` 护栏会拒绝重建**：`aegis_clip/cli/rematch.py` 里 `ValueError('Candidate ID already registered with different predictions')`。推理与 9/9 校验都会正常完成、包也会写出，**只有登记表不会自动改写**。这是设计，不是失败。
6. **旧阶段资产按 500 类 / `train_dedup/` 路径键**：`outputs/phase/phase3/oof/`、`outputs/phase4/purification/*`、`outputs/data/d3_strict/` 等**仍在磁盘且被 git 跟踪，但新阶段一律不得读取**。危险在于路径存在，误引用**不会报 FileNotFoundError**，而是静默接错数据、或跑到一半才 `KeyError` —— 失败会伪装成"方法无效"。可迁移的只有代码与公式。
7. **autocast 下换 batch 形状会翻转边界样本**：我在特征漂移工具里把 `batch_size` 默认成 64（记录那次是 128），val micro 就差 2 张（0.7285618 vs 0.7286962）。**修法是从 config 取 `evaluation.batch_size`，不是放宽容差** —— 放宽容差等于废掉锚点这个护栏。
8. **我的 RM_LORA 结果没入库**：本机 GPU 八轮（macro 64.1239 / micro 65.1747，选 epoch 8）跑完在 09-22 20:48，但**产物在 gitignore 的 `outputs/` 里**，`results/` 与文档里只有 NPU 那次。两者相差 0.02pp，是很好的跨后端复现证据 —— 值得补一条记录（纯记录，无计算）。

---

## 9. 需要人类决策的未决项

**9.1 RM_LP 登记表行与今天的分数**
registry 的 RM_LP 行 `zip_sha256=db30ff7a…` 描述的是另一台机器上已删除的包；本机重建是 `ae654d8f…`。今天的平台分**记不上去**。两条路：**(a) 改那一行**（= 谱系变更，需说明）或 **(b) 开新 candidate ID**。**这不阻塞上传**，但阻塞如实回填。同理 `RM_FT_NPU` / `RM_FT_NPU_TUNED_E2` 两行的包也在那台机器上。

**9.2 登记表里 `submission_id` / 带时区提交时间 / `platform_period` 一直缺失**
按"不伪造 `valid`"的原则，这些字段保持空。若平台能提供，补上才算完整回执。

**9.3 剩下的两三周往哪走**
现在唯一有信息量的测量就是今天的 LP 槽位。它给出方向之后才有下一步：
- 若 **≥60%** → LP 本身成为新 baseline，方向转为在冻结特征上做文章；
- 若 **53.3–60%** → 冻结迁移更好 → 考虑**部分微调**（只调后段 / 冻结更多 backbone）；
- 若 **<53.3%** → FT 家族对 → 在 FT 之上找**鲁棒性**杠杆（而非再加容量）。

**9.4 主线是否继续走"效率/工程"（V3）还是回"精度"**
V2 已经说明：大 batch（1024）掉精度 4.5pp，batch32 最优但只比 GPU 快 1.59 倍。V3 想找 64/128 附近的折中点，**那是工程交付，不是平台提分**（V3 文档自己也写了"不自动声称平台更优、不自动上传"）。要不要把剩下的时间投在它上面，是人该决定的。

---

## 10. 一句话交接

**今天的槽位是唯一还在等的测量，先做它。** 别重测已闭环的四条（§4），别自行恢复 V3，别动 RM-FULL。计算走 NPU（选空闲卡、独立目录/分支），记录走 `results/`（`outputs/` 不入库），推送走 Windows 侧 git。
