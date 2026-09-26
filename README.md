# Noise-Aware Parameter-Efficient CLIP Fine-Tuning

面向噪声标签数据的细粒度图像识别。初赛数据为 500 类、约 103K 训练图；2026-09-21 已将本地数据替换为复赛数据：**750 类、148,695 张训练图、37,444 张测试图**。数据来源、路径、完整性校验和使用前准备见[复赛数据集元信息](docs/rematch_dataset_20260921.md)。复赛数据准备、RM-LP、RM-FT 与 RM-LT 已完成；独立验证 FT macro **71.8362%** / micro **72.9032%**，LT macro **71.8655%** / micro **72.6949%**。提交包已校验；RM-FT 为正式 baseline、RM-LT 为采样对照，RM-FULL 已取消；见[当前复赛执行记录](docs/rematch750_execution_20260921.md)。2026-09-22 两次平台分数已回填：RM-FT **60.96570879179575%（22,828 / 37,444）**、RM-LT **60.885589146458706%（22,798 / 37,444）**，FT 高 0.080119645pp / 30 张；见[平台结果记录](docs/rematch750_platform_results_20260922.md)。本地比较材料见[FT/LT 比较材料](docs/rematch750_local_comparison_20260922.md)。用户另完成 FT94/LT06 logits 融合研究，本地相对 FT +0.0806pp，但该候选因 RULE-05 禁止多模型 logits 加权且未过 +0.30pp 内部门槛而关闭，未上传平台；见[研究与止损记录](docs/rematch750_ft_lt_blend_research_20260922.md)。2026-09-22 训练侧 OOF 连续降权（RM-OOFW）完成本地配对裁决：以本机独立复现的 RM-FT 为对照（macro 71.8054%，与登记值 71.8362% 差 0.03pp），**主判据固定训练尾部 75 类 macro +0.0113pp 未过 +0.20pp 晋级门（t=+0.04），整体 macro −0.1311pp**，止损未触发；本轮据此关闭、不派生参数扫描、**无提交包**，详见[策略文档与结果](docs/rematch750_strategy_oof_downweight_20260921.md)。2026-09-22 推理侧探针（关闭方向，无候选、无提交包）：①**Flip TTA 全融合规则扫描**，本地 micro **+0.155~+0.242pp**（9/9 规则同号，配对 t 1.19~1.68），收益全在 head/medium、尾部 ≈0，量化了「推理侧不是答案」这一既有决定；②**输入几何假设证伪** —— 测试集 807 种形状且各维 ≤500px（375×500 占 37.6%）而训练/验证集长边 800–1390+，但把验证集压到测试集几何只掉 **0.17~0.31pp**，即本地/平台 **11.90pp** 差距不是输入几何造成的，推理侧改预处理关不掉。详见[推理侧探针](docs/rematch750_inference_side_probe_20260922.md)，其中含两处订正（上一轮的 TTA 推荐与既有记录冲突；RM-OOFW 预注册的**本地**晋级门违反 lessons §一.1「不设本地晋级门槛」）。2026-09-22 复赛 LP–FT 迁移对照预注册（**2026-09-23 提交槽位发 RM-LP**）：两包均为**完整合规的真提交包、非占位**，但明天那一次**按探针使用** —— 无在产候选有理由超过 60.97%，该槽位买信息不买分数，故**绝对分不判定策略、迁移率才判定**（LP 拿 55% 比 FT 拿 60% 信息量更大，仅「≥60%」一档是分数直接定策略；单种子噪声 0.90pp 量级，<1pp 差异不过度解读）。两臂本机重建并过 9/9 校验（RM-LP csv `5d564b6d…` / zip `ae654d8f…`；RM-FT csv `7e390c53…` / zip `8d395cbf…`），两臂在测试集上预测**不一致 28.72%**（10,755 / 37,444 张，检验力充足）。三条测量：①**val 是训练池的忠实 iid 抽样** —— 连「长边 ≤500 且 portrait」这一档测试集长相的切片都按同比例存在（train 8.50% / val 8.58%），故**不存在任何本地跨来源代理**，11.90pp 对 val 上一切指标结构性不可见，GPT-6 的「留一来源」路线在复赛数据上不成立，**本地解释这条线正式关闭**；②**FT 相对 LP 的 +9.23pp 池内增益是均匀的**（按 train_samples 五分位 +7.97~+9.99pp，750 类中 560 好 / 162 平 / 28 差），否掉「FT 只拟合头部池内特性」，但池内均匀增益仍无法在本地区分「表征更好」与「更彻底地拟合同源域」；③**预注册判据改用迁移率**（平台/本地）：FT 0.8366、LT 0.8375，故 **RM-LP 平台分 > 53.30% 即冻结特征比微调更抗来源漂移**，三档判定与后续动作见[预注册](docs/rematch750_lp_ft_transfer_prereg_20260922.md)。另注：FT/LT 两个不同模型平台上只差 0.080pp 而本地 LT 的 macro 反而更高，**full-FT 这一族已贴自身天花板，无在产候选有理由超过 60.97%**。2026-09-22 特征漂移分析（测量轮：不改判据、不派生候选、**无提交包**，测试集未触碰）：对 val 逐类算 `drift = 1 − cos(FT 特征, 冻结 OpenAI 特征)` 与 `Δrecall` 配对，**两处锚点均逐位通过**（val micro `0.7286962270736694` 复现记录值；mean drift `0.0379748` 与训练日志 `0.0379748` 前 7 位吻合，独立确认 drift 定义一致）。结果：Spearman `drift–Δrecall` **+0.4013**（Pearson +0.0199），drift 五分位上 Δmicro 由 **+5.05pp 单调升到 +15.10pp**。分辨检验：**`drift ⊥ train_samples`（+0.033，且在 support 五分位上 drift 恒定 0.0380/0.0376/0.0372/0.0382/0.0387）而 `drift vs LP_recall` 强负（−0.603）** —— 「池内域适配」读法（应集中在高样本类）被否掉，「真实表征改善」读法（应集中在冻结 CLIP 最弱的类）对上，故 **FT 的增益更可能迁移**。**据此收回**我当日口头给过的反向判断（「正相关 → 预测 RM-LP 平台分偏高」是错的，它默认了「特征被移动 = 域适配」）；正确预期是 **RM-LP 更可能落在低档（< 53.30%）**。判据与明天的选包决定均不变，仅读数预期改变。详见[特征漂移分析](docs/rematch750_feature_drift_20260922.md)。下文既有实验结果属于历史数据阶段。 2026-09-22 NPU 两轮迁移验收通过：910B2 单卡完成 20 轮 LP 和用户授权的 2 轮 FT（保留 8 轮学习率计划），FT 独立验证 macro **68.1837%**、micro **69.2742%**，相对 GPU 第 2 轮 macro **−0.072pp**；8,364 次更新与 checkpoint 重载审计通过，37,444 行 CSV/ZIP 在远端和本地均校验通过，**未上传平台**。FT batch 32 实测 **0.2265 秒/步**，本机 GPU 0.1266 秒/步，NPU 当前耗时约 **1.79 倍**；迁移正确性已验收；以上为优化前速度，未完成八轮精度复现。见[迁移记录](docs/rematch750_npu_migration_20260922.md)。后续 NPU 性能调优的 batch 32 两轮验收通过：**0.079585 秒/步**，较原 NPU **快 2.85 倍**、较本机 GPU **快 1.59 倍**；独立验证 macro **68.3531%** / micro **69.3817%**，8,364 次更新与重载审计通过，37,444 行 CSV/ZIP 双端校验通过，**未上传平台**。用户最终选用 **batch 1024 / workers 40 / prefetch 4**，两次吞吐实测均约 **1,832 张/秒**、张量显存 **23.52 GiB**，当时大 batch 尚无完整精度结果；后续结论见本段末V2记录，上述两轮提交包对应 batch 32。配置与证据见[调优记录](docs/rematch750_npu_tuning_20260922.md)。 **2026-09-23 REMATCH750_V2 已完成**：NPU batch1024八轮macro **67.3704%**、micro **68.4207%**（1,048次更新），固定配方关闭；同LP的NPU batch32八轮macro **71.9047%**、micro **72.9839%**（33,456次更新），保留为效率配置，未达+0.20pp新候选门；LoRA跑满八轮、按macro选epoch6，macro **64.1064%**、micro **65.1478%**，关闭。三组checkpoint/optimizer/逐类重载审计通过，无新提交包、无平台上传；原RM-FT包37,444行及ZIP字节一致性复核通过。完整曲线、逐类、计时与结论见[V2执行记录](docs/rematch750_v2_execution_20260922.md)。 **2026-09-23 V3按用户新指令恢复实施**：在B64×8、B128×16外新增B1024×16/LR×4补偿点，按同NPU B32的macro下降≤0.1pp筛选，再比较完整CLI耗时；用户随后删除不使用NPU0的限制，现只选当时空闲的NPU。B64已完成审计，macro 70.8299%、完整CLI 1946.7秒，精度未达线；B128已完成审计，macro 71.9320%达线但完整CLI 3663.7秒慢于B32；B1024×16/LR×4已完成并通过审计，macro 72.3472%、micro 73.4140%、完整CLI 2055.3秒，较B32 macro +0.4425pp且快1.397倍，是本轮效率胜者；条件B64_LR2按规则跳过。37,444行CSV/ZIP与9/9提交校验均完成，平台分数未知。见[V3执行记录](docs/rematch750_v3_tradeoff_20260923.md)。 **2026-09-25 F05 focus L05 交付保护与校准绑定**：平台最高分包 `L05_T14_P060`（平台 **66.94797564362783%**，center-only +0.5582pp）登记为不可覆盖产物并复核 CSV/ZIP/manifest 哈希，未重训未重交；`run_l05_local_retrain.py` 补上真正生效的 `--train-seed`（split seed 42、RM-LP parent SHA、配方均不变），出包脚本默认拒绝覆盖，新增 `tta_prior_binding.py` 把校准缓存绑定到 checkpoint/映射/split/内容组/精度/TTA 与 prior bias SHA；在既有缓存上按固定配方（T=1.4、prior=0.60）做内容组 3 折条件交叉拟合，pooled OOF Macro **0.7556778** / Micro **0.7639113**（相对 flip-TTA 无 prior **+0.1407pp / −0.0739pp**，纠正 144 / 破坏 155，逐折 ΔMacro 全负），故平台增益不能归因于 prior；见[记录](docs/f05_focus_l05_calibration_binding_20260925.md)。 **2026-09-26 L05 单卡新候选搜索（不做复验版）已闭环**：从 L05 checkpoint 独立续训三个候选（seed 42、各最多 3 个新增 epoch、`parent_kind=same_split_continue`、不重划数据、不重跑 A0/N1/O3/PTA、不补 seed3407/2026、不扫参）。固定协议（flip + `mean_probabilities` + T=1.4 + 本阶段验证集 prior 0.60）下 decode macro 相对 L05 winner（0.7582651）的差：NEW01 原图 ROI **−0.0335pp**（fail）、NEW02 融合目标 **+0.0144pp**（weak）、NEW03 冲突组集合监督 **+0.0615pp**（weak，唯一同时提高 micro，+0.0470pp）。三者均未过 +0.30pp 晋级门，**不替换 L05 winner、不生成提交候选、未上传平台**；按预注册口径只记为「阶段二续训配方收益」；见[实现与结果](docs/l05_new_candidates_implementation_20260925.md)。

**2026-09-23 冻结：** V3 winner `RM_V3_B1024_E16_LR4` 已由用户指令冻结，搜索关闭，不再追加 LR/epoch/batch/worker 扫描；远端 best/last checkpoint 与提交文件已设为只读并写入 `FROZEN.json`，机器可读记录见 [freeze.json](results/rematch750_v3/freeze.json)。

## 当前状态

**上面那段是本项目约定的状态落点：每完成一次交付就在原地更新它**（完成后做了什么、独立验证指标、是否有平台成绩）。阶段相关的一切都以它为准，不要在别处重复。

细节与下一步：

- **→ [docs/current_execution_plan.md](docs/current_execution_plan.md)** —— 当前执行入口 + 历史记录
- 当轮详细方案与固定配方：该文件顶部「当前执行入口」所指向的文档
- 提交登记表：[results/rematch_submission_registry.csv](results/rematch_submission_registry.csv)

本文件的其余部分描述阶段之间不变的内容。

## 快速上手

```bash
# 当前主线（Aegis 框架）的总入口
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.rematch --help
# 子命令：prepare / verify / cache / train / infer / run / record / prepare-full
# 完整可重放命令序列见 docs/current_execution_plan.md 顶部所指向的当轮执行文档

# 提交校验（--num-classes 或 --class-mapping 二者必给其一）
python3 scripts/check_submission.py \
  --test_dir <测试集目录> --class-mapping <class_to_idx.json> \
  --csv <pred_results.csv> --zip <submission.zip>

# Aegis 测试
PYTHONPATH=reproducibility/aegis_f1 python3 -m pytest reproducibility/aegis_f1/tests -q
```

## 项目结构

```
common/                      稳定骨架：数据集、配置/随机种子/日志、提交生成、PEFT、噪声鲁棒工具
experiments/                 早期「一个方法一个子目录」的实验骨架（现多为薄封装）
configs/                     实验 YAML（注意：存在多套互不兼容的 schema，见 CLAUDE.md）
scripts/                     数据准备、提交校验、审计，实验无关
reproducibility/aegis_f1/    当前主线实验框架（Aegis），绝大多数近期工作在这里
results/                     每轮实验的结果记录与提交登记表
docs/                        方案、预注册、执行记录
outputs/                     产物（*.pt 与 cache/ 不入 git）
```

`common/` + `experiments/` 是上一阶段的骨架；`reproducibility/aegis_f1/` 是当前前沿，自带独立的 `pyproject.toml` 与测试套件。

## 比赛约束

完整规则见 [COMPETITION_RULES_AGENT.md](COMPETITION_RULES_AGENT.md)。要点：

- 骨干固定 **CLIP ViT-B/32**，权重限 **OpenAI 官方**
- **禁止跨阶段复用**数据、checkpoint、特征缓存、伪标签、拟合的 prior
- **测试集只读**：禁止测试图入训、TTT、用测试分布调参
- **禁止集成**：多模型 / 多 checkpoint / 多 seed / 多头融合均不允许。最终 = 单 checkpoint + 单确定性推理脚本 + 一份 `pred_results.csv`
- 多尺度 + Flip TTA 已裁定合规（单 checkpoint + 单确定性流程）
- 提交格式：`文件名, 0001`（逗号 + 空格 + 4 位补零），ZIP 只含 `pred_results.csv` 且内外字节一致

## 经验与方法论

上一阶段（初赛）跑了几十轮实验，有大量可迁移的结论 —— 哪些方向有效、哪些被证伪、以及若干会让人得出**错误结论**的陷阱（本地验证与平台反相关、单 seed 不可靠、split 谱系不一致会伪造 +8pp 假信号等）。

**→ [docs/lessons_learned.md](docs/lessons_learned.md)**

建议在任何一轮新实验开始前先读一遍，尤其是「陷阱」一节。

## 文档地图

| 文档 | 用途 |
|---|---|
| [docs/current_execution_plan.md](docs/current_execution_plan.md) | **当前状态权威入口** + 历史记录 |
| [docs/lessons_learned.md](docs/lessons_learned.md) | 可迁移经验与方法论教训 |
| [docs/README.md](docs/README.md) | 文档索引与结果状态约定 |
| [COMPETITION_RULES_AGENT.md](COMPETITION_RULES_AGENT.md) | 比赛规则全文 |
| [CLAUDE.md](CLAUDE.md) | 面向 agent 的工作指引（含协作约定） |

## Git 策略

- ✅ 跟踪：`.json` / `.csv` / `.log` / `.yaml` / `.md` 结果与记录文件
- ❌ 忽略：`.pt` 检查点、`cache/`、`train/` `train_dedup/` `test/`

协作约定（提交推送纪律、避免重复劳动）见 [CLAUDE.md § 协作约定](CLAUDE.md)。
