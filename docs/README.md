# 文档索引与状态约定

**核对时间**：2026-09-29

## 当前权威文档

现行状态、计算资源、搜索纪律与下一步统一见 [current_execution_plan.md](current_execution_plan.md)。
**后续仅用本机 CPU/CUDA，NPU 不再使用。** 旧交接与长篇沿革移至
[历史索引](history/README.md)，不再把“待 NPU”“服务器恢复后继续”列为当前任务。

| 文档 | 用途 |
|---|---|
| [current_execution_plan.md](current_execution_plan.md) | **当前执行权威入口**：状态、资源、固定在途边界与下一步 |
| [strategy_v1_v2_20260929.md](strategy_v1_v2_20260929.md) | 团队自主策略的正式命名、当前验证及交付 |
| [v1.md](v1.md) | v1：噪声筛选、LoRA与固定推理 |
| [v2.md](v2.md) | v2：分辨率/LR阶梯与全量续训 |
| [p75_error_budget_policy_20260929.md](p75_error_budget_policy_20260929.md) | 主要误差、可恢复收益与完整训练的前置门禁 |
| [p75_full_sam_execution_20260928.md](p75_full_sam_execution_20260928.md) | 已启动本机固定实验的收尾与每小时监控，不自动衍生搜索 |
| [p75_supported_ce_preparation_20260929.md](p75_supported_ce_preparation_20260929.md) | CPU 实现准备与证据边界，不构成开训许可 |
| [rematch_dataset_20260921.md](rematch_dataset_20260921.md) | 当轮数据集元信息：规模、路径、SHA-256 与迁移状态 |
| [lessons_learned.md](lessons_learned.md) | 可迁移经验与方法论教训（**开新实验前建议先读**） |
| [../README.md](../README.md) | 最简通用项目说明 |
| [../AGENTS.md](../AGENTS.md)、[../CLAUDE.md](../CLAUDE.md) | 面向 agent 的指引、搜索与 Git 约定 |
| [../COMPETITION_RULES_AGENT.md](../COMPETITION_RULES_AGENT.md) | 比赛规则全文 |
| [../results/rematch_submission_registry.csv](../results/rematch_submission_registry.csv) | 平台提交登记表 |

## 历史快照

以下内容记录的是**当时**的预注册方案、阶段性结果或执行上下文，**不得改写成当前结论**。如状态与上方权威文档不一致，以权威文档为准。

- [history/README.md](history/README.md) —— 原执行计划、README 沿革、旧 brief、退出执行清单的 NPU 交接及历史实测；不恢复旧队列
- [rematch750_execution_20260921.md](rematch750_execution_20260921.md) —— 首轮复赛划分、RM-LP/FT/LT 与旧全量命令，仅供追溯
- [rematch750_local_comparison_20260922.md](rematch750_local_comparison_20260922.md)、[rematch750_platform_results_20260922.md](rematch750_platform_results_20260922.md) —— 早期 FT/LT 配对与平台证据，不是当前胜者
- [rematch750_ft_lt_blend_research_20260922.md](rematch750_ft_lt_blend_research_20260922.md) —— 已关闭融合研究，未上传
- `docs/preliminary_75_*`、`docs/rematch*`、`docs/repechage_prep_20260831.md` —— 各轮预注册与执行方案
- `docs/superpowers/plans/`、`docs/superpowers/specs/` —— 早期设计文档与实施计划
- `docs/lqh/`、`docs/phase4_results.md`、`docs/e20_e21_posthoc.md`、`docs/aegis_independent_experiments_2026-07-22.md`、`docs/team_assignments_and_experiment_configs_2026-07-22.md` —— 早期阶段与团队记录
- `../results/*.md`、`../results/*.json`、`../results/*.csv` —— 每轮实测结果记录（**证据，不可重写**）
- `../CURRENT_STAGE_ACCEPTANCE.md`、`../progress.md`、`../findings.md` —— 上一阶段的验收与进度日志
- `../reproducibility/aegis_f1/docs/` —— Aegis 子工程的协议库
- `../results/prelim_storage_cleanup_20260921.md` —— 上一阶段权重与缓存的删除清单

原 `phase4_plan.md` 已有意删除，由 `phase4_results.md` 取代。

## 结果口径

- 本地 noisy validation 用于复现、配对归因、分组误差预算与止损；普通本地 accuracy 小涨不能推算平台收益，平台提升只能依据实测声明。
- Bare、Flip TTA、M1/M3 等是不同推理协议，必须分栏比较，不做跨协议归因。
- 历史测试批统计量拟合结果须单独登记，不得与合规结果混排或充当基线；当前禁止用测试集预测分布调参。
- 未通过预注册 gate 的实验保留负结果，不自行晋级或占用平台名额；已关闭或已归档的计划不自动恢复。

**证据状态标记**（用于 `results/` 中的登记行）：

| 标记 | 含义 |
|---|---|
| `audited` | 本仓库有完整 checkpoint / 提交产物哈希 |
| `audited_incomplete` | 平台分数与 checkpoint 可核验，但 prediction/ZIP 字段缺失 |
| `reported_incomplete` | 分数见于历史实验回填，但缺对应注册行 |
| `reported_unverified` | 仅有团队回填分数，缺本仓库可验证的提交包 |
