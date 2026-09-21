# 文档索引与状态约定

**核对时间**：2026-09-21

## 当前权威文档

**阶段相关的一切只在 [current_execution_plan.md](current_execution_plan.md) 一处维护。** 以下文件必须与项目现状同步：

| 文档 | 用途 |
|---|---|
| [current_execution_plan.md](current_execution_plan.md) | **当前状态唯一权威入口**：当前阶段、执行入口、下一步 + 历史记录 |
| [rematch750_execution_20260921.md](rematch750_execution_20260921.md) | 当轮复赛划分、三阶段配置、提交登记与全量重训命令 |
| [rematch_dataset_20260921.md](rematch_dataset_20260921.md) | 当轮数据集元信息：规模、路径、SHA-256 与迁移状态 |
| [lessons_learned.md](lessons_learned.md) | 可迁移经验与方法论教训（**开新实验前建议先读**） |
| `../README.md` | 项目入口、快速上手、文档地图 |
| `../CLAUDE.md` | 面向 agent 的工作指引（含协作约定） |
| `../COMPETITION_RULES_AGENT.md` | 比赛规则全文 |
| `../results/rematch_submission_registry.csv` | 平台提交登记表 |

## 历史快照

以下内容记录的是**当时**的预注册方案、阶段性结果或执行上下文，**不得改写成当前结论**。如状态与上方权威文档不一致，以权威文档为准。

- `docs/preliminary_75_*`、`docs/rematch*`、`docs/repechage_prep_20260831.md` —— 各轮预注册与执行方案
- `docs/superpowers/plans/`、`docs/superpowers/specs/` —— 早期设计文档与实施计划
- `docs/lqh/`、`docs/phase4_results.md`、`docs/e20_e21_posthoc.md`、`docs/aegis_independent_experiments_2026-07-22.md`、`docs/team_assignments_and_experiment_configs_2026-07-22.md` —— 早期阶段与团队记录
- `../results/*.md`、`../results/*.json`、`../results/*.csv` —— 每轮实测结果记录（**证据，不可重写**）
- `../CURRENT_STAGE_ACCEPTANCE.md`、`../progress.md`、`../findings.md` —— 上一阶段的验收与进度日志
- `../reproducibility/aegis_f1/docs/` —— Aegis 子工程的协议库
- `../results/prelim_storage_cleanup_20260921.md` —— 上一阶段权重与缓存的删除清单

原 `phase4_plan.md` 已有意删除，由 `phase4_results.md` 取代。

## 结果口径

- **本地 noisy validation 只用于安全 gate，平台分数才用于最终排序。** 历史上本地最高分的候选在平台最差。
- Bare、Flip TTA、M1/M3 等是不同推理协议，必须分栏比较，不做跨协议归因。
- 携带测试集统计量（如测试批拟合的 prior）的结果属于**不同来源协议**，必须单独登记，不得与合规结果混排或充当基线。
- 未通过预注册 gate 的实验必须保留负结果，但不得生成平台候选。

**证据状态标记**（用于 `results/` 中的登记行）：

| 标记 | 含义 |
|---|---|
| `audited` | 本仓库有完整 checkpoint / 提交产物哈希 |
| `audited_incomplete` | 平台分数与 checkpoint 可核验，但 prediction/ZIP 字段缺失 |
| `reported_incomplete` | 分数见于历史实验回填，但缺对应注册行 |
| `reported_unverified` | 仅有团队回填分数，缺本仓库可验证的提交包 |
