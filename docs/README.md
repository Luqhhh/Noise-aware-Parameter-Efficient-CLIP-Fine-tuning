# 文档索引与状态约定

**核对时间**：2026-07-30
**最新本仓库实验时间**：2026-07-22

## 当前权威文档

以下文件必须与项目现状同步：

| 文档 | 用途 |
|---|---|
| `../README.md` | 项目入口、当前最佳结果和下一步 |
| `../CURRENT_STAGE_ACCEPTANCE.md` | 当前验收状态、已关闭方向和未决事项 |
| `../progress.md` | 按时间追加的执行记录；顶部是最新状态 |
| `../findings.md` | 按时间追加的研究结论；顶部是最新结论 |
| `phase4_results.md` | Phase 4 P0–P4 最终报告 |
| `team_assignments_and_experiment_configs_2026-07-22.md` | 独立研发仓库候选和平台回填状态 |
| `../outputs/README.md` | 输出目录和平台结果索引 |
| `../results/submission_registry.csv` | 已上传或已报告的平台提交登记 |
| `../results/current_platform_summary.csv` | 跨推理协议的平台结果摘要 |
| `../results/phase4_experiments.csv` | Phase 4 机器可读结果 |

## 历史快照

`docs/superpowers/plans/`、`docs/superpowers/specs/`、`docs/lqh/`、`../results/*.md`、`../reproducibility/aegis_f1/docs/` 以及其他文件名带日期的实验计划/报告，记录的是当时的预注册方案、阶段性结果或执行上下文。它们不应被改写成当前结论；如其状态与当前权威文档不同，以本索引上方列出的文件为准。`../reproducibility/aegis_f1/README.md` 顶部也已标明该子工程的快照边界。

原 `phase4_plan.md` 已有意删除，由 `phase4_results.md` 取代。

## 结果口径

- Bare、Flip TTA、M1/M3 是不同推理协议，必须分栏比较。
- `audited` 表示本仓库有完整 checkpoint/提交产物哈希；`audited_incomplete` 表示平台分数和 checkpoint 可核验、但 prediction/ZIP 字段仍缺失；`reported_incomplete` 表示分数见于本仓库历史实验回填但缺对应注册行；`reported_unverified` 表示只有团队回填分数，缺少本仓库可验证的提交包。
- 本地 noisy validation 只用于安全 gate，平台分数才用于最终排序。
- 未通过预注册 gate 的实验必须保留负结果，但不得生成平台候选。
