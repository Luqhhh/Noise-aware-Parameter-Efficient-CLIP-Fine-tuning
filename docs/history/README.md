# 历史文档索引

归档日期：2026-09-29。**当前执行以[现行入口](../current_execution_plan.md)为准。**
后续仅使用本机 CPU/CUDA，NPU 不再使用；历史状态、旧授权和待跑安排都不能恢复任务。

## 状态沿革与旧 brief

| 归档 | 内容 |
|---|---|
| [原执行计划](execution_plan_before_local_only_20260929.md) | 资源更新前的完整执行计划，含初赛/复赛各轮结果与分工沿革 |
| [README 原状态段](readme_status_before_local_only_20260929.md) | 此前交付、独立验证指标与平台回填 |
| [交接 brief](handover_brief_20260923.md) | 旧跨机器/NPU 分工与历史资产边界 |
| [策略 brief](strategy_brief_20260923.md) | 旧最佳分、搜索目标、候选与算力假设 |
| [GPT-6 搜索 brief](gpt6_search_brief_20260923.md) | 旧搜索提议、变更与争议记录，包含后来撤回的并行/多 seed 假设 |

## 已退出执行清单的交接

以下只保留旧实现与预注册，停止等待 NPU/服务器恢复，不自动迁为本机高成本搜索：

- [WD relaxation](../rematch750_wd_relax_prereg_20260923.md)
- [Head L2-SP](../rematch750_head_l2sp_prereg_20260924.md)
- [F05 evidence transfer](../rematch750_f05_transfer_prereg_20260924.md)
- [Decay filter](../rematch750_decay_filter_prereg_20260924.md)
- [Full data control](../rematch750_full_ft_prereg_20260923.md)：原关闭决定继续有效。

## 保留的历史执行证据

- [NPU 迁移验收](../rematch750_npu_migration_20260922.md)
- [NPU 吞吐调优](../rematch750_npu_tuning_20260922.md)
- [V2 已完成实验](../rematch750_v2_execution_20260922.md)
- [V3 已完成并冻结](../rematch750_v3_tradeoff_20260923.md)
- [早期本机搜索批次](../rematch750_search_batch1_20260923.md)

原文链接已按归档目录重定位，字面命令/路径仍描述原执行环境。指标、哈希、结果 JSON/CSV
与原始代码不因资源停用而重写；它们用于审查历史，不表示当前资源仍可访问或候选仍待执行。
