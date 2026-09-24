# REMATCH750_SEARCH_V4 结果汇总（2026-09-24）

本记录整理 2026-09-23 至 2026-09-24 远端 NPU 队列的最终状态。82 个 trial
均已生成机器可读结果。

## 总体状态

- 计划 trial：82
- passed_training：67
- already_complete：1（F06，迁移到 NPU0 后被原队列识别）
- failed_rc_1：14
  - B05、E02
  - C01–C12

## 最新本地 top5

V3 基线：macro 0.7234723568 / micro 0.7341398001。

| 排名 | trial | selected epoch | macro | micro | Δmacro | Δmicro |
|---:|---|---:|---:|---:|---:|---:|
| 1 | F05 | 14 | 0.7443073988 | 0.7544354796 | +2.0835pp | +2.0296pp |
| 2 | F06 | 16 | 0.7439153790 | 0.7543010712 | +2.0443pp | +2.0161pp |
| 3 | F03 | 16 | 0.7406703234 | 0.7508736849 | +1.7198pp | +1.6734pp |
| 4 | F04 | 16 | 0.7400628924 | 0.7505376339 | +1.6591pp | +1.6398pp |
| 5 | G01 | 16 | 0.7329644561 | 0.7438843846 | +0.9492pp | +0.9745pp |

F05/F06 均同时满足旧 +2.0pp 门槛和新 +1.0pp 保留规则。其 checkpoint 已通过
hardlink 保护在 `outputs/rematch750_search_v4/rule_1pp_preserved/`。

## 当前已知平台分

| 候选 | 平台分 | 正确数 | 备注 |
|---|---:|---:|---|
| F03 | 64.41352419613288% | 24,119 / 37,444 | 当前已知 platform leader |
| FULL01 / RM_FULL | 61.6360% | 约 23,079 / 37,444 | 全量训练重叠诊断，不能与 F03 本地分直接比较 |
| RM_FT | 60.96570879179575% | 22,828 / 37,444 | 独立验证 |
| RM_LT | 60.885589146458706% | 22,798 / 37,444 | 独立验证 |

## 失败原因

### C01–C12

12 个 C 组 trial 全部在数据集初始化阶段失败：

```text
ValueError: Trust bundle misses 14880 split samples;
first=0000/095017bc930947d0917b451e527ea15b.jpg
```

质量资产 `C01.pt`–`C12.pt` 覆盖 `train_dev.csv` 的 133,815 个样本，但没有覆盖
`val_dev.csv` 的 14,880 个样本。训练代码把 trust bundle 同时传给了验证集，
导致验证集构造时 fail-closed 覆盖率检查失败。

### B05

训练已完成 16 轮并保存 best checkpoint，但在 `per_class_report` 阶段失败：

```text
FileNotFoundError: logs/longtail_epoch_1.json
```

训练日志目录/长尾账本在训练期间被移动或清理，导致最终报告无法读取完整账本。

### E02

训练期间日志目录被外部移动/清理，写 `logs/progress.jsonl` 时失败：

```text
FileNotFoundError: .../RM_V4_E02/seed42/logs/progress.jsonl
```

## 后续

1. 修复训练器：验证集不再挂 trust bundle。
2. 增强日志目录与 `per_class_report` 的健壮性，避免日志被外部清理后直接失败。
3. 重跑 B05、E02、C01–C12。
4. 重跑后重新生成本地 top 与平台候选清单。
