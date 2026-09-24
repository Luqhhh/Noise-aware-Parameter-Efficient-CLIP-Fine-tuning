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

---

# 2026-09-24 03:30 UTC 重跑进展

## 第一轮失败重跑已完成

以下 trial 已完成并覆盖旧失败结果：

| trial | status | selected epoch | macro | micro | Δmacro | Δmicro |
|---|---|---:|---:|---:|---:|---:|
| B05 | passed | 16 | 0.723607 | 0.734274 | +0.0135pp | +0.0134pp |
| E02 | passed | 16 | 0.725303 | 0.733938 | +0.1831pp | −0.0202pp |
| C01 | passed | 16 | 0.723252 | 0.733938 | −0.0221pp | −0.0202pp |
| C04 | passed | 16 | 0.723286 | 0.733938 | −0.0187pp | −0.0202pp |
| C07 | passed | 16 | 0.723649 | 0.734341 | +0.0176pp | +0.0202pp |
| C10 | passed | 16 | 0.723477 | 0.734207 | +0.0005pp | +0.0067pp |

这些结果均接近 V3 基线，没有明显本地增益；C 组 soft repair / hard relabel / unlabel consistency
在当前点上未表现出显著收益。

## 当前正在运行的 trial

| NPU | trial | 进度 |
|---|---|---|
| 0 | C02 | epoch 12/16，step 1572 |
| 1 | C03（迁移） | epoch 9/16，step 1140 |
| 2 | C05 | epoch 11/16，step 1441 |
| 4 | C06（迁移） | epoch 9/16，step 1060 |
| 5 | C08 | epoch 11/16，step 1441 |
| 7 | C11 | epoch 12/16，step 1480 |

队列调整：
- C03 从 NPU0 原队列迁移到 NPU1。
- C06 从 NPU2 原队列迁移到 NPU4。
- NPU0/NPU2 原队列 runner 已暂停，待迁移 trial 完成后恢复并跳过重复 trial。
- C09、C12 仍保留在 NPU5、NPU7 原队列中。

## 当前结果汇总

- with_result：82
- passed_training：73
- already_complete：1（F06）
- failed_rc_1：8（C02、C03、C05、C06、C08、C09、C11、C12 的旧失败结果，待覆盖）

本地 top5 未变化：

| trial | macro | micro |
|---|---:|---:|
| F05 | 74.4307% | 75.4435% |
| F06 | 74.3915% | 75.4301% |
| F03 | 74.0670% | 75.0874% |
| F04 | 74.0063% | 75.0538% |
| G01 | 73.2964% | 74.3884% |

---

# 2026-09-24 03:52 UTC 重跑进展（第二批）

第二批 C 组重跑已完成并覆盖旧结果：

| trial | status | selected epoch | macro | micro | Δmacro vs V3 | Δmicro vs V3 |
|---|---|---:|---:|---:|---:|---:|
| C02 | passed | 16 | 0.723429 | 0.734140 | −0.0043pp | 0.0000pp |
| C03 | already_complete | 16 | 0.723228 | 0.733938 | −0.0245pp | −0.0202pp |
| C05 | passed | 16 | 0.723477 | 0.734207 | +0.0005pp | +0.0067pp |
| C06 | already_complete | 16 | 0.723477 | 0.734207 | +0.0005pp | +0.0067pp |
| C08 | passed | 16 | 0.723649 | 0.734341 | +0.0176pp | +0.0202pp |
| C11 | passed | 16 | 0.723513 | 0.734207 | +0.0041pp | +0.0067pp |

这些结果同样接近 V3 基线，C 组修复/重标/一致性机制目前没有明显本地增益。

当前仍在运行：

- `C09`：NPU5，epoch 2/16，step 262
- `C12`：NPU7，epoch 3/16，step 393

尚未覆盖的旧失败结果只剩 `C09`、`C12`。

---

# 2026-09-24 04:23 UTC 全部重跑完成

V4 全部 82 个 trial 均已有最终结果：

| 状态 | 数量 |
|---|---:|
| passed_training | 79 |
| already_complete | 3（F06、C03、C06） |
| failed_rc_1 | 0 |

C09、C12 最终结果：

| trial | selected epoch | macro | micro | Δmacro vs V3 | Δmicro vs V3 |
|---|---:|---:|---:|---:|---:|
| C09 | 16 | 0.723513 | 0.734207 | +0.0041pp | +0.0067pp |
| C12 | 16 | 0.723477 | 0.734207 | +0.0005pp | +0.0067pp |

结论：
- B05、E02 以及全部 C01–C12 重跑均已完成。
- 当前没有失败 trial。
- C 组所有结果均接近 V3 基线，没有显著本地增益。
- 本地 top5 仍为 F05、F06、F03、F04、G01。
- 当前本任务 NPU 队列已清空。
