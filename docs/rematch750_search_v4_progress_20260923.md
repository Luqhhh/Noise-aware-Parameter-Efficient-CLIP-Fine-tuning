# REMATCH750_SEARCH_V4 当前进度（2026-09-23 16:16 UTC）

本记录对应远端 NPU 节点 `vllm-lqh-86`（8×Ascend 910B2）的队列快照，时间为
2026-09-23 16:16 UTC（北京时间 2026-09-24 00:16）。

## 总体状态

- 计划 trial：82
- 已生成机器可读结果：68
  - 训练通过：66
  - 失败：2（B05、E02）
- 尚缺结果：C01–C12、F05、F06
  - F05：正在 NPU5 训练
  - F06：排队等待
  - C01–C12：等待 Quality Asset OOF 完成

## 当前 NPU 队列

| NPU | 任务 | 状态 |
|---|---|---|
| 2 | WD03 | epoch 16/16，完成最后训练，正在收尾 |
| 5 | F05 | epoch 5/16，约 27% |
| 1/4/6/7 | 空闲 | — |
| 3 | 外部进程占用 | 约 57.2 GB HBM，非本任务 |

CPU 端 Quality Asset OOF 进度约 `650/1046`，仍在运行。

## 当前本地主要结果

V3 基线：macro 0.7234723568 / micro 0.7341398001。

| trial | selected epoch | macro | micro | Δmacro | Δmicro |
|---|---:|---:|---:|---:|---:|
| F03 | 16 | 0.7406703234 | 0.7508736849 | +1.7198pp | +1.6734pp |
| F04 | 16 | 0.7400628924 | 0.7505376339 | +1.6591pp | +1.6398pp |
| G01 | 16 | 0.7329644561 | 0.7438843846 | +0.9492pp | +0.9745pp |
| F01 | 16 | 0.7326631546 | 0.7430779338 | +0.9191pp | +0.8938pp |
| F02 | 16 | 0.7310260534 | 0.7414650321 | +0.7554pp | +0.7325pp |

当前没有 trial 同时达到 V4 内部显著性门槛（macro ≥ +2.0pp 且 micro ≥ +1.0pp）。

## F03 重训与推理

- F03 重训完成：selected epoch 16，macro 0.7406703234（74.0670%），
  micro 0.7508736849（75.0874%）。
- 重训结束后已完成 37,444 行推理，`check_submission.py` 全部通过。
- 提交包 SHA256：
  `ba9b9c974f612b2050b13707a8da66b507bd426eaa82babbf834af144ac3a2f5`
- 桌面副本：`C:\Users\lqh22\Desktop\submission\F03_submission.zip`。
- 该包未上传平台，平台分数未知。

## FULL01_submission.zip 平台分

用户回报：**FULL01_submission.zip = 61.6360%**。

- 候选：RM_FULL
- 训练策略：全量训练后 full fine-tune，224 center crop，无 TTA、无 prior
- ZIP SHA256：
  `e6c39f299a69351d352dced314beddaff5cc6676d426db3fadd82804b5a70e1a`
- checkpoint SHA256：
  `255c967ffc9dcbc689343909013f12fafbf29393407565a85a55bae90ff16f75`
- 用户未提供 platform submission ID、带时区的提交时间和 reset-period，
  因此正式 `results/rematch_submission_registry.csv` 仍保留 `ready`，
  不伪造 `valid` 回执；分数-only 记录见
  `results/rematch750_full01_platform_20260923.json`。

## 后续

- 等 F05 完成后继续 F06。
- 等 Quality Asset OOF 和 C01–C12 完成。
- FULL01/F03 的平台上传仍由用户执行；agent 不占据平台额度。
