# CURRENT STAGE ACCEPTANCE — 2026-08-02

## Scope

本验收状态覆盖主仓库截至 `7c8b966` 的历史代码、截至 2026-07-22 的训练产物，以及 2026-07-30 新增的 M1 attention-local 推理优化、局部特征 Adapter 实验、同 split E2→F1 严格重建、M1 + Flip 四视图融合、2026-07-31 的 balanced-prior 推理校准，和 2026-08-02 的 A12（LoRA 全 12 block，广度）训练。A2 STRICT Adapter 未通过 gate；F1 重建通过 gate，M1 平台为 62.9791%；M1 + Flip + balanced-prior 0.25 平台实测 65.5786%，其上的 prior 1.0（W060 checkpoint）平台 67.2007%；温度峰值 1.5 × prior 0.85（W060）平台 **67.5812%**；A12（LoRA 广度）+ M1/Flip + temp1.5 + prior 0.85 平台 **67.6173%**，新的审计完整平台最佳。

原 `docs/phase4_plan.md` 已确认有意删除，并由以下最终产物取代：

- `docs/phase4_results.md`
- `results/phase4_experiments.csv`
- `results/current_platform_summary.csv`

## Current Platform Anchors

不同推理协议分别管理，不能把 M1、Flip TTA 和 Bare 混成同一消融。

### M1 inference anchors

| Experiment | Inference | Platform | Evidence status |
|---|---|---:|---|
| A12 seed 42 | M1/Flip 0.40/0.50 + temp1.5 + balanced-prior 0.85 | **67.6173%** | `audited`（新平台最佳）：checkpoint/prediction/ZIP 哈希齐全 |
| W060 seed 42 | M1/Flip 0.40/0.50 + temp1.5 + balanced-prior 0.85 | 67.5812% | `audited`：checkpoint/prediction/ZIP 哈希齐全 |
| W060 seed 42 | M1/Flip 0.40/0.50 + balanced-prior 0.8 | 67.3329% | `audited`：checkpoint/prediction/ZIP 哈希齐全 |
| W050 seed 42 | M1/Flip 0.40/0.50 + balanced-prior 0.75 | 67.2848% | `audited`：checkpoint/prediction/ZIP 哈希齐全 |
| W060 seed 42 | M1/Flip 0.40/0.50 + balanced-prior 1.0 | 67.2007% | `audited`：checkpoint/prediction/ZIP 哈希齐全 |
| F1 REBUILD R1 seed 42 | M1/Flip 0.40/0.50 + balanced-prior 0.25 | 65.5786% | `audited`：checkpoint/prediction/ZIP 哈希齐全 |
| F1 REBUILD R1 seed 42 | crop160 / top5 / local 0.40 + Flip 0.50 | 63.7802% | `audited`：checkpoint/prediction/ZIP 哈希齐全 |
| AEGIS F1 | M1 attention-guided local crop | 63.3276% | `reported_unverified`：本仓库缺 ZIP SHA-256 |
| F1 REBUILD R1 seed 42 | crop160 / top5 / global 0.65 + local 0.35 | 62.9791% | `audited`：checkpoint/prediction/ZIP 哈希齐全 |
| A2 STRICT seed 42 | crop160 / top5 / global 0.65 + local 0.35 | 62.6870% | `audited`：checkpoint/prediction/ZIP 哈希齐全 |
| A2 | M1 attention-guided local crop | 62.6747% | `reported_unverified`：缺 checkpoint/ZIP 哈希 |
| A2 | M3 | 62.0259% | `reported_unverified`：缺推理 manifest/哈希 |

### Audited Bare / Flip TTA anchors

| Experiment | Inference | Platform | Status |
|---|---|---:|---|
| A2 `NR_CL_KNN_DROP` seed 42 | horizontal flip | **61.2128%** | checkpoint audited；prediction/ZIP hashes unavailable |
| A2 STRICT seed 42 | horizontal flip mean-prob T=0.5 | **61.1487%** | registered with ZIP SHA-256 |
| AEGIS F1 | horizontal flip mean-prob T=0.5 | 61.1007% | registered with ZIP SHA-256 |
| A2 STRICT seed 42 | Bare | **60.6521%** | registered with checkpoint/ZIP SHA-256 |
| A2 STRICT seed 3407 | Bare | **60.6441%** | registered with checkpoint/prediction/ZIP SHA-256 |
| AEGIS F1 | Bare | 60.5159% | registered with checkpoint/ZIP SHA-256 |

权威机器可读摘要：`results/current_platform_summary.csv`。

A2 STRICT + M1 weight 0.35 的最终平台分数为 62.6870%，相对同 checkpoint Bare +2.0349pp，但低于 F1 + M1 0.6406pp，判定为 `platform_valid_not_promoted`。完整证据见 `results/m1_localization_optimization_20260730.md`。

## Local Feature Adapter Acceptance

A2 STRICT 局部特征 Adapter 已完成首轮和六个有界消融。最佳 clean-core 增益
`+0.1821pp`，低于预注册 `+0.20pp`，因此状态为 `best_not_promoted`，没有测试
提交包。完整证据见 `results/local_adapter_a2_strict_20260730.md`。

## F1 Rebuild Acceptance

同源 split 的 E2→F1 重建已完成。E2 raw `70.3470%`；F1 相对 epoch 0
clean-core `+0.4092pp`、raw `+0.2230pp`、漂移 `0.3982%`、500 类，
promotion PASS。M1 weight 0.35 相对 F1 global raw `+1.2602pp`、
clean-core `+0.9548pp`，trusted/proxy 同向。

提交包 `02d37906accdf6b49e40733b4e675220f0177b5d71c5662984de68df5e781bb6`
已审计通过，平台 **62.9791%**，比 A2 STRICT + M1 高 `0.2921pp`，比已报告
原 F1 + M1 低 `0.3485pp`，状态 `platform_valid_not_promoted`。
完整证据见 `results/f1_rebuild_20260730.md`。

## M1 + Flip Candidate Acceptance

F1 REBUILD R1 上固定 `crop160/top5` 的 8 点有界扫描完成。选中
`local_weight=0.40, flip_weight=0.50`：相对 62.9791 对应的 M1 weight 0.35，
raw `+0.3296pp`、trusted macro `+0.1520pp`、proxy macro `+0.1340pp`、
clean-core micro `+0.2728pp`、clean-core macro `+0.0931pp`，覆盖 500 类。

提交包 `67f4eda57291e34096edcb0545b142fd0a3114fb1c76eb1e17996afe87d692e0`
经 24,967 条预测、500 类、ZIP 单文件和哈希一致性审计通过。平台实测
**63.7802%**，状态 `platform_valid_promoted`，新的审计完整平台最佳：相对
62.9791 对应协议 `+0.8011pp`、相对已报告原 F1 + M1 63.3276% `+0.4526pp`。
这是带水平翻转 TTA 的单 checkpoint 四视图概率融合，不能与无 Flip M1 混成
相同推理协议。完整证据见 `results/m1_flip_optimization_20260730.md`。

## Balanced-Prior Calibration Acceptance

`align_logits_to_prior` 均衡先验校准（IPF 类别偏置拟合）在本任务价值巨大：
模型测试预测严重不均衡（最差类 2 / 最好类 ~190 个预测，均衡期望 ~50/类），
平台测试类别均衡。已实测四个强度：

- **strength 0.6**（W060）：平台 66.9404%；
- **strength 0.75**（W050）：平台 67.2848%；
- **strength 0.8**（W060）：平台 **67.3329%**，新的审计完整平台最佳；
- **strength 1.0**（W060）：平台 67.2007%。

曲线 0.8 仍上升、1.0 回落，**峰值在 0.8~1.0 之间**（0.85/0.9 待测）。测试集
非完美均衡（24,967 = 467×50 + 33×49），1.0 略过矫正。0.6 低于 0.75/0.8
（校准不足）。

校准只改推理 logits，不训练、不改模型；累计把 local→platform gap 由 8.35pp
收窄至 ~4.9pp，距离 70 分 `2.67pp`。完整证据见
`results/prior_alignment_20260731.md` 与 `results/70p_campaign_20260731.md`。

## Phase 4 Final Acceptance

| Phase | Mechanism | Evidence | Acceptance |
|---|---|---|---|
| P0 | Multiprototype / LDA / Ridge | proxy 改善伴随 raw 回退；Ridge 基线胜出 | CLOSED |
| P1 | Same-trajectory checkpoint averaging | SWA-1/2/3 不及 epoch 6；greedy soup 等价 epoch 6 | CLOSED |
| P2 | Clean-Routed LoRA | hard gate 无独立干预；soft gate +0.042pp clean-core，低于 +0.20pp 门槛 | CLOSED |
| P3 | Trusted Prototype-Contrastive | clean-core −0.084pp；499/500 类 | CLOSED |
| P4 | Dynamic Trust Refresh | 刷新后所有 epoch 低于刷新前 | CLOSED |

Phase 4 没有平台候选，不补多 seed，不继续 threshold/rank/lr 网格。完整报告见 `docs/phase4_results.md`。

## Noise-Robust Wave A Acceptance

| Experiment | Result | Acceptance |
|---|---|---|
| A0 OOF zero p<0.001 | Bare 59.90%，TTA 60.31% | CONTROL |
| A2 CL+kNN consensus drop 991 | seed42 TTA 61.21%；seed3407 TTA 60.31% | BEST FROZEN, seed-sensitive |
| A3 consensus relabel 100 | TTA 59.89% | CLOSED |
| A1 classwise drop 8680 | TTA 59.55% | CLOSED |

机器可读记录见 `results/noise_robust_wave.csv`。

## Tests and Documentation

- 根目录 `tests/` 当前可收集 405 个测试；2026-07-30 实跑结果为 **402 passed、1 skipped、2 failed**。
- `tests/test_integration.py::test_full_pipeline_smoke`：CPU 单进程 DataLoader 配置了非零 `timeout`，触发 `_SingleProcessDataLoaderIter requires timeout == 0`。
- `tests/test_oof_soft_targets.py::test_oof_targets_map_stable_image_keys_and_return_probabilities`：float32 返回值与十进制字面量进行精确列表相等比较，`0.8000000119... != 0.8`。
- 标准命令为 `pytest -q tests`；直接运行裸 `pytest` 会扫描被忽略的 `.claude/worktrees/`，造成重复模块收集冲突，因此不能作为本仓库验收命令。
- 早期设计和执行计划保留为历史快照；当前文档入口及优先级见 `docs/README.md`。
- Phase 4 原始 `reproducibility/aegis_f1/outputs/` 仍按子工程规则忽略；关键指标已抽取进受跟踪的 Markdown/CSV。

## External Candidate Status

截至本仓库现有证据：

| Candidate | Status |
|---|---|
| F2 + M1 | submission package audited; platform result not recorded |
| O1 + M1 | submission package audited; platform result not recorded |
| N3 + M1 | submission package audited; platform result not recorded |
| O3 | stopped before training because reference reproduction audit failed |

这些候选位于 `/home/x28639/projects/...` 独立仓库，当前机器不存在对应目录。不得把“待平台”擅自更新为完成；获得真实分数后必须回填平台分数、上传时间和 ZIP SHA-256。

## Remaining Documentation Backfill

- [x] Phase 4 最终结果进入受跟踪报告和 CSV
- [x] README/progress/findings/outputs 索引同步
- [x] A2 STRICT seed 3407 的 60.64% 状态修正
- [x] Wave A 结果表补齐
- [x] 历史计划与当前权威文档分层
- [ ] F1/A2 + M1/M3 补齐提交 ZIP SHA-256
- [ ] F2/O1/N3 获得平台反馈后回填
- [x] A2 STRICT seed 3407 补齐 checkpoint/prediction/ZIP 哈希
- [ ] 修复并复跑当前 2 个 pytest 失败项
- [x] A2 STRICT + M1 weight 0.35 平台 62.6870% 已回填
- [x] A2 STRICT 局部特征 Adapter 负结果与 gate 证据已归档
- [x] E2→F1 重建与 M1 新候选完成审计
- [x] F1 重建 M1 平台 62.9791% 已按审计哈希回填
- [x] F1 重建 M1 + Flip 候选平台 63.7802% 已回填，新的平台最佳
