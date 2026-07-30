# CURRENT STAGE ACCEPTANCE — 2026-07-30

## Scope

本验收状态覆盖主仓库截至 `7c8b966` 的代码和截至 2026-07-22 的本地实验产物。2026-07-30 的工作只同步文档与结果索引，没有启动新训练。

原 `docs/phase4_plan.md` 已确认有意删除，并由以下最终产物取代：

- `docs/phase4_results.md`
- `results/phase4_experiments.csv`
- `results/current_platform_summary.csv`

## Current Platform Anchors

不同推理协议分别管理，不能把 M1、Flip TTA 和 Bare 混成同一消融。

### Reported external inference anchors

| Experiment | Inference | Platform | Evidence status |
|---|---|---:|---|
| AEGIS F1 | M1 attention-guided local crop | **63.3276%** | `reported_unverified`：本仓库缺 ZIP SHA-256 |
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
