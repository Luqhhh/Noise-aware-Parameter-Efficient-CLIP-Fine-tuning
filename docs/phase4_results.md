# Phase 4 最终实验报告

**实验完成时间**：2026-07-22
**文档核对时间**：2026-07-30
**训练基线**：A2 STRICT + AEGIS visual LoRA，seed 42
**结论**：P0–P4 均未达到预注册晋级门槛，不生成 Phase 4 平台提交。

原 `docs/phase4_plan.md` 是执行前计划，已按项目决定删除。本报告是 Phase 4 的最终权威状态；机器可读明细见 `results/phase4_experiments.csv`。

## 结果总览

| 阶段 | 候选 | 关键结果 | 判定 |
|---|---|---|---|
| P0 | Multiprototype | proxy macro 79.4571% → 79.7169%，但 raw micro 69.6377% → 69.2114%；修复 88、破坏 132，净 −44 | 关闭 |
| P0 | Shrinkage LDA | proxy macro 79.4571% → 79.6188%，但 raw micro 降至 69.2017% | 关闭 |
| P0 | Corrected Ridge | sweep 最优仍是原始 linear head（alpha=0） | 关闭 |
| P1 | SWA-1/2/3 | clean-core 分别为 83.1793% / 83.1513% / 83.1653%，均低于 epoch 6 的 83.2073% | 关闭 |
| P1 | Greedy soup | 只保留 epoch 6，等价于未平均 | 关闭 |
| P2 | CR-1 hard gate | raw +0.0388pp，clean-core −0.0140pp | 关闭 |
| P2 | CR-2 soft gate | clean-core +0.0420pp，但 raw −0.0387pp、proxy macro −0.0825pp；远低于 +0.20pp gate | 关闭 |
| P3 | Prototype-Contrastive | clean-core 83.1232%，比 CR-0 低 0.0840pp；只覆盖 499 类，promotion 失败 | 关闭 |
| P4 | Dynamic Trust Refresh | epoch 2 刷新后最佳 clean-core 82.7731%，低于刷新前 82.9972%，也低于 CR-0 83.2073% | 关闭 |

## P0：结构化分类头

多原型和 LDA 都能提高 proxy 指标，但会在 raw validation 上破坏更多样本。多原型 sweep 的 proxy 最优候选改变 556 个预测，其中修复 88、破坏 132；LDA 的 proxy 最优候选 raw micro 下降约 0.436pp。Ridge 没有找到优于 alpha=0 基线的候选。

因此，P0 的代理指标改善不能解释为可靠泛化增益，不进入平台。

本地产物：

- `reproducibility/aegis_f1/outputs/phase4/p0_multiprototype/sweep.json`
- `reproducibility/aegis_f1/outputs/phase4/p0_structural_lda/sweep.json`
- `reproducibility/aegis_f1/outputs/phase4/p0_structural_ridge/sweep.json`

## P1：同轨迹 Checkpoint Averaging

逐 epoch 训练完整运行 6 epoch。单 epoch 最佳 raw 出现在 epoch 2（69.7055%），最佳 clean-core 出现在 epoch 6（83.2073%）。三种等权平均均不能同时达到这两个锚点；greedy soup 拒绝其余四个 checkpoint，只保留 epoch 6。

结论：同轨迹平均没有降低当前模型选择折中，关闭该方向。

## P2：Clean-Routed LoRA

CR-0、CR-1、CR-2 均完成 seed 42。Hard gate 的阈值与现有训练样本选择阈值重合，没有形成新的有效训练干预；Soft gate 只产生 +0.0420pp clean-core 波动，同时损失 raw/proxy 指标。

预注册要求 clean-core 至少 +0.20pp 且两个 seed 同方向。seed 42 已明显不过 gate，因此不补 seed 3407、不生成平台包。

## P3：Trusted Prototype-Contrastive

最佳 epoch 的 clean-core 为 83.1232%，低于 CR-0 的 83.2073%；`predicted_class_count=499`，`promotion.json` 的 class coverage 检查失败。该候选不具备平台提交资格。

## P4：Dynamic Trust Refresh

训练在 epoch 2 执行一次动态 trust 刷新。刷新后的 clean-core 全部低于刷新前 epoch 1；最终 selector 选择的仍是刷新前 checkpoint。这说明当前刷新规则造成负干预，而不是改善静态 trust。

## 最终决策

Phase 4 没有产生新的平台候选。当前已审计平台锚点保持：

| 模式 | 实验 | 平台 |
|---|---|---:|
| Bare | A2 STRICT seed 42 | 60.6521% |
| Bare | A2 STRICT seed 3407 | 60.6441% |
| Flip TTA | A2 `NR_CL_KNN_DROP` seed 42 | 61.2128% |
| Flip TTA | A2 STRICT seed 42 | 61.1487% |

独立研发侧另报告 F1 + M1 为 63.3276%，但本仓库没有对应 ZIP SHA-256，必须与上述已审计结果分开标注。

## 产物与可复现性说明

Phase 4 原始运行目录受 `reproducibility/aegis_f1/.gitignore` 的 `outputs/` 规则控制，不进入 Git。为避免克隆后只剩提交说明，本报告和 `results/phase4_experiments.csv` 保存关键指标、判定和本地证据路径；配置、训练实现和 averaging 工具已由提交 `509ace1` 跟踪。
