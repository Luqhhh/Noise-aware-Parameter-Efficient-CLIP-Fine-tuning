# 平台结果总表与下一步提交决策

日期：2026-07-22

本页只把官网返回的真实成绩写入“平台分数”。机器可读权威记录为 [`../../../results/aegis_independent_platform_results.csv`](../../../results/aegis_independent_platform_results.csv)，其中 `platform_accuracy` 使用 0–1 小数；本文表格使用百分数。待评测、未运行和仅本地审计项目保持空分，不以本地 validation 代填。

## 完整历史锚点

| 提交 | 推理 | 平台分数 | 备注 |
|---|---|---:|---|
| D1 bare | 单中心视图 | `<59.8500%` | 仅确认低于 Flip，精确值不可得 |
| D1 Flip | 水平翻转 TTA | 59.8500% | 首次正式平台闭环 |
| E20 | epoch-35/44 线性头插值 + K7 + Flip | 60.1794% | 单主干、单线性头插值 |
| E21 | 全量低学习率续训 + K7 + Flip | 60.2195% | 从原模型最佳检查点出发 |

完整实验状态、哈希和未评测候选统一见团队根目录的 [`docs/aegis_independent_experiments_2026-07-22.md`](../../../docs/aegis_independent_experiments_2026-07-22.md) 与 [`results/aegis_independent_platform_results.csv`](../../../results/aegis_independent_platform_results.csv)。

## 已确认结果

| 提交 | 单检查点 | 固定推理 | 平台分数 | 状态 |
|---|---|---|---:|---|
| F1 + M1 | F1 visual LoRA | center + attention-local，1:1 概率均值 | **63.3276** | 当前最佳 |
| A2 + M1 | A2 kNN-drop | center + attention-local，1:1 概率均值 | **62.6747** | 强复现 |
| A2 + M3 | A2 kNN-drop | A2 Flip 分支 + M1 分支，1:1 概率均值 | **62.0259** | 有效但劣于纯 M1 |
| A2 + Flip | A2 kNN-drop | center/flip logits 均值 | 61.2128 | 旧最佳基线 |
| F1 + Flip | F1 visual LoRA | center/flip 概率均值 | 61.1007 | F1 基线 |
| F1 bare | F1 visual LoRA | 单中心视图 | 60.5159 | F1 裸推理基线 |

## 因果读数

1. M1 在 A2 和 F1 上分别相对其已知 Flip 基线提升 `+1.4619pp` 与 `+2.2269pp`，是当前最可靠的平台正信号。
2. F1 + M1 比 A2 + M1 高 `0.6529pp`，表明局部细节推理与视觉 LoRA 的表示适配存在正协同。
3. M3 本地优于 A2 + M1，但平台低 `0.6488pp`；不能继续依据带噪本地验证把更多视图当成更好。
4. 后续候选应优先使用纯 M1，并把 M3 仅作为消融，不再默认叠加 Flip。

## 当前仍可评测的独立候选

- F2 + M1：主工作树 `outputs/F2_VISUAL_LORA_FULLFIT/seed42/submissions/m1/submission.zip`，SHA-256 `4c0b5ea229f300d080e7cd6b1a5a93bd0c743a3b886a206f2a2f44f690f7a690`。F2 是当前平台最佳 F1 的预注册全量重放，固定使用完全相同的 M1；审计通过。
- O1 + M1：`outputs/O1_A2_ADAPTFORMER_MIXUP_GATE/seed42/submissions/m1/submission.zip`，SHA-256 `73cb20eda8063306071c09f40c2aeab0318549506bf71c05d8f570e37e5f0043`。O1 是 A2 上的 6 层 AdaptFormer + 平台已验证 MixUp；审计通过。
- N3 + M1：`outputs/N3_A2_ADAPTFORMER_GATE/seed42/submissions/m1/submission.zip`，SHA-256 `0b189c0669d1844787e860c936ede6eb1b1dfbacac07159e8623a7bc9b6fcbbb`。N3 是不含 MixUp 的严格配对 AdaptFormer；审计通过。
- N3 + M3：`outputs/N3_A2_ADAPTFORMER_GATE/seed42/submissions/m3/submission.zip`，SHA-256 `36baaa64b1b2c725191c3462e2d31f55c0ed67b5eb922addfaad24b5b02cbe45`。尚无平台分数；由于已知 M3 弱于 M1，不应作为下一优先项。

O1 与 N3 的纯 M1 验证 clean-core 均为 `83.5875%`；O1 的 trusted macro/raw micro 为 `81.3115%/70.7809%`，分别比 N3 高 `0.0447pp/0.0775pp`。这只作为安全审计，不视为平台分数估计。

下一次有限平台名额的优先顺序为：**F2 + M1 → O1 + M1 → N3 + M1**。F2 + M1 是当前平台最佳的最小风险全量重放；O1/N3 用于检验 AdaptFormer 表示能否进一步放大 M1。不要优先上传 N3 + M3 或旧的裸推理/Flip 包。

## 已预注册或完成本地审计、但尚未形成提交包

- R1：F1+M1 Part-Token 局部残差。协议、实现、真实 F1 epoch-0 逐位复现审计已完成；正式 GPU cache 与训练尚未启动，当前无 ZIP、无平台分数。最新团队整合快照完整回归 `201 passed`。权威协议见 [`R1_F1_M1_PART_TOKEN_RESIDUAL_PROTOCOL_2026-07-22.md`](R1_F1_M1_PART_TOKEN_RESIDUAL_PROTOCOL_2026-07-22.md)。
- T0/T1：F1 可信梯度子空间严格配对。T0 只使用可信标签梯度，T1 仅加入不确定标签梯度在近期可信梯度 rank-8 子空间中的投影；两份配置除实验编号和处理模式外逐字段一致。实现和自动 gate 已完成，但两个训练臂均未启动，当前无 ZIP、无平台分数。权威协议见 [`T1_F1_TRUST_SUBSPACE_GRADIENT_PROTOCOL_2026-07-22.md`](T1_F1_TRUST_SUBSPACE_GRADIENT_PROTOCOL_2026-07-22.md)。
- U0：冻结 OpenAI CLIP ViT-B/32 的数字类别 Prompt 可行性审计已完成且独立重跑逐字节一致。raw/clean-core validation 为 `0.232648%/0.229854%`，90%/99% 能量秩为 `1/5`，因此直接数字类名共享 CoOp 路线关闭。U0 没有训练、没有读取 test、没有 ZIP，也不是平台成绩。权威记录见 [`U0_NUMERIC_CLASS_PROMPT_FEASIBILITY_AUDIT_2026-07-22.md`](U0_NUMERIC_CLASS_PROMPT_FEASIBILITY_AUDIT_2026-07-22.md)。

以上三项均不进入当前 **F2 + M1 → O1 + M1 → N3 + M1** 的平台上传顺序；只有预注册训练门禁通过并生成审计合格的单模型包后，才可新增平台候选。
