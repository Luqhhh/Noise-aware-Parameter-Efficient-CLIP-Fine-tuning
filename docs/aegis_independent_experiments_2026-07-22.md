# Aegis 独立实验线总账

更新时间：2026-07-22

## 结论先行

- 当前已确认平台最高分为 **F1 + M1：63.3276%**。
- M1（中心视图 + 注意力定位局部视图的 1:1 概率融合）在 F1、A2 两个不同检查点上均显著提升平台分数，是目前最强且有跨模型复现的平台信号。
- M3 在本地排序中曾表现更好，但平台仅 62.0259%，低于 A2 + M1 的 62.6747%；因此 M3 只保留作消融，不再默认叠加 Flip。
- F2、O1、N3 已完成训练、审计与打包，当前只缺平台评测，不需要新增训练算力。
- O3 原方案在训练前复现审计中停止；O3-R1、Q1A 与 R1 仅完成预注册/实现，尚未运行，必须另行获得算力与执行授权。R1 逐位锚定当前平台最佳 F1+M1，但目前没有提交包或平台分数。

机器可读平台记录见 [`../results/aegis_independent_platform_results.csv`](../results/aegis_independent_platform_results.csv)，团队统一提交登记见 [`../results/submission_registry.csv`](../results/submission_registry.csv)。

## 平台实测与待评测包

| 实验 | 检查点/训练策略 | 推理策略 | 平台分数 | 状态 | ZIP SHA-256 |
|---|---|---|---:|---|---|
| D1 bare | 全量 7 epoch gated adapter | 单中心视图 | `<59.8500%` | 仅知上界，精确值缺失 | `092e3d11ea71f7e15e1838d9796c9c764ffc442de7163a839fb43c0ef88fd5ad` |
| D1 Flip | 同一 D1 检查点 | 水平翻转 TTA | 59.8500% | 已确认 | `e3ab2e85b37f9dfb34b35521c41410342395d9c7acf8df110606a6ee2689b5a0` |
| E20 | epoch 35/44 线性头插值 + K7 原型 | 水平翻转概率融合 | 60.1794% | 已确认 | `37e524ef7aae81880b825595a26c90d272bb16753eac20222bf08349a5771a97` |
| E21 | 最佳原模型检查点低学习率全量续训 + K7 原型 | 水平翻转概率融合 | 60.2195% | 已确认 | `52a4ed874a745c818790de09b1b287c9845c36f13c402523e46e9662ffc97b5c` |
| F1 bare | 四个视觉块 Q/V/out rank-8 LoRA + clean-core + feature anchoring | 单中心视图 | 60.5159% | 已确认 | `6c81b7e38d5688cd67c36cb50868c2de507e0fc4fef3b69b9180c65f29f7a363` |
| F1 Flip | 同一 F1 检查点 | 水平翻转概率融合，T=0.5 | 61.1007% | 已确认 | `5773f52944af998ac349b7091386282484d8c7dcbc8af296461ae1978dd96657` |
| A2 Flip | kNN 共识删除冻结模型 | 水平翻转 | 61.2128% | 已确认的父基线 | 登记表未保存 ZIP 哈希 |
| A2 + M1 | 同一 A2 检查点 | 中心 + attention-local，1:1 概率均值 | 62.6747% | 已确认 | `b73eed1f826b37433962cce547cbfa6f15e57afd7d83b3c56557ce2ab399ecbd` |
| A2 + M3 | 同一 A2 检查点 | A2 Flip 分支 + M1 分支，1:1 概率均值 | 62.0259% | 已确认；低于纯 M1 | `8f757c6590e9d92ce7655e716d72eb36397d8f302e14c94f691b45e5e184ef4b` |
| **F1 + M1** | 同一 F1 检查点 | 中心 + attention-local，1:1 概率均值 | **63.3276%** | **当前最佳** | `eca9e7c6269c6a4a1cdb213228fa11e881a7ed9795df14da721d6799a1dab63c` |
| F2 + M1 | F1 配方的预注册全量重放 | 固定 M1 | — | 已训练、已审计、待平台 | `4c0b5ea229f300d080e7cd6b1a5a93bd0c743a3b886a206f2a2f44f690f7a690` |
| O1 + M1 | A2 + 6 层 AdaptFormer + MixUp | 固定 M1 | — | 已训练、已审计、待平台 | `73cb20eda8063306071c09f40c2aeab0318549506bf71c05d8f570e37e5f0043` |
| N3 + M1 | A2 + 严格配对 AdaptFormer | 固定 M1 | — | 已训练、已审计、待平台 | `0b189c0669d1844787e860c936ede6eb1b1dfbacac07159e8623a7bc9b6fcbbb` |
| N3 + M3 | 同一 N3 检查点 | 固定 M3 | — | 已审计、低优先级 | `36baaa64b1b2c725191c3462e2d31f55c0ed67b5eb922addfaad24b5b02cbe45` |

平台提交建议顺序：**F2 + M1 → O1 + M1 → N3 + M1**。

## 全部实验状态索引

这里的“通过/失败”指各自预注册的本地门禁，不等同于平台分数。完整参数、哈希和停止规则以链接协议为准。

| 实验族 | 方法与结论 | 状态/证据 |
|---|---|---|
| S0 / A0-A3 / B0-B5 / C0-C4 / CR0-CR2 / D0-D1 | 完成缓存、噪声可信度、feature adapter、ELR、classwise dual-GCE、残差头与全量重训等早期探索；C1 是早期本地最强，C4、B3/B4、B5 均未晋级；D1 完成首次平台闭环 | [`VERIFICATION.md`](../reproducibility/aegis_f1/docs/VERIFICATION.md) 与 `configs/` |
| E2 / E7 / E8 / E20 / E21 | MixUp/CE5 复现、跨 seed、全量续训、epoch-35/44 线性头插值与 K7 视觉原型 | E20/E21 平台 60.1794/60.2195；[`E2_E21_RESULTS`](../reproducibility/aegis_f1/docs/E2_E21_RESULTS_2026-07-18.md) |
| F1 | clean-core 视觉 LoRA，四个视觉块 Q/V/out rank-8，GCE q=0.5 与特征锚定 | 成功；bare/Flip/M1 均已平台评测；[`F1 report`](../reproducibility/aegis_f1/docs/F1_SUBMISSION_REPORT_2026-07-18.md) |
| F2 | F1 配方全量重放 | 已训练、已审计，M1 待平台；[`F2`](../reproducibility/aegis_f1/docs/F2_FULLFIT_PLAN_2026-07-19.md) |
| F3 | 冲突样本 keep/drop 严格配对 | 负结果，关闭；[`F3`](../reproducibility/aegis_f1/docs/F3_CONSENSUS_CONFLICT_PROTOCOL_2026-07-19.md) |
| F4 | 288 高分辨率评估 | Phase 1 未过门禁，关闭；[`F4`](../reproducibility/aegis_f1/docs/F4_HIGHRES_PROTOCOL_2026-07-19.md) |
| F6 | A2 disjoint visual LoRA | 接近但未过预注册门禁，关闭；[`F6`](../reproducibility/aegis_f1/docs/F6_A2_DISJOINT_LORA_GATE_2026-07-20.md) |
| F7 | A2 fixed full-fit fallback | 仅准备，未因证据不足启动；[`F7`](../reproducibility/aegis_f1/docs/F7_A2_FIXED_FULLFIT_PROTOCOL_2026-07-20.md) |
| G1 | train-only visual memory | 负结果，关闭；[`G1`](../reproducibility/aegis_f1/docs/G1_TRAIN_ONLY_VISUAL_MEMORY_PROTOCOL_2026-07-20.md) |
| H0-H3 | AdaptFormer control、noise-tolerant contrastive、KTA anchor/cyclic damping | 门禁未显示可转移收益，关闭；[`H1`](../reproducibility/aegis_f1/docs/H1_NOISE_TOLERANT_CONTRASTIVE_GATE_2026-07-20.md)、[`H3`](../reproducibility/aegis_f1/docs/H3_KTA_CYCLIC_GATE_2026-07-20.md) |
| I0 | OOF logit 重建 | 重建一致性通过，作为后续诊断证据；[`I0`](../reproducibility/aegis_f1/docs/I0_OOF_LOGIT_RECONSTRUCTION_PROTOCOL_2026-07-20.md) |
| I1 | cross-fitted structured selection/correction | 未过门禁，关闭；[`I1`](../reproducibility/aegis_f1/docs/I1_CROSSFIT_STRUCTURED_ALLOCATION_PROTOCOL_2026-07-20.md) |
| J0/J1 | broad LoRA control + trust-aligned projection | J1 有诊断性正信号但未过晋级门槛，关闭；[`J1`](../reproducibility/aegis_f1/docs/J1_TRUST_ALIGNED_BROAD_LORA_PROTOCOL_2026-07-20.md) |
| K1 | SNSCL | 未过门禁，关闭；[`K1`](../reproducibility/aegis_f1/docs/K1_SNSCL_PROTOCOL_2026-07-20.md) |
| L0-L4 / L1 protocol | LoRA 位置、clean/distill 与 cross-fitted prior correction | 未形成平台可晋级收益，关闭；[`L1`](../reproducibility/aegis_f1/docs/L1_CROSSFITTED_PRIOR_CORRECTION_PROTOCOL_2026-07-20.md) |
| M1 | attention-local/global inference | 跨 F1/A2 平台强复现；F1 + M1 63.3276%；[`M1`](../reproducibility/aegis_f1/docs/M1_ATTENTION_LOCAL_GLOBAL_TTA_PROTOCOL_2026-07-20.md) |
| M2 | discriminative multi-region inference | 有正向本地信号但未达门槛，关闭；[`M2`](../reproducibility/aegis_f1/docs/M2_DISCRIMINATIVE_MULTI_REGION_PROTOCOL_2026-07-20.md) |
| M3 | Flip 与 M1 互补融合 | 本地晋级但平台弱于纯 M1，保留消融；[`M3`](../reproducibility/aegis_f1/docs/M3_FLIP_M1_COMPLEMENTARY_TTA_PROTOCOL_2026-07-20.md) |
| N1 | learned local residual head | 未过门禁，关闭；[`N1`](../reproducibility/aegis_f1/docs/N1_LEARNED_LOCAL_RESIDUAL_HEAD_PROTOCOL_2026-07-20.md) |
| N2 | trust-weighted local class prototype | 未过门禁，关闭；[`N2`](../reproducibility/aegis_f1/docs/N2_TRUST_WEIGHTED_LOCAL_CLASS_PROTOTYPE_PROTOCOL_2026-07-20.md) |
| N3 | A2 AdaptFormer | 已训练、通过安全审计；M1/M3 已打包，待平台；[`N3`](../reproducibility/aegis_f1/docs/N3_A2_ADAPTFORMER_GATE_PROTOCOL_2026-07-20.md) |
| N4 | full-image letterbox | Phase 0 失败，关闭；[`N4`](../reproducibility/aegis_f1/docs/N4_FULL_IMAGE_LETTERBOX_GATE_2026-07-20.md) |
| N5 | N3 high-resolution adaptation | 未过门禁，关闭；[`N5`](../reproducibility/aegis_f1/docs/N5_N3_HIGHRES_ADAPTATION_PROTOCOL_2026-07-20.md) |
| N6/N7 | deep/DSPT visual prompt | N7 屏幕实验否决继续运行 N6，方向关闭；[`N6`](../reproducibility/aegis_f1/docs/N6_A2_DEEP_VISUAL_PROMPT_PROTOCOL_2026-07-20.md)、[`N7`](../reproducibility/aegis_f1/docs/N7_A2_DSPT_VISUAL_PROMPT_PROTOCOL_2026-07-20.md) |
| N9 | full-depth fixed-budget adapter | 未过门禁，关闭；[`N9`](../reproducibility/aegis_f1/docs/N9_A2_FULL_DEPTH_FIXED_BUDGET_ADAPTER_PROTOCOL_2026-07-20.md) |
| N10 | cross-fitted fine-conflict cap | 未过门禁，关闭；[`N10`](../reproducibility/aegis_f1/docs/N10_CROSSFITTED_FINE_CONFLICT_CAP_PROTOCOL_2026-07-20.md) |
| N11 | class-retention balanced softmax | 未过门禁，关闭；[`N11`](../reproducibility/aegis_f1/docs/N11_CLASS_RETENTION_BALANCED_SOFTMAX_PROTOCOL_2026-07-21.md) |
| N12 | active-forgetting noise suppression | 未过门禁，关闭；[`N12`](../reproducibility/aegis_f1/docs/N12_ACTIVE_FORGETTING_NOISE_SUPPRESSION_PROTOCOL_2026-07-21.md) |
| O1 | A2 AdaptFormer + MixUp | 已训练、审计通过；M1 待平台；[`O1`](../reproducibility/aegis_f1/docs/O1_ADAPTFORMER_MIXUP_GATE_PROTOCOL_2026-07-21.md) |
| O2 | F1 attention-local training | 未过门禁，关闭；[`O2`](../reproducibility/aegis_f1/docs/O2_F1_ATTENTION_LOCAL_TRAINING_PROTOCOL_2026-07-22.md) |
| O3 | F1 local-only feature adapter | 原方案在训练前复现审计停止；[`O3`](../reproducibility/aegis_f1/docs/O3_F1_LOCAL_ONLY_FEATURE_ADAPTER_PROTOCOL_2026-07-22.md) |
| O3-R1 | O3 batch-size 复现修订 | 已预注册，尚未运行；需要明确算力授权；[`O3-R1`](../reproducibility/aegis_f1/docs/O3_R1_BATCH_REPRODUCIBILITY_AMENDMENT_2026-07-22.md) |
| P1/P3/P4 | balanced-prior 计划与团队 Phase 4 联合探索 | 独立线仅保留 train-only 计划/配置；不得把本地先验假设写成平台证据；[`P1`](../reproducibility/aegis_f1/docs/P1_BALANCED_PRIOR_PLAN_2026-07-19.md) |
| Q1A | cross-fitted wrong-event trajectory | 代码、测试与协议完成，尚未运行；需要明确轻量 GPU 授权；[`Q1`](../reproducibility/aegis_f1/docs/Q1_CROSSFITTED_WRONG_EVENT_PROTOCOL_2026-07-22.md) |
| R1 | F1+M1 Part-Token 局部残差 | 协议、实现、源工程 188 项/团队快照 189 项完整回归及真实 F1 epoch-0 逐位复现审计完成；GPU cache/训练未启动，无提交包、无平台分数；[`R1`](../reproducibility/aegis_f1/docs/R1_F1_M1_PART_TOKEN_RESIDUAL_PROTOCOL_2026-07-22.md) |

## 合规边界

- 训练监督只来自比赛官方训练集；没有引入外部训练图像、外部标签或测试集伪标签。
- 主干固定为允许的 OpenAI CLIP ViT-B/32；每个提交由一个检查点和一条固定、确定性的推理流水线生成。
- 测试集只用于推理和格式审计，不参与训练、选样、调参或类别先验估计。
- M1/M3/Flip 属于同一模型的固定多视图推理。由于赛事对 TTA 的文字解释可能需要组委会最终确认，登记中保留 `tta_risk_acknowledged`，不把合规解释风险隐藏掉。
- 不把带噪本地验证分数当作平台分数；不把全量训练的回代准确率当作独立泛化证据；未知分数保持空缺。

## 合并范围与复现

此次整合把 Aegis 从共同基线 `d542fc6` 到独立提交 `beaa81f` 的新增/修改源代码、配置、测试与协议合并到 `reproducibility/aegis_f1/`，并保留团队在该目录后续加入的 A2 STRICT、Phase 4 与 A2 LoRA 消融内容。未提交 `.pt`、缓存、数据集、预测 CSV 或 ZIP 大文件。

详细来源见 [`PROVENANCE.md`](../reproducibility/aegis_f1/PROVENANCE.md)。
