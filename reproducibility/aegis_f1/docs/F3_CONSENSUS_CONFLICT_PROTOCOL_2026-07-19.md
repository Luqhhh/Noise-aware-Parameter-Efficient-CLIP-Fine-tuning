# F3 协议：折外共识冲突剔除

日期：2026-07-19

## 动机

赛题严格限定 OpenAI 官方 CLIP ViT-B/32，不能通过更大骨干或其他预训练权重突破当前瓶颈。F1 的有效策略是只让 `clean_probability >= 0.70` 的样本承担分类监督，但进一步审计发现：部分被判为 clean-core 的样本，其折外类别原型和折外线性探针却高置信一致指向另一个类别。

在 seed 42 的独立开发划分中：

- 训练集 clean core：65,473；
- 任意强度的双视图共识冲突：3,358，占 clean core 的 5.1288%；
- `pseudo_confidence >= 0.85` 的高置信冲突：1,842，占 clean core 的 2.8134%；
- 验证集对应高置信冲突为 145 个。

这说明单一的连续可信概率仍会保留内部证据相互矛盾的样本。F3 先测试最保守处理：只把高置信冲突样本的分类权重设为 0，不修改标签、不让伪标签直接产生梯度；所有样本仍保留 F1 的冻结特征蒸馏约束。

## 实验门控

先运行两个冻结特征快速对照，它们都从同一个 E2 epoch-44 检查点开始，训练集、学习率、GCE、轮数和随机种子完全一致：

1. `F3A_CONFLICT_KEEP_CACHED_DEV`：保留冲突，作为严格对照；
2. `F3B_CONFLICT_DROP_CACHED_DEV`：只剔除 `pseudo_confidence >= 0.85` 且存在折外修正证据的冲突样本。

主指标仍为独立验证集 clean-core micro，裁决顺序为 clean-core micro、clean-core macro、trusted macro、raw macro。只有 F3B 相对 F3A 在 clean-core micro 提升至少 0.10pp，且 trusted macro 不下降超过 0.05pp，才允许把相同开关迁移到视觉 LoRA。阈值 0.85 在运行前固定，不根据平台分数扫描。

若通过门控，LoRA 阶段仍使用 F1 的开发划分和 4-epoch 最佳轮数，先验证再做全量重放；若未通过则停止该方向，不尝试硬伪标签。

## 合规性

- 所有冲突证据均来自官方训练集上按内容组隔离的折外预测；没有外部数据或人工改标。
- 测试集不参与筛选、训练或超参数选择。
- 推理仍来自单个 OpenAI CLIP ViT-B/32 检查点。
- 该策略属于赛题明确鼓励的自动样本筛选和语义标签验证方向，同时比直接标签修正更保守。

研究依据：

- TrustCLIP（Semantic Label Verification 与 Trust-aligned Gradient Projection）：https://doi.org/10.1145/3746027.3755415
- JoAPR（联合自适应划分与标签翻新）：https://openaccess.thecvf.com/content/CVPR2024/html/Guo_JoAPR_Cleaning_the_Lens_of_Prompt_Learning_for_Vision-Language_Models_CVPR_2024_paper.html
- TURN（先保护表征、再在降噪子集上微调）：https://www.ijcai.org/proceedings/2024/403

## 执行结果

两个缓存对照均从同一个 E2 epoch-44 检查点确定性运行 6 个 epoch，73 项自动测试在执行前全部通过。

| 指标 | F3A 保留冲突（best epoch 2） | F3B 剔除冲突（best epoch 3） | F3B - F3A |
|---|---:|---:|---:|
| clean-core micro | 80.8275% | 80.8275% | 0.0000pp |
| clean-core macro | 81.4191% | 81.4111% | -0.0080pp |
| trusted macro | 79.9889% | 79.9696% | -0.0193pp |
| proxy micro | 78.4453% | 78.4447% | -0.0006pp |
| raw micro | 70.2598% | 70.2501% | -0.0097pp |

F3B 每个 epoch 实际剔除 2,125 个训练出现次数对应的高置信冲突样本，但主指标没有提升，未达到预注册的 `+0.10pp` 晋级线。结论：**停止该方向，不迁移到视觉 LoRA，也不继续扫描冲突阈值或尝试更激进的硬伪标签。**
