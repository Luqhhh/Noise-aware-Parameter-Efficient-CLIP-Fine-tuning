# 战略转向审计：从冻结特征后处理转向噪声感知视觉 PEFT

日期：2026-07-18

## 结论

此前 E20–E30 的主要搜索空间仍是冻结 CLIP 图像特征上的分类头、原型、插值和推理后处理。这类方法能改决策边界，却不能产生新的细粒度视觉表征，因而不应继续作为冲击显著平台增益的主线。

训练集/本地验证集来自带噪网络分布，而平台测试集标签干净、类别均衡且分布不同。团队登记结果中，本地指标与平台分数并未呈稳定正相关。因此，本地 noisy-val 约 70% 不是模型能力上限，也不能作为唯一选模依据。

主线改为：正确实现的视觉 LoRA + 高置信干净样本选择 + 表征保持 + 双视图一致性审计。该路线改变视觉表征，同时用小参数量和漂移约束控制对 CLIP 预训练知识的破坏。

## 规则边界

- 仅使用官方 OpenAI CLIP ViT-B/32 预训练权重。
- 仅使用当前赛段官方训练数据；测试集只做推理。
- 最终仍是单模型、单次推理，不做模型集成或投票。
- 官方数据没有可靠的类别语义名称映射，因此暂不采用依赖 class name 的文本 Prompt 方法，也不从外部补类别语义。

## 为什么不再围绕 noisy-val 70% 做局部优化

截至审计时，21 条同时记录本地和平台成绩的异质实验记录给出 Pearson 约 -0.13、Spearman 约 -0.17。该统计含配置差异和 7 条格式异常记录，不能作严格因果结论，但足以说明：继续把 noisy-val 微小上升当成平台提升代理是不安全的。

已有信任估计显示，阈值 0.70 时训练集仍保留约 70.6% 样本、覆盖全部 500 类；这为“高精度干净核心训练、低置信样本不直接监督”提供了可行基础。

## F1：VISUAL_LORA_CLEAN_CORE

配置：`configs/f1_visual_lora_clean_core.yaml`

- 从 E2 epoch 44 检查点初始化。
- 在 ViT 最后 4 个 Transformer block 的注意力 Q、V 与输出投影安装 rank-8 LoRA。
- 仅约 40.4 万可训练参数，占总参数约 0.46%。
- 高置信阈值为 0.70；拒绝样本监督权重为 0。
- 使用 GCE、弱随机裁剪和水平翻转；不做 MixUp，不进行激进伪标签覆盖。
- 用冻结父模型特征作锚点，限制视觉特征漂移。
- 主选择指标为高置信干净核心准确率；同时审计 noisy-val、翻转预测一致性和特征漂移。

工程安全验证：完整测试 63 项通过；真实权重在 epoch 0 的 logits 与父模型严格相同；真实 GPU 前向、反向和 LoRA 梯度均通过；原生多头注意力保持可运行。

阶段结果：

| Epoch | noisy-val micro | clean-core micro | flip agreement | val feature drift |
|---:|---:|---:|---:|---:|
| Parent E2 e44 | 70.2307% | 80.7599% | 88.5711% | 0.0001% |
| 1 | 70.5021% | 81.1385% | 88.8523% | 0.2528% |
| 2 | 70.5409% | 81.2872% | 88.7166% | 0.3414% |
| 3 | 70.5312% | 81.2737% | 88.8426% | 0.3812% |
| 4 | 70.6766% | 81.5306% | 88.8426% | 0.4081% |
| 5 | 70.6669% | 81.4494% | 88.7456% | 0.4216% |
| 6 | 70.6281% | 81.4494% | 88.8232% | 0.4260% |

最终判断：epoch 4 为最佳；epoch 5–6 连续未刷新，已按两轮耐心停止。漂移始终显著低于 1% 风险线，未发生表征崩坏。上述指标只能作为安全门控，不能直接推断平台分数。

同口径独立复评确认，epoch 4 相对初始化父模型：noisy-val micro +0.4459 个百分点、clean-core micro +0.7707 个百分点、clean-core macro +0.7074 个百分点、flip agreement +0.2714 个百分点；最终特征漂移为 0.4081%。训练内评估与独立评估逐项一致。交叉熵由 1.5309 增至 1.5853，说明概率校准未同步改善，后续不应基于置信度做激进融合。

## 后续优先级

1. 在 F1 出现干净核心收益停滞或一致性持续恶化时早停，保留最佳检查点。
2. 对父模型和 F1 最佳检查点用同一套 clean-core、flip consistency 指标复评，消除“缺少 epoch 0 基线”的比较盲区。
3. 只有通过门控才生成提交包；优先提交裸推理，TTA 仅作受控对照。
4. 若 F1 证明视觉 PEFT 有效，再做 F2：EMA teacher + 双视图一致性 + 分阶段伪标签，但仍保持单模型推理和官方数据边界。
5. 停止大规模后处理权重扫描；每个实验必须回答一个可证伪问题。

## 研究依据

- 比赛页：https://www.aicomp.cn/tracks/tracks-1/3714.html
- PTNL（ICCV 2023）：https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Why_Is_Prompt_Tuning_for_Vision-Language_Models_Robust_to_Noisy_ICCV_2023_paper.html
- JoAPR（CVPR 2024）：https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_JoAPR_Cleaning_the_Lens_of_Prompt_Learning_for_Vision-Language_Models_CVPR_2024_paper.pdf
- DeFT（NeurIPS 2024）：https://proceedings.neurips.cc/paper_files/paper/2024/hash/6af08ba9468f0daca4b8dd388cb95824-Abstract-Conference.html
- SNSCL（CVPR 2023）：https://openaccess.thecvf.com/content/CVPR2023/html/Wei_Fine-Grained_Classification_With_Noisy_Labels_CVPR_2023_paper.html
