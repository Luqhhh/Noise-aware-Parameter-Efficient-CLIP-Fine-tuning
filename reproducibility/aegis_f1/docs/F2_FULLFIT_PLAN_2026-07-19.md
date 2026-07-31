# F2 方案：F1 Visual LoRA 全量重放

日期：2026-07-19

## 决策

F1 已由平台确认：裸推理 **60.5159%**，水平翻转概率均值 TTA **61.1007%**。下一轮优先做低风险的 `F2_VISUAL_LORA_FULLFIT`：不改变已经有效的模型方向，而是把 F1 开发阶段保留的 10% 验证样本重新纳入训练。

## 固化策略

- 仍为单个 OpenAI CLIP ViT-B/32；仅最后四个视觉 Transformer block 的 Q/V/out 使用 rank-8 LoRA。
- 分类监督仍只来自交叉拟合信任分数不低于 0.70 的 clean core；其余样本只承担对原始 CLIP 特征的蒸馏约束。
- GCE q=0.5、弱随机裁剪与水平翻转、学习率、批大小、特征蒸馏权重全部保持 F1 不变。
- 训练集由开发阶段的 92,902 张扩展为全部 103,218 张官方训练图像。
- 总轮数固定为 **4**，这是 F1 在平台评测前由独立开发划分选出的最佳 epoch。
- 旧验证划分已经包含在全量训练中，因此其指标只做故障诊断。`selection_policy: last_epoch` 强制选择 epoch 4，禁止用重叠验证集挑选提交检查点。

## 合规性

- 不使用外部数据、测试标签、测试时训练或多模型投票。
- 信任分数来自官方训练集内部的交叉拟合预测，测试集只在最终推理时读取。
- 提交仍是一个模型检查点；TTA 只融合同一检查点的原图与水平翻转概率。
- F2 的训练轮数和超参数来自 F1 开发实验，不根据 F1 的平台标签或 F2 的平台反馈调整。

## 执行结果

F2 已在团队任务释放显卡后独立完成，未修改或占用团队项目的工作目录。训练命令为：

```bash
cd /home/x28639/projects/AegisCLIP-Noise-Robust
python -m aegis_clip.cli.train --config configs/f2_visual_lora_fullfit.yaml
```

训练共完成 4 个 epoch，最终检查点按预注册策略固定为 epoch 4。`best.pt` 与 `last.pt` 虽然文件级哈希不同，但二者的 178 个模型张量逐项完全一致；提交统一使用 `best.pt`。

- checkpoint SHA-256：`7904312e7ca13b5ea6ea01d47dca5b3e59df64c374f4bdb69af9c7672a0042b6`
- 训练图像：103,218；测试图像：24,967；类别：500
- 可训练参数：403,956；LoRA 位置：视觉 block 8–11 的 Q/V/out
- 训练/测试数据审计：`external_data=false`，`test_usage=inference_only`

由于旧验证集已经参与 F2 训练，下面的数值只能用于检查训练是否退化，不能用于选择 epoch 或宣称泛化提升：

| epoch | raw micro | clean-core micro | train feature drift |
|---:|---:|---:|---:|
| 1 | 70.5506% | 81.3277% | 0.032600 |
| 2 | 70.7639% | 81.5982% | 0.033562 |
| 3 | 70.8996% | 81.7199% | 0.033436 |
| 4（固定） | 70.7929% | 81.6658% | 0.033477 |

epoch 3 的重叠诊断略高，但没有据此换检查点；这保留了 `last_epoch=4` 的预注册约束，避免全量重放阶段发生隐性挑选。

## 提交产物

以下两个常规包均通过 24,967 行、500 类覆盖、ZIP/CSV 哈希和单检查点来源审计：

| 候选 | 推理方式 | ZIP SHA-256 | 状态 |
|---|---|---|---|
| F2 bare | 单视图裸推理 | `daa914ae7f75121e7169bf4481ec03538fc5adf45076e7fd432f70fcaa09a813` | 可平台评测 |
| F2 flip TTA | 原图与水平翻转的概率均值，T=0.5 | `433e5eafd2fe91926bcc54355806640a3f10aac6bb8b9621a650ee442d41e7ce` | 首选平台评测 |

F2 TTA 相对 F2 bare 改变 2,057/24,967 条预测（8.2389%）。F2 TTA 相对已获 61.1007% 的 F1 TTA 改变 579 条（2.3191%），因此它是保守的全量数据增量实验，而不是模型方向突变。

在重叠诊断集上，F2 bare 的 clean-core micro 为 81.6658%，固定 F1 TTA 方式为 81.8145%；额外扫描的熵加权 TTA 达到 81.9227%，但该差异来自已经参与训练的数据，不能作为替换当前 TTA 规格的独立证据。

## 后续高风险分支

若 F2 没有超过 F1，再进入 F3：将 clean-core 之外的样本视为无标签数据，只施加冻结父模型的高置信度一致性监督，不重新信任原始噪声标签。该方向与“检测/筛除噪声后做参数高效适配”及“将疑似噪声样本作为无标签数据利用”的文献结论一致，但实现和超参数风险高于 F2，必须保留 F1/F2 对照。

- DeFT: https://arxiv.org/abs/2409.19696
- MOIT: https://openaccess.thecvf.com/content/CVPR2021/html/Ortego_Multi-Objective_Interpolation_Training_for_Robustness_To_Label_Noise_CVPR_2021_paper.html
- Fine-Grained Classification with Noisy Labels: https://openaccess.thecvf.com/content/CVPR2023/html/Wei_Fine-Grained_Classification_With_Noisy_Labels_CVPR_2023_paper.html
