# N1：冻结 A2 的学习式注意力局部残差头

日期：2026-07-20

## 动机

训练自由 M1/M3 已证明 attention-local 与全局/翻转视图存在稳定互补，但固定平均无法学习不同类别应如何使用局部细节。2026-07-15 的预印本 *CLIP-Guided Label-Free Discriminative Region Scoring for Fine-Grained Classification*（https://arxiv.org/abs/2607.13437）在多个细粒度数据集上使用全局与局部 CLIP 表示训练轻量线性分类头，显示学习式全局—局部组合可能远强于单全局特征。

原论文使用类别文本和干净基准数据。N1 不使用文本、类别名、外部数据或测试图像，只把我们已验证的 M1 attention-local 表示用于训练一个受严格约束的局部线性残差。

## 数据审计

- 源训练集：A2-kept `90,204` 张；
- 与固定 D3 validation 的规范化图像键重叠：`0`；
- 固定训练选择：cross-fitted trust `clean_probability >= 0.70`；
- 选择后：`66,929` 张，覆盖全部 `500` 类，每类 `31–184` 张；
- 不使用 validation label 选择训练样本，不使用 test image。

## 冻结表示

- checkpoint：A2 exact converted checkpoint；
- checkpoint SHA-256：`1e2c1a4a274c5e466b716ded41ccf58bebf167a2fadbeba75693d08bdb4f039c`；
- global feature：A2 原生 224×224、L2-normalized 512 维视觉特征；
- local feature：M1 固定最后视觉 block 12-head 平均 attention、top-5 patch 加权中心、160×160 裁剪放大回 224×224后的 L2-normalized 512 维特征；
- 视觉骨干与 A2 global classifier 在训练全过程冻结；特征缓存为 float16，进入损失前转 float32；
- train/validation 各缓存一次，不做图像增强或重复随机视图。

## 冻结模型

`logits_N1 = logits_A2_global + Linear_local(local_feature)`

- `Linear_local`：512→500，weight/bias 全零初始化；
- 训练时仅对 local feature 使用 dropout `0.1`；推理时关闭；
- 只训练 `256,500` 个局部头参数；A2 global 路径在 epoch 0 与原模型逐元素完全一致；
- 无门控、类别先验、原型、额外非线性或多层头。

## 冻结优化

- seed：42；
- 训练标签：高可信子集的原始标签；
- loss：standard cross entropy，无 label smoothing、class weighting、mixup 或 distillation；
- optimizer：AdamW，LR `1e-3`，weight decay `1e-4`；
- batch size：`1024`；
- max epochs：`30`；
- scheduler：cosine decay 到 0，无 warmup；
- validation：每 epoch；selector 为 clean-core micro；
- early stopping patience：`5`；
- epoch 0 必须记录并与 A2 online center 对齐；
- 不扫描阈值、dropout、LR、weight decay、epoch 或损失。

## 实现与效果门槛

1. train/validation cache 路径唯一、顺序唯一、零重叠；
2. cache global logits 必须与 A2 online center cache 最大绝对差 `0`、预测一致率 `100%`；
3. N1 epoch-0 logits 必须与 cache global logits 最大绝对差 `0`、预测一致率 `100%`；
4. best N1 clean-core micro 相对 A2 M1 至少 `+0.30pp`；
5. trusted macro 不得低于 A2 M1，raw micro 不得下降超过 `0.10pp`；
6. 任一门槛失败即关闭：不扫描超参数、不运行测试集、不生成提交包；
7. 通过后才允许单独预注册 N1 与水平翻转/局部 TTA 的组合。

## 合规性

N1 仍是单个 OpenAI CLIP ViT-B/32 加一个线性 PEFT 分类头；所有监督来自官方训练集。验证集仅用于模型选择，测试集仅保留给通过门控后的固定推理。团队仓库保持只读，全部缓存、训练和检查点位于独立工作树。

## 执行结果与结论

状态：**CLOSE（未通过预注册门槛）**。

- 高置信训练切分审计通过：`66,929` 张、`500` 类、每类 `31–184` 张、与 D3 validation 重叠 `0`；
- validation 双视图缓存为 `10,322 × 512` global/local features 与 `10,322 × 500` global logits；
- 新缓存 global logits 相对既有 A2 online-center cache 的最大绝对差为 `0`，预测一致率 `100%`；
- epoch 0 相对缓存 global logits 的最大绝对差为 `0`，预测一致率 `100%`；
- early stopping 在 epoch 28 触发，clean-core selector 的最佳点为 epoch 23。

| 候选 | clean-core micro | trusted macro | raw micro |
|---|---:|---:|---:|
| A2 center / N1 epoch 0 | 82.5755% | 80.3732% | 69.4536% |
| A2 + M1 固定融合基线 | 83.1716% | 80.8197% | 70.1124% |
| N1 best（epoch 23） | 83.3241% | 80.6988% | 69.0661% |
| N1 best − M1 | **+0.1525pp** | **−0.1209pp** | **−1.0463pp** |

N1 虽然把 clean-core 相对 A2 center 提高了 `+0.7485pp`，证明 attention-local 特征确有可学习信号，但没有达到相对 M1 的 `+0.30pp` clean-core 增益门槛；trusted macro 也下降，raw micro 更下降 `1.0463pp`，超过允许的 `0.10pp`。因此不运行测试集、不生成提交包，也不扫描学习率、正则、残差尺度或 epoch。

残差参数范数从 `0` 持续增长到 epoch 23 的约 `52.14`，与此同时 clean-core 小幅增加但 raw 持续下降。这支持以下诊断：自由的 500 类局部线性残差逐渐覆盖 A2 全局判别边界，并对高置信子集中残留的标签噪声和裁剪偏差过拟合。后续方向应保留局部互补性，但避免让一个无锚定的高自由度残差直接接管全局 logits；优先检验由训练集聚合得到、容量受限且尺度自校准的类别级局部表征。
