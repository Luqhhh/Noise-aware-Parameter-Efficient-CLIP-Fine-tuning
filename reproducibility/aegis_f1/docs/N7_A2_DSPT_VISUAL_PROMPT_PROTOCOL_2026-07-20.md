# N7：A2 深层视觉 Prompt + 双 Softmax 梯度抑制

日期：2026-07-20

## 核心判断

N3 证明轻量视觉适配可以改进 A2，但此前所有在线视觉实验仍依赖 GCE 和人工信任权重。
2026 年 ICML 工作 *Intrinsic Gradient Suppression for Label-Noise Prompt Tuning in
Vision-Language Models* 指出：CLIP 强先验与错标冲突时，普通交叉熵反而给错标最大梯度，
迅速覆盖预训练知识；其 DSPT 损失直接使用

`CE(softmax(logits), target)`，

即对 logits 先做一次 softmax，再把所得概率作为外层交叉熵的输入。高置信但与噪声标签
不一致的样本因此进入梯度饱和区，而不需要噪声率、温度或裁剪阈值。

论文：https://arxiv.org/abs/2605.00591

本实验不是对论文文本 Prompt 实现的逐字复现。赛题没有语义类别名，因此 N7 将同一损失
机制迁移到 N6 的视觉 Prompt；A2 epoch 48 提供已被平台验证的强视觉分类先验。

## 严格配对

N7 与 N6 的数据、A2 父检查点、12 层 × 5 个视觉 token、初始化、训练轮数、学习率、
信任权重、增强和特征蒸馏完全一致，唯一差异是把 GCE q=0.5 换成 DSPT 双 Softmax
交叉熵。MixUp、标签修正、类别先验调整和额外 loss cap 均关闭。

由于 DSPT 是针对 Prompt 噪声训练的直接方法，GPU 空闲后的优先次序为 N7 先于 N6；
这项优先级在查看任何 N6/N7 模型结果前冻结，不根据平台反馈改变。

## 工程门槛

1. 双 Softmax 必须严格等于论文的两阶段概率定义；
2. 所有损失计算使用 fp32，即使主前向启用 AMP；
3. 对一个高置信、预测与标签冲突的合成样本，其 logit 梯度范数必须低于普通 CE 的
   `1e-6`；
4. 真实 batch 64 反向审计仍必须证明只有 46,080 个 prompt 参数和 256,500 个线性头
   参数可训练，全部 CLIP 原生参数无梯度；
5. 初始检查点、数据路径和类别映射必须与 N6/A2 审计完全一致。

真实 batch-64 数值审计发现，PyTorch AMP 默认初始 loss scale `65536` 只在深层视觉
Prompt 张量上产生溢出；同一 batch 的 scale `1/128/1024/8192` 均为有限梯度，且
scale 1 时 46,079/46,080 个元素非零、prompt 梯度范数 `11.5692`，证明计算图和 DSPT
本身有效。N7 因此固定 `amp_initial_scale=8192`；该值只改变反向数值表示，反缩放后的
梯度、损失与优化目标不变，也不根据验证精度选择。训练前必须用该固定值重新通过审计。

### 训练前真实梯度体检

在不运行新模型、只读取 A2 epoch 48 头和冻结 CLIP 训练缓存的条件下，对 A2-kept 的
90,204 张样本逐一计算 logit 梯度。A2 与训练标签一致 74,278 张，不一致 15,926 张：

- 一致样本的 DSPT/CE 梯度范数比中位数为 `0.9562`，多数有效梯度被保留；
- 不一致样本的中位数仅 `0.00628`；
- 8,677 个 `p(noisy label)<0.01` 的高置信冲突样本，平均 DSPT/CE 比为 `0.00174`；
- 这 8,677 个冲突覆盖 494/500 类，前 10 个类只占 `8.44%`，不是少数坏类造成的假象；
- 低信任且不一致的 11,012 张样本，平均 DSPT/CE 比为 `0.0353`；
- 按 GCE q=0.5 的解析梯度比较，DSPT 对高置信冲突再缩小约 23 倍，对低信任冲突再
  缩小约 5.3 倍，而 trusted-match 只再缩小约 5%。

因此论文所需的“强先验 + 冲突样本梯度浪涌”条件在本项目真实训练缓存中成立，N7 通过
CPU 因果前门，可以进入真实 batch 反向审计。

### 与 CVPR 2025 PromptMAE 的去冗余审计

CVPR 2025 Highlight *NLPrompt* 的官方实现把 PromptMAE 写成
`GeneralizedCrossEntropy(q=1.0)`，即其相对 CE 的逐样本 logit 梯度缩放恰为
`p(noisy label)`。论文与代码：

- https://openaccess.thecvf.com/content/CVPR2025/html/Pan_NLPrompt_Noise-Label_Prompt_Learning_for_Vision-Language_Models_CVPR_2025_paper.html
- https://github.com/qunovo/NLPrompt

在同一 90,204 张 A2-kept 缓存上，PromptMAE 与 DSPT 的逐样本梯度范数比相关系数为
`0.99999845`，平均绝对差仅 `0.002186`。匹配样本的平均 DSPT/CE 与 MAE/CE 分别为
`0.88343/0.88606`，不匹配样本为 `0.05739/0.05728`；两者在本任务的 500 类强先验
区间几乎等价。因此不另开一个只把 GCE q 改成 1 的 N8 完整训练，避免把机制重复当成
独立探索；N7 是这一强抑制损失族的代表性 GPU 检验，N6 q=0.5 保留为较弱抑制对照。

## 效果门槛

N7 进入平台候选必须同时满足：

1. 相对 A2 center，clean-core micro 至少 `+1.00pp`；
2. trusted macro 至少 `+0.50pp`，raw micro 至少 `+0.25pp`；
3. flip agreement 不低于 A2 的 `87.37%`；
4. 平均 feature drift 不超过 `1%`；
5. 最佳轮次相对随机 Prompt 初始化至少恢复 `+1.50pp` clean-core；
6. 若之后执行 N6-GCE，N7 只有在 clean-core 至少高 `0.30pp` 且 trusted macro不低时，
   才能把收益归因于 DSPT，而不能只归因于视觉 Prompt 架构。

失败后不扫描 softmax 温度、prompt 数、层数、学习率或训练轮数。通过时只生成同一检查点
的裸推理与固定水平翻转；Prompt 的 attention-local 路径尚未通过审计，不使用 M3。

## 反方审查

原论文附录明确指出 DSPT 并非在所有中等噪声条件下都优于 GCE，而且论文训练的是带
语义类别名的文本 Prompt；N7 则是视觉 Prompt 加学习式线性头。因此训练缓存中的梯度
分离只证明机制适用，不等同于精度必然提升。为防止极小梯度让训练完全停滞，必须同时
报告随机 Prompt 初始点、首步 prompt/分类头梯度范数和逐轮恢复幅度；若最佳点未比初始化恢复
1.50pp，直接判定为过度抑制，不能以“更稳定”为由放宽效果门槛。

## 合规性

只使用官方训练图像、官方 OpenAI CLIP ViT-B/32 和单个线性分类头；无类别语义名称、
外部数据、测试时训练、模型融合或测试集统计。团队工程与 A2 资产保持只读。

## 执行结果：主门槛失败并关闭

真实 batch-64 反向审计在固定 AMP scale 8192 下通过：46,080 个 Prompt 参数与线性头
均获得有限非零梯度，冻结 CLIP 参数无梯度。随机深层 Prompt 的初始点却把 clean-core
从 A2 的 `82.5755%` 降至 `48.1979%`，feature drift 达 `29.0664%`，说明该结构在
初始化时并不保持 A2 强先验。

四轮训练全部完成，最佳为 epoch 3：

| 指标 | A2 center | N7 epoch 3 | 变化 |
|---|---:|---:|---:|
| clean-core micro | 82.5755% | 83.2132% | +0.6377pp |
| clean-core macro | 82.4547% | 83.0459% | +0.5912pp |
| trusted macro | 80.3732% | 80.9758% | +0.6026pp |
| raw micro | 69.4536% | 70.0639% | +0.6103pp |
| flip agreement | 87.3668% | 89.5078% | +2.1410pp |
| feature drift | 0.0000% | 0.6280% | +0.6280pp |

N7 从随机初始化恢复 clean-core `+35.0153pp`，trusted/raw/flip/drift 护栏均通过；但
主门槛要求相对 A2 clean-core 至少 `+1.00pp`，实际只有 `+0.6377pp`。它相对 N3
center 也仅约 `+0.0832pp`，不足以证明 Prompt 架构形成了新的能力级别。

按预注册协议关闭：不追加 epoch，不运行 N6，不扫描 Prompt 数/层数/LR/温度，不生成
测试预测或提交包。机制结论是 DSPT 能把被随机 Prompt 严重破坏的 A2 拉回安全区域，
但“先破坏再恢复”不如零初始化 Adapter 直接保持强先验，下一结构实验必须 epoch 0
严格等价于 A2。
