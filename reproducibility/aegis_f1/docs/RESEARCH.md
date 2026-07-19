# 研究依据与设计取舍

检索日期：2026-07-16。以下只采用会议官方页面或官方论文，代码中的实现是针对本赛题约束的工程化推导，不声称逐字复现论文。

## 1. DeFT：VLM 可成为噪声标签检测器

[Vision-Language Models are Strong Noisy Label Detectors, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6af08ba9468f0daca4b8dd388cb95824-Abstract-Conference.html)

DeFT 通过每类正、负文本提示建立噪声检测器，并用 PEFT 适配视觉编码器。它证明了“先检测可信样本，再参数高效适配”是一条有效路线。

本赛题没有语义类名，因此 AegisCLIP 不使用文本提示。对应替代是：

- 正证据：样本对其噪声标签 OOF 类原型的支持；
- 独立复核：OOF 线性探针对同一标签的支持；
- 负证据：两视图共同拒绝噪声标签或发生高置信分歧。

这保留了 DeFT 的检测—适配思想，但不伪造数字类别的文本语义。

## 2. DKAF：不要盲信 CLIP 自己的先验

[Mitigating Endogenous Confirmation Bias in Noisy Label Learning for Vision-Language Models, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39641)

DKAF 指出，仅依赖 CLIP 零样本预测或自训练会产生内生确认偏差；其策略是聚合不同信号，分阶段做干净样本选择、标签细化和模型适配。

AegisCLIP 的对应控制：

- 每个样本只能由未见过它的模型评估；
- 原型与探针必须同时同意另一个类别，才允许软修正；
- 软修正强度不超过 0.45；
- 每个噪声类最多修正 20%；
- GCE 主实验的最低样本权重为 0.75，避免与鲁棒损失叠加后过度过滤；
- 训练前五轮保持原锚点配方，不立即相信伪标签。

## 3. Early Cutting：警惕“错误但容易”的样本

[Enhancing Sample Selection Against Label Noise by Cutting Mislabeled Easy Examples, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e826f682178d9830a3b704141b4989e-Abstract-Conference.html)

该工作指出，早期被模型正确预测的错标样本尤其有害，并用后期状态重新校准早期选择。

AegisCLIP 的视觉类比是：冻结原型认同噪声标签、但适配后的 OOF 探针以足够置信度拒绝它时，将其标记为 `mislabeled_easy` 并显著降低可信概率。这里不是对 Early Cutting 的完整复现，而是把“后期证据重新审判早期容易度”落实到静态 CLIP 特征场景。

## 4. OGC：梯度阈值必须随训练动态变化

[Optimized Gradient Clipping for Noisy Label Learning, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33025)

OGC 说明固定梯度裁剪阈值忽略了干净与噪声梯度分布随训练变化的问题。

AegisCLIP 没有声称复现 OGC 的完整估计器，而是采用两个更适合 8 GB GPU 的机制：

- 以高可信样本损失的运行分位数为阈值，对极端损失使用平滑对数尾部；
- 当整体梯度与高可信锚点梯度冲突时，把冲突分量投影到非负半空间。

前者保留非零梯度，后者只在 A3 的 PEFT 阶段启用。

## 5. FINE：从特征几何而不是训练损失识别噪声

[FINE Samples for Learning with Noisy Labels, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/ca91c5464e73d3066825362c3093a45f-Abstract.html)

FINE 强调潜在表征与类分布几何可用于噪声检测，避免完全依赖一个容易记忆噪声的分类器。AegisCLIP 的冻结类原型视图承接了这一几何优先原则，但选择了更可扩展的原型相似度，而不是为 10 万样本执行完整 Gram 特征分解。

## 6. Balanced Softmax：给后续长尾赛段留出受控入口

[Balanced Meta-Softmax for Long-Tailed Visual Recognition, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html)

复活赛和半决赛的数据规模说明明确存在长尾分布。Balanced Softmax 针对训练与测试类别先验不一致修正 Softmax 的训练偏差。AegisCLIP 实现了可关闭的训练期类先验 logit 校正：

- 默认 `tau=0`，初赛不在没有证据时启用；
- 后续赛段只比较 `0/0.5/1.0` 三个预注册值；
- 校正仅进入训练损失，验证和推理仍使用原始单模型 logits；
- 必须同时改善 macro 代理指标和多种子稳定性才能晋级。

这不是引入额外数据，也不是测试时重标定；类别频率完全来自当前赛段官方训练集。

## 为什么不是更多“前沿模块”

高分工程最危险的做法，是把论文名词同时堆进一个配置。AegisCLIP 坚持：

- A1 只检验可信度与软修正；
- A2 只新增动态损失上限；
- A3 才新增 PEFT、蒸馏与梯度投影；
- 每个机制都可以单独关闭；
- 未真实运行的收益不写入结果表。

这使榜单提升能被归因，也能在验证与线上不一致时快速回退。

## 7. TaskRes：保留强基座，只学习任务残差

[Task Residual for Tuning Vision-Language Models, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Task_Residual_for_Tuning_Vision-Language_Models_CVPR_2023_paper.html)；[官方代码](https://github.com/geekyutao/TaskRes)

TaskRes 冻结预训练分类器权重，并学习尺度受控的任务残差。本赛题的类别只有匿名数字编号，不能可靠建立文本分类器，因此 C4 使用 B0 的鲁棒视觉分类头作为冻结锚点。这是名称无关的工程化迁移，不是论文的文本原型复现。

C4 的真实结果为 70.2501%，低于 B0 的 70.3373% 和 C1 的 70.9965%。证据表明，仅改变分类头参数化不足以带来提升；TaskRes 思想可以保留为组件，但不再优先投入算力。

## 8. NLPrompt / DeFT：借净化机制，不借匿名类别文本

[NLPrompt, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Pan_NLPrompt_Noise-Label_Prompt_Learning_for_Vision-Language_Models_CVPR_2025_paper.html)；[官方代码](https://github.com/qunovo/NLPrompt)；[DeFT 官方代码](https://github.com/HotanLee/DeFT)

这类方法依赖有语义的类别名称、正负提示或文本原型来发现错标。对 `0000`—`0499` 直接构造文本提示不会产生可靠类别先验，而且通过外部网站或模型恢复类别语义会触碰赛规。因此不能直接照搬。

可合法迁移的部分是：用训练集内部的交叉拟合视觉原型替代文本原型，以类内配额或最优传输生成“可靠/可疑”分区；可靠样本使用 CE/GCE，可疑样本继续保留但使用更鲁棒的 MAE/GCE，而不是简单降权或丢弃。该路线与已经负收益的 B1/B2/C2 样本降权有实质区别。

该思想的保守版本已由 B5 验证：每类最低可信 20% 使用 GCE `q=1.0`，其余使用 `q=0.5`。B5 raw 69.6588%、proxy macro 78.8073%，均低于用同一可信资产重评估的 B0（70.3373%、78.9397%）。因此仅靠现有标量可信度做类内配额仍不足；除非重建并保存完整 OOF 类概率以支持真正的最优传输/confident-joint，否则不继续调整配额。

## 9. 下一轮优先级

1. 若继续净化路线，先重建并保存完整 OOF 类概率，再比较视觉原型/FINE、最优传输和 classwise confident-joint；现有标量可信度已不足；
2. 只有新选择器在类别覆盖、稳定性和困难干净样本保留率上通过门禁，才再次运行双损失；
3. 同步记录每轮训练预测波动，用 Label Wave 思路辅助检查点选择，解决本地验证排序与平台排序不一致；
4. ELR 已在团队工程实现但尚缺严格单变量配置，适合作为团队侧低成本对照；
5. 暂不投入 DivideMix/ProMix/DISC 等双网络或半监督重训练方案：算力和改造成本高，且单模型/外部信息边界需要额外合规审查。

## 10. ELR：官方符号核验与本任务负结果

[Early-Learning Regularization 官方实现](https://github.com/shengliu66/ELR)在示例中将 `log(1 - p·t)` 直接加到分类损失；该项为负，最小化时会增大当前预测与历史目标的一致性。

团队工程的只读审计发现，`common/elr.py` 当前返回 `-log(1 - p·t)`，而训练循环又以正权重加到总损失，因此优化方向与官方实现相反。尚未发现已报告平台结果启用 ELR，所以该问题不影响当前已知成绩；在修正符号并补充方向性测试之前，团队侧不应启动 S-ELR。

AegisCLIP 实现了正确方向、稳定样本索引、MixUp 对齐门禁和 ELR 状态恢复。真实对照中，`lambda=3.0` 为 66.8864%，按损失尺度校准到 `0.3` 后为 69.9399%，仍低于 B0 的 70.3373%。因此 ELR 不再是独立工程的优先路线。
