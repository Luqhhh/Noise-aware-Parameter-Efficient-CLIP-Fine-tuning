# N11：类别保留偏差与训练期 Balanced Softmax

日期：2026-07-21

## 动机与诊断

官方原始训练集 500 类接近均衡，每类 201--213 张；去重、固定验证划分与 A2 净化之后，
实际训练子集 90,204 张，每类仅 36--192 张，最大/最小频比为 5.33。现有实验协议规定频比
达到 3 时应独立比较 `class_prior_adjustment_tau ∈ {0, 0.5, 1.0}`，但此前所有配置均为 0。

N3 的固定干净核心验证显示明显的类别级关联：

- 训练保留数与每类 clean-core accuracy 的 Spearman 相关系数为 0.541；
- 保留数最低 50 类的聚合 clean-core accuracy 为 47.95%，最高 50 类为 94.26%；
- A2 线性头权重范数与训练保留数的 Spearman 相关系数为 0.399。

该关联不能直接证明删样导致低准确率，因为困难类也更容易被筛选器判为低可信；但它满足
CVPR 2026 Debiased Sample Selection 所指出的类别级确认偏差模式：容易类被过度选择，困难类
的有效监督进一步减少。Balanced Softmax 只在训练损失中加入经验类先验，迫使稀有类学习
更高的原始 logit；推理仍使用单模型原始 logits。

主要依据：

- https://openaccess.thecvf.com/content/CVPR2026/html/Pan_Debiased_Sample_Selection_for_Learning_with_Noisy_Labels_CVPR_2026_paper.html

## 严格配对设计

复用已完成的 N10C 作为 `tau=0` 控制：冻结 CLIP 特征、A2 epoch 48 初始化、原始 KTA trust、
GCE q=0.5、相同 90,204 条训练记录、batch 512、6 epoch、相同学习率和 checkpoint selector。
N11 只运行两个预注册处理：`tau=0.5` 与 `tau=1.0`。除训练期
`class_prior_adjustment_tau` 外不得改变任何字段，不做连续 tau 扫描。

控制检查点 SHA-256：
`5488b595176312fe62267827a5185c30dbda28f1ee75fbf4996e70707215f0b9`。

## 门槛

只有至少一个处理同时满足以下条件，才允许进入一次在线视觉适配复验：

1. clean-core macro 相对 N10C 至少 `+0.40pp`；
2. clean-core micro 相对 N10C 至少 `+0.25pp`；
3. trusted macro 不下降，raw micro 不下降超过 `0.10pp`；
4. 保留数最低 50 类的聚合 clean-core accuracy 至少提高 `+3.0pp`；
5. 保留数最高 50 类的聚合 clean-core accuracy 不下降超过 `1.0pp`。

若两种 tau 都通过，以 clean-core macro 较高者进入在线复验；若差异小于 `0.10pp`，优先
选择较弱的 `tau=0.5`。任一全局指标的提升若完全来自 noisy raw accuracy 而 clean-core 与
trusted macro 不支持，则关闭。

在线复验仍以 A2 为 parent，只改变 N3 的训练期 tau，不重用缓存处理后的线性头，不叠加
新标签修正或新 TTA。在线 clean-core 相对 N3 至少 `+0.50pp` 且 trusted/raw 安全指标通过，
才允许执行固定 M3；M3 再相对 N3+M3 至少 `+0.30pp` 才能生成测试候选。

## 合规与泄漏边界

仅使用官方训练图像、官方 OpenAI CLIP ViT-B/32、训练划分的类别计数和单模型损失；测试集
不参与估计先验、选择 tau、训练或门控。Balanced Softmax 是训练期损失修正，不是测试批次
分布适配，也不改变单模型推理要求。团队仓库保持只读。

## 执行结果

两种处理均从空输出目录独立确定性运行；曾有一次外层等待超时留下同配置 CPU 子进程，发现
日志交错后立即终止，仅清理 N11 输出并从空目录重跑。以下结果只来自重跑后的单进程产物。

| 指标 | N10C tau=0 | N11 tau=0.5 | 相对控制 | N11 tau=1.0 | 相对控制 |
|---|---:|---:|---:|---:|---:|
| clean-core macro | 82.5461 | 82.5837 | +0.0376pp | 82.5955 | +0.0494pp |
| clean-core micro | 82.6726 | 82.7142 | +0.0416pp | 82.7280 | +0.0554pp |
| trusted macro | 80.4551 | 80.4909 | +0.0358pp | 80.5016 | +0.0466pp |
| raw micro | 69.4827 | 69.5117 | +0.0291pp | 69.5117 | +0.0291pp |
| 最低保留 50 类 clean-core | 47.9491 | 48.3734 | +0.4243pp | 48.5149 | +0.5658pp |
| 最高保留 50 类 clean-core | 93.9920 | 93.9920 | 0.0000pp | 93.9920 | 0.0000pp |

检查点 SHA-256：

- `tau=0.5`：`d87875423606f5c2686936b887e2dc4e3d9fecdcaedb835511dcedf9e903fd9c`
- `tau=1.0`：`83d48a3725c3e2ef5fd4115257ffbcd819869da30cb3f8121b1a0a6286226cfb`

## 判定

**关闭 N11，不进入在线视觉适配。** 两种 tau 的所有安全指标方向为正，且主要收益确实落在
低保留类，说明类别保留偏差是真实信号；但 clean-core macro/micro 和最低 50 类增益分别远低于
`+0.40pp`、`+0.25pp`、`+3.0pp` 的预注册门槛。不得因为方向正确就跳过门控消耗一次在线训练。

本实验排除的是“在已经收敛且带偏的 A2 线性头上做小幅训练期先验修正”这一弱干预，并没有
排除类别偏差本身。结果更支持下一层因果解释：低保留类缺少的是多样化图像监督与表示学习，
仅在冻结特征头上重新标定梯度不足以恢复。后续若继续该主线，应检验从训练初期进行的可信
类均衡采样/困难类恢复，而不是继续扫描 tau 或做测试期先验校正。
