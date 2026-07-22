# N10：交叉拟合 FINE 几何冲突上限

日期：2026-07-20

## 结构性问题

A2 实际只删除 991/91,195 张训练样本，而 OOF 与原标签冲突达到 30.55%。既有
OOF、prototype 和 kNN 又来自同一 CLIP 表征，I1/K1 已证明直接让它们互相投票会形成
自确认闭环。进一步审计发现，KTA bundle 会把 40,649 个原标签锚点和 601 个严格改标
锚点的 `clean_probability` 同时设为 1，无法表达“对替代标签很可靠”和“原标签可用于
监督”之间的区别。

N10 引入 FINE 的类内主方向几何，但不照搬在本数据上不稳定的二成分 GMM。论文与官方
代码：

- https://proceedings.neurips.cc/paper/2021/hash/ca91c5464e73d3066825362c3093a45f-Abstract.html
- https://github.com/Kthyeon/FINE_official

## 冻结方法

1. 使用 A2 的固定 5 折划分；对每个类别和每个 held-out fold，只用其余四折的官方
   OpenAI CLIP ViT-B/32 冻结特征做 20 次确定性 power iteration，得到首主方向；
2. held-out 样本分数为其归一化特征对该方向投影的平方，保证样本不参与自己的方向拟合；
3. 只有当 OOF prototype 与 probe 的 top-1 相同且共同反对原标签时，才令新的
   `clean_probability = min(old_probability, fine_alignment)`；其余样本完全不变；
4. 不改标签、不启用 KTA correction、不丢弃低分样本、不引入阈值、GMM 或验证集调参。

逐类 GMM 已在任何 N10 训练前作为方法适用性筛查关闭：若逐类强制拟合两个高斯，部分
类别不收敛，clean-core 正误区分 AUC 仅 0.536，低于直接连续几何分数的 0.672。因此
N10 只迁移 FINE 的几何量，不声称复现其完整训练算法。

## 资产审计

- 来源训练样本 91,195，500 类，5 折；
- 双视图同一替代类冲突 36,784；
- 几何上限实际降低 18,856 张（20.6766%）的信任值；
- 全集平均信任值 `0.760677 → 0.721598`；
- 冲突样本平均 FINE 分数 `0.685474`；
- 原标签固定锚点平均仅降低约 0.0081，严格改标锚点平均降低约 0.3041；
- bundle SHA-256：`8182b96fa19c9c79f244c2fc9b41aaed162d19cde964ad03de3de057d9532053`。

## CPU 因果门控

N10C 与 N10F 均从同一 A2 epoch-48 检查点出发，使用同一 90,204 张 A2 kept 数据、
冻结特征、GCE q=0.5、连续权重、LR 5e-5、batch 512 和固定 6 epochs。唯一差异是
N10C 使用原 KTA trust，N10F 使用几何冲突上限 bundle。两者都在 CPU 运行，不与视觉
实验争用 GPU。

N10F 只有同时满足以下条件才允许迁移到 N3 Adapter：

1. clean-core micro 相对 N10C 至少 `+0.30pp`；
2. trusted macro 不低于 N10C，raw micro 不得下降超过 `0.10pp`；
3. 相对未继续训练的 A2 clean-core 至少 `+0.50pp`；
4. 任一失败即关闭，不扫描冲突定义、power iteration、上限函数、LR 或 epoch。

若通过，在线阶段只把 N3 的 trust bundle 换成 N10 bundle，其余 N3 结构和训练变量保持
完全一致；必须相对 N3 center 再提高 clean-core `+0.30pp`，并通过 trusted/raw/flip/
drift 护栏，才允许固定 M3 推理和平台候选。

## 合规边界

只使用官方训练图像的冻结 OpenAI CLIP ViT-B/32 特征与严格 OOF 证据；测试集不参与
几何拟合、冲突定义、训练或选择。最终若晋级仍是单个 ViT-B/32、单个线性头和一组
Adapter 权重。团队仓库与资产只读。

## CPU 门控结果：失败并关闭

两组均确定性完成 6 epochs，初始点严格复现 A2：clean-core micro `82.5755%`、
raw micro `69.4439%`。N10C 最佳为 epoch 4，N10F 最佳也为 epoch 4：

| 指标 | N10C 原 trust | N10F FINE cap | N10F − N10C |
|---|---:|---:|---:|
| clean-core micro | 82.6726% | 82.6587% | -0.0139pp |
| clean-core macro | 82.5461% | 82.5343% | -0.0118pp |
| trusted macro | 80.4551% | 80.4418% | -0.0133pp |
| raw micro | 69.4827% | 69.4633% | -0.0194pp |

N10F 相对 A2 的 clean-core 仅 `+0.0832pp`，未达到 `+0.50pp`；相对严格控制也
没有正收益，更未达到 `+0.30pp`。处理组训练损失约 `0.269`、控制约 `0.278`，说明
权重变化确实进入优化，而不是实现未触发。结论是：FINE 连续几何对风险诊断有效，
但在冻结 CLIP 特征和已收敛 A2 线性头上，单纯降低冲突监督权重不能产生新的判别信息。

按预注册协议关闭 N10：不迁移到 N3、不扫描冲突定义/上限函数/阈值/学习率/轮数，
不生成测试预测或平台提交。
