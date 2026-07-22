# H1：噪声容忍对比表征成对门控

日期：2026-07-20

## 动机

平台 61.2128% 与领先结果约 90% 的差距不能由线性头微调解释。团队 OOF 显示 30.6% 的
训练标签与跨折预测冲突，而 SNSCL 在细粒度噪声标签基准上报告了方法级收益：其核心不是
继续调 robust loss，而是让可信同类表征可分，同时阻止不可信标签污染对比队列。

本项目先做一个便宜但严格的因果门控，不声称复现完整 SNSCL。H1 将论文思想适配到官方
CLIP 冻结缓存：每个样本生成一个小幅随机特征视图；所有样本只把自身另一视图视为正对；
只有 `clean_probability>=0.70` 的样本才额外使用同标签正对。低可信样本因此参与无标签
一致性学习，但不会传播其类别标签。

## 成对协议

- H0：A2 epoch48 初始化 + 128 维近恒等残差 feature adapter + GCE；
- H1：与 H0 完全相同，仅新增噪声容忍对比损失；
- 数据：F6 adapt 9,135；evaluation 9,315，且路径/内容组隔离；
- 分类监督：固定 OOF clean threshold 0.70，共 6,492 张，500/500 类；
- 对比参数直接冻结：weight=1、temperature=0.1、feature noise std=0.01；
- 30 epochs、batch256、LR2e-4；同一初始化、随机种子、调度与 clean-core selector；
- H0/H1 均使用全 FP32。两次 H1 AMP 尝试均在首次参数更新前因适配器梯度溢出而被安全审计终止，未形成实验结果；为保持唯一变量仍是对比目标，H0 也按 FP32 重跑；
- 无测试数据、外部数据、平台反馈或超参数扫描。

## 预注册判据

1. 先运行 H0，随后 H1；
2. H1 best clean-core micro 必须比 H0 至少高 **1.0pp**，trusted macro 不得下降；
3. mean feature drift 必须不超过 1%；
4. 未同时通过则终止，不调 temperature、noise std 或 loss weight；
5. 通过后才实现在线双视图 LoRA 和训练集全量版本，缓存门控本身不生成平台包。

## 反方审查

缓存特征扰动不等同于像素增强，feature adapter 也不能证明视觉 LoRA 必然受益。因此本轮只
检验“对比几何是否提供显著增量”。即使通过，也必须由在线视觉实验再次验证。若只提高
0.1–0.5pp，按当前平台差距仍视为失败。

## 结果（冻结协议）

- H0 初始 A2：clean-core micro 82.9321%，trusted macro 80.6099%，feature drift 0%；
- H0 学习后最佳（epoch 1）：clean-core micro 82.7006%，trusted macro 80.3284%，raw micro 69.1358%，feature drift 0.4152%；
- H1 学习后最佳（epoch 1）：clean-core micro 81.5741%，trusted macro 78.9467%，raw micro 67.7939%，feature drift 12.7922%；
- H1 相对 H0 最佳为 **-1.1265pp**，相对未训练 A2 为 **-1.3580pp**；trusted macro 同时下降 1.3818pp，且漂移超过 1% 门槛 12.79 倍；
- H0 checkpoint SHA-256：`3856e88fc1250e63045a920017b9c78e04e8e97f147826e63e94ca0bd6083d67`；
- H1 checkpoint SHA-256：`1c39b4c7ac932896546d248f86b24247cd774b13c4e6dccf87207ee7d47e453f`。

结论：H1 严格失败并关闭，不做 loss weight、temperature 或 noise std 扫描，不生成平台提交包。缓存特征上的该对比目标快速破坏 A2 判别几何；它不支持继续投入在线 LoRA 版本。
