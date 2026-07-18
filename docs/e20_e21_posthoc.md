# E20/E21：严格单模型后处理实验（2026-07-18）

## 结论先行

- **E20 平台分数为 60.1794%**。这是有效、可复现的单模型提交，但比团队当前最佳
  `S_MIXUP_CE5 + Flip TTA = 60.4758%` 低 **0.2964 个百分点**，因此不替换主提交。
- **E21 平台分数为 60.2195%**。它比 E20 的 60.1794% 高 **0.0401 个百分点**，但仍比团队当前最佳
  `S_MIXUP_CE5 + Flip TTA = 60.4758%` 低 **0.2563 个百分点**，因此同样保留但不替换主提交。
  E21 与 E20 的测试集预测有 1,660/24,967（6.649%）不同，说明策略确有互补性，但本次增益有限。
- 两个实验均不使用外部数据、测试集拟合、多模型投票或测试时训练。部署产物均为一个
  CLIP ViT-B/32 骨干、一个分类头和一个检查点；TTA 仅使用同一检查点的原图与水平翻转。

## E20 策略

E20 以 `E2_MIXUP_CE5_REPLICA` 为父实验：冻结 OpenAI CLIP ViT-B/32，只训练线性头；
前 5 个 epoch 使用交叉熵，随后使用 GCE `q=0.5`，并采用 pixel MixUp
`alpha=0.2, probability=0.2`。

后处理分三步：

1. **严格线性头权重汤**：逐张量验证 epoch 35 与 epoch 44 检查点除
   `classifier.weight/bias` 外完全一致，然后计算
   `0.6 × head_epoch35 + 0.4 × head_epoch44`。该操作等价于同一骨干上两个线性头
   logits 的固定插值，但最终仍保存为一个线性头、一个检查点。
2. **每类 7 个视觉原型**：只在开发训练集上，将原图与水平翻转的冻结 CLIP 特征取均值，
   使用确定性 K-means 拟合每类 7 个单位向量原型；不使用置信度筛样（uniform weight），
   每类取与查询最相似的原型。最终分数为
   `linear_logits + 0.4 × 58.307098 × prototype_similarity`。
3. **概率级 Flip TTA**：同一检查点分别推理原图和水平翻转图，温度 `T=1`，对两者
   softmax 概率取均值后输出 Top-1。

严格本地在线验证 raw micro 从权重汤的 70.9190% 提升到最终 71.1710%；平台为
60.1794%。本地—平台差距再次说明本地含噪验证分数不能直接预测隐藏测试集排名。

## E21 策略

E21 从 E2 epoch 44 初始化，在全部 103,218 张官方训练图上以 head LR `5e-4`
低学习率续训 3 个 epoch。随后使用全部训练图拟合每类 7 个原型，并沿用 E20 的
`max` 原型聚合、残差系数、尺度和 `T=1` 概率级水平翻转 TTA。

关键区别是：E21 的原验证集已经进入全量续训，因此训练内 70.8123% 只是健康检查，
**不是独立验证分数，也不能用于调参或泛化声明**。E21 只能由平台分数判断是否有效。

最终平台结果为 **60.2195%**：相对 E20 提升 0.0401 个百分点，但相对团队最佳仍低 0.2563 个百分点，
因此判定为“有效但不晋级”。

## 产物与指纹

| 产物 | SHA-256 |
|---|---|
| E20 epoch 35 源检查点 | `aa5e6098078bfd7626e06dbcf1c8b9e57b8348bf03072af4cedf3d1444fc3a34` |
| E20 epoch 44 源检查点 | `50e05d09921a0f9bf852589cae848b926c61892e6952f1082dfa25daae2e3ff6` |
| E20 严格权重汤 | `1a38694a1eb788b0a60dd83c6e262d52074c7f4a17c8f3d8937fc8bf44466f2d` |
| E20 最终检查点 | `4ef590196d52dfcfe13016d2f72ac72cc81114575596edffc8a1945931a2ffa6` |
| E20 `pred_results.csv` | `f42bc99ec46b58302a05e9eb3b5d2f4823e785f2a5c207c3f3e7e64e35890b42` |
| E20 `submission.zip` | `37e524ef7aae81880b825595a26c90d272bb16753eac20222bf08349a5771a97` |
| E21 全量续训源检查点 | `d5c7e215a6993342c6b9eb6bf679b44f1e890392096c2a75f85ea6894f6dfad0` |
| E21 最终检查点 | `eda1501455d44bd9327ffad934f5f4e9f522e303d11420a39160ea0e547835e8` |
| E21 `pred_results.csv` | `673cefc0315ab6859eda1ee77646861d2713939e886c90769b6654aecef05334` |
| E21 `submission.zip` | `52a4ed874a745c818790de09b1b287c9845c36f13c402523e46e9662ffc97b5c` |

两个提交均为 24,967 条预测、500 个预测类别、0 张损坏图像，ZIP 根目录只包含
`pred_results.csv`，并已通过允许 TTA 的提交审计。

## 团队工程中的复现入口

E20 配方：`configs/e20_posthoc.yaml`

E21 配方：`configs/e21_posthoc_fulltrain.yaml`

```bash
python -m experiments.posthoc.build --config configs/e20_posthoc.yaml
python -m experiments.posthoc.infer --config configs/e20_posthoc.yaml

python -m experiments.posthoc.build --config configs/e21_posthoc_fulltrain.yaml
python -m experiments.posthoc.infer --config configs/e21_posthoc_fulltrain.yaml
```

构建器会在插值前强制验证非分类器权重完全相同，并验证原图/翻转特征缓存逐样本对齐；
任一条件不满足都会中止。配置中引用的不可入库大型检查点位于同级独立工程
`../AegisCLIP-Noise-Robust/`，其 SHA-256 作为跨工程交接依据。

## 平台决策规则

1. E20 已判定为“可用但不晋级”：保留方法与结果，不替换 60.4758% 主提交。
2. E21 已判定为“有效但不晋级”：平台 60.2195%，比 E20 高 0.0401 个百分点，但仍低于团队最佳 0.2563 个百分点。
3. E20/E21 均保留方法与结果，后续优先验证与它们有更大预测差异、且严格本地门禁更强的候选；
   不得根据训练内分数选择全量训练超参数。
