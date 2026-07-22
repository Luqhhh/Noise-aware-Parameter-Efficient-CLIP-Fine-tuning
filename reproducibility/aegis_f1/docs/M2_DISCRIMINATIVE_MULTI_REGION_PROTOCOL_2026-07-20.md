# M2：伪标签判别性多局部区域 top-k 聚合

日期：2026-07-20

## 研究依据与适配边界

2026-07-15 提交的预印本 *CLIP-Guided Label-Free Discriminative Region Scoring for Fine-Grained Classification*（https://arxiv.org/abs/2607.13437）报告：在 CLIP ViT-B/32 上，多随机局部区域经过全局伪标签引导的判别性评分与 top-k 聚合，可明显优于单一全局特征；低排名随机区域会引入噪声，因此只聚合少量高分区域。

该论文尚未经过正式同行评审，原方法还使用类别文本嵌入和新训练的线性头，不能直接用于本比赛。M2 只采用其“多区域、伪标签、top-k”结构思想：不使用类别名、文本塔、SAM、外部模型或新训练头，所有区域评分与分类都来自同一个 A2 线性分类模型。

## 冻结输入

- A2 exact converted checkpoint SHA-256：`1e2c1a4a274c5e466b716ded41ccf58bebf167a2fadbeba75693d08bdb4f039c`；
- A2 M1 cache SHA-256：`cea5f7651bf0d24828b7da8b252cb47b52288fa2cb7638a1c6d32359c0c92698`；
- validation CSV SHA-256：`607e019165912bb0639efb456b7e8dea122b3e8579a2344dedb8109798921eae`。

## 冻结候选区域

输入始终为 CLIP 原生 224×224 预处理图像，所有局部区域均固定为 160×160 并双线性放大回 224×224。

每张图只生成以下 9 个候选：

1. M1 的最后一个视觉 block、12-head 平均 CLS→patch attention top-5 加权中心裁剪；
2. 8 个固定网格中心裁剪，中心坐标依次为 `(80,80)`、`(80,112)`、`(80,144)`、`(112,80)`、`(112,144)`、`(144,80)`、`(144,112)`、`(144,144)`。

不使用随机种子搜索、不同尺度、不同长宽比或更多区域。

## 冻结评分与融合

1. 原生全局 logits 的 top-1 类别作为该图的伪标签 `y*`；
2. 对每个局部候选 logits `z_i`，计算 soft-negative margin：`s_i = z_i[y*] - Σ_{c≠y*} softmax(z_i[c]) z_i[c]`；负类 softmax 温度固定为 1；
3. 按 `s_i` 选择 9 个候选中的 top-5；
4. 对 top-5 的 `s_i` 再做温度 1 的 softmax，按该权重聚合五个局部候选的类别概率；
5. 聚合局部概率与原生全局概率固定 1:1 平均，取对数作为最终分数；
6. 不扫描候选数、top-k、crop size、坐标、温度或全局/局部权重。

## 实现审计

- M2 global logits 必须与 A2 online center cache 最大绝对差 `0`、预测一致率 `100%`；
- M2 的 attention 候选 logits 必须与 A2 M1 cache 中的 local logits 最大绝对差 `0`、预测一致率 `100%`；
- 任一项失败即判实现无效，不读取效果门控。

## 增量效果门槛

M2 的比较基线是已经通过迁移的 A2 M1，而不是 A2 center：

1. clean-core micro 相对 A2 M1 至少 `+0.15pp`；
2. trusted macro 不得下降超过 `0.05pp`；
3. raw micro 不得下降超过 `0.10pp`；
4. M2 与 M1 至少改变 `1%` 的验证预测，证明多区域选择产生了非平凡作用；
5. 任一门槛失败即关闭 M2：不扫描参数、不运行测试集、不生成提交包。

## 合规性

M2 是单个 OpenAI CLIP ViT-B/32 检查点的确定性多视图推理。测试图像不会参与训练、参数更新、先验估计或方法选择；所有开发选择仅在冻结验证拆分上一次性评估。团队仓库只读，全部实现、缓存和日志位于独立工作树。

## 正式结果

双路径实现审计均通过：

- M2 global vs A2 online center：最大绝对 logit 差 `0.0`，预测一致率 `100%`；
- M2 attention candidate vs A2 M1 local：最大绝对 logit 差 `0.0`，预测一致率 `100%`。

相对 A2 M1 的增量结果：

| 指标 | A2 M1 | M2 | 变化 |
|---|---:|---:|---:|
| clean-core micro | 83.1716% | 83.2548% | **+0.0832pp** |
| clean-core macro | 82.9844% | 83.0962% | +0.1119pp |
| trusted micro | 81.1766% | 81.3462% | +0.1696pp |
| trusted macro | 80.8197% | 81.0212% | +0.2015pp |
| raw micro | 70.1124% | 70.2480% | +0.1356pp |
| raw macro | 70.1100% | 70.2481% | +0.1381pp |

M2 相对 M1 改变 `928 / 10,322` 个预测（`8.9905%`），满足非平凡作用门槛；但 clean-core micro 增益 `+0.0832pp` 未达到预注册的 `+0.15pp`。其余保护门槛通过。

产物哈希：

- M2 validation cache：`b152a7235268b1bc652f74b9bc4d6872199f5d403d8e0dfce162ad9acf1ad870`；
- M2 evaluation JSON：`a340d47c85ffd0d68c2c2eadac312c23a7878d792a794294362cea58da66d816`；
- 测试：`127 passed`。

## 门控结论

M2 的实现有效且方向为正，但未达到预注册的增量 clean-core 门槛，结论为 **CLOSE**。不扫描候选区域数、top-k、crop、坐标、温度或融合权重，不运行测试集，不生成提交包。正式平台候选保持 A2 + M1。
