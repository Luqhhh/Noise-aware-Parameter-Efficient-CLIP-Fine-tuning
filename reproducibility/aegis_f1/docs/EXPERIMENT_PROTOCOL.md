# 实验、晋级与停止协议

## 固定资产

所有主实验固定以下资产：

- OpenAI CLIP ViT-B/32；
- `artifacts/stages/<stage>/seed42` 中由代码生成的训练/验证划分；
- 同目录的 `class_to_idx.json` 和内容 SHA-256 分组；
- 同一份冻结 CLIP 特征缓存；
- 训练随机种子 42，复核种子 17 和 73；
- 官方测试集只在最终单模型推理时读取。

任何资产哈希变化都视为新实验族，不与原结果直接合并。

冻结模型允许对未触发 MixUp 的 batch 复用同一 OpenAI CLIP 缓存特征；一旦发生像素 MixUp，必须走真实视觉前向。任何 PEFT batch 都不得走缓存捷径。

## 阶段 0：工程门禁

必须全部通过：

```bash
bash scripts/run_stage.sh test
bash scripts/run_stage.sh prepare
DEVICE=cuda bash scripts/run_stage.sh cache
bash scripts/run_stage.sh audit
```

检查内容包括配置闭合性、官方样本数量、特征覆盖、内容组隔离、训练/验证交叉、MixUp 对齐、类别修正上限和提交原子性。

## 阶段 1：可信度资产

`prepare` 已按图片内容 SHA-256 分组。随后分别对训练与验证做五折 OOF：

```bash
DEVICE=cuda bash scripts/run_stage.sh trust
```

必须检查：

- 每个样本恰有一个 OOF 折；
- 内容重复样本不能跨 OOF 折；
- `correction_rate` 不超过每类 20% 的硬上限；
- 原型/探针一致率和各自 noisy-label accuracy 没有异常坍塌；
- 可信概率不是几乎全 0 或全 1。

如果统计异常，先调整可信度检测，不启动 50 epoch 主训练。

## 阶段 2：单种子消融

顺序固定为 A0 → A1 → A2 → A3。

建议的晋级门槛：

- A1 相对 A0：`proxy_macro` 至少提高 0.20 个百分点，且 `raw_micro` 不下降超过 0.15 个百分点；
- A2 相对 A1：selector 至少提高 0.10 个百分点，否则保留 A1；
- A3 相对最佳 frozen 模型：selector 至少提高 0.10 个百分点，且平均特征漂移不超过 0.5%；
- 任一方法触发早停且最佳点早于第 3 epoch，优先视为不稳定而不是“高效收敛”。

阈值是实验决策规则，不是统计显著性结论。

## 阶段 3：多种子复核

只复核单种子排名前两名。复制配置并将 seed 改为 17、73，不改变其他超参数。比较：

- selector 均值；
- raw micro / macro；
- trusted 与 proxy macro；
- 最佳 epoch 方差；
- A3 的特征漂移；
- 不同种子是否给出一致的模型排序。

若平均优势小于种子间标准差，不宣称方法稳定优于。

## 阶段 4：提交

每个候选生成独立提交目录，禁止覆盖：

- 主提交：最佳多种子判断对应的单模型裸推理；
- 不做跨检查点集成、融合或投票；
- 上传前核对 `manifest.json` 的检查点和提交哈希。

## 后续长尾赛段

复活赛和半决赛必须先更新 stage、类别数、官方训练/测试数量，再完整重建划分、缓存和 OOF 可信度，绝不沿用初赛资产。先报告训练集最大/最小类频比：

- 频比小于 3：保持 `class_prior_adjustment_tau: 0.0`；
- 频比达到 3：单独比较 `tau ∈ {0.0, 0.5, 1.0}`；
- 只在 macro 代理指标和多种子结果共同改善时启用；
- 类先验校正只作用于训练损失，推理仍使用单模型原始 logits。

## 停止条件

满足任一条件即停止扩展方法：

- 连续两个新增机制未越过晋级门槛；
- 验证提升只来自 raw noisy-label accuracy，而可信代理指标下降；
- A3 漂移超过预算且线上没有可靠收益证据；
- 新方法需要使用测试标签、测试分布自适应或其他赛规不明确的信息。

停止后把算力用于复核最佳候选，而不是继续堆叠模块。
