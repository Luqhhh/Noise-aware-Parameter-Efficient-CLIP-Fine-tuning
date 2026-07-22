# AegisCLIP-Noise-Robust

面向噪声标签细粒度图像识别比赛的独立工程。目标不是堆叠论文名词，而是在赛规允许的 OpenAI CLIP ViT-B/32 上建立一条可复现、可消融、可回退的高分路线。

团队仓库内的可复现快照位置：

```text
reproducibility/aegis_f1
```

独立开发来源为本机 `AegisCLIP-Noise-Robust` / `AegisCLIP-F6-A2LoRA`；本目录已增量整合到来源提交 `0e06f0a`。只把团队项目目录中的 `train/` 与 `test/` 当作本机官方数据位置；划分、特征缓存、噪声估计、训练、检查点、审计与提交产物均由本工程生成。数据、缓存、检查点、预测 CSV 与提交 ZIP 不进入 Git。

## 方法：Cross-Fitted Visual Trust

赛题类别名只有 `0000`–`0499`，不存在可用的类别语义。AegisCLIP 因而不把数字标签伪装成文本提示，而是让两个互补的视觉证据源交叉审核每个样本：

1. 冻结 CLIP 特征上的折外类别原型，保留预训练几何；
2. 只在其他折训练的线性探针，提供任务适配证据；
3. 类内 2-GMM 把双视图支持度转为连续可信概率；
4. 只有两个视图高置信同意另一个类别时才进行有上限的软修正；
5. 对“原型认为容易、后期探针却拒绝”的疑似错标容易样本额外降权。

训练阶段依次验证可信加权、CE→GCE、MixUp、动态损失尾部抑制，以及 LayerNorm-only PEFT、冻结特征蒸馏和可信梯度投影。每个新增机制都有独立配置，失败时可回退到上一阶段。

冻结视觉编码器时采用严格等价的混合执行：未触发像素 MixUp 的 batch 复用已审计的冻结特征，触发 MixUp 的 batch 仍执行真实图像前向；PEFT 阶段始终执行图像前向。训练日志记录缓存前向占比，避免计算优化悄悄改变方法。

## 赛规边界

- 只允许官方 OpenAI CLIP ViT-B/32；
- 不使用外部数据；
- 测试图像只参与最终推理；
- 不依赖人工删除清单或人工改标签；
- 一个提交只绑定一个模型检查点，不做跨模型融合或投票；
- `pred_results.csv` 严格输出 `filename,0001`，无前导空格；
- 任何图片缺失、解码失败、数量不符或 ZIP 内容不一致都会停止发布。

完整对应关系见 [赛规合规矩阵](docs/COMPETITION_COMPLIANCE.md)。

## 已完成的工程验证

- 独立 `.venv` 已按锁定依赖创建；
- 180 项隔离自动测试通过；
- 官方训练数据审计：103,218 张、500 类；
- 内容哈希：101,980 个唯一组，1,238 个重复样本被组级隔离；
- seed 42 划分：92,902 训练、10,316 验证；
- 新工程重建的 train/val CSV 与既有全量划分逐字节一致；
- 新工程独立重建 103,218 × 512 的官方 CLIP 冻结特征；
- 特征路径索引与 103,218 张官方训练图像完全一致；
- OOF 可信度、快速闭环和完整消融结果只在真实运行后记录，不把本地分数或未评测候选冒充平台结果；
- 当前已确认平台最佳为 **F1 + M1：63.3276%**，A2 + M1 为 62.6747%，A2 + M3 为 62.0259%。

## 从零复现

```bash
cd reproducibility/aegis_f1

uv sync --extra dev --locked

bash scripts/run_stage.sh test
bash scripts/run_stage.sh prepare
DEVICE=cuda bash scripts/run_stage.sh cache
DEVICE=cuda bash scripts/run_stage.sh trust
bash scripts/run_stage.sh audit
bash scripts/run_stage.sh smoke
```

`prepare` 和 `cache` 只读取官方训练根目录。更换赛段时，必须更新类别数和官方样本数，然后完整重建这两项资产，禁止复用初赛可信度。

## 主实验顺序

```bash
bash scripts/run_stage.sh a0
bash scripts/run_stage.sh a1
bash scripts/run_stage.sh a2
bash scripts/run_stage.sh a3
```

| 阶段 | 相对上一阶段唯一新增因素 | 决策用途 |
|---|---|---|
| A0 | 全量官方数据上的冻结 CLIP 分类头锚点 | 建立可比较基线 |
| A1 | CVT 可信加权与双视图共识软修正 | 验证噪声估计本身 |
| A2 | 平滑动态损失上限 | 抑制晚期极端噪声梯度 |
| A3 | LN-only PEFT、蒸馏、可信梯度投影 | 谨慎适配视觉编码器 |

先用 seed 42 排序，只对前两名补跑 seed 17 和 73。详细晋级与停止条件见 [实验协议](docs/EXPERIMENT_PROTOCOL.md)。

后续长尾赛段可单独消融 `class_prior_adjustment_tau ∈ {0.0, 0.5, 1.0}`。该校正只进入训练损失，推理仍使用原始 logits；初赛默认值为 `0.0`，未通过消融不得开启。

## 单模型推理与提交

```bash
.venv/bin/python -m aegis_clip.cli.infer \
  --checkpoint outputs/A1_CVT_SOFT/seed42/checkpoints/best.pt \
  --output-dir outputs/submissions/a1_bare
```

输出目录包含：

- `pred_results.csv`；
- 只含该 CSV 的 `submission.zip`；
- 记录检查点、CSV、ZIP SHA-256 和完整性检查的 `manifest.json`。

上传前再次执行只读审计：

```bash
.venv/bin/python -m aegis_clip.cli.audit_submission \
  --config configs/a1_cvt_soft.yaml \
  --submission-dir outputs/submissions/a1_bare
```

## 文档

- [赛规合规矩阵](docs/COMPETITION_COMPLIANCE.md)
- [研究依据与设计取舍](docs/RESEARCH.md)
- [实验、晋级与停止协议](docs/EXPERIMENT_PROTOCOL.md)
- [R1：F1+M1 Part-Token 残差预注册协议](docs/R1_F1_M1_PART_TOKEN_RESIDUAL_PROTOCOL_2026-07-22.md)
- [已执行验证记录](docs/VERIFICATION.md)
- [平台结果与待评测顺序](docs/PLATFORM_RESULTS_2026-07-22.md)
- [来源与增量合并说明](PROVENANCE.md)
- [团队根目录独立实验总账](../../docs/aegis_independent_experiments_2026-07-22.md)

本工程提高的是获得高分的概率与实验效率，不虚构未运行的榜单成绩。每个真实结果必须绑定配置、数据划分、特征缓存、可信度资产、有效参数清单与检查点哈希。
