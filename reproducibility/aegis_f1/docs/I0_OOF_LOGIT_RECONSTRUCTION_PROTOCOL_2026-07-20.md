# I0：完整 OOF logits 严格复建协议

日期：2026-07-20

## 目的

历史三折 OOF 的 91,195×500 完整 logits 已被清理，只留下逐样本 top-1 与若干标量质量信号。下一阶段的结构化标签推断需要完整候选类别分布，因此必须先按原固定协议复建，并证明新轨迹与历史结果一致；不得用全量模型对自身训练样本打分代替折外概率。

## 固定输入

- 三折划分 SHA-256：`eb2409fdd252450cc8750b8e9ceb5ac6f2d6dd1e9c829b81bc3a609f6824c138`
- 冻结特征 SHA-256：`707b45bedba030a3851737427f664291ec83ad2d2b7d20dfd5964b960efbd143`
- 特征路径索引 SHA-256：`c77194e18065a5a133e7484bff74248194d015fda6b1669a3d7ce0dcad9388df`
- 特征标签索引 SHA-256：`634ebeeca5b3e3182a93959a8bbbb890a6a6027a4faecd705042d0d279cebe90`
- 历史 KTA 质量表 SHA-256：`d0f4c0441ee827e455131ffa0c16ef153c3b152d9fc6ae0a82badbdacf9acda2`

所有输入来自官方训练集。团队仓库只读，复建输出只写入 AegisCLIP 独立工作树。

## 冻结训练协议

- 每折训练另外两折，严格固定 50 epochs，不查看本折指标选 epoch；
- OpenAI CLIP ViT-B/32 的 512 维冻结特征，只训练单个 500 类线性头；
- GCE `q=0.5`、AdamW、LR `0.005`、weight decay `1e-4`；
- batch 128、warmup 2 epochs、余弦衰减、梯度裁剪 1.0；
- fold seeds 42/43/44，CUDA AMP 与历史实现一致；
- 输出 float16 logits，但概率与审计统一转为 float32 计算。

## 复建门槛

1. 91,195 个 sample ID 与折号完整且每张仅出现一次；
2. 所有 logits 有限，三折训练均未使用 holdout 选择 epoch；
3. 与历史 `oof_top1` 的逐样本一致率至少 99.5%；
4. 复建 OOF accuracy 与历史 69.4479% 的绝对差不超过 0.10pp；
5. `p_original_label` 与 `p_top1` 的平均绝对误差均不超过 0.01，p99 不超过 0.03；
6. 任一门槛失败则不得进入结构化标签分配，先定位复建漂移。

## 下一门控而非最终假设

I0 只恢复无泄漏的完整证据，不把复建本身视为分数提升。I1 才会以这些 OOF 分布为局部成本，结合 KTA、kNN、原型锚点与训练集类别流量约束做候选标签分配。测试图像、平台反馈和验证集标签均不进入分配器。

## 实测结果

| 项目 | 历史值 | 复建值 | 差异/一致率 |
|---|---:|---:|---:|
| fold 0 accuracy | 69.3246% | 69.3279% | +0.0033pp |
| fold 1 accuracy | 69.7085% | 69.7085% | 0.0000pp |
| fold 2 accuracy | 69.2907% | 69.2907% | 0.0000pp |
| 全局 OOF accuracy | 69.4479% | 69.4490% | +0.0011pp |
| 逐样本 top-1 | — | — | 99.9989%（仅 1/91,195 不同） |

概率复建误差：

- `p_original_label`：MAE `1.48096e-05`，p99 `3.76431e-04`，最大 `0.00368348`；
- `p_top1`：MAE `2.22027e-05`，p99 `6.76408e-04`，最大 `0.00373638`；
- `top1_margin`：MAE `3.26087e-05`，p99 `0.00109200`，最大 `0.00741985`。

全部 91,195 行均被恰好一个折外模型填充，所有 logits 有限，所有输入 SHA-256 与冻结协议一致。门槛 1–5 全部通过，I0 可晋级为 I1 的证据输入。

## 产物

- 完整 OOF logits：`outputs/I0_OOF_LOGIT_RECONSTRUCTION/seed42/oof_logits.pt`
- 完整 OOF logits SHA-256：`b2bb6f72c0f44961cd1665d81443dc5afe58b264702c1c5e9672e55e399440b3`
- 复建审计：`outputs/I0_OOF_LOGIT_RECONSTRUCTION/seed42/reproducibility_audit.json`
- 复建审计 SHA-256：`a7bc4dfc0c2933dfc7db4d598eb68e181681901f4bbaa2701da7f8910289fc7c`
