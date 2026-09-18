# 初赛 75 分冲刺 v5：抽样概率融合监督

版本：2026-09-17

计划 ID：`PRELIM75_V5_20260917`

核对远端：`01658a23e7e031affa964eaa09b676bed6dae2a6`

## 状态

本文主体保留 2026-09-17 开工时的固定方案语义。实际执行已于 2026-09-18 闭环：F0/F1 平台成绩分别为 **69.0752% / 69.1393%**，F1 成为当前无 prior 胜者；v5 已关闭，75% 未达到。实际命令、指标、哈希和停止决定见 [执行记录](../results/prelim75_v5_execution_20260918.md) 与 [最终记录](../results/prelim75_v5_final_20260918.json)，不要再把下文的开工前措辞理解为当前待办。

## 固定候选

两组均从 v4 C0 独立加载完整 visual、shared head、O3 和 PTA：

- F0：global/local 分支分别使用温度 1.5 的 GCE。
- F1：50% 分支独立 GCE + 50% global/local 概率融合 GCE。

两组各 3 epoch，seed 42，有效 batch 32，固定最后一轮；sample epoch 为 10/11/12。F1 不从 F0 继续训练。本轮不新增教师缓存、伪标签、可信样本筛选或完整十视图训练，不扫描温度、融合比例、学习率、epoch 或 prior。

## 损失定义

对原 500 维软目标 `y` 和样本权重 `w`，固定：

```text
T = 1.5
p_g = softmax(z_g / T)
p_l = softmax(z_l / T)
p_mix = 0.6 * p_g + 0.4 * p_l

GCE(p, y) = sum_c y_c * (1 - max(p_c, 1e-7)^0.5) / 0.5
ell_sep = 0.6 * GCE(p_g, y) + 0.4 * GCE(p_l, y)
ell_mix = GCE(p_mix, y)

F0: ell = ell_sep
F1: ell = 0.5 * ell_sep + 0.5 * ell_mix
```

完整有效 batch 使用共同分母 `max(sum(w), 1e-8)`。原 global 同像素官方 CLIP anchor 保持系数 2.0，并按实际 batch 图像数归一；local 不增加整图 anchor。零分类权重样本仍只保留 anchor。

实现必须融合概率而不是 logits，不得在 `p_mix` 上再次 softmax；两个实时学生分支都保留梯度。F1 融合项本身必须对 global/local logits、visual projection、shared head、O3 和 PTA 产生有限非零梯度。

F1 的损失按定义通常可低于 F0，因此训练 loss 不作为晋级证据。执行记录分别报告 global、local、独立分支、融合和总分类项。

## 固定训练与推理

| 项目 | 设置 |
|---|---|
| 权重父模型 | v4 C0 |
| Backbone LR / WD | `1e-6 / 0` |
| Shared head LR / WD | `1.5e-7 / 1e-4` |
| O3/PTA LR / WD | `3e-6 / 0` |
| Optimizer | fresh AdamW |
| Scheduler | fresh cosine，18 epoch horizon，只运行 3 epoch |
| GCE | `q=0.5`, `epsilon=1e-7` |
| Global/local | `0.6 / 0.4` |
| Anchor | 同像素 global，系数 2.0 |
| 冻结范围 | 继承 C0；conv1、位置编码继续冻结 |
| 精度 | FP32，沿用确定性设置 |
| 最终推理 | 原生 224、112/128/144/160、Flip、T=1.5、无 prior |

训练只抽样一个方向和一个 local 尺度，尺度概率为 0.2/0.3/0.4/0.1，继续使用已绑定的冻结 V1 boxes。它不是完整十视图损失的无偏估计：有限 hash epoch 不保证恰好实现期望，且 `E[L(p)]` 一般不等于 `L(E[p])`。最终推理由候选自身重新计算 attention，不读取训练 boxes 或第二模型。

## 资产绑定

- C0 checkpoint：`outputs/prelim75_v4_20260916/C0/candidate.pt`

  SHA-256：`48d4f4ccec8758f93b126e059790107981c2359a94963d5d0147d49f05613f8a`
- C0 fallback ZIP：`b85d680781407bcf90b9b8b37bc1540387c42296d4e1ba97b78a10720d7e27a0`
- C0 diagnostic：`458a043a73ace962328c8420099bac7b9f15691a98e5e2359190b714c4e9e2cf`
- V1 geometry manifest：`93e35940e19bbc371a30e67f0c12a429ecd48e88ca784be0983d4c8eacfc86e4`
- V1 paths：`3a9bbfe4e3049aff1f081962c798be713dd17bef354612ad86321327dcf3c50c`
- V1 boxes：`9041665389e51b8da6af21b457a6d8a69eb365495d17558e9a5dedb4a7ed2ff1`
- 原监督：`7fc6dcf0efb87e0a402114a7ff7ef9421e7f1c8fedcc06db10ba6d6485c606c4`

权重父模型与几何来源分别绑定，不能因二者不同而删除身份检查。训练不读取 v3 teacher probabilities、recovery weights/mask 或 v4 trusted-CE mask。

## 执行与停止

```bash
PYTHONPATH=reproducibility/aegis_f1 \
  python3 -u scripts/run_prelim75_v5_queue.py \
  --config configs/prelim75_v5.yaml --execute
```

队列默认只读，显式 `--execute` 才运行。先做最重 F1 路径 smoke；若 batch32 首次无更新 OOM，则两组统一使用 16×2。GPU 总预算上限为 28,800 秒，包含 smoke、训练、诊断、推理和中断尝试。

固定末轮重叠诊断只防止 raw 或 clean-core 相对 C0 下降超过 2.0pp 的明显失效，不选择 epoch。通过后每组生成一个单 checkpoint、无 prior 的 24,967 行 CSV/ZIP，并检查四位标签、全覆盖、无重复和 ZIP 内外字节一致。

平台直接基线为 C0 68.4544%。F0/F1 均不超过 C0 时保留 C0；F0 最好时不宣称融合有效；F1 同时超过 C0/F0 时保留 F1。F1 比二者较高者至少高 0.30pp 仅记为达到投入门槛，不自动启动 F2。本轮结束后不自动追加训练、参数扫描、上传或 push。

历史 prior0.90 的 70.352866% 属于不同 checkpoint/校准协议，不能迁移或加到新候选上。当前正式候选不使用测试批先验拟合。
