# N12：主动遗忘已记忆噪声，而不是再次猜标签

日期：2026-07-21

## 大方向

A2 已训练到 epoch 48。此前 N10 只改变可疑样本权重，I1 直接把同源 OOF/prototype/kNN
投票变成新标签；两者都没有改善，说明“继续相信同一表示给出的伪标签”容易形成自确认闭环。
CVPR 2026 FINE 提出了不同干预：对被筛选为噪声的原标签施加反向交叉熵，主动降低模型已经
记住的错误关联；再以随机互补标签负学习抑制后续噪声吸收，而不推断真实类别。

官方论文：

- https://openaccess.thecvf.com/content/CVPR2026/html/Sheng_Revisiting_Learning_with_Noisy_Labels_Active_Forgetting_and_Noise_Suppression_CVPR_2026_paper.html

本实现遵循论文 Eq. (2)--(5)：AFMU 为 `log p(y_noisy|x)`，NSNL 为单个非原标签上的
`-log(1-p_complement)`；采用论文真实噪声实验默认权重 `beta=0.001`、`gamma=0.1`。
互补标签用 sample index 与 epoch 的确定性整数哈希选择，保证恢复训练与审计可复现，且永不
等于原标签。

## 固定噪声队列

不在线拟合 GMM，也不使用测试数据。固定训练 trust 中 `clean_probability <= 0.0500001` 的
14,362 张为可疑队列；阈值对应 KTA bundle 已有的最低置信度地板，不在结果后扫描。A2 当前
仍以约 0.2875 的样本权重训练这些记录，因此 N12 同时回答两个问题：

- N12C：从基础 GCE 中完全去掉该队列，仅作为硬丢弃控制；
- N12F：基础 GCE 与 N12C 完全相同，并只对同一队列增加 AFMU+NSNL。

两组均冻结 CLIP 特征，从同一 A2 epoch 48 初始化，使用相同 90,204 条训练表、原 KTA trust、
GCE q=0.5、batch 512、6 epoch、AdamW、学习率与 clean-core selector。两份配置除 experiment id
和 `active_forgetting.enabled` 外必须逐字段相同。

## 工程门槛

1. 164 项现有测试全部通过；
2. N12F 每轮可疑样本计数必须恰为 14,362，AFMU/NSNL 均有限；
3. 第一步分类头梯度与参数更新必须非零，冻结视觉参数不得变化；
4. N12C/N12F epoch 0 logits 必须完全一致并复现 A2 起点；
5. 不允许 MixUp、伪标签修正、类先验校正、adapter、LoRA 或新 TTA 与本实验叠加。

## 效果门槛

主动遗忘路线只有在 N12F 相对 N12C 同时满足以下条件时晋级：

1. clean-core micro 与 macro 均至少 `+0.30pp`；
2. trusted macro 至少 `+0.20pp`；
3. raw micro 不下降超过 `0.20pp`。

硬丢弃路线独立判断：N12C 相对既有 N10C 的 clean-core micro/macro 均至少 `+0.40pp`，
trusted macro 至少 `+0.30pp`，raw micro 不下降超过 `0.20pp`，才允许晋级。

任一路线通过后，才允许以 N3 的最后 6 层 adapter 做一次两轮在线复验，且只加入获胜处理。
在线 clean-core 相对 N3 至少 `+0.50pp`、trusted macro 不降、raw micro 不下降超过 `0.10pp`，
才执行固定 M3；M3 再相对 N3+M3 至少 `+0.30pp` 才生成测试候选。失败后不扫描 beta、gamma、
阈值或遗忘轮数。

## 合规边界

只使用官方训练图像、官方 OpenAI CLIP ViT-B/32、训练划分的交叉拟合 trust 和一个最终模型。
测试集不参与噪声选择、损失、超参数或门控。AFMU/NSNL 只存在于训练期，推理仍为一个 CLIP
ViT-B/32 PEFT 模型及单线性头；团队仓库保持只读。

## 执行结果

工程门槛全部通过。N12C/N12F 的 epoch 0 指标一致；N12F 六轮均恰好处理 14,362 个固定
可疑样本，AFMU 约为 `-4.316` 至 `-4.325`，NSNL 约为 `0.00125` 至 `0.00179`，全部有限。

| 指标 | N10C 原控制 | N12C 硬丢弃 | N12F 主动遗忘 | N12F 相对 N12C |
|---|---:|---:|---:|---:|
| clean-core micro | 82.6726 | 82.6726 | 82.6864 | +0.0139pp |
| clean-core macro | 82.5461 | 82.5461 | 82.5601 | +0.0140pp |
| trusted macro | 80.4551 | 80.4544 | 80.4636 | +0.0091pp |
| raw micro | 69.4827 | 69.4730 | 69.4827 | +0.0097pp |

检查点 SHA-256：

- N12C：`f99dd13b9fab3335681b2a5cd061a12e56d1da54f26dd4b8632606201a6a6d12`
- N12F：`5f1109e174630eb903a1d01ccb71ae260c4502a66b6fdddc8f18d2e4d82e1eef`

## 判定

**关闭 N12，不进入在线视觉复验。** N12C 相对 N10C 的两个 clean-core 指标完全相同，硬
丢弃路线未通过；N12F 相对 N12C 虽然所有方向均为正，但 clean-core 与 trusted 增益只有约
`0.01pp`，远低于 `+0.30pp/+0.20pp` 门槛。不得扫描 beta、gamma、阈值或遗忘轮数。

本结果只否定“在 A2 epoch 48 后对冻结线性头做短程遗忘”。FINE 的论文机制依赖从 warmup
结束后持续阻止噪声记忆；A2 已经收敛后，视觉表示中的错误关联无法由极小的线性头更新撤销。
因此后续若继续主动遗忘，只能作为从训练早期开始的完整单模型流程，而不能继续作为 A2 后处理。
