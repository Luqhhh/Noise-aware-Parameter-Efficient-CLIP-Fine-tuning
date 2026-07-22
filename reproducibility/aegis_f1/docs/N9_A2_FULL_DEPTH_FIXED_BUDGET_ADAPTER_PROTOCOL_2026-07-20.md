# N9：固定参数预算的全深度零初始化 Adapter

日期：2026-07-20

## 要回答的问题

N3 已证明最后 6 个视觉 block 的零初始化并联 MLP adapter 能稳定改善 A2，但仍有
1,044 个 clean-core 样本在 center/M1/M3 三种视图下全部预测错误。细粒度类别不仅依赖
末层语义，也依赖早层纹理、边缘与局部形态。N9 检验：在**不增加 adapter 总容量**的
条件下，把容量分布到全部 12 层，是否比只改最后 6 层形成更有效的任务表征。

零初始化残差 adapter 保持预训练函数的动机同时得到 AdaptFormer 与 2026 AdapterTune
的支持：

- https://proceedings.neurips.cc/paper_files/paper/2022/hash/69e2f49ab0837b71b0e0cb7c555990f8-Abstract-Conference.html
- https://arxiv.org/abs/2603.14706

AdapterTune 尚为预印本，N9 不采用其宣称收益，只采用“固定容量、零初始化、跨层分配”
这一可证伪结构假设。

## 与 N3 的严格配对

- parent 均为 A2 epoch 48；
- N3：最后 6 层、bottleneck 64，adapter 参数 604,032；
- N9：全部 12 层、bottleneck 31，adapter 参数 599,412；这是能使整数 bottleneck
  与 N3 预算最接近且进入 1% 门槛的取值，相对 N3 少 4,620（`-0.765%`）；
- scale 0.1、dropout 0.1、ReLU、零初始化 Up、训练数据、trust、GCE q=0.5、特征
  蒸馏 2.0、batch 64、两轮、学习率与 selector 全部不变；
- 唯一方法变量是相同容量在视觉深度上的分布，不叠加 Prompt、LoRA、标签修正或新 TTA。

## 工程门槛

1. 12 个 adapter 必须位于 block 0--11，原生 CLIP 参数全部冻结；
2. epoch 0 对 A2 的全验证 logits 最大绝对差必须为 0、预测一致率 100%；
3. 真实 batch-64 反向必须证明所有 12 层 Up 获得有限非零梯度，冻结参数无梯度；
4. 参数预算相对 N3 差异不超过 1%，否则不能把结果归因于层分布。

## 效果门槛

N9 只有同时满足以下条件才晋级：

1. clean-core micro 相对 A2 至少 `+0.75pp`，且相对 N3 至少 `+0.30pp`；
2. trusted macro 不低于 N3，raw micro 不得比 N3 低超过 `0.10pp`；
3. flip agreement 不得比 N3 的 `89.07%` 低超过 `0.20pp`；
4. 相对 A2 的 feature drift 不超过 `0.75%`；
5. 任一失败即关闭，不改 bottleneck、层数、scale、LR 或 epoch。

通过训练门槛后才允许执行固定 M3 验证；只有 M3 相对 N3+M3 的 clean-core 至少
`+0.30pp`、trusted macro 不降且 raw 不下降超过 `0.10pp`，才生成一个单 checkpoint
测试候选。

## 合规边界

只使用官方 OpenAI CLIP ViT-B/32、官方训练图像与单个线性头；所有 adapter 都属于
同一模型的 PEFT 参数。测试集不参与训练、选层、选容量或推理门控，团队仓库保持只读。

## 执行结果（2026-07-21）

工程门槛全部通过：12 个 block 的 Up 参数均获得有限非零梯度，冻结参数无梯度；
epoch 0 与 A2 的 10,322 条验证 logits 最大绝对差为 0，argmax 完全一致；adapter 参数为
599,412，相对 N3 少 0.765%。

| 指标 | A2 起点 | N3 | N9 epoch 2 | N9 相对 A2 | N9 相对 N3 |
|---|---:|---:|---:|---:|---:|
| clean-core micro | 82.5755 | 83.1300 | 83.3102 | +0.7347pp | +0.1802pp |
| clean-core macro | 82.4547 | 83.0355 | 83.1802 | +0.7255pp | +0.1447pp |
| trusted macro | 80.3732 | 80.8917 | 81.0079 | +0.6347pp | +0.1161pp |
| raw micro | 69.4536 | 69.8605 | 70.0639 | +0.6103pp | +0.2034pp |
| flip agreement | 87.3668 | 89.0719 | 89.4207 | +2.0539pp | +0.3488pp |
| feature drift | 0.0001 | 0.4365 | 0.5268 | +0.5267pp | +0.0902pp |

最佳检查点为 epoch 2，SHA-256：
`c99fa4a92bd11cfd4db598536695c1c845681b6835b3282e5639b579eb49e5db`。

## 判定

**关闭 N9。** N9 未同时达到 clean-core micro 相对 A2 至少 `+0.75pp`、相对 N3
至少 `+0.30pp` 的预注册主门槛（实际分别为 `+0.7347pp` 与 `+0.1802pp`）。其余安全
指标均通过，但不能用次要指标替代主门槛。因此不执行 M3、不运行测试推理、不生成提交，
也不在结果出现后修改 bottleneck、层数、学习率或训练轮数。

这次消融说明全深度分配确有增益，但相对 N3 的边际收益已经较小；下一步不再围绕同一
adapter 预算做局部搜索，而转向训练数据的类别级确认偏差与均衡监督问题。
