# O1：AdaptFormer × Pixel MixUp 严格交互门控

日期：2026-07-21

## 核心问题

当前平台上可重复的正信号主要来自 `MixUp(alpha=0.2, p=0.2)`、极低置信样本剔除和水平翻转 TTA。N3 则证明 6 个末层、零初始化的 AdaptFormer 分支能在同一 A2 检查点上显著改善干净核与可信验证指标，但 N3 没有使用 MixUp。O1 只改变这一项，检验训练正则化和表示适配是否互补。

该实验不声称能由本地指标估计平台分数。它首先是一个严格因果交互门控；只有通过安全门槛才生成测试提交。

## 冻结配对

- 父检查点：团队只读 A2 `NR_CL_KNN_DROP` epoch 48/best，SHA-256 `74ad2856e4449a42397edbda599ae79e8a4c6a6fa923624ef4e91a35e20a2a4c`；
- 对照：N3 `N3_A2_ADAPTFORMER_GATE`；
- 唯一训练差异：N3 的 `mixup_probability=0` 改为 O1 的像素级 `mixup_alpha=0.2, mixup_probability=0.2`；
- 其余数据、A2/KTA 信任包、弱 RRC+翻转、6 个末层、瓶颈 64、scale 0.1、dropout 0.1、GCE q=0.5、特征蒸馏权重 2.0、学习率、batch、seed 和 2 epoch 完全一致；
- 官方测试集仅可在门控通过后用于一次候选推理，不参与训练、选择或统计拟合。

## 预注册门槛

N3 的冻结参照为 clean-core micro `83.1300%`、trusted macro `80.8917%`、raw micro `69.8605%`、flip agreement `89.0719%`、mean drift `0.4365%`。

O1 进入互补 M3 推理的必要条件：

1. clean-core micro 不低于 `82.7800%`（相对 N3 最多回退 0.35pp）；
2. trusted macro 不低于 `80.6000%`；
3. raw micro 不低于 `69.3600%`；
4. mean feature drift 不超过 `0.75%`，且所有训练损失、梯度与预测均有限；
5. 配置审计确认除 MixUp 外不存在未登记差异。

若安全门槛通过，即使 clean-core 未超过 N3，也允许生成一个 O1+M3 单模型提交候选，因为已有平台记录表明 MixUp 的本地排序与平台排序并不单调。若任一安全条件失败，则关闭 O1，不做测试推理、不扫描 MixUp 参数。

## 合规边界

O1 仍为单个 OpenAI CLIP ViT-B/32、单个线性分类头和该模型内部的参数高效适配分支；MixUp 仅使用官方训练图像。没有外部数据、测试时训练、多模型融合、投票或集成。若晋级，M3 仍是同一模型的固定单一推理流程。

## 结果

O1 从与 N3 完全相同的 A2 初始点完成两轮训练，初始评估逐项复现。最佳检查点为 epoch 2：

| 指标 | N3 epoch 2 | O1 epoch 2 | O1−N3 |
|---|---:|---:|---:|
| clean-core micro | 83.1300% | **83.1577%** | +0.0277pp |
| trusted macro | 80.8917% | **80.9381%** | +0.0464pp |
| raw micro | 69.8605% | **69.9186%** | +0.0581pp |
| flip agreement | **89.0719%** | 88.6456% | -0.4263pp |
| mean feature drift | 0.4365% | 0.4650% | +0.0285pp |

全部安全门槛通过，损失、梯度和预测均有限。O1 最佳检查点 SHA-256 为 `14a60190d329bd51eda3e5f7409b1022d0d2d3c1ac4da6df7cdabe9748f8382b`。

固定 M3 验证相对 O1 center：clean-core `+0.8040pp`、trusted macro `+0.8170pp`、raw `+1.0948pp`；clean-core 249 个预测变化中净纠正 58 个。O1+M3 clean-core 为 `83.9756%`，比 N3+M3 的 `84.0726%` 低 `0.0970pp`。中心与 M3 global 路径最大 logit 差为 `0`、预测一致率为 `100%`。

产物哈希：

- center cache：`e37bf4dd6250ffbd7932adac0701d5d9c220b026c5e5270ffb4c6f272672f1e6`；
- complementary cache：`c2794740b218c08ece205420aec2887824b6222281259d818d1eaa49e3540c06`；
- complementary evaluation：`b7474f4e181ec3a4a6dc6cbd41a46beead431dd60e65b6225f465735d5ed9f90`。

在测试推理开始前，新平台结果表明 A2+M1 `62.6747` 显著高于 A2+M3 `62.0259`。因此不继续生成 O1+M3 包，改为后续生成 O1+纯 M1；这是由同日独立平台消融直接支持的推理规格变更，而不是根据测试预测调参。
