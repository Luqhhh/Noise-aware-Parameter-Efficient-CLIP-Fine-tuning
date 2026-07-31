# N3：A2 上的零初始化 AdaptFormer 表征门控

日期：2026-07-20

## 核心问题

现有证据显示：冻结特征后处理、样本级 kNN、自由局部残差与单裁剪类别原型都不能解释平台 `61→90` 的差距；A2 上的视觉 LoRA 在多个隔离集合上方向一致为正，但增益偏小。N3 不再修改 logits 或拼接弱预测器，而是检验一种与 attention-LoRA 不同的视觉表征适配：在 ViT block 的 MLP 旁路加入零初始化瓶颈残差，使模型能够学习任务域的形状、纹理和局部组合，同时保持原 CLIP block 完全冻结。

依据：AdaptFormer，NeurIPS 2022，官方论文与实现：

- https://proceedings.neurips.cc/paper_files/paper/2022/hash/69e2f49ab0837b71b0e0cb7c555990f8-Abstract-Conference.html
- https://github.com/ShoufaChen/AdaptFormer

## 严格配对

- parent：A2 `NR_CL_KNN_DROP` epoch 48；
- control：J0 `A2 + last-4 attention Q/V/out LoRA`；
- train：A2 实际保留的 `90,204` 张，validation 为未见 D3 `10,322` 张，规范路径重叠 `0`；
- trust、GCE、augmentation、feature distillation、训练轮数和优化器均与 J0 保持一致；
- N3 相对 J0 只替换视觉 PEFT 结构，不修改清洗、损失或数据。

## 固定结构

对 CLIP ViT-B/32 的最后 6 个 residual blocks，在 attention residual 之后、原 MLP residual 旁路并联：

`adapter(x) = 0.1 × Up(Dropout(ReLU(Down(LayerNorm(x)))))`

`x_out = x + MLP(LN_2(x)) + adapter(x)`

- bottleneck：`64`；
- internal dropout：`0.1`；
- activation：ReLU；
- scale：`0.1`；
- `Down` 使用 Kaiming uniform，`Up` weight/bias 全零初始化；
- 原始 ViT 参数全部冻结；仅训练 6 个 adapter、原 A2 线性分类头；
- epoch 0 必须与 A2 logits 逐元素一致；checkpoint 重建必须 strict-load。

## 固定训练

- seed `42`，online `weak_rrc_flip`；
- GCE `q=0.5`；continuous cross-fitted trust 权重、minimum `0.25`、selection threshold `0.05`；不改标签；
- frozen-parent feature distillation weight `2.0`；
- AdamW；head LR `5e-5`，adapter LR `1e-4`；head WD `1e-4`，adapter WD `0`；
- batch `64`，2 epochs，1 epoch warmup，cosine schedule；无 MixUp；
- selector：clean-core micro；同时审计 trusted macro、raw micro、flip agreement、feature drift；
- 不扫描 bottleneck、block 数、scale、dropout、LR、loss、threshold 或 epoch。

## 预注册门槛

1. epoch 0 与 A2 online validation logits 最大绝对差 `0`、预测一致率 `100%`；
2. N3 best clean-core micro 相对 A2 至少 `+0.50pp`；
3. N3 best clean-core micro 相对 J0 至少 `+0.25pp`；
4. trusted macro 不低于 A2，raw micro 不得比 A2 低超过 `0.20pp`；
5. feature drift 不超过 `0.75%`，flip agreement 不得比 A2 低超过 `0.20pp`；
6. 任一门槛失败即 CLOSE，不追加 epoch、不改变结构或超参、不生成测试提交；全部通过才允许测试集单 checkpoint 推理及 M3 互补门控。

## 训练前证据与实现审计

类别级诊断进一步支持 N3：A2 的前 50 个最差类别只覆盖 38.66% 的 clean-core 错误，M1/M3 与 A2 的逐样本 oracle 上限也只有 +2.95pp，仍有 1,044 个样本三种推理视图均错。这说明当前主要瓶颈是广泛的视觉边界而不是少数类别或推理融合。完整证据见 `docs/PLATFORM_GAP_CLASS_DIAGNOSTIC_2026-07-20.md`。

实现审计已经完成：6 个 adapter 位于 block 6–11，共 604,032 个 adapter 参数；连同分类头总可训练参数 860,532，原始视觉编码器 87,849,216 个参数全部冻结。真实 OpenAI CLIP ViT-B/32 和 A2 checkpoint 的两张图像 CPU 审计得到 adapter-on 与 frozen-A2 logits 最大绝对差 `0`、tensor equality `true`、预测一致率 `100%`。完整 validation 的 epoch-0 等价性仍将在 GPU 空闲后、正式训练前执行。

## 合规边界

只使用官方 OpenAI CLIP ViT-B/32 和官方训练图像；新增 adapter 低于模型参数的 2%，属于单模型 PEFT。测试集只保留给通过门控后的固定推理，不进行训练、自适应或分布统计。团队仓库及其资产保持只读，代码、日志和 checkpoint 均位于独立工作树。

## 执行结果

### 零初始化与反向边界

- materialized epoch-0 checkpoint 通过 production strict-load；
- 全部 10,322 张 validation 与 A2 online center logits 逐元素相等：最大/平均绝对差均为 `0`，预测一致率 `100%`；
- 真实训练 batch `64` 的前向/反向峰值显存 `1,025,415,168` bytes；
- adapter `604,032`、分类头 `256,500`，合计可训练 `860,532`；冻结参数无梯度、无 optimizer step 时无参数变化；
- 产物：`outputs/N3_ADAPTFORMER_INITIAL_AUDIT/seed42/`。

### 两轮训练

| 方案 | clean-core micro | trusted macro | raw micro | flip agreement | feature drift |
|---|---:|---:|---:|---:|---:|
| A2 / epoch 0 | 82.5755% | 80.3732% | 69.4536% | 87.3668% | 0.0001% |
| N3 epoch 1 | 82.9914% | 80.7956% | 69.8314% | 88.2581% | 0.3331% |
| N3 epoch 2 | **83.1300%** | **80.8917%** | **69.8605%** | **89.0719%** | **0.4365%** |

epoch 2 被固定 selector 选中。相对 A2，clean-core `+0.5545pp`、trusted macro `+0.5186pp`、raw `+0.4069pp`、flip agreement `+1.7051pp`；相对 J0 clean-core `+0.2911pp`。全部训练门槛通过。

### 固定 M3 推理门控

用同一 `batch=64` 复核 center 与 complementary 内部 global 路径，logits 最大差 `0`、预测一致率 `100%`。固定 M3 将 N3 clean-core 从 `83.1162%` 提至 **`84.0726%`**：

- clean-core `+0.9565pp`；
- trusted macro `+0.9747pp`；
- raw micro `+1.1432pp`；
- 250 个 clean-core 预测变化中纠正 119、伤害 50，净增 69；
- 推理方式固定为 `complementary_flip_local_global`，不扫描融合权重。

机器判定文件 `outputs/N3_A2_ADAPTFORMER_GATE/seed42/gate_report.json` 的 `training_gate`、`fixed_M3_inference_gate` 与 `overall_passed` 均为 `true`。测试集没有参与 checkpoint 或推理方式选择。

### 最终测试提交

- 单 checkpoint：N3 epoch 2，SHA-256 `5771de6aea834ff5a1444cf0f7b51130bb2d613134d07edd1f4ccd81a96afe6c`；
- 固定推理：`complementary_flip_local_global:crop=160:top5:branch_mean_probabilities`；
- 预测：`24,967/24,967`，类别数 `500`，损坏图像 `0`；
- `submission.zip` SHA-256：`36baaa64b1b2c725191c3462e2d31f55c0ed67b5eb922addfaad24b5b02cbe45`；
- 独立 `audit_submission --allow-tta` 状态：`passed`；
- 路径：`outputs/N3_A2_ADAPTFORMER_GATE/seed42/submissions/m3/submission.zip`。
