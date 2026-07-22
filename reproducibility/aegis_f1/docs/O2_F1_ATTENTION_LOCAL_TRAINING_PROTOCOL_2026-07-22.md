# O2：F1 训练时注意力局部—全局对齐门控

日期：2026-07-22

## 平台驱动的假设

固定 M1 在 A2 与 F1 上分别比各自 Flip 基线高 `1.4619pp` 与 `2.2269pp`，F1+M1 达到当前平台最佳 `63.3276`。这比此前任何损失函数或冻结头修补都更强，说明主要瓶颈是 CLIP ViT-B/32 在 224px/7×7 patch 下的全局表示忽略局部细节，而不是线性头拟合不足。

O2 不再增加测试视图，而是在训练阶段把已经平台验证的 M1 局部图作为同一 F1 LoRA 的辅助监督，使视觉适配显式学习局部判别特征。相关一手证据包括：

- ICML 2025 ABS：<https://proceedings.mlr.press/v267/cai25d.html>，用注意力选择的局部与全局信息缓解随机裁剪和全局偏置；
- 2026 CLIP-MHAdapter：<https://arxiv.org/abs/2602.16590>，指出仅依赖 CLIP 全局嵌入会遗漏细粒度局部属性，并通过轻量局部 patch 适配改善分类。

O2 是由本项目平台因果证据驱动的训练门控，不声称复现上述论文。

## 严格配对

- 共同父模型：平台 F1 checkpoint，SHA-256 `7da95e427b959e85cbbf37c99d47d9909b941032e836fc219aaea8e690d72cc4`；
- 共同数据：F1 原始 92,902 张训练拆分与 10,316 张从未用于该训练的验证拆分；
- 共同监督：OOF clean probability `>=0.70` 的样本承担 GCE q=0.5 分类监督，其余样本权重为 0；
- 共同模型：最后四个视觉 block 的 Q/V/out rank-8 LoRA，feature distillation=2.0；
- 共同优化：2 epoch、batch64、head LR `2e-5`、LoRA LR `1e-5`、无 warmup、固定选择 epoch2；
- O2C：只做原有全局分类继续训练；
- O2L：唯一新增训练干预为固定 `crop=160/top5` 的注意力局部监督与全局→局部一致性。

## O2L 训练公式

对每个无 MixUp 的在线图像批次：

1. 原图正常前向，得到全局 logits 与特征蒸馏损失；
2. 同一当前模型在无梯度路径读取最后层 12-head 平均 CLS→patch attention；
3. 用 M1 固定 top-5 加权中心裁剪 160×160，并放大回 224×224；top-k/裁剪位置完全 detach；
4. 局部图由同一模型再次前向；
5. 分类项为 `0.5 * GCE_global + 0.5 * GCE_local`；
6. 额外加入 `0.25 * KL(stopgrad(p_global) || p_local)`，温度为 1；
7. 分类和一致性均只由同一 OOF clean-core 权重门控。

没有第二个教师模型、外部数据、测试时训练或标签传播。无梯度注意力路径只是同一当前模型的确定性裁剪器。

## 预注册门槛

两组都固定使用 epoch2，不根据验证集挑 epoch。训练后分别生成同规格 center 与纯 M1 validation cache；O2L 晋级测试推理必须同时满足：

1. O2L M1 clean-core micro 至少比 O2C M1 高 `+0.20pp`；
2. O2L M1 trusted macro 不得比 O2C 低超过 `0.05pp`；
3. O2L M1 raw micro 不得比 O2C 低超过 `0.10pp`；
4. O2L center clean-core 不得比 O2C 低超过 `0.25pp`；
5. O2L mean feature drift 不超过 `1.0%`，局部分类、一致性、梯度和预测均有限；
6. 若失败，不扫描 crop/top-k/权重/温度，也不生成测试包。

若通过，只生成 O2L+纯 M1 单检查点提交；不叠加已被平台证伪为次优的 M3。

## 结果

两组均从共同 F1 父检查点出发，按预注册固定运行 2 epoch，并使用各自 epoch2 检查点生成同规格 center/M1 缓存。未读取测试集。

| 预注册指标 | O2C | O2L | O2L - O2C | 门槛 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| M1 clean-core micro | 82.1525 | 82.4229 | **+0.2704pp** | `>= +0.20pp` | 通过 |
| M1 trusted macro | 81.2081 | 81.4987 | **+0.2905pp** | `>= -0.05pp` | 通过 |
| M1 raw micro | 71.6072 | 71.9174 | **+0.3102pp** | `>= -0.10pp` | 通过 |
| center clean-core micro | 81.4224 | 80.7869 | **-0.6355pp** | `>= -0.25pp` | **失败** |
| mean feature drift | 0.4216% | 0.5807% | +0.1591pp | `<= 1.0%` | 通过 |

补充诊断显示，O2L 的纯局部 clean-core micro 从 O2C 的 `77.0416` 提高到 `78.9346`（`+1.8929pp`），局部—全局融合的净修正样本也从 `54` 增至 `121`；说明注意力局部监督确实强化了局部判别能力。代价是共享 LoRA/分类头同时被局部目标牵引，导致整图路径显著回退。

相对未经继续训练、且已有平台结果的原始 F1+M1，O2L+M1 的验证 clean-core micro、trusted macro、raw micro 分别提高 `+0.1758pp`、`+0.2252pp`、`+0.2811pp`，但 center clean-core micro 下降 `-0.7437pp`。这进一步排除了“控制组继续训练退化”这一单一解释，同时也不足以推翻预注册的整图安全门槛。

### 门控结论

O2L 未同时满足全部预注册门槛，**判定失败**。严格按协议：

- 不生成 O2 测试集预测或提交包；
- 不事后扫描 crop、top-k、监督权重或温度；
- 保留负结果，下一步改为隔离局部参数，使 F1 全局路径按构造保持完全不变，而局部视图使用同一检查点内的轻量残差适配器。

### 可复现证据

- O2C checkpoint SHA-256：`5d23fbd57eb8539c165218a6d3755e830f52e480d4dce8d9e3037b3b166f3b23`；
- O2L checkpoint SHA-256：`32c463df68fa71cac20768162bae16105137865478e41677faa5965bbdb1fc6d`；
- O2C center/M1 cache SHA-256：`be93f3ebbd8fa6ffab9462fc3750c2bbed607543104801e790a2c89034a01505` / `ebdb4889d2e5cf0b8b51986cc4b4f02e8132a1cb6378846139fd6e59f03a8cd9`；
- O2L center/M1 cache SHA-256：`3df4aa27f510b0ff115669b2a4951239c3d381a08fc9f14c2579f61a61b1b5d9` / `bd601978ada0df085173e2418b41b90155423fcbb85ee7e1630350b4ba18c524`；
- 两份比较报告均确认 `global_path_max_abs_logit_difference=0`、`global_path_prediction_agreement=1`，即缓存内部的整图路径一致性校验通过；
- 训练过程局部分类、一致性、梯度和预测均为有限值，未出现崩溃或空类别。
