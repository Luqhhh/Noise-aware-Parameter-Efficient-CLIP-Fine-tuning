# T1：F1 可信梯度子空间投影协议

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-22
- Verification Status: UNVERIFIED（实现与静态审计完成；正式训练未运行）
- Version Label: code_plan_v1

## Experiment Overview

- **Title**：F1 label-only trust-gradient subspace continuation
- **Objective**：检验能否在不直接采用不确定标签梯度的前提下，从 F1 当前丢弃的 `27,429` 个官方训练样本中提取与可信样本一致的表示学习信号，并提升平台最佳 F1+M1 路线。
- **Hypothesis**：不确定样本梯度中与近期可信梯度张成空间一致的分量含有可利用的细粒度信息；正交分量更可能导致噪声漂移，应在优化器更新前删除。
- **Type**：paired single-model training gate
- **Status**：`pre_registered / not_run`

## 1. 为什么不是继续微调损失权重

比赛说明引用了 JoAPR 与 TrustCLIP。JoAPR 的动态两阈值划分、增强预测软标签和迭代重训练具有启发性，但把其最终时刻划分静态套在本项目 OOF logits 上会一次改变约 `24,350` 个标签，远高于现有严格 KTA 修正的 `601` 个；在本项目已经观察到静态 relabel/drop 负结果的情况下，风险过大。

TrustCLIP 摘要公开的核心思想更适合作为第一步：可信样本正常监督，不确定样本的梯度只保留与可信梯度子空间对齐的部分。当前 J1 并未实现该机制：它将**全批总梯度**与一条聚合可信梯度比较，只有点积为负才删除冲突分量；两轮训练 `projection_count=0`，因此 J1 与 J0 参数和结果相同。该负结果只能否定当前半空间实现，不能否定真正的子空间方案。

本实验明确称为 **label-only TrustCLIP-inspired surrogate**，不声称忠实复现 TrustCLIP：比赛类别只有 `0000...0499` 数字标识，无法执行语义类名驱动的 SLV；TrustCLIP 全文实现细节也未取得。依据仅限其公开摘要所描述的 SLV/TGP 机制边界。

一手资料：

- TrustCLIP DOI：<https://doi.org/10.1145/3746027.3755415>
- JoAPR CVPR 2024 正文：<https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_JoAPR_Cleaning_the_Lens_of_Prompt_Learning_for_Vision-Language_Models_CVPR_2024_paper.pdf>
- 比赛说明：<https://www.aicomp.cn/tracks/tracks-1/3714.html>

## 2. 严格因果配对

| 项目 | T0 control | T1 treatment |
|---|---|---|
| 父模型 | F1 best checkpoint | 完全相同 |
| 数据、顺序、增强 | F1 train/val、seed42、weak RRC+flip | 完全相同 |
| 模型 | last-4 Q/V/out rank-8 visual LoRA + 单线性头 | 完全相同 |
| 可信集合 | cross-fitted `clean_probability >= 0.70` | 完全相同 |
| 分类损失 | 可信样本 GCE q=0.5 | 完全相同 |
| 特征锚定 | 全批 OpenAI CLIP feature distillation=2.0 | 完全相同 |
| 学习率、epoch、batch | 2e-5 head / 1e-5 LoRA；2 epoch；64 | 完全相同 |
| 唯一干预 | 不使用不确定标签梯度 | 加入不确定梯度的可信子空间投影 |

自动测试会移除 `experiment_id` 与 `mode` 后逐字段比较两份配置；任何其他差异均失败。配置：

- `configs/t0_f1_trust_subspace_control.yaml`
- `configs/t1_f1_trust_subspace_projection.yaml`

## 3. 固定数学定义

对批次大小 `B`，令 `C={i:p_clean(i)>=0.70}`、`U` 为其补集，`ell_i` 为 GCE，`d_i` 为当前特征与冻结 OpenAI CLIP 特征的余弦距离：

```text
L_C   = (1/B) * sum_{i in C} ell_i
L_D   = (2/B) * sum_{i=1..B} d_i
L_ref = (1/B) * sum_{i in C} (ell_i + 2*d_i)
g0    = grad(L_C + L_D)
gU    = grad((1/B) * sum_{i in U} ell_i)
```

在线基 `Q_t` 由最近可信参考梯度用两遍 modified Gram-Schmidt 构建，固定 FIFO rank=8：

```text
Q_t = FIFO_MGS8(Q_{t-1}, grad(L_ref))
g_T0 = g0
g_T1 = g0 + Q_t * (Q_t^T * gU)
```

所有和式都除以完整批次 `B`，所以 T0/T1 不会因每批可信比例不同而偷偷改变归一化。T1 不设置可扫描的 `lambda`，也不直接使用 `gU`；AMP 缩放在三个梯度上完全一致，投影后再统一 unscale、clip 和 AdamW step。

## 4. 固定输入及审计

| 输入 | 数量 / SHA-256 |
|---|---|
| F1 parent checkpoint | `7da95e427b959e85cbbf37c99d47d9909b941032e836fc219aaea8e690d72cc4` |
| train CSV | 92,902；`a726b8a3ca8bc5857136106aca80f01d557104d3661ef92ccedfb2c0ea087875` |
| validation CSV | 10,316；`54a790b35f836cfba4c19cbb5fe38c4b1b37aab62cc9d477f9285496b2d5568e` |
| CVT trust bundle | `52e59a991a5eb3c57abdfabee5647423726f51fbdd3da2ce377467664d173608` |
| 可信 / 不确定 train | 65,473 / 27,429；两边均覆盖 500 类 |
| 可信 / 不确定 validation | 7,396 / 2,920；两边均覆盖 500 类 |
| train/validation 路径交集 | 0 |
| F1 可训练参数 | 403,956；模型总参数 88,253,172 |

按代码实际的 seed42 DataLoader shuffle 模拟两轮共 `2,904` 个批次，可信样本每批最少 `28`，不确定样本每批最少 `8`；预注册的 `minimum_trusted_samples=8`、`minimum_uncertain_samples=8` 均不会跳过批次。

## 5. Setup 与执行卡

> 团队仓库映射：下列命令保留独立来源工程的原始工作目录以维持 provenance。团队仓库复现时应进入 `reproducibility/aegis_f1/`，将 `PYTHONPATH` 指向该目录；已整合的 T0/T1 配置把数据、trust、F1 checkpoint 与输出根路径规范化为相对路径。任何路径替换不得改变数据/检查点哈希或实验参数。

- **Working Directory**：`/home/x28639/projects/AegisCLIP-F6-A2LoRA`
- **Python**：`/home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python`
- **Framework**：PyTorch；单张空闲 GPU；不得与团队训练并行
- **Timeout**：每个训练任务 3 小时硬上限；只监控各自输出目录
- **执行前条件**：队长/用户明确授权，且 `nvidia-smi` 无团队任务
- **额外常驻显存**：rank-8 × 403,956 × FP32 = `12,926,592` bytes（约 `12.33 MiB`）；T1 每批执行 reference/shared/uncertain 三次反向，计算而非基向量显存是主要额外成本

固定执行顺序（当前尚未执行）：

```bash
PYTHONPATH=/home/x28639/projects/AegisCLIP-F6-A2LoRA \
/home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
-m aegis_clip.cli.train --config configs/t0_f1_trust_subspace_control.yaml

PYTHONPATH=/home/x28639/projects/AegisCLIP-F6-A2LoRA \
/home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
-m aegis_clip.cli.train --config configs/t1_f1_trust_subspace_projection.yaml
```

不得使用 `--overwrite` 静默覆盖现有实验。中断后只允许从相应 `last.pt` 使用 `--resume`，在线基已纳入 `training_aux_state`，缺失或 rank/epsilon 不同会 fail closed。

两组 epoch2 完成且 Gate 0 机械条件成立后，按相同 batch 和视图生成四份 validation cache：

```bash
PYTHONPATH=/home/x28639/projects/AegisCLIP-F6-A2LoRA \
/home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
-m aegis_clip.cli.cache_validation_logits \
--checkpoint outputs/T0_F1_TRUST_SUBSPACE_CONTROL/seed42/checkpoints/best.pt \
--output outputs/T0_F1_TRUST_SUBSPACE_CONTROL/seed42/validation_center.pt \
--batch-size 128 --num-workers 4 --view-mode center

PYTHONPATH=/home/x28639/projects/AegisCLIP-F6-A2LoRA \
/home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
-m aegis_clip.cli.cache_validation_logits \
--checkpoint outputs/T0_F1_TRUST_SUBSPACE_CONTROL/seed42/checkpoints/best.pt \
--output outputs/T0_F1_TRUST_SUBSPACE_CONTROL/seed42/validation_m1.pt \
--batch-size 128 --num-workers 4 --view-mode attention_local_global

PYTHONPATH=/home/x28639/projects/AegisCLIP-F6-A2LoRA \
/home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
-m aegis_clip.cli.cache_validation_logits \
--checkpoint outputs/T1_F1_TRUST_SUBSPACE_PROJECTION/seed42/checkpoints/best.pt \
--output outputs/T1_F1_TRUST_SUBSPACE_PROJECTION/seed42/validation_center.pt \
--batch-size 128 --num-workers 4 --view-mode center

PYTHONPATH=/home/x28639/projects/AegisCLIP-F6-A2LoRA \
/home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
-m aegis_clip.cli.cache_validation_logits \
--checkpoint outputs/T1_F1_TRUST_SUBSPACE_PROJECTION/seed42/checkpoints/best.pt \
--output outputs/T1_F1_TRUST_SUBSPACE_PROJECTION/seed42/validation_m1.pt \
--batch-size 128 --num-workers 4 --view-mode attention_local_global
```

随后只能由固定 gate 程序做决定，禁止手工挑指标：

```bash
PYTHONPATH=/home/x28639/projects/AegisCLIP-F6-A2LoRA \
/home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
-m aegis_clip.cli.evaluate_trust_subspace_gate \
--original-m1 outputs/M1_F1_TRANSFER/seed42/f1_attention_local_global.pt \
--t0-center outputs/T0_F1_TRUST_SUBSPACE_CONTROL/seed42/validation_center.pt \
--t0-m1 outputs/T0_F1_TRUST_SUBSPACE_CONTROL/seed42/validation_m1.pt \
--t1-center outputs/T1_F1_TRUST_SUBSPACE_PROJECTION/seed42/validation_center.pt \
--t1-m1 outputs/T1_F1_TRUST_SUBSPACE_PROJECTION/seed42/validation_m1.pt \
--t0-initial outputs/T0_F1_TRUST_SUBSPACE_CONTROL/seed42/checkpoints/initial_evaluation.json \
--t1-initial outputs/T1_F1_TRUST_SUBSPACE_PROJECTION/seed42/checkpoints/initial_evaluation.json \
--t0-evaluation outputs/T0_F1_TRUST_SUBSPACE_CONTROL/seed42/checkpoints/best_evaluation.json \
--t1-evaluation outputs/T1_F1_TRUST_SUBSPACE_PROJECTION/seed42/checkpoints/best_evaluation.json \
--t0-metrics outputs/T0_F1_TRUST_SUBSPACE_CONTROL/seed42/logs/metrics.csv \
--t1-metrics outputs/T1_F1_TRUST_SUBSPACE_PROJECTION/seed42/logs/metrics.csv \
--output outputs/T1_F1_TRUST_SUBSPACE_PROJECTION/seed42/t0_t1_gate.json
```

## 6. Expected Outputs 与监控

| 输出 | 路径 | 成功条件 |
|---|---|---|
| T0 metrics | `outputs/T0_F1_TRUST_SUBSPACE_CONTROL/seed42/logs/metrics.csv` | 2 epoch 完整；所有数值有限 |
| T1 metrics | `outputs/T1_F1_TRUST_SUBSPACE_PROJECTION/seed42/logs/metrics.csv` | 2 epoch 完整；所有数值有限 |
| T0/T1 checkpoints | 各自 `seed42/checkpoints/{last,best}.pt` | epoch2；父规格不变；aux state 可恢复 |
| M1 validation caches | 训练门禁后才生成 | 路径、标签、融合公式严格一致 |
| test submission | 仅所有门禁通过后 | 单 checkpoint + 固定 M1；24,967 行；审计通过 |

每个 epoch 必须记录 basis rank/update、可信/不确定样本数、投影步数、原始/投影不确定梯度范数、保留范数比例、center 验证指标与 feature drift。进程存活、日志更新时间和 GPU 显存同时监控；异常只报告，不自动重跑。

## 7. 预注册门禁

### Gate 0：实现与复现

1. 全部单元/回归测试通过；
2. T0/T1 除 `experiment_id` 和 `mode` 外完全相同；
3. 两者 `initial_evaluation.json` 逐字段相同，并锚定同一 F1 SHA；
4. 训练中无 NaN/Inf、无空预测类、无数据交集；
5. T0 `projection_steps=0`；T1 basis 最终 rank=8、`skipped_steps=0`、投影非零步数至少占 eligible steps 的 95%，平均 retained norm ratio 严格位于 `(0,1)`。

### Gate 1：固定 M1 validation 因果收益

使用 epoch2 checkpoint 和完全相同的 `attention_local_global` 缓存流程。T1 必须同时满足：

1. 相对 T0，M1 clean-core micro `>= +0.20pp`；
2. 相对 T0，M1 trusted macro `>= -0.05pp`；
3. 相对 T0，M1 raw micro `>= -0.10pp`；
4. 相对 T0，center clean-core micro `>= -0.20pp`；
5. 相对原始 F1+M1，M1 clean-core micro `>= +0.10pp`；
6. mean feature drift `<=1.0%`，500 类均有预测，所有缓存值有限。

任何一项失败：T1 关闭，不读取 test，不扫描 threshold/rank/LR/epoch/投影权重。若通过，才生成一个 T1+M1 测试包；不叠加 Flip/M3、先验校准或第二模型。

### Gate 2：平台验证

平台包只有在 Gate 0/1 全过后才可上传。平台得分必须与 ZIP SHA-256 一并登记。只有真实平台分数高于当前 `63.3276`，才能称为新的高分方案；本地通过不等同于平台提升。

## 8. 合规边界

- 仅使用官方 train 图像与其数字标签；信任度来自 train 内交叉拟合；
- validation 只用于预注册门禁，test 只在门禁通过后做一次固定推理；
- 最终模型仍是单一 OpenAI CLIP ViT-B/32 checkpoint；M1 是同一模型的确定性中心/局部视图概率均值；
- 不使用外部图像、外部类名、文本教师、测试伪标签、测试时训练或多模型投票；
- 当前文档只证明方案已预注册和实现，**不证明有效、未生成提交包、没有平台分数**。
