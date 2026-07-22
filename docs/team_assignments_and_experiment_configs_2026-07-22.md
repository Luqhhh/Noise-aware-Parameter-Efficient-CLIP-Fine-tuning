# 团队分工与当前实验配置

日期：2026-07-22

团队仓库本次增量同步基线：`origin/main`，commit `9ecb289`（已包含 Aegis 完整账本、平台分数回填与 A2 bare 归属修正）

独立研发来源：`/home/x28639/projects/AegisCLIP-F6-A2LoRA`，已按来源提交 `9f9126a` 完整增量整合到团队目录 `reproducibility/aegis_f1/`；文件级审计确认 246 个相关源文件缺失数为 0

本文用于统一人员分工、待评测候选、固定实验配置与合规边界。新增训练、缓存和提交包保留在独立 Aegis 工程中；团队当前开发工作树不被占用或覆盖。任何 GPU 任务启动前必须确认无人占卡，并禁止多人写入同一输出目录。

## 一、人员分工

| 负责人 | 当前职责 | 必须回传的证据 | 禁止事项 |
|---|---|---|---|
| 队长 | 最终方案与合规复核；决定平台上传顺序；登记平台分数 | 包名、平台分数、上传时间、ZIP SHA-256 | 不以本地带噪 validation 代替平台结论；不临时混合多个检查点 |
| x28639 | 平台包上传与结果同步；维护实验总表；把平台反馈通知研发侧 | 平台截图/分数、对应包 SHA-256、是否为有效评测 | 不重命名或重新压缩已审计 ZIP；不上传低优先级包替代高优先级包 |
| JJT | 在收到新的明确执行卡后协助运行尚未执行的新实验；保留完整日志和产物 | 执行命令、配置 SHA、日志、checkpoint/gate SHA-256 | 不重复运行 F2/O1/N3；O3 原案不得重跑；不得临时改超参数或融合权重 |
| Codex（研发支持） | 方法设计、代码实现、预注册门控、结果审计、文档与提交包生成 | commit、测试结果、门控表、提交包 manifest/audit | 门控失败不得生成测试提交；不得与团队训练抢 GPU |

如果队长调整人员，只调整“执行者”，不得同时改变已经预注册的配置与判定门槛；否则必须建立新实验编号并重新预注册。

### 1.1 算力任务状态（避免把“待评测”误解成“待训练”）

| 实验 | 训练状态 | 是否还需要算力 | 当前动作 | 执行者 |
|---|---|---|---|---|
| F2 + M1 | 已训练、已审计、已生成 ZIP | 否 | 只做平台上传与分数回填 | x28639 上传；队长决定顺序 |
| O1 + M1 | 已训练、已审计、已生成 ZIP | 否 | 只做平台上传与分数回填 | x28639 上传；队长决定顺序 |
| N3 + M1 | 已训练、已审计、已生成 ZIP | 否 | 只做平台上传与分数回填 | x28639 上传；队长决定顺序 |
| O3 原案 | 训练前复现审计失败，未更新参数 | 否，且禁止重跑原案 | 保留失败证据 | 无 |
| O3-R1 | 修订协议已预注册，尚未执行 | **是**：先重建一次 batch-128 validation cache；门控通过后才做 CPU Adapter 训练 | 等队长明确批准和空闲 GPU 窗口 | 待队长确认；可由 JJT 协助 |
| Q1A 交叉拟合轨迹 | 新诊断方案已实现并通过微型测试，真实运行尚未开始 | **是**：三折冻结特征线性头、50 epoch；远轻于完整 CLIP 微调 | 等代码完整回归、明确授权和空闲 GPU | 暂未分配；可由队长协助执行固定命令 |
| R1 F1+M1 Part-Token 残差 | 协议、实现、源工程 188 项/团队快照 189 项回归与真实 F1 epoch-0 逐位复现已通过；尚未生成正式 cache/训练 | **是**：先生成固定 batch-128 train/validation cache；随后 Adapter 在 CPU 训练 | 等队长明确批准和空闲 GPU 窗口 | 暂未分配；可由 JJT 按固定命令协助 |
| T0/T1 可信梯度子空间 | 严格配对配置、实现和自动 gate 已完成；两个训练臂均未运行 | **是**：T0、T1 各 2 epoch；之后仅在 Gate 0 通过时生成 center/M1 validation cache | 等队长明确批准和空闲 GPU 窗口；T0→T1 串行 | 暂未分配；可由 JJT 按固定协议协助 |
| U0 数字 Prompt 审计 | 已完成并确定性重跑；raw/clean-core `0.232648%/0.229854%` | 否 | 关闭 direct numeric shared-context CoOp；仅保留审计证据 | 无 |
| V1/V2 均衡先验传输 | 已在 F1+M1/A2+M1 两份固定 validation cache 上完成并复现；跨检查点门禁失败 | 否 | 关闭该固定传输方向；无测试推理、无提交包 | 无 |

截至本文当前版本，没有任何默认分配给队长或 JJT 的运行中任务。若要使用两人的算力，只能从 O3-R1、Q1A、R1、T0/T1 中按队长确认的优先级领取；必须先在群内确认实验编号、执行者和 GPU 时间窗。未经确认，不启动，也不重复训练已经完成的 F2/O1/N3。U0、V1、V2 均已结束，不需要 GPU。

## 二、平台评测任务

当前有限平台名额建议按以下顺序使用：

1. **F2 + M1**；
2. **O1 + M1**；
3. **N3 + M1**；
4. N3 + M3 暂不上传，除非前三项均完成且队长明确批准。

每次上传后立即记录真实平台分数，再决定下一次；不得依据测试集预测分布挑包。

| 优先级 | 候选 | ZIP（x28639 主机） | SHA-256 | 当前状态 |
|---:|---|---|---|---|
| 1 | F2 + M1 | `/home/x28639/projects/AegisCLIP-Noise-Robust/outputs/F2_VISUAL_LORA_FULLFIT/seed42/submissions/m1/submission.zip` | `4c0b5ea229f300d080e7cd6b1a5a93bd0c743a3b886a206f2a2f44f690f7a690` | 审计通过，待平台 |
| 2 | O1 + M1 | `/home/x28639/projects/AegisCLIP-F6-A2LoRA/outputs/O1_A2_ADAPTFORMER_MIXUP_GATE/seed42/submissions/m1/submission.zip` | `73cb20eda8063306071c09f40c2aeab0318549506bf71c05d8f570e37e5f0043` | 审计通过，待平台 |
| 3 | N3 + M1 | `/home/x28639/projects/AegisCLIP-F6-A2LoRA/outputs/N3_A2_ADAPTFORMER_GATE/seed42/submissions/m1/submission.zip` | `0b189c0669d1844787e860c936ede6eb1b1dfbacac07159e8623a7bc9b6fcbbb` | 审计通过，待平台 |

已知平台锚点：F1 + M1 为 `63.3276`，A2 + M1 为 `62.6747`，A2 + M3 为 `62.0259`。本地 validation 标签带噪，平台测试标签干净，因此本地结果只用于安全门控，最终排序以平台分数为准。

全部独立实验、失败/停止状态、平台分数和提交哈希统一见 [`aegis_independent_experiments_2026-07-22.md`](aegis_independent_experiments_2026-07-22.md)；本文件只负责当前分工、待执行算力与提交顺序，避免重复维护两份总账。

## 三、公共 M1 推理配置

| 项目 | 固定值 |
|---|---|
| 模型数 | 1 个 checkpoint |
| 输入 | OpenAI CLIP ViT-B/32 原生 224×224 |
| 全局分支 | 原图 center/global forward |
| 局部定位 | 最后视觉 Transformer block 的 CLS→patch attention |
| attention 聚合 | 12 个 head 平均 |
| 局部 patch | attention top-5，加权中心 |
| 裁剪 | 160×160，双线性放大到 224×224 |
| 融合 | 全局与局部类别概率 1:1 平均 |
| 禁止 | Flip 叠加、先验校准、测试时训练、外部数据、第二模型或投票 |

## 四、固定实验配置

### 4.1 F2 + M1

F2 是当前平台最佳 F1 方向在全部官方训练图上的固定重放。

| 模块 | 配置 |
|---|---|
| 训练数据 | 官方 train 全量 `103,218` 张；不使用外部数据 |
| 分类监督 | cross-fitted trust `clean_probability >= 0.70`；其余样本分类权重 0 |
| 视觉 PEFT | 最后 4 个 ViT block，Q/V/out LoRA，rank 8，alpha 8 |
| 损失 | GCE `q=0.5` + feature distillation `2.0` |
| 增强 | weak random-resized-crop + horizontal flip；无 MixUp |
| 优化 | 4 epoch，batch 64，head LR `5e-5`，LoRA LR `2e-5`，warmup 1 epoch |
| 选择 | 固定最后 epoch 4，不用与训练重叠的 validation 挑 checkpoint |
| 推理 | 公共 M1 配置 |

权威配置：`/home/x28639/projects/AegisCLIP-Noise-Robust/configs/f2_visual_lora_fullfit.yaml`。

### 4.2 N3 + M1

N3 用 MLP 并行 Adapter 替换 attention LoRA，检验不同视觉 PEFT 结构。

| 模块 | 配置 |
|---|---|
| 父模型 | 团队 A2 checkpoint；训练时只读加载 |
| 训练数据 | A2-kept `90,204` 张；固定独立 validation |
| 视觉 PEFT | 最后 6 个 ViT block 的 AdaptFormer |
| Adapter | bottleneck 64，residual scale 0.1，dropout 0.1，零输出初始化 |
| trust | KTA trust；minimum weight 0.25，selection threshold 0.05 |
| 损失 | GCE `q=0.5` + feature distillation `2.0`；无 MixUp |
| 优化 | 2 epoch，batch 64，head LR `5e-5`，Adapter LR `1e-4`，warmup 1 epoch |
| 选择 | clean-core micro 最佳点，固定 seed 42 |
| 推理 | 公共 M1 配置 |

权威配置：`reproducibility/aegis_f1/configs/n3_a2_adaptformer_gate.yaml`。

### 4.3 O1 + M1

O1 与 N3 严格配对，唯一训练干预是加入已预注册的像素 MixUp。

| 模块 | 配置 |
|---|---|
| 父模型、数据、Adapter、trust | 与 N3 完全相同 |
| MixUp | alpha `0.2`，触发概率 `0.2` |
| 损失 | GCE `q=0.5` + feature distillation `2.0` |
| 优化 | 2 epoch，batch 64，head LR `5e-5`，Adapter LR `1e-4`，warmup 1 epoch |
| 选择 | clean-core micro 最佳点，固定 seed 42 |
| 推理 | 公共 M1 配置 |

权威配置：`reproducibility/aegis_f1/configs/o1_a2_adaptformer_mixup_gate.yaml`。

### 4.4 O3 原案 / O3-R1：F1 局部专用特征 Adapter

O3 的目标是保留 O2L 已证明有效的局部学习，同时从计算图上禁止全局 F1 回退。

| 模块 | 配置 |
|---|---|
| 父模型 | F1，SHA-256 `7da95e427b959e85cbbf37c99d47d9909b941032e836fc219aaea8e690d72cc4` |
| 冻结项 | F1 视觉 LoRA、全局路径、唯一线性分类头全部冻结 |
| 训练样本 | F1 train 中 trust `>=0.70` 的 `65,473` 张，500 类 |
| 局部视图 | 与 M1 完全相同：last-block / mean-12-head / top-5 / crop160 |
| Adapter | `512→32→512`，scale 0.25，dropout 0.1，上投影零初始化，约 34,336 参数 |
| 最终输出 | `0.5*softmax(global) + 0.5*softmax(adapted_local)` |
| 损失 | fused GCE `q=0.5` + `0.25`×local GCE + `2.0`×feature L2 anchor |
| 优化 | AdamW，LR `5e-4`，weight decay `1e-4`，batch 1024，最多 20 epoch，patience 5 |
| 选择 | 只在 trusted/raw/drift 安全条件通过的 epoch 中按 clean-core micro 选择 |
| 晋级门槛 | clean-core `>=+0.20pp`、trusted macro 不降、raw micro `>=-0.10pp`、drift `<=1%`、全局路径保持一致 |

截至 2026-07-22，两份特征缓存已完成且通过样本数、路径唯一性、类别维度、父 checkpoint SHA 审计。训练前参考复现审计未通过：center 最大 logit 差 `0.041015625`、预测一致率 `99.9128%`；M1 最大 logit 差 `8.243944`、预测一致率 `99.6413%`。程序已按预注册规则在训练前停止，未更新任何模型参数、未生成测试提交。任何人均不得自行重跑或放宽标准，待确认旧 M1 参考与新缓存是否使用完全相同的数值/融合流程后，再建立修订版协议。

原 O3 已关闭，不允许重跑。O3-R1 只修正唯一已确认的复现差异：把 validation cache 的 batch size 从 64 恢复为旧 F1/M1 参考所用的 128；模型、数据、Adapter、损失、选择指标和门槛均不改变。O3-R1 当前仅为预注册状态，尚未执行；首先需要一次 GPU 特征缓存，严格复现门控通过后才允许进行 CPU Adapter 训练。

完整协议（团队仓库已整合）：

- `reproducibility/aegis_f1/docs/O3_F1_LOCAL_ONLY_FEATURE_ADAPTER_PROTOCOL_2026-07-22.md`
- `reproducibility/aegis_f1/docs/O3_R1_BATCH_REPRODUCIBILITY_AMENDMENT_2026-07-22.md`

### 4.5 Q1A：交叉拟合 Wrong-Event 训练轨迹（尚未执行）

Q1A 不是直接生成提交包的新模型，而是验证时间维噪声信号是否比最终一次 OOF 置信度提供更多信息。

| 模块 | 固定配置 |
|---|---|
| 输入 | 官方 train 的冻结 OpenAI CLIP ViT-B/32 512 维特征；不使用 test/外部数据 |
| 交叉拟合 | 固定三折；每个样本每个 epoch 只由未见过它的折头预测一次 |
| 线性头 | GCE `q=0.5`，AdamW，LR `0.005`，weight decay `1e-4` |
| 训练 | 50 epoch，batch 128，2 epoch warmup + cosine，折种子 42/43/44 |
| 轨迹 | Wrong-Event、预测变化、原标签概率、top-5 类/概率、最终 logits |
| 复现门控 | 第 50 epoch 对 I0 top-1 一致率 `>=0.9999`；原标签概率 MAE `<=0.002`、P99 `<=0.01` |
| 结果边界 | Q1A 只产生 train-only 诊断；门控失败或信号高度冗余时不进入 Q1B |

完整协议（团队仓库已整合）：`reproducibility/aegis_f1/docs/Q1_CROSSFITTED_WRONG_EVENT_PROTOCOL_2026-07-22.md`。Q1A 尚未获得真实运行授权，不得凭本表自行补命令运行。

### 4.6 R1：F1+M1 Part-Token 局部残差（尚未执行）

R1 保持已经平台验证的 F1+M1 全局/局部路径不变，只在同一次局部前向中读取 patch token，并用零初始化的 `512→32→512` 残差器学习 `part-local` 差值。最终局部 logits 使用冻结 F1 线性头的锚定残差，因此 epoch 0 与 F1+M1 逐位一致。

| 模块 | 固定配置 |
|---|---|
| 父模型 | 单一 F1 checkpoint，SHA-256 `7da95e427b959e85cbbf37c99d47d9909b941032e836fc219aaea8e690d72cc4` |
| 局部视图 | M1 last-block / mean-12-head / top-5 / crop160 |
| Part pool | 49 个 patch 中按 CLS cosine 稳定取 top-8，temperature `0.07` |
| Adapter | `512→32→512`，scale `0.25`，dropout `0.1`，上投影零初始化 |
| 训练样本 | cross-fitted clean probability `>=0.70` 的 `65,473` 张；固定 validation `10,316` 张 |
| 缓存 | train/validation 图像 batch 均固定 `128`；父 checkpoint、pool spec 与路径交集 fail-closed 审计 |
| 损失 | fused GCE `q=0.5` + `0.25×` local GCE + `2.0×` feature L2 anchor |
| 晋级门槛 | clean-core `>=+0.20pp`、trusted macro 不降、raw micro `>=-0.10pp`、drift `<=1%`、空预测类 0 |
| 当前证据 | R1 合并时独立源工程 `188 passed`、团队快照 `189 passed`；最新 V1/V2 增量后团队完整回归 `210 passed`；真实 F1 的 R1 epoch-0 对 F1+M1 最大 logit 差 `0`、预测一致率 `100%` |
| 当前边界 | 尚未生成正式 GPU cache、尚未训练、尚无提交包和平台分数 |

完整协议：`reproducibility/aegis_f1/docs/R1_F1_M1_PART_TOKEN_RESIDUAL_PROTOCOL_2026-07-22.md`。只有预注册门控 `passed=true` 后才能读取 test 并生成提交包。

### 4.7 T0/T1：F1 可信梯度子空间严格配对（尚未执行）

T0/T1 从同一 F1 best checkpoint 出发，数据顺序、增强、可信集合、学习率、epoch、batch 与特征锚定全部相同。唯一干预是：T0 不使用不确定标签梯度；T1 只保留该梯度在近期可信参考梯度构成的 rank-8 子空间中的分量。

| 模块 | 固定配置 |
|---|---|
| 父模型 | F1 best，SHA-256 `7da95e427b959e85cbbf37c99d47d9909b941032e836fc219aaea8e690d72cc4` |
| 可信/不确定 train | clean probability `>=0.70`：`65,473 / 27,429`；两边均覆盖 500 类 |
| 模型 | 最后 4 个视觉块 Q/V/out rank-8 LoRA + 单线性头 |
| 损失 | 可信样本 GCE `q=0.5` + 全批 feature distillation `2.0` |
| 优化 | 2 epoch，batch 64，head LR `2e-5`，LoRA LR `1e-5` |
| T1 投影 | 最近可信参考梯度、FIFO 两遍 MGS、rank 8；不确定梯度正交分量全部丢弃 |
| 本地晋级 | 相对 T0 的 M1 clean-core `>=+0.20pp`，trusted macro `>=-0.05pp`，raw `>=-0.10pp`；并满足复现、漂移与类别覆盖门禁 |
| 当前状态 | 团队整合完整回归 `210 passed`；T0/T1 均 `not_run`，无提交包、无平台分数 |

配置与协议：

- `reproducibility/aegis_f1/configs/t0_f1_trust_subspace_control.yaml`
- `reproducibility/aegis_f1/configs/t1_f1_trust_subspace_projection.yaml`
- `reproducibility/aegis_f1/docs/T1_F1_TRUST_SUBSPACE_GRADIENT_PROTOCOL_2026-07-22.md`

### 4.8 U0：数字类别 Prompt Tuning 可行性审计（已完成）

U0 冻结 OpenAI CLIP ViT-B/32，用唯一固定模板 `a photo of a 0000` 至 `a photo of a 0499` 检查数字 ID 是否可作为标准共享 CoOp 的语义锚点。它没有训练、没有读取 test、没有生成提交包。

| 指标 | 结果 |
|---|---:|
| raw / clean-core validation | `0.232648% / 0.229854%` |
| 有预测类别 | `210 / 500` |
| 文本非对角余弦均值 | `0.978551` |
| 90% / 99% 能量秩 | `1 / 5` |
| 审计 JSON SHA-256 | `694a114eb3dced2b7b298d50948e5738988bc3965d58b0779b7c19efb261cff1` |

结论：关闭 direct numeric shared-context CoOp；不能把该本地审计写作平台成绩，也不能外推为“所有 Prompt 方法无效”。权威记录：`reproducibility/aegis_f1/docs/U0_NUMERIC_CLASS_PROMPT_FEASIBILITY_AUDIT_2026-07-22.md`。

## 五、平台结果回填格式

每次平台评测后，在团队项目的结果文档中追加一行，并在群内按以下格式同步：

```text
实验：F2+M1 / O1+M1 / N3+M1
平台分数：
提交时间：
ZIP SHA-256：
是否有效评测：是/否
备注：不得根据测试预测分布补充解释
```

## 六、合规共识

1. 只使用官方训练数据；测试集只做最终推理；
2. 最终提交始终是单一 OpenAI CLIP ViT-B/32 checkpoint 的固定推理流水线；
3. TTA/M1 只能是同一模型的确定性视图，不得做多模型融合；
4. 不使用外部数据、类别文本、外部教师、测试标签或测试时参数更新；
5. 任何新方案先写配置与门槛，再训练；失败结果保留，不事后降低门槛。
