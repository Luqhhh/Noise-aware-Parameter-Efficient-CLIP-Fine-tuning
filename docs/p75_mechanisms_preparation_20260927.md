# P75 训练机制候选：2026-09-27 仅 CPU 准备

> **2026-09-29 搜索纪律覆盖说明**：本文件保留当时的准备、启动与结果记录，不能作为待跑队列。
> Patch Readout 已在下文记录关闭，相关邻域搜索冻结；Full SAM 只收尾已启动固定实验，
> 不衍生 rho/ASAM/GSAM 或联合训练。Hard Support 的代理群体尚不构成可恢复收益证据，
> 后续先按[主要误差预算门禁](p75_error_budget_policy_20260929.md)归因，再决定是否训练。

状态：**实现与只读/CPU 检查完成；未启动 GPU 训练、未产生新候选 checkpoint、验证分或平台分。**
分支 `codex/p75_mechanisms_20260927`，基于 `origin/main` 的 `e443a63`，方案身份为
`P75_HARD_SUPPORT`、`P75_PATCH_READOUT`、`P75_FULL_SAM`。本段遵照用户指令只做准备。
现役仍是单 checkpoint `L05_T14_P060`，平台 **66.94797564362783%**。37,444 张测试图中
正确 25,068 张；75% 需要 28,083 张，尚需净增加 3,015 张，约为当前 12,376 个错误的 24.36%。
本地固定解码 75.8265% macro / 76.6532% micro 只作候选筛选基线，不推算平台分。

## 开工核对与代码基线

已 fetch 远端，检查所有本地/远端分支、近期 `origin/main` 与在途方案。现有
`codex/l05_oof_hard_filter` 是**删除约 1% 低可信训练样本并重新训练 LP/FT**，不是对有
额外证据的困难原标签样本恢复 CE；其远端记录显示筛后 LP 完成、FT 曾启动，最终结果
尚不能从本方案据以宣称。没有复用或修改它的训练输出。

当前 `main` 缺少 L05 实际使用的 focus 分支训练语义，因此本方案从 L05 checkpoint 内保存的
本机运行配置生成候选配置，运行器校验 focus 提交
`f050ecb59e0a885da0b43cdc6c8c316953fb5e7d`，并将该提交导出到**本方案独立**
`outputs/codex/p75_mechanisms_20260927/framework_{patch,sam}/`。不修改 focus 工作目录。
输入绑定当前阶段 `20260921` 的训练/验证 CSV、750 类映射、原 RM-LP 父权重
`d5cb8f…de23e24fefbcf616c747f2cab13e4dc2201fdd689b`，以及 L05 控制 checkpoint
`35d17c…1c04c9da72312923b983f17bb1d311ca32710aedc8f`。脚本核对完整 SHA 和
训练/验证内容组不相交，输出使用独立目录。测试集没有参与诊断或拟合。

## 0. 困难样本训练预算：有代理群体，尚无 L05 逐样本耦合证据

实际 L05 代码的 `confidence_gate=0.7` 对**全局 logits 的最大 softmax 概率**判断，
低于门限时，把局部分类的逐样本损失替换为全局损失。它既不是原标签正确概率，也没有
独立的局部图像质量检测。第 5 轮局部激活 94,883/133,815（70.9061%），第 16 轮
112,349/133,815（83.9585%）；这些是训练日志的**汇总**，无法与指定样本对齐。

CPU 预算表已登记在 [机器可读结果](../results/p75_training_budget_20260927.json)，本地完整路径为
`outputs/codex/p75_mechanisms_20260927/training_budget.json`。它用当前阶段
RM-LP 冻结特征和原分类头计算原标签概率、GCE 的 `sqrt(p_y)` 缩放，并与已存在的当前阶段
五折类别原型/岭原型折外记录逐行绑定。`p_y<0.3`、RM-LP 最大置信度 `<0.7` 且两路
折外类别原型均支持原标签者 **2,139 张、覆盖 530 类**。这只是可疑的**代理规模**；
RM-LP 是 224px 冻结读出，不能替代 L05 384px 训练期逐样本 logits。折外类别原型支持
也不能冒充内容组外近邻证据。当前没有同一训练行的弱视图稳定性缓存。

因此本次有限诊断的结论是：**不能确认 L05 同时压低 GCE 梯度和局部监督的可信困难群体；
P75_HARD_SUPPORT 不进入 GPU 队列。** 未重新训练教师、未重建大规模 OOF、未使用旧阶段
缓存。代码层面已实现固定 CSV 校验、`r=0.5` 的原标签 CE/GCE 混合、低置信样本的局部
恢复入口和空间方差质量检查，并在私有 trainer 副本中通过严格锚点补丁与编译检查；
没有符合证据门禁的支持 CSV 时，训练会 fail closed。后续若已有当前阶段逐样本 L05
记录、组外近邻与弱视图证据，再按同一训练行生成权重，不在本轮扩展阈值网格。

## 1. P75_PATCH_READOUT：历史准备记录，已关闭

固定 OpenAI CLIP ViT-B/32，同一次视觉 Transformer 前向读取最终 patch token；一个
线性评分器 softmax 汇聚 patch，经**零初始化**残差映射加入原 CLS 特征，最后仍只走原
`classifier` 一个分类头。保留 L05 的 384px、16 轮、有效 batch 1024、CE warmup→GCE、
feature anchor、从第 5 轮开始的 attention-local、学习率、增强与按中心 macro 选模。
单 checkpoint、无外部视觉权重、无独立分类器概率融合。与历史冻结特征分类头不同，此
读出与视觉层一起训练。

实现位于 `scripts/p75_runtime/p75_patch_readout.py`，准备身份与配置路径见
[准备记录](../results/p75_preparation_20260927.json)，只在配置显式
`model.patch_readout={enabled:true,version:1}` 时挂载。父权重加载只允许**完整缺失**
新模块参数；候选 checkpoint 严格重载，不接受部分读出状态。读出参数进入原 head 优化器
组。真实 RM-LP 权重在两次 CPU 首次前向检查中，与普通路径的 logits 最大绝对差
**≤1.67×10⁻⁶**、top1
完全一致；CPU 单测检查残差梯度、优化器组和错误 checkpoint 拒绝。该初始恒等结果不
等于完整训练已验证。方案配置与运行入口：

```bash
python3 scripts/p75_prepare.py --phase prepare
python3 scripts/run_p75_patch_readout.py --phase preflight
PYTHONPATH=./scripts/p75_runtime:./outputs/codex/p75_mechanisms_20260927/framework_patch/reproducibility/aegis_f1 python3 scripts/p75_cpu_smoke.py
```

正式 GPU 轮须用户后续解除本次暂停；入口分别为
`python3 scripts/run_p75_patch_readout.py --phase train|cache|evaluate|deliver --execute-gpu`。
其中 evaluate 产生固定解码报告与相对现役的逐张修正/退化、逐类覆盖报告；deliver 只在
macro ≥现役 +0.30pp 且 micro 不退化时出包，随后仍需 9/9 提交校验。平台优先队列门槛
是 macro ≥+1.00pp 且 micro 不退化，0.30–1.00pp 只保留本地候选。不得拿早期第 4 轮
与现役第 16 轮比较；应看同轮 L05 曲线。单次提升不声明稳定性证明。

## 2. P75_FULL_SAM：历史准备记录，固定实验已启动

现有 focus trainer 的 SAM 完整有效 batch 路径会**拒绝 attention-local**；其第二遍损失
只有 global + anchor。直接开启 `train.sam` 不构成 L05 完整配方上的 SAM 对照。本方案在
私有 trainer 副本中补上局部项：第一遍为原 L05 全损失并累积 1024 个样本的梯度；保存
每个 microbatch 的原图张量和**第一遍已确定的 attention 裁剪几何**到 CPU；一次全局
L2 扰动后，按同样图像与裁剪几何重放第二遍 global/local/anchor；精确恢复原参数后只
进行一次 AdamW 更新。固定标准 SAM `rho=0.05`，不扩展 ASAM/GSAM/半径扫描。异常
时 `finally` 恢复参数，第二遍的 RNG 消耗必须等于第一遍一遍随机流。CPU 数值测试覆盖
局部项、完整有效 batch 单次更新和异常恢复。

当前已绑定的 SAM 实现**要求 FP32**，所以固定 SAM 配置将 `train.amp` 从 L05 的 true
改为 false。这是除优化机制外的数值协议差异，正式结果必须明示，不能称严格单变量
SAM 因果对照。正式训练前还需独立工程 smoke 核实 8GB 显存、first/second 局部
路径及整批回放；工程 smoke 不用来提前淘汰方法。配置保留 L05 384px、16 轮与有效
batch 1024，输出单模型。CPU 准备入口：

```bash
python3 scripts/run_p75_full_sam.py --phase preflight
python3 -m pytest tests/test_p75_sam.py -q
```

GPU 入口同 Patch：`python3 scripts/run_p75_full_sam.py --phase train|cache|evaluate|deliver --execute-gpu`。
准备轮用户明确暂不启动 GPU，故当时未调用。原设想是两主线都出现独立正收益后再准备
联合训练；**2026-09-29 已撤销该自动派生路径**，没有联合配置，后续先满足主要误差预算门禁。

## 验证边界与保底交付

本轮 CPU 测试：`python3 -m pytest tests/test_p75_patch_readout.py tests/test_p75_hard_support.py tests/test_p75_sam.py -q`
为 **7 passed**；私有 focus 源补丁编译通过；两份配置及 RM-LP 父血缘预检通过。
配对验证脚本用现役同一缓存作自比较，复现 75.82651238982523% / 76.65322580645161%，
修正/退化均为 0。现役 L05_T14_P060 的 37,444 行 CSV/ZIP 重新通过 9/9 校验，路径为
`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/`。
本次**没有新候选预测 CSV/ZIP**；仅上述既有保底包可提交，平台今天两次额度已用完。
本次只交付代码、CPU 诊断、配置、门禁和可重放命令，不声称训练候选已完成。

## GPU 启动（2026-09-28，用户解除暂停）

用户指令「启动 p75 训练」后，本轮只启动 **P75_PATCH_READOUT** 的 train 阶段；同一张
GPU 上不并行启动 FULL_SAM（FP32 全批回放，显存与时长都不同量级），后者排队待定。

- **运行器 bug（本轮实测发现并修复）**：两个队列脚本都用 `log.parent.mkdir()` 把捕获日志
  建在 RUN 里，而 `trainer.train` 见到已存在的 RUN 会直接抛
  `FileExistsError: Run directory exists: ... Use --overwrite or --resume.`，导致**任何一次
  全新 `--phase train` 都在第一步之前失败**（06:27 首次启动即复现）。修法是新增
  `phase_log()`，把阶段 stdout 写到 RUN 旁边（同一实验目录），RUN 仍由 trainer 独占。
  改动 `scripts/run_p75_patch_readout.py`、`scripts/run_p75_full_sam.py`，新增
  `tests/test_p75_runner_logs.py`；定向测试 **9 passed**。`run_p75_*.py` 不在
  `project.p75_implementation_files` 内，冻结身份不变，preflight 仍通过。
- 启动前检查：preflight 通过；`test_p75_patch_readout.py`/`test_p75_hard_support.py`/
  `test_p75_sam.py` 7 passed；CPU smoke 最大 logit 误差 9.54e-07、top1 一致；
  `_gpu_smoke.py` 90 秒 **SMOKE OK**（针对本机历史 `cudaErrorUnknown` 的健康探针）。
- 实测已启动：screen 会话 `p75_patch`，第一步审计通过（head_grad=0.035027、
  visual_grad=31.851227），读出模块 `single_head_zero_residual_v1` 已挂载，可训练参数
  86,098,927；GPU 占用约 3.34 GiB。第 40 次更新时速率约 **20 步/204 秒**（≈131 步/轮）。
- **本轮没有新候选 checkpoint、验证分、提交包或平台分**；机器可读记录见
  [启动记录](../results/p75_patch_readout_launch_20260928.json)。

## P75_PATCH_READOUT 结果：未过门，线关闭（2026-09-28 深夜）

16 轮跑满（2,096 次更新，10.8 小时，零崩溃），按 center macro 选中 **epoch 14**。
接着跑完 `--phase cache` 与 `--phase evaluate`，拿到预注册的固定
Flip / T=1.4 / prior0.60 解码与相对现役的逐张配对。

| 判据 | 现役 L05_T14_P060 | P75_PATCH_READOUT | 差 |
|---|---:|---:|---:|
| 固定解码 macro | 75.8265% | 75.8339% | **+0.0074pp** |
| 固定解码 micro | 76.6532% | 76.6667% | +0.0134pp |
| 过门线（macro +0.30pp） | — | — | **差 0.2926pp** |

配对逐张：修正 184 / 退化 182 / **净 +2 张**（共 14,880 张验证图）。逐支持段：
头部 −3、中部 0、尾部 +5；没有任何一类的净变化超过 ±2 张。**与零无法区分。**

中心视图上 patch readout 是**真实且八轮同号**的正效应（八轮领先 L05 同轮
+0.049 ~ +0.438pp，均值约 +0.21pp；e14 center macro 75.3523% 已超 L05 最终
75.2497%），**但这个增益被固定解码整个吃掉了**——解码是预注册的比较口径，故
本配方关闭：不出包、不占平台名额、不派生读出变体扫描、不为 2 张图的效应用于多种子复现。
机器可读结果见[结果记录](../results/p75_patch_readout_result_20260928.json)。

**本轮结论的迁移含义**：在这个解码协议下，「推理前的小幅表示增益」会被
Flip + 温度 + 先验解码抹平。后续任何机制候选都应**直接以固定解码分为判据**，
不要用 center 分预筛。
