# V2_PREPROJECTION768_FULL_20261003：完整 768 维路线实现

**2026-10-04方向更新**：后续优化主线转向v1；本完整v2+768实现保留，长训练未启动，
不列入默认后续队列。提交基准继续为v2六视图logits求和＋固定bias的74.89317380621728%，
见[当前方向与晋级规则](v1_priority_20261004.md)。以下保留本段实现和验证记录。

用户已明确“768”为投影前特征，并要求“实现 v2+768”。本段交付从官方 CLIP
初始化的完整四阶段实现、配置与 CPU 验证，未启动真实数据训练或测试集推理；没有新准确率、
平台分或新候选提交包。分支为 `codex/v2_preprojection768_full_20261003`。

## 实现与此前短续训的关系

复用已推送分支 `codex/v2_preprojection768_pair_20261002` 的 `61ecddb` 中
`V2FeatureClassifier` 与投影折叠实现，补上完整路线的模型工厂、配置绑定、阶段衔接、
checkpoint 校验、推理及控制器入口。此前固定 512 步短续训对控制净 +4、对父净 −17，
仍按原结论关闭；这些数字既不证明也不否定本次完整路线的收益。没有重复启动旧实验。

本路线直接使用 `ln_post` 后、`visual.proj` 前的 768 维 CLS 特征，分类头为
`Linear(768, 750)`，移除 768→512 投影。视觉骨干从 SHA 固定的官方 OpenAI
ViT-B/32 初始化，全参数训练。复用的读出实现使用 FP32，视觉塔保留原 BF16 autocast
策略；因此与原 v2 比较时，除维度/投影变化，还要披露读出精度变化。
默认旧配方仍走原 `V2Classifier`，没有改变其数值计算。

`feature_dim`、特征来源和读出精度写入 recipe、准备计划、各阶段配置、checkpoint
顶层及 binding；同时检查分类头形状和投影是否存在。旧 512 权重不能通过修改标签混入
新路线；后续阶段严格加载选出的单份 raw/EMA 768 权重。固定 raw 快照和可选 SWA
导出保留同一架构信息，本段没有执行 SWA。

## 固定配置与执行边界

配置：[v2_preprojection768_full_20261003.json](../configs/v2_preprojection768_full_20261003.json)。
保留原 v2 日程、增强、优化器、EMA 0.9995、阶段间重置和 full 末轮 raw 策略。
本机 microbatch 为 2，启用 activation checkpointing；CUDA 显存和耗时尚未实测。

| 阶段 | 图像尺寸 | 轮数 | 逻辑 batch | 更新数 | 训练/留出张数 |
| --- | ---: | ---: | ---: | ---: | --- |
| s1_384 | 384 | 10 | 96 | 13,930 | 133,815 / 14,880 |
| s2_448 | 448 | 6 | 96 | 8,358 | 133,815 / 14,880 |
| s3_576 | 576 | 4 | 80 | 6,688 | 133,815 / 14,880 |
| full_576 | 576 | 5 | 80 | 9,290 | 148,695 / 0 |

合计 38,266 次逻辑更新。full 已纳入原留出，不能报独立验证准确率。
本路线推理入口保留原 v2 四视图概率求和取 argmax，生成 raw CSV/ZIP 和架构记录。
此前六视图 logits 求和的现役 512 双包已单独交付，本段没有把它冒充为 768 产物。

独立工作目录：`/home/lux1/noise/worktrees/v2_preprojection768_full_20261003`。
下列 CPU 命令已在该目录执行：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python3 scripts/check_v2_cpu.py \
  --recipe configs/v2_preprojection768_full_20261003.json --backward \
  --output outputs/codex/v2_preprojection768_full_20261003/cpu_model_check.json

PYTHONPATH=reproducibility/aegis_f1 python3 -m v2.plan prepare \
  --recipe configs/v2_preprojection768_full_20261003.json \
  --output outputs/codex/v2_preprojection768_full_20261003/prepared

PYTHONPATH=reproducibility/aegis_f1 python3 -m v2.plan verify \
  --plan outputs/codex/v2_preprojection768_full_20261003/prepared/plan.json

PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python3 -m pytest tests/test_v2.py tests/test_v2_768_full.py tests/test_v2_swa.py \
  tests/test_v2_continuation.py tests/test_ndcw.py tests/test_v2_fixed_prior.py \
  tests/test_v2_sixview_bias.py tests/test_v2_sixview_logit_sum.py -q \
  --junitxml=outputs/codex/v2_preprojection768_full_20261003/targeted_tests.xml

PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python3 -m pytest reproducibility/aegis_f1/tests -q \
  --junitxml=outputs/codex/v2_preprojection768_full_20261003/aegis_tests.xml
```

`prepare` 拒绝覆盖已有目录；已准备好的计划只需 `verify`。其他 checkout 应使用新的独立
输出目录重新准备，不复制带有旧绝对路径的计划。准备不占用 GPU，也不创建训练授权。

以下为已实现但**本段未执行**的本机完整运行入口：

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -m v2.controller \
  --plan outputs/codex/v2_preprojection768_full_20261003/prepared/plan.json \
  --from-official
```

该入口要求新 768 配方与官方初始父权重，禁止导入旧 S1；串行执行四阶段，各 DEV 阶段先做
8 次逻辑更新及完整留出的成本探针，full 复用 S3 同分辨率成本，再执行推理及提交检查。
运行前等待已有 CUDA 任务，异常或不完整旧目录会停止并要求诊断，不自动重复。
后续应先核对本机成本和投入依据，再决定长训练；当前未加入运行队列，未连接远端或 NPU。

## 实测验证

- 148 项针对性测试全部通过，覆盖旧 v2、ND-CW 兼容性，768 梯度、混合精度、
  错误架构拒绝、raw/EMA 阶段继承、full 快照、SWA 导出以及合成图的真实 CSV/ZIP 校验流程。
- 官方模型 CPU 前向：224 原生路径与新特征路径差值 0；384/448/576 均输出 `[1, 750]`。
  参数量 88,032,750；全部 FP32 存储，153 个参数张量梯度有限。合成反向无 optimizer 更新，
  不将随机输入 loss 作为模型效果。
- 正式数据准备校验通过：133,815/14,880，内容组交叉为 0；四阶段均绑定 768 架构。
  计划 SHA 为 `4be63bf6c770e34fc2d5ec892a3837cf3f78f8f32a488afbe434a52a1a732756`。
- Aegis 套件：797 passed、15 skipped；2 项失败均来自 `test_scope_protocol.py`
  缺少历史 `scope_k2/.../best.pt`（其中一项期待摘要不匹配但先遇到文件缺失），符合
  `CLAUDE.md` 已登记的资产缺失例外。不是全套全绿。

核验资料：[模型检查](../results/v2_preprojection768_full_20261003/cpu_model_check.json)、
[计划摘要](../results/v2_preprojection768_full_20261003/plan_summary.json)、
[测试与文件摘要](../results/v2_preprojection768_full_20261003/validation.json)。

## 现役产物与检查点

现役平台分仍为用户报告的 **74.6688387992735%**，包为
`C:\Users\lqh22\Downloads\v2_continuation_20261001\submission_bias_sixview\submission.zip`，
本段重新核对 ZIP SHA 为 `7164b26d9662a8c0e78c9c0636393efe0927739cb84df8dc88ed33e35f47659a`。
37,444 行及正式九项提交检查来源：[平台登记](../results/v2_sixview_platform_20261003/record.json)。
本次实现没有新的准确率结论。方案验证后推送方案分支，在 main 集成目录采用
`git pull --rebase --autostash origin main` 自动模式同步、合并、重验及推送，随后暂停。
