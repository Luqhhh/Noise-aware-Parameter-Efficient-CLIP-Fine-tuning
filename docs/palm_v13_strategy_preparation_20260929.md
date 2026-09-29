# V13 策略迁移：分辨率 / 学习率阶梯与全量续训

实验标识 `P75_RESOLUTION_LADDER`。2026-09-29 用户授权本地生成代码，随后明确只迁移策略。
本段交付为 `engineering_ready`，项目训练状态 `not_started`；**未启动 GPU 实验**。
CPU 检查通过不代表识别提分，本项目的新 local / platform 指标均为空。

策略参考 [Palm 的全量微调方案](https://github.com/Palm0palM/aic-noisy-clip/tree/fcc37813c8bea3f2761707a145b2955ce95c097e/solutions/clip-vitb32-full-finetune)，
固定来源 commit `fcc37813c8bea3f2761707a145b2955ce95c097e`。参考仓库报告的 76.5890% 仅作背景，
本段没有重放该成绩。保留四份原始参数配置及 SHA256 来源清单供审查；辅助函数沿用 MIT 许可和版权声明。

## 迁移后的策略与数据协议

从现有 OpenAI 官方 CLIP ViT-B/32 开始，训练整个视觉塔和 512→750 线性分类头。
使用项目现有原始图像与 FP32 参数、训练时 BF16；位置编码复用 Aegis 的插值实现。
没有新增 Python 依赖，模型初始化不下载权重。

保留 RRC `[0.35,1.0]`、翻转、ColorJitter 0.5、RandAugment 2/7、RandomErasing 0.3，
以及 smoothing 0.15、Mixup 0.2 / CutMix 1.0、混合概率 0.8 下的 soft CE。
按训练类频次的平方根倒数加权、有放回采样；类别的总抽样权重仍与频次平方根成正比，不能称完全均匀类别先验。
AdamW `weight_decay=0.15`，bias / 一维参数不衰减；每段重新建立 warmup+cosine、优化器和 EMA 0.9995。

前三段固定使用本项目 `133,815 train_dev / 14,880 val_dev`，已有 decoded-RGB 内容组跨划分交集为 0。
这些是像素内容组，不能称完整真实来源隔离。父模型按固定中心视图的验证 macro 选择 raw/EMA，再以
`macro + 0.5×micro` 选择该段 epoch；每段全部轮数完成后才封存可供下一段使用的父权重。

最后一段使用全部 **148,695** 张官方训练图，包括此前验证图；不执行验证、不用训练内分数选择 checkpoint。
最终固定取第 5 轮 raw `last.pt`。没有沿用外部排除清单、纠正标签、JPEG 缓存、任务权重或作者的 95/5 随机划分。

## 固定起始配置

参数是策略迁移的起始配置，尚未实测本机 GPU 成本或验证最优性。

| 段 | 分辨率 | 轮数 | 训练行数 | 逻辑批次 | 骨干 / 分类头 LR | warmup 轮数 | 更新次数 |
|---|---:|---:|---:|---:|---|---:|---:|
| s1_384 | 384 | 10 | 133,815 | 96 | 3e-5 / 5e-4 | 1.0 | 13,930 |
| s2_448 | 448 | 6 | 133,815 | 96 | 1.5e-5 / 3e-4 | 0.5 | 8,358 |
| s3_576 | 576 | 4 | 133,815 | 80 | 8e-6 / 2e-4 | 0.4 | 6,688 |
| v13_final | 576 | 5 | 148,695 | 80 | 4e-6 / 1.2e-4 | 0.3 | 9,290 |

共 **38,266** 次逻辑更新，按有放回采样和 `drop_last=True` 计算。`min_lr_ratio=0.02`、梯度裁剪 1.0。
当前资源配置为 micro batch 2、2 个数据 worker、activation checkpointing。
Mixup/CutMix 在完整逻辑批次上只做一次，各 micro batch 按实际行数累积平均损失，每个逻辑批次只更新一次优化器 / scheduler / EMA。
这些设置已验证 CPU 逻辑，8GB 级显存下的 BF16 前向、反向与速度仍待后续获准的本地 CUDA 检查。

## 已交付入口与本段实际命令

方案分支 `codex/palm_v13_prepare_20260929`，独立目录
`/home/lux1/noise/worktrees/palm_v13_prepare_20260929`。
主配置为 [recipe.json](../configs/palm_v13_20260929/recipe.json)，准备代码在
[palm_v13](../reproducibility/aegis_f1/palm_v13/)，结构化核验记录为
[results JSON](../results/palm_v13_strategy_preparation_20260929.json)。

以下命令在该方案目录执行，只使用 CPU：

```bash
PYTHONPATH=reproducibility/aegis_f1 CUDA_VISIBLE_DEVICES='' python3 -m palm_v13.plan prepare \
  --output outputs/codex/palm_v13_prepare_20260929/prepared
PYTHONPATH=reproducibility/aegis_f1 CUDA_VISIBLE_DEVICES='' python3 -m palm_v13.plan verify \
  --plan outputs/codex/palm_v13_prepare_20260929/prepared/plan.json
CUDA_VISIBLE_DEVICES='' python3 scripts/check_palm_v13_strategy_cpu.py \
  --output outputs/codex/palm_v13_prepare_20260929/cpu_forward.json
PYTHONPATH=reproducibility/aegis_f1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python3 -m pytest \
  tests/test_palm_v13_strategy.py reproducibility/aegis_f1/tests/test_model.py -q
```

准备输出包含 manifest、固定 split、四段 JSON 和 `plan.json`，绑定官方数据审计、750 类映射、源代码与官方权重 SHA256；拒绝覆盖已有段。
唯一交付计划位于该独立目录的 `outputs/codex/palm_v13_prepare_20260929/prepared/plan.json`。
准备过程复用既有解码审计，不扫描测试图或生成测试预测。运行前训练图逐文件 SHA256 校验另计入预算。

实测 **45 passed**，其中 26 项为新策略检查。覆盖真实 CPU 合成图像的小模型段衔接、检查点保存 / 封存 / 严格加载、
学习率计数、分辨率插值、激活重计算梯度、完整批次与 micro batch 梯度等价、CutMix 面积、采样分布、内容组隔离、阶段血缘、
默认授权拒绝、预算停止，以及提交的四位标签与 ZIP/CSV 字节一致性。
合成图像检查不使用项目图像训练。

真实官方 CLIP 的 CPU 224 输出与原视觉实现最大绝对差 **0.0**；384 / 448 / 576 前向均得到有限 `[1,750]` logits。
88,233,966 个训练参数均为 FP32，全程 `torch.cuda.is_initialized()==False`。
这项实测是接口与数值一致性检查，没有识别准确率、GPU 性能或提分结论。

## 后续执行入口与本次停止边界

`palm_v13.runtime` 提供 `probe / train / infer`，本段均未调用 GPU 执行。
`probe` 只允许 3–10 次更新，成本包含原始图像解码 / 增强 / DataLoader 等待、完整 raw+EMA 验证及实际检查点写入。
成本快照用于测量后删除，不作为候选模型。`train` 按段独立执行；训练开始后异常或时限后写 `interrupted.pt` 和状态，
未完成段不接受为父权重，不自动恢复、串行启动下一段或开启后台服务。

后续启动需用户的新 GPU 指令、当前误差 / 有限探索证据和有限预算。
[authorization.example.json](../configs/palm_v13_20260929/authorization.example.json) 默认 `authorized=false`；
运行入口要求操作、阶段和 plan SHA256 一致，以及测得的完整成本满足 20% 预算余量。
dev 阶段可用 probe 的完整验证成本；全量段的验证成本为 0，仍须单独测量 576 的更新成本和检查点开销。
本段没有现成训练授权或捏造 GPU 时长。可用 `python3 -m palm_v13.runtime --help` 查看参数，需设置上述 `PYTHONPATH`。

未来推理使用同一个最终 raw checkpoint 的固定四视图 `resize576,center,flip,resize806.4`，概率等权求和后 argmax。
不拟合测试统计、prior 或温度，不读取测试标签。输出 `pred_results.csv` / `submission.zip`，调用现有 750 类提交检查器。
最终全量段没有独立本地分；平台分仅在实际平台回报后登记。

本段没有新预测包。现役包仍为
`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/{pred_results.csv,submission.zip}`。
本次 CPU 复核为 **37,444 行、9/9 校验通过、ZIP/CSV 字节一致**，具体摘要见 results JSON；现役平台仍为 **66.94797564362783%**。
方案验证后推送方案分支，在 main 集成目录以**自动模式** pull / 合并 / 复核 / push，保留原有 `.gitignore` 本地修改，
不执行额外 `stash pop`；提交及合并 SHA 在交付消息记录。本检查点停止，不启动 GPU、上传或追加训练。
