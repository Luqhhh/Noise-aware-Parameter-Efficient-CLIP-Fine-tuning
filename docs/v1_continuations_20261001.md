# WFT448、LR512 与 V2_FULL_LAST3_SWA：新增策略实现

用户要求实现固定训练路线。本段为**工程准备与真实 CPU 预检**，没有启动新训练、推理或平台提交。
起点仍是用户回填的 full v1 SWA **70.98600576861446%**；75% 的缺口约 1,503 张净修正，
来自成绩换算而非独立平台计数回执。不能将本段检查解释为新候选识别收益。
方案分支 `codex/v1_continuations_20261001`，独立目录
`/home/lux1/noise/worktrees/v1_continuations_20261001`；训练时每条路线另建独立方案分支/目录。

## 固定实现

入口位于 `reproducibility/aegis_f1/v1_continuation/`，四份配置位于 `configs/v1_continuation/`。
DEV 从原 train_dev SWA4–12 EMA 出发、复用其目标；FULL 从 full SWA 出发、复用 full 目标。
当前官方数据、类别顺序、父模型/SWA窗口、目标哈希和归档代码均校验。
归档 dev 的 `v1_pipeline.py` 与现在的差异仅是未使用的 `calibrate` 报告字段。
实现分别保留父绑定与当前绑定，并用 AST 核对其余所有有效代码一致；不跳过源码校验。

| 项目 | WFT448 | LR512 |
|---|---|---|
| 初始权重 | 当前阶段 v1 SWA，先加载、后转换 | 当前阶段 v1 SWA |
| 转换 | 合并全部48个实际参数化LoRA，全视觉塔解冻 | 保留rank32/alpha64，448位置编码aligned插值至512 |
| 分类头 | 原cosine head、projection、norm、logit scale保留 | 同父模型 |
| 实际输入 | 448×448 / 197 tokens | 512×512 / 257 tokens |
| 轮数 | 固定4 | 固定4 |
| 视觉/LoRA LR；head LR | 5e-6；5e-5 | 5e-5；2.5e-4 |
| 优化器 | AdamW，矩阵WD .05，bias/一维/标量不衰减 | 原v1 AdamW，LoRA WD0、head WD .01 |
| 新日程 | warmup .25轮 + cosine至0 | 原v1 OneCycle，重新建立4轮日程，warmup比例 .1 |
| 监督、样本、增强 | 原v1目标/可靠度；仅保留正权重样本；原RRC .8、Flip、RA2/7、Mixup .2、LS .1 | 相同 |
| batch | 原实际逻辑batch32；micro2、累积；原v1的最后不足batch保留 | 相同 |
| 平均 | EMA .999，末轮EMA + 固定EMA2–4权重均值 | 相同 |
| DEV评估 | 原六视图：resize448/512/576→crop448，各加flip | 固定512 center+flip |
| 校准 | 两条路线均先交付未校准输出；本实现bias关闭 | 相同 |

逻辑batch内只生成一次Mixup配对与比例，以全batch可靠度总和归一化；
micro拆分只影响显存，每个逻辑batch更新一次optimizer/scheduler/EMA。
使用activation checkpointing，不改变父模型的函数或头。
不恢复父optimizer、已经结束的scheduler或旧EMA累计状态。
DEV正权重119,074/133,815行，FULL正权重133,594/148,695行；
DEV每轮3,722次、共14,888次更新，FULL每轮4,175次、共16,700次更新。
正权重数量不是已确认错标数量或可恢复收益。

训练前自动做FP32/eval零更新核对。WFT比较选定父SWA与合并后的模型；
LR先核对原448函数，再变更实际分辨率。
LR的完整14,880张**未训练512父模型基线**必须在首次更新前落盘，
随后raw/EMA与它使用完全相同的512推理协议，区分输入变化和训练收益。
WFT同样先评估父模型。两个候选不会继承旧bias。

DEV每轮raw/EMA与两份固定末端导出均输出全量micro/macro、逐张修正/退化/净修正、预测变化。
训练前冻结尾部75类、头部75类、其余类、独立内容支持≤5的类；报告保留定义和各类有效监督量。
轨迹记录EMA更新数、初始化残留系数和相对父模型退化；最终配对CSV与每轮预测NPZ均绑定。
这些统计相对验证原标签；支持组和监督代理不是真实干净标签。
固定报告状态为 `evidence_requires_review`：小幅含噪local下降不自动淘汰，也不自动启动FULL。

FULL准备要求对应DEV完整四轮报告、证据文件哈希与非空支持分析，记录为何值得扩大投入。
运行中没有独立验证回调，禁止用已经纳入训练的val_dev选轮、选平均窗口或拟合bias。
FULL导出末轮EMA `training/selected.pt`，另保留固定 `training/ema_swa.pt`；
推理只接受其中一份明确checkpoint。提交阶段生成CSV/ZIP并运行9项校验。
不融合logits，不平均不同seed/分支，不用测试预测拟合参数。

## 资源与队列

真实GPU入口要求显式 `--execute`，只用本机CUDA。`nvidia-smi`发现在用进程就拒绝新任务，
没有守护排队、自动恢复、抢占或远端/NPU回退；释放资源后由执行者按顺序启动。
本次只读核对看到v3 PID8916在用CUDA，未操作它。
**仓库当前v2是 `engineering_ready / not_started`，并不是正在训练**；不据用户建议中的运行描述自动启动它。
v3维持384/两臂2轮、固定第二轮EMA的原主判据，raw曲线仅辅助解释短轨迹/EMA滞后。

队列：原v2/v3按既有授权协议收尾 → v2末段固定SWA（若快照齐全） → WFT448_DEV → LR512_DEV。
两个FULL均在各自DEV证据支持后另开分支；`SUPERVISION_LADDER`保持条件候选，
等待v3配对结果支持，未新增自动阶梯入口。v3两轮探针不优先占平台名额。
成熟包的平台优先级为v2完整末轮、WFT448_FULL、LR512_FULL；v2固定SWA按届时质量安排。
每日最多两次；平台达75%即冻结，不自动派生搜索或上传平台。

## V2_FULL_LAST3_SWA

v2保留原full_576第5轮raw `last.pt`的默认提交和原训练配方。
仅新增保存该段第3/4/5轮raw快照的能力：`epoch_03_raw.pt` 至 `epoch_05_raw.pt`。
`python3 -m v2.swa` CPU导出固定一次RAW3–5算术平均为单checkpoint，
绑定同一完整full_576、父权重、plan、类别、配置、更新次数、快照SHA和原last.pt。
不读取EMA作平均，不混分辨率、不扫描窗口；缺快照就拒绝，不重训凑快照。
`v2.runtime infer --checkpoint <export>/selected.pt`是独立候选入口；不提供该参数仍使用原last.pt。
已有v2 plan须按原配方重新prepare才能绑定新增快照代码；不改变v3运行目录或其冻结plan。

```bash
PYTHONPATH=reproducibility/aegis_f1 CUDA_VISIBLE_DEVICES='' python3 -m v2.swa \
  --plan <v2-prepared>/plan.json --output <independent-v2-swa-export>
PYTHONPATH=reproducibility/aegis_f1 python3 -m v2.runtime infer \
  --plan <v2-prepared>/plan.json --authorization <fixed-infer-authorization.json> \
  --checkpoint <independent-v2-swa-export>/selected.pt --output <independent-submission>
```

## 可重放命令

合并后每条真实实验另建worktree。例如WFT448_DEV（LR512另建`codex/lr512_dev_20261001`）：

```bash
git fetch origin
git worktree add -b codex/wft448_dev_20261001 /home/lux1/noise/worktrees/wft448_dev_20261001 origin/main
cd /home/lux1/noise/worktrees/wft448_dev_20261001
PYTHONPATH=reproducibility/aegis_f1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python3 -m v1_continuation.plan prepare \
  --config configs/v1_continuation/wft448_dev.yaml --output outputs/codex/wft448_dev_20261001/prepared
PYTHONPATH=reproducibility/aegis_f1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python3 -m v1_continuation.plan verify \
  --plan outputs/codex/wft448_dev_20261001/prepared/plan.json
PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 python3 -m v1_continuation.runtime train \
  --plan outputs/codex/wft448_dev_20261001/prepared/plan.json \
  --output outputs/codex/wft448_dev_20261001/run --execute
PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 python3 -m v1_continuation.runtime infer \
  --plan outputs/codex/wft448_dev_20261001/prepared/plan.json \
  --run-root outputs/codex/wft448_dev_20261001/run \
  --checkpoint outputs/codex/wft448_dev_20261001/run/training/selected.pt \
  --output outputs/codex/wft448_dev_20261001/submission --execute
```

LR512采用`configs/v1_continuation/lr512_dev.yaml`和独立输出；不另接旧512裁剪bias。
开发证据支持后，另建FULL worktree，使用`wft448_full.yaml`或`lr512_full.yaml`：

```bash
PYTHONPATH=reproducibility/aegis_f1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python3 -m v1_continuation.plan prepare \
  --config configs/v1_continuation/wft448_full.yaml --output outputs/codex/wft448_full_20261001/prepared \
  --development-report <verified-dev-run>/report.json \
  --support-assessment '记录总体、固定分组及raw/EMA轨迹的支持、退化与未知项'
```

占位文字不能代替真实证据分析；本段没有准备或启动FULL。prepare只读核对父FULL资产不是FULL授权。

## 本段验证与交付

- **142项CPU测试通过**：真实参数化合并、分类头保留、全塔梯度、真实512输入、全batch加权Mixup梯度、
  短micro/短最后batch、四轮更新、末轮EMA/EMA2–4均值、FULL验证阻断、忙卡拒绝、旧代码兼容、V2快照血缘/缺失门。
- 真实官方ViT-B/32 + 当前DEV SWA预检：WFT48模块合并、88,346,113可训练参数，
  FP32/eval logits最大绝对误差 **0**；LR5,102,593可训练参数、实际257 tokens，转换前原448核对误差 **0**。
- DEV/FULL父模型与各自目标覆盖均核对；逻辑batch32。CUDA没有初始化；新训练/测试推理均未启动。
- 两份DEV plan已CPU prepare/verify；证据见
  [校验记录](../results/v1_continuations_20261001/validation.json)、
  [真实预检](../results/v1_continuations_20261001/cpu_preflight.json)。没有新local或平台成绩。

CPU命令（在工程worktree执行）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python3 scripts/check_v1_continuation_cpu.py \
  --output outputs/codex/v1_continuations_20261001/cpu_preflight.json
PYTHONPATH=reproducibility/aegis_f1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python3 -m pytest \
  tests/test_v1_continuation.py tests/test_v2_swa.py tests/test_v2.py tests/test_v3.py \
  reproducibility/aegis_f1/tests/test_v1.py reproducibility/aegis_f1/tests/test_v1_export.py -q
```

测试曾被沙箱禁止本机multiprocessing socket；获准在沙箱外重跑后142项通过。
现役可提交包仍为
`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/{pred_results.csv,submission.zip}`。
37,444行、9项通过、ZIP/CSV一致的来源为
[full v1验证](../results/v1_full_swa_20260930/final_validation.json)与
[平台包复核](../results/v1_full_swa_platform_20261001/artifact_verification.json)。
本段引用既有校验，不为工程准备另训练新包。完成提交、方案push及main集成后，在该工程检查点暂停。

收尾补充：v3原controller在2026-10-01 01:26:03 CST自行回报`completed / delivered`，
CSV/ZIP与末轮EMA摘要写入其原`status.json`，平台分仍为空。本工程没有干预该任务，
该状态回报不代替独立配对/提交校验登记，也没有据此自动启动新训练。
