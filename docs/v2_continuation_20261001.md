# V2_CONTINUATION_20261001

用户提供 `v2_s1_384_bundle_20261001_r3.zip`，授权准备完成后继续固定v2。
最新要求是每20分钟监控；全部完成后下载权重和提交包，核验成功后关闭服务器。
此前“保持开机”已被2026-10-02的新指令覆盖；完成并核验下载之前仍保持运行。
独立分支 `codex/v2_continuation_20261001`，
本机目录 `/home/lux1/noise/worktrees/v2_continuation_20261001`。
本记录封存实现、CPU验证和真实启动；正式阶段训练尚在持续执行。

## 2026-10-02进度与收尾策略

北京时间11:10实测full_576已完成8,200/9,290次更新（88.3%），处于第5轮；
controller PID5764存活，健康状态running，磁盘余量5.95GB。尚未生成最终提交包。
S2已完成6轮/8,358次更新，独立留出所选EMA micro76.955645%、macro76.101112%；
S3已完成4轮/6,688次更新，所选EMA micro77.473118%、macro76.592670%。
上述是14,880张独立DEV指标，full阶段没有独立留出；没有新平台分。
full于04:26:12启动，预计训练及固定推理在12:00–15:00结束，下载另计；非保证截止时间。

用户最新明确授权“完成后把权重和提交包下载到本地，然后关机”。
收尾服务 `noise-v2-completion-20261002.service` 只由已完成、已核验CSV/ZIP的监控触发，
使用 [deliver_v2_39385.py](../scripts/deliver_v2_39385.py)。私有配置绑定上述固定plan SHA，
目标Windows目录为 `C:\Users\lqh22\Downloads\v2_continuation_20261001`。
保存S2/S3 best、full last、full RAW第3/4/5轮快照、CSV/ZIP和报告；另保存配置、binding、
状态、历史、holdout及日志的 `delivery_metadata.tar.gz`。预计权重约5.3GB。
32MiB完整块断点续传，不完整块重试，最终逐文件核对服务器SHA-256。
实际提交的checkpoint SHA必须与full last一致；所有阶段complete、冻结plan匹配、
本地正式9项检查通过并写入本地收据后，才再次检查controller完成且GPU无其他任务，
执行 `sync && /usr/bin/shutdown`，随后3次SSH不可达检查。失败不提前关机、不自动重训。
此收尾服务不改正在运行的冻结代码、plan、配置及推理政策。
最终仍为第5轮raw单checkpoint、原四视图，无固定均衡先验bias，也未切换SWA。

11项收尾测试通过，包括未完成/未授权/plan不匹配、缺少或损坏权重、提交校验失败的关机阻断，
中断续传、最终SHA与续传身份检查，以及收据先于关机和关机幂等性；原始记录见
[completion_tests.log](../results/v2_continuation_20261001/completion_tests.log)。
这仅说明收尾实现已验证，不能据此声称当前训练、下载或关机已完成。

## 实际启动检查点

北京时间2026-10-01 15:56:56持久controller PID5764启动，进入s2_448成本检查。
服务器代码commit `ca590906c786c6796ed2f54d3d9281cc44fbd2ae`，原s1已导入；
新冻结plan SHA `596454f233830c3dde0fb2cdfa5252de4b0207c58e6717d72fb4331d2ee3428e`。
服务器96项回归通过（4.13秒），[原日志](../results/v2_continuation_20261001/server_tests.log)。
最初小时watcher PID5804与小时timer激活，首次服务成功，16:00也成功核查；
[历史启动记录](../results/v2_continuation_20261001/health_watcher.json)、[首次健康记录](../results/v2_continuation_20261001/initial_health.json)。
随后用户明确改为每20分钟：旧watcher已停止、旧timer disabled/inactive，
新watcher PID6439、interval1200；[新进程记录](../results/v2_continuation_20261001/health_watcher_20min.json)。
新timer `noise-v2-monitor-20261001.timer` enabled/active，calendar每小时00/20/40分，
首次服务Result=success/ExecMainStatus=0，下一次北京时间16:20；
[定时器核验](../results/v2_continuation_20261001/monitor_schedule_20min.log)、
[20分钟监控首次真实报告](../results/v2_continuation_20261001/health_20min_initial.json)。

16:20的本机自动读取遇SSH断连，已保留异常历史，未影响controller或服务器独立监控。
随后精简健康传输字段、启用SSH压缩/显式cipher与无QoS标记，并为SSH子进程设置60秒内部超时、
外层健康读取75秒超时，使三次重试能在service的300秒内给出结果。
16:32:54同一systemd服务实测成功、告警清除，健康状态running、原controller存活，
已读到s2 epoch0、600次真实更新、loss3.2110589445、elapsed1468.289秒。
[修复后真实监控报告](../results/v2_continuation_20261001/health_20min_recovered.json)。
连接设置保存在私有transport，不改动训练代码或plan；无需等待旧每小时定时器。

s2探针529.304秒完成，8次更新全部通过有限梯度检查，完整14,880张raw/EMA holdout两遍，
验证447.775秒；峰值2,717,040,640 bytes（约2.53 GiB），checkpoint写盘3.930秒，
更新计时包含数据加载。保守取除首步外最慢4.574836秒/更新；
[成本原输出](../results/v2_continuation_20261001/s2_cost.json)。
估算s2 41,033.761秒（11.398小时）、上限57,448秒（15.958小时），
北京时间16:05:49已自动正式开训；[原预算](../results/v2_continuation_20261001/s2_authorization.json)。
启动日志已观察到epoch0、200/8,358次更新，loss3.6736203432、elapsed529.795秒，后续监控已到600次；
这是正式训练进度，尚无完整阶段验证结果、完整候选或新提交包。
前200步含初始化均摊2.649秒/更新，低于探针保守上界，不能当作576速度测量。
完整v2（含固定推理）暂估还需18–36小时，即10月2日上午至10月3日凌晨；
该区间基于448实测、三个后续阶段共24,336次计划更新与未测576路径，s3探针后再校准，非保证截止时间。

## 原始s1核验

源文件 `/mnt/c/Users/lqh22/Downloads/v2_s1_384_bundle_20261001_r3.zip`，
1,349,808,241 bytes，SHA-256 `a12bf57cb79763dd45f70fda5919b1436abc23ccacbc2650bd9755d8c1f16b10`。
已传到服务器并校验解压，来源目录只读：
`/root/autodl-tmp/noise/handoff/s1_pending/v2_s1_384_bundle_20261001`。
全部归档CRC、45项JSON资产和46项SHA列表通过；详见
[完整性记录](../results/v2_continuation_20261001/bundle_integrity.json)。

原 `best.pt` SHA-256 `3643b9a10a8cf3b330db6537a4b1d9c4cc630a78649099fce9f81b7f1538b817`。
s1完成10轮、13,930次更新，所选epoch9 raw；独立留出micro 75.1546%、macro 74.2294%。
这是队友原始s1结果，没有本段新候选/平台分。88233966参数严格加载逐张量一致，raw/EMA全部有限，
CPU448前向 `[1,750]` 通过；[验证记录](../results/v2_continuation_20261001/parent_cpu_validation.json)。

交接包导出plan SHA `1aea2bbf37162ba44d7741a908fd540070d585658c1fbb360e1740426da4c727`，
checkpoint记录plan SHA `be3206eff072bd54c3af2f25fb11aea56daa187aaa02d16f54a0bb0aadcb873b`，二者不同。
不能声称导出的plan就是当时训练plan。实际冻结输入、源代码、stage配置、manifest、split、
完成终点和原sidecar分别核验；保留两种SHA及原文件，没有改写checkpoint或其binding。
来源实现与基线 `bde43f7f1fc9f21c2172c7f295babf0368bbb455` 归一化换行相等；
共享v1_strategy只比对V2实际使用的WeightAverage/using_weights AST。

迁移入口保留原s1配置/manifest/split的完整字节与原checkpoint/sidecar/status，
后续s2/s3/full配置只转换资源路径与Linuxworkers，训练参数逐字段相等。
实验血缘ID仍为 `V2_20260929` / `v2_recipe_1` / `20260921`，本段执行标识另记。
禁止在导入工作区重新train/probe s1；既有read_checkpoint/check_checkpoint防护保持原样。
[本机导入收据](../results/v2_continuation_20261001/s1_import.local.json)、
[服务器原收据](../results/v2_continuation_20261001/s1_import.server.json)显式记录哈希差异。
[实际服务器plan](../results/v2_continuation_20261001/server.plan.json)保留执行参数和绑定。

## 固定执行与边界

服务器独立worktree：`/root/autodl-tmp/noise/worktrees/codex/v2_continuation_20261001`。
独立workspace：`/root/autodl-tmp/noise/runs/codex/v2_continuation_20261001/prepared`。
使用已有独立venv、官方CLIP、当前750类数据和冻结133815/14880 split；
recipe为 [continuation_39385.recipe.json](../configs/v2/continuation_39385.recipe.json)。
micro16、worker2、梯度checkpoint；原EMA/优化器/LR/监督/增强/日程不变。
Torch2.12.1/cu130与队友Windows Torch2.6/cu126不同，不声称逐位复现。

```bash
cd /root/autodl-tmp/noise/worktrees/codex/v2_continuation_20261001
export PYTHONPATH="$PWD/reproducibility/aegis_f1"
export OMP_NUM_THREADS=2
PY=/root/autodl-tmp/noise/venv/bin/python
$PY -m v2.plan prepare --recipe configs/v2/continuation_39385.recipe.json \
  --output /root/autodl-tmp/noise/runs/codex/v2_continuation_20261001/prepared
$PY -m v2.import_parent \
  --bundle /root/autodl-tmp/noise/handoff/s1_pending/v2_s1_384_bundle_20261001 \
  --plan /root/autodl-tmp/noise/runs/codex/v2_continuation_20261001/prepared/plan.json
$PY -u -m v2.controller \
  --plan /root/autodl-tmp/noise/runs/codex/v2_continuation_20261001/prepared/plan.json
```

controller按s2_448 → s3_576 → full_576串行执行。每个DEV先8次真实逻辑更新、完整raw/EMA holdout、
真实checkpoint写入的成本检查，探针不产候选。训练估算包含全部更新/每轮验证/写盘；
预算为ceil(估算×1.4)，保留至少20%。full使用s3实测576成本，不对full人口额外探测/选模。
推理只用原固定full第5轮raw last、四视图，37,444行CSV/ZIP与9项检查；平台上传由用户决定。
遇其他GPU任务等待，已有部分训练/失败探针要求诊断，不能隐式重复或删除产物。

## 每20分钟监控与交付

服务器 `python -u -m v2.health --watch --interval 1200 --plan ...` 持久进程每20分钟记录，
独立flock防重复；读取真实updates/loss/epoch、进程命令、GPU、磁盘、日志新鲜度和失败状态。
报告在workspace的 `monitoring/latest.json` 与各次JSON，完成后停止服务器监控进程；
关机顺序以本页2026-10-02最新收尾策略为准。
本机systemd用户timer `noise-v2-monitor-20261001.timer` 同时每20分钟SSH读取，
连接失败重试3次；异常写 `ALERT.json`，不会自动重跑未完成训练。
本机timer依赖本机Linux运行；服务器监控独立于本机。原小时timer不再运行。
训练授权JSON保留生成时的小时监控说明作为历史，实际周期以1200参数和当前timer为准；
修改监控周期未修改冻结的训练源文件、plan、训练配置或正在执行的controller PID5764。

本机报告目录：`outputs/codex/v2_continuation_20261001/monitoring/`。
最终自动取回CSV/ZIP/report/服务器检查记录，核对SHA，执行本机9项正式检查；
通过后触发权重下载与验证服务，再停本机timer，小包放同一执行目录的 `submission/`；
完整下载交付Windows下载目录，全部核验后按最新授权关机。平台上传由用户决定。
所有认证配置只在私有输出目录，禁止入Git。

CPU回归77项通过（3.69秒），另19项官方模型检查通过（1.42秒）；含导入s1重跑阻断、训练参数迁移、manifest移植、
成本绑定、full无holdout预算及死进程/过期日志/真实更新识别；见
[CPU记录](../results/v2_continuation_20261001/cpu_tests.json)。本机GPU留给已有v1任务。
尚无本段可提交包，现役包和9项校验继续见[当前入口](current_execution_plan.md#交付与历史入口)。
