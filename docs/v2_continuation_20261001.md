# V2_CONTINUATION_20261001

用户提供 `v2_s1_384_bundle_20261001_r3.zip`，授权准备完成后继续固定v2，以及每小时监控。
最新要求是保持服务器开机。独立分支 `codex/v2_continuation_20261001`，
本机目录 `/home/lux1/noise/worktrees/v2_continuation_20261001`。
本记录先封存实现和CPU验证；GPU成本检查、正式启动与监控激活另记实测。

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
[导入收据](../results/v2_continuation_20261001/s1_import.local.json)显式记录哈希差异。

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

## 每小时监控与交付

服务器 `python -u -m v2.health --watch --interval 3600 --plan ...` 持久进程每小时记录，
独立flock防重复；读取真实updates/loss/epoch、进程命令、GPU、磁盘、日志新鲜度和失败状态。
报告在workspace的 `monitoring/latest.json` 与逐小时JSON，完成后停止监控进程，保持服务器开机。
本机systemd用户timer `noise-v2-hourly-20261001.timer` 同时每小时SSH读取，
连接失败重试3次；异常写 `ALERT.json`，不会自动重跑未完成训练。
本机timer依赖本机Linux运行；服务器小时记录独立于本机。

本机报告目录：`outputs/codex/v2_continuation_20261001/monitoring/`。
最终自动取回CSV/ZIP/report/服务器检查记录，核对SHA，执行本机9项正式检查；
通过后停本机timer，产物放同一执行目录的 `submission/`。不关机、不平台上传。
所有认证配置只在私有输出目录，禁止入Git。

CPU回归77项通过（3.69秒），含导入s1重跑阻断、训练参数迁移、manifest移植、
成本绑定、full无holdout预算及死进程/过期日志/真实更新识别；见
[CPU记录](../results/v2_continuation_20261001/cpu_tests.json)。本机GPU留给已有v1任务。
尚无本段可提交包，现役包和9项校验继续见[当前入口](current_execution_plan.md#交付与历史入口)。
