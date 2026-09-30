# v3第2轮后暂停：V3_PAUSE_EPOCH2_20260930

用户明确要求“第二轮完成后暂停”。当前执行的是原标签对照臂第2轮，
因此暂停边界为该臂第2轮验证及可恢复epoch checkpoint完成后。
这是对此前“v3继续”的新约束，不继续进入v1监督臂，不自动恢复。
不改配方、模型、优化器、原固定计划或训练实现，保留原内存现场。

## 已启用监控

2026-09-30 15:55:21 CST，用户服务`noise-v3-pause-epoch2-20260930.service`
已启动，独立监控进程PID471892，状态`armed_waiting_epoch`。
该启动快照当时为等待状态；后续已达到边界并暂停，见下方实际暂停结果。
实时监控状态为
`/home/lux1/noise/worktrees/v3_pause_epoch2_20260930/outputs/codex/v3_pause_epoch2_20260930/pause_status.json`。
最终暂停时该文件自动记录时间、全部暂停PID、T状态、checkpoint SHA和两轮指标，
并更新原v3 `handoff_exif_recovery/status.json`为`paused_by_user`。

触发必须同时满足：

- 原计划`original/epoch02.pt`和`epoch02.binding.json`存在。
- `history.json`已写完第2轮验证，末记录为epoch2、2786次有效更新。
- checkpoint sidecar绑定原计划SHA、original臂、`probe=false`、`complete=true`。

监控只识别原runner PID361828和训练PID361861，启动时核对命令行和进程生命周期。
每50ms检查已落盘边界，达到条件后先STOP训练，再STOP原runner及其当前子进程，
过滤僵尸进程，不操作其他任务。暂停后核对T状态和epoch02 checkpoint实际SHA。
保留原随机数、模型、优化器、数据迭代和文件现场；恢复需用户新指令。
当前训练实现先写epoch checkpoint和history，再导出selected及进入下一臂，
监控选择前一个已验证epoch边界，避免在写epoch checkpoint期间暂停。
5项CPU检查通过，覆盖缺失输出不触发、轮次与更新数/绑定一致，以及PID身份解析。

确切启动代码`ee28edf`，独立分支/worktree `codex/v3_pause_epoch2_20260930`。
固定计划SHA保持`615fc646d20cf5d7b02f64e3adcc4d659bd607036302b895d08901140fa9e224`。
运行入口[scripts/pause_v3_after_epoch.py](../scripts/pause_v3_after_epoch.py)、
[检查](../tests/test_v3_pause_watcher.py)、[执行状态](../results/v3_pause_epoch2_20260930/execution.json)
与[启动快照](../results/v3_pause_epoch2_20260930/watcher_initial_status.json)。

## 可重放命令

工作目录`/home/lux1/noise/worktrees/v3_pause_epoch2_20260930`。
服务已启动，不应再次运行相同暂停监控。

```bash
python3 -m pytest tests/test_v3_pause_watcher.py -q
systemd-run --user --unit=noise-v3-pause-epoch2-20260930 \
  --property=WorkingDirectory=/home/lux1/noise/worktrees/v3_pause_epoch2_20260930 \
  --property=StandardOutput=append:/home/lux1/noise/worktrees/v3_pause_epoch2_20260930/outputs/codex/v3_pause_epoch2_20260930/watcher.log \
  --property=StandardError=inherit /usr/bin/python3 -u scripts/pause_v3_after_epoch.py \
  --status /home/lux1/noise/worktrees/v3_after_v1_20260930/outputs/codex/v3_after_v1_20260930/handoff_exif_recovery/status.json \
  --arm-dir /home/lux1/noise/worktrees/v3_after_v1_20260930/outputs/codex/v3_after_v1_20260930/prepared_exif_recovery/runs/train/original \
  --epoch 2 \
  --report /home/lux1/noise/worktrees/v3_pause_epoch2_20260930/outputs/codex/v3_pause_epoch2_20260930/pause_status.json
```

## 交付范围

本段交付已验证的边界暂停机制及已运行监控，不是v3两臂训练完成。
用户新暂停指令优先于原计划持续运行至两臂出包的约定。v3最终候选与提交包尚未完成；
不为凑新包继续跑v1监督臂。现役已校验包仍是
`/mnt/c/Users/lqh22/Desktop/v1_full_swa_submission.zip`，
SHA256 `1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`，
[9项校验](../results/v1_full_swa_20260930/submission_check.log)及
[full v1交付记录](v1_full_swa_20260930.md)保留。

## 实际暂停结果

2026-09-30 **16:25:58 CST**自动触发，状态`paused_by_user`。
原标签对照臂第2轮验证、epoch02 checkpoint及history已完成，2786次有效更新。
原runner/训练/资源追踪及当前两个加载进程PID
361828、361861、362190、461544、461551均为T，随后再次只读核对五进程T状态。
两个加载PID随轮次变化，由监控动态识别，原先僵尸加载进程未被操作。
`v1_supervision`目录尚未创建，没有启动监督臂，内存训练现场保留，不自动恢复。

第2轮raw macro/micro **59.6075% / 60.5914%**，EMA **53.0370% / 54.0995%**。
这些是14,880张val_dev上的对照臂指标，两臂比较及v3最终候选尚未完成。
Checkpoint：
`/home/lux1/noise/worktrees/v3_after_v1_20260930/outputs/codex/v3_after_v1_20260930/prepared_exif_recovery/runs/train/original/epoch02.pt`。
实际SHA256与binding侧车重新核对一致：
`08d4986158d388e51a311093c3fdb7c36d36ea478defb0a308ee099c3d5073e9`。

暂停[实际结果](../results/v3_pause_epoch2_20260930/pause_result.json)、
[原任务状态](../results/v3_pause_epoch2_20260930/paused_v3_status.json)、
[checkpoint绑定](../results/v3_pause_epoch2_20260930/epoch02.binding.json)、
[两轮对照臂指标](../results/v3_pause_epoch2_20260930/original_history.json)及
[已更新执行记录](../results/v3_pause_epoch2_20260930/execution.json)已归档。
现役可提交包仍引用上述full v1 SWA及9项校验；没有生成v3新提交包，
按用户新指令保留此提前暂停状态，等待后续恢复指令。
