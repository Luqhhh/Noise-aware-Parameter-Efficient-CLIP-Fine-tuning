# V3_RESUME_CHECKPOINT_20260930

用户“继续”，恢复原v3固定实验。2026-09-30恢复检查发现原暂停进程与
`noise-v3-after-v1-exif-recovery-20260930.service`已不存在；不推断消失原因。
磁盘完整保留original的第2轮checkpoint、history和全部2786条更新轨迹。
本段新增恢复包装入口，不修改冻结训练实现，不重新训练已完成的对照臂。

## 实现与协议

- 独立工作目录：`/home/lux1/noise/worktrees/v3_resume_checkpoint_20260930`。
- 方案分支：`codex/v3_resume_checkpoint_20260930`，基于最新`origin/main`的`d31d721`。
- 冻结代码目录：`/home/lux1/noise/worktrees/v3_after_v1_20260930`。
- 原prepared目录：冻结代码目录下`outputs/codex/v3_after_v1_20260930/prepared_exif_recovery`。
- 原plan SHA：`615fc646d20cf5d7b02f64e3adcc4d659bd607036302b895d08901140fa9e224`。
- 恢复epoch02 SHA：`08d4986158d388e51a311093c3fdb7c36d36ea478defb0a308ee099c3d5073e9`。
- 两臂共享初始模型SHA：`1095bab4f55b749f6b37aa2feda3d0cd712b756c1e535f1a99ff5f209a4beba0`。

`recover_v3_checkpoint.py`验证授权、冻结输入、完整checkpoint绑定、history及轨迹序列，
在核对官方训练图像字节后，把保存的original末轮EMA重新评估到全部14880张val_dev。
必须与原记录的macro/micro完全一致，才新写`selected.pt`和`predictions.npz`。
原epoch checkpoint、history、trace保持不变；拒绝覆盖已有导出或已启动的监督臂。
之后从同一官方初始化运行未修改的`v3.runtime.fit_arm(v1_supervision)`，检查配对轨迹逐字节相等，
生成配对报告，然后调用原冻结`v3.delivery`，完成37444张测试推理和9项提交校验。
所有新增训练与controller产物分别在原实验未完成的输出位置、新段独立日志目录中，不覆盖其他方案产物。

固定train_dev v1 targets/reliabilities，384、2轮/臂、1393更新/轮，logical batch96、
micro batch2、workers2、EMA .9995、last EMA、384 center crop，无bias/TTA。
不改用full v1监督，不派生阶梯、续训搜索或自动重试，不上传平台。
controller沿用plan绑定的旧`PYTHONPATH`；当前main的v1实现已扩展，不能代入原绑定。
此前STOP的旧服务和暂停监控不重建；新服务独立运行到交付，systemd按控制组管理子进程。

## 可重放启动命令

此命令是原目录一次性续接，拒绝在已有新产物时重复执行：

```bash
systemd-run --user --unit=noise-v3-checkpoint-resume-20260930 \
  --property=WorkingDirectory=/home/lux1/noise/worktrees/v3_resume_checkpoint_20260930 \
  --property=KillMode=control-group --property=TimeoutStopSec=40 \
  /usr/bin/python3 -u \
  /home/lux1/noise/worktrees/v3_resume_checkpoint_20260930/scripts/run_v3_checkpoint_resume.py \
  --plan /home/lux1/noise/worktrees/v3_after_v1_20260930/outputs/codex/v3_after_v1_20260930/prepared_exif_recovery/plan.json \
  --frozen-code-root /home/lux1/noise/worktrees/v3_after_v1_20260930 \
  --checkpoint-sha256 08d4986158d388e51a311093c3fdb7c36d36ea478defb0a308ee099c3d5073e9 \
  --output /home/lux1/noise/worktrees/v3_resume_checkpoint_20260930/outputs/codex/v3_resume_checkpoint_20260930 \
  --execute
```

controller：`outputs/codex/v3_resume_checkpoint_20260930/status.json`；
进度日志：同目录`recover_train.log`；之后`infer.log`。
原handoff的`paused_by_user`是旧现场历史，后续进度以新controller为准。
原prepared目录`runs/train/v1_supervision/history.json`保存监督臂轮次指标。
最终产物预定在原prepared目录`submission/pred_results.csv`、`submission/submission.zip`、
`submission/submission_check.log`及`submission/report.json`；这些当前尚未交付。

## 实测验证与当前状态

CPU真实预检通过：冻结plan及输入哈希、原第2轮绑定、全部2786条连续更新、
原初始模型摘要，以及测试StageContext绑定、37444项测试文件名均一致。
原checkpoint的第2轮raw macro/micro **59.6075% / 60.5914%**，
EMA **53.0370% / 54.0995%**。这些为已有对照结果，本段尚无新的监督臂指标。
完整[预检记录](../results/v3_resume_checkpoint_20260930/preflight.json)。

```bash
python3 -m pytest tests/test_v3_checkpoint_recovery.py tests/test_v3.py \
  tests/test_v3_delivery.py tests/test_v3_after_v1.py -q
```

**44 passed in 5.63s**。首次沙箱内测试中数据加载worker的本地socket被环境禁止，
在授权的宿主环境重跑后全部通过；不修改测试或训练实现绕过该限制。
恢复新增测试覆盖保存EMA选择、原始产物不变、SHA变更、残缺/乱序轨迹、
拒绝覆盖导出以及验证不一致时阻止产物生成。

23:02:45 CST用户服务启动，runner PID8852、worker PID8916，状态`running / recover_train`。
启动核对日志显示`verify_training_pixels`；后续GPU恢复与监督臂进度以实时controller/log为准。
启动快照见[状态](../results/v3_resume_checkpoint_20260930/resume_status.json)。
当前工程恢复完成并后台继续原训练；整个v3实验尚未完成，没有新的平台结果。

现役可提交包：`/mnt/c/Users/lqh22/Desktop/v1_full_swa_submission.zip`，
源ZIP为`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/submission.zip`，
SHA `1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`。
9项既有校验见[校验日志](../results/v1_full_swa_20260930/submission_check.log)和
[最终验证](../results/v1_full_swa_20260930/final_validation.json)。最高用户报告平台分仍为train_dev v1 SWA **70.4091%**。

## Git集成

代码、测试、记录在方案分支提交与推送后，main集成目录采用
**自动模式**`git pull --rebase --autostash origin main`，再合并、重新校验并推送。
集成信息见[执行记录](../results/v3_resume_checkpoint_20260930/execution.json)及本次最终回复；
此工程检查点不暂停已按新指令恢复的后台训练。
