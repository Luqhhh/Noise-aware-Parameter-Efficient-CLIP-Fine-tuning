# 本机串行续训最初启动记录

本次最初服务已因第75个batch AMP缩放溢出停止，无完整候选；后续以[数值修复和新队列](continuations_serial_recovery_20261001.md)为准。

2026-10-01 02:09:39 CST启动本机持久服务`noise-continuations-serial-20261001.service`。
顺序WFT448_DEV → LR512_DEV → 有完整DEV支持的WFT448_FULL → 有支持的LR512_FULL。
CPU/CUDA串行；父模型、目标和原始数据只读；不启动v2、监督阶梯或平台上传。

启动状态running，controller PID 85846、WFT CUDA子进程85849；当前为未续训父模型独立验证基线。
实际服务内FP32零更新核对通过：48个LoRA已合并、最大logits误差0、cosine head/类别顺序保留。
该启动记录不表示4轮训练已经完成，不含新识别成绩或新提交包。

[冻结队列](../results/continuations_serial_start_20261001/queue.json)、
[服务启动与血缘](../results/continuations_serial_start_20261001/service_start.json)、
[配方/成本/FULL门禁/交付](continuations_serial_20261001.md)。
源码方案提交`c62a08e`，启动源码`2db9e98`；148项测试通过。
WFT CUDA预检和LR512 CUDA预检各3个实际逻辑更新通过，探针权重不保存。
逻辑batch32、micro8；父模型与候选均用相同FP32确定性评估batch32。
两个DEV四轮、每轮raw/EMA与固定EMA2–4、全量测试推理合计短测估计约6.4小时，实际耗时待记录。

精确启动入口：

```bash
/usr/bin/python3 -u /home/lux1/noise/worktrees/wft448_dev_20261001/scripts/run_serial_continuations.py \
  --queue /home/lux1/noise/worktrees/wft448_dev_20261001/outputs/codex/continuations_serial_20261001/queue.json --execute
systemctl --user status noise-continuations-serial-20261001.service --no-pager
```

由systemd user服务持久运行，PATH包含本机nvidia-smi，OMP/MKL/OPENBLAS线程均为2。
失败不自动重启，其他CUDA计算进程存在时等待，不抢占。
controller会逐段完成CSV/ZIP、9项校验和独立配对复算，提交推送方案分支及main后进入下一段。
自动FULL依据预先冻结的总体、分组和raw/EMA轨迹共同支持条件；混合证据保留DEV包待复核。
没有单纯以本地accuracy小降否定候选，当前未知平台迁移收益。

实时状态：`worktrees/wft448_dev_20261001/outputs/codex/continuations_serial_20261001/status.json`。
WFT日志：`worktrees/wft448_dev_20261001/outputs/codex/wft448_dev_20261001/train.log`。
LR日志由下一阶段创建：`worktrees/lr512_dev_20261001/outputs/codex/lr512_dev_20261001/train.log`。

当前现役提交包仍为
`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/submission.zip`；
37,444行、9项校验，来源见[既有包复核](../results/v1_full_swa_platform_20261001/artifact_verification.json)。
最高分70.9860%仍仅为用户回填，不新增平台分声明。
