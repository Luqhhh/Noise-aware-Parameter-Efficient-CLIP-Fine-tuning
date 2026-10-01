# V1_768_DELIVERY_RECOVERY_20261001

用户原授权的768 full与后续两项队列继续有效；2026-10-01进度检查发现交付复核失败。
本段修复脚本、补独立复核与归档，并恢复尚未启动的原两项队列。训练和测试推理已完成，
不重跑训练、推理或改变预测，不增加方案/强度/轮数；原失败状态和日志保留。

## 原因与已完成部分

768 full于北京时间16:28完成12轮、六视图推理、固定200次测试均衡bias，以及原始/校正两包。
两份37,444行CSV/ZIP原校验均为 `All checks passed`。
随后独立校验器返回0且成功信息在stderr，复核代码只检查stdout，误报
`Independent submission checker did not pass`；finisher和依赖队列因此停止，后续两项没有GPU更新。
这是复核脚本缺陷，不是识别结果；当前平台分未知。

修复：同时检查stdout/stderr且保留非零退出失败；真实CLI小样本回归分别验证成功和无效标签拒绝。
桌面原始包只在不存在时创建，已有副本必须SHA相等；不覆盖不同文件。
完成入口新增显式 `--recover-failed`，仅允许已交付轨迹的failed完成状态恢复，拒绝训练失败或在途恢复。
单独恢复日志前缀，保留旧失败日志；恢复使用独立分支/worktree，原核心模型/训练源码不修改。

## 完成与队列恢复协议

先对原logits/bias/checkpoint和两份CSV/ZIP重新完整校验并独立NumPy200次复算，
归档指标、桌面包、配置、失败证据；提交/push方案、main自动autostash同步、合并、82项复核及push成功，
才将原 `completion_status.json` 置completed。原training status保持completed。

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -u scripts/finalize_v1_768_full.py \
  --config /home/lux1/noise-worktrees/v1_768_full_test_bias_20261001/configs/v1_768_full_test_bias_experimental_20261001.yaml \
  --integration-root /home/lux1/noise \
  --desktop-raw /mnt/c/Users/lqh22/Desktop/v1_768_full_raw_submission.zip \
  --scheme-branch codex/v1_768_delivery_recovery_20261001 --recover-failed
```

后续恢复配置：[固定队列](../configs/v1_post768_serial_recovery_20261001.json)。
仅改变controller输出目录，仍锁定原512对照和DEV768头两项配置、父模型、commit、预算60/90分钟与顺序。
旧服务/状态不删除；新服务等本轮交付及main集成成功，再运行原定未启动的各项。
失败不自动重试、没有新增full train或平台上传。

方案分支 `codex/v1_768_delivery_recovery_20261001`；worktree
`/home/lux1/noise-worktrees/v1_768_delivery_recovery_20261001`。
原有两包、原失败日志与状态见[诊断记录](../results/v1_768_delivery_recovery_20261001/failure_diagnosis.json)。
现役包及已有正式校验见[当前交付入口](current_execution_plan.md#交付与历史入口)。

## 已验证恢复与后续服务

2026-10-01T08:57:47.490666+00:00 UTC 完成独立复核、归档与main推送；方案commit `4a6ddb267041e785110dd1715d45484ba226c800`，main `da40a391cb4e7fd48d1d131fcb09b7fcdd3fca43`。
两份37,444行CSV/ZIP再次通过校验，NumPy全部预测一致，独立200次bias拟合最大误差6.7861328e-06，重拟合预测37,444/37,444一致。
bias改变6,775条预测，类别预测数从0–169变为43–55，不代表准确率提升；新平台分待用户回填。

桌面候选：`/mnt/c/Users/lqh22/Desktop/v1_768_full_test_bias_submission.zip`；无bias：`/mnt/c/Users/lqh22/Desktop/v1_768_full_raw_submission.zip`。
包摘要、原训练记录与完整复核：[交付报告](../results/v1_768_full_test_bias_20261001/delivery_summary.json)。

首次后续恢复服务因systemd缺少WSL `/usr/lib/wsl/lib` PATH，在资源查询前退出；原记录保留，没有启动GPU工作。
显式补齐服务PATH后，以新服务 `noise-v1-post768-serial-recovery-env-20261001.service` 启动固定队列；核验状态 `running`，仍按60/90分钟预算及原顺序执行。
[修正环境后的配置](../configs/v1_post768_serial_recovery_env_20261001.json)；[恢复及服务实测](../results/v1_768_delivery_recovery_20261001/recovery_validation.json)。

运行目录仍为本方案worktree，精确启动：

```bash
systemd-run --user --unit=noise-v1-post768-serial-recovery-env-20261001 \
  --property=WorkingDirectory=/home/lux1/noise-worktrees/v1_768_delivery_recovery_20261001 \
  --property=Restart=no \
  --setenv=PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --setenv=PYTHONPATH=/home/lux1/noise-worktrees/v1_768_delivery_recovery_20261001/reproducibility/aegis_f1 \
  --setenv=OMP_NUM_THREADS=2 --setenv=MKL_NUM_THREADS=2 --setenv=OPENBLAS_NUM_THREADS=2 \
  /usr/bin/python3 -u scripts/run_v1_post768_queue.py \
  --queue configs/v1_post768_serial_recovery_env_20261001.json --execute
```

实时controller：`/home/lux1/noise-worktrees/v1_768_delivery_recovery_20261001/outputs/codex/v1_768_delivery_recovery_20261001/controller_env/status.json`。
