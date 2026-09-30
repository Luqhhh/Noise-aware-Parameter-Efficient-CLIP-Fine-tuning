# 串行续训数值执行修复（2026-10-01）

最初WFT448执行在第75个逻辑batch因非有限梯度停止；已完成74次更新，无完整checkpoint/提交包。
父模型三尺度flip基线已完整保存，14,880×750 logits有限；此前的启动记录是历史，不表示仍在运行。
旧服务/日志/基线/失败状态保留，禁止拿未完成探针作为成熟候选。

## 修复与实测

问题精确复现于真实shuffle/增强/目标/逻辑batch的第75个batch。
初始AMP loss scale65536溢出；GradScaler正常backoff到32768后，重算同一batch得到有限梯度norm50.5729。
此后成功完成128个连续逻辑更新，probe不保存任何拟合权重，正式输出仍从原v1 DEV SWA开始。
CPU测试150项通过，新增测试检验重算保持forward RNG、只更新一次、真正非有限梯度有界失败且不更新权重。
证据：[真实CUDA复现与验证](../results/continuations_serial_recovery_20261001/wft_cuda_probe.json)、
[首轮失败](../results/continuations_serial_recovery_20261001/initial_failure.json)、
[首轮基线核验](../results/continuations_serial_recovery_20261001/initial_baseline_validation.json)、
[测试及边界](../results/continuations_serial_recovery_20261001/validation.json)。

`logical_update`仅修复数值执行：保持已加载图像、一次全局Mixup、lambda/permutation不变，
溢出时恢复CPU/CUDA forward RNG、清空无效梯度，按标准GradScaler backoff重新前反向。
每个逻辑batch只发生一次optimizer/scheduler/EMA更新，失败尝试不跳过训练样本。
最多17次尝试；scale≤1仍非有限或非有限forward loss会明确停止，不无限重启。
每个重算batch及scale/norm落盘到`training/amp_retries.json`，最终报告绑定其SHA。
没有改变4轮、学习率、逻辑batch32、micro8、监督、采样、增强、剪裁、EMA衰减、导出窗口或评估协议。

```bash
PYTHONPATH=reproducibility/aegis_f1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python3 -m pytest \
  tests/test_v1_continuation.py tests/test_serial_continuations.py tests/test_v2_swa.py tests/test_v2.py \
  tests/test_v3.py reproducibility/aegis_f1/tests/test_v1.py reproducibility/aegis_f1/tests/test_v1_export.py -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u scripts/check_continuation_cuda.py \
  --plan outputs/codex/wft448_dev_20261001_r2/prepared_amp/plan.json \
  --report outputs/codex/wft448_dev_20261001_r2/cuda_recovery_final.json --updates 128 --execute
```

## 新执行目录与边界

固定串行顺序仍为WFT448_DEV→LR512_DEV→有支持的WFT448_FULL→有支持的LR512_FULL。
[修复后队列](../results/continuations_serial_recovery_20261001/queue.json)使用新controller目录
`worktrees/wft448_dev_20261001/outputs/codex/continuations_serial_20261001_r2/`，
WFT输出`worktrees/wft448_dev_20261001/outputs/codex/wft448_dev_20261001_r2/`。
新plan绑定修复后的源码；LR也须重新prepare并预检后启动，不能复用绑定旧runtime的plan。
旧74次更新不恢复，重新从v1 SWA评估基线并执行完整4轮，旧产物不覆盖。
仍仅本机CPU/CUDA串行、有已有CUDA进程则等待；无NPU/远端，无v2/监督阶梯/平台上传。
FULL须满足[预先固定的多项支持判断](continuations_serial_20261001.md#full投入边界)。
失败不由服务自动重启；本次是在用户已授权范围内修复可复现的运行错误后明确重新启动。

现役仍为full v1 SWA70.9860%用户回填包，
`worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/submission.zip`，
37,444行、9项通过，引用[既有复核](../results/v1_full_swa_platform_20261001/artifact_verification.json)。
本段验证的是执行修复，没有新正式4轮候选、提交包或平台成绩。
