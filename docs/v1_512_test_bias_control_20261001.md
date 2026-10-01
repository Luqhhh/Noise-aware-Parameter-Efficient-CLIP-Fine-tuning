# V1_512_TEST_BIAS_CONTROL_20261001

状态：prepared_not_started；用户已授权后续串行队列，等待768 full完整交付及集成。
本方案是无训练的512均衡bias对照，独立分支 `codex/v1_512_test_bias_control_20261001`。

固定配置：[YAML](../configs/v1_512_test_bias_control_20261001.yaml)；
真实父产物/当前阶段数据与官方权重预检：[preparation.json](../results/v1_512_test_bias_control_20261001/preparation.json)。
父checkpoint来自148,695张full_train的v1 SWA4–12，750类，512维；只复制原有张量并重绑独立解码血缘，
保存父checkpoint SHA，不修改原checkpoint或模型参数。没有新的训练准确率/平台分。

448/512/576×flip共六视图logits求和，固定200次、强度1测试均衡bias，
只利用用户转述官方许可的无标签测试类别均衡先验，不扫描强度。
分别交付同模型原始和bias校正两个37,444行CSV/ZIP，9项校验及独立NumPy全预测重放。
包含推理、bias拟合、独立校验和桌面副本的墙钟预算60分钟；失败不重试，不上传。

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 scripts/run_v1_post768_job.py \
  --config configs/v1_512_test_bias_control_20261001.yaml \
  --prepared results/v1_512_test_bias_control_20261001/preparation.json --execute
```

实际启动仅由[持久串行controller](v1_post768_serial_20261001.md)安排，避免与在途768 full重叠。
产物根：`/home/lux1/noise-worktrees/v1_512_test_bias_control_20261001/outputs/codex/v1_512_test_bias_control_20261001`。
桌面前缀：`/mnt/c/Users/lqh22/Desktop/v1_512_test_bias_control_20261001_512_{raw,test_bias}_submission.zip`。
未产生新候选；现役可提交包及已有校验见[当前入口](current_execution_plan.md#交付与历史入口)。

## 已验证交付

2026-10-01T09:25:44.344332+00:00 完成，用时21.92分钟。原始标签诊断不证明平台改善；平台分待回填。

- submission: `/home/lux1/noise-worktrees/v1_512_test_bias_control_20261001/outputs/codex/v1_512_test_bias_control_20261001/submission/submission.zip`；SHA `260fc98bff216236e2d1bb4b4d6423f10a779f14bd6e508e9bcf0c9a8ae11048`；37,444行，9项校验通过，独立NumPy预测与CSV/ZIP一致。
- submission_raw: `/home/lux1/noise-worktrees/v1_512_test_bias_control_20261001/outputs/codex/v1_512_test_bias_control_20261001/submission_raw/submission.zip`；SHA `774e3d94837b72fa9449786bc610cb8c318ce7087fd5458d26ed2114dfdc7d9e`；37,444行，9项校验通过，独立NumPy预测与CSV/ZIP一致。

[完整指标与血缘](../results/v1_512_test_bias_control_20261001/delivery_report.json)。
