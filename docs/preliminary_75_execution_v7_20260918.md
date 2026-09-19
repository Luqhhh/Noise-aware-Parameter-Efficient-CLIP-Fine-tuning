# PRELIM75 v7 执行方案：一次固定收尾退火对照

计划 ID：`PRELIM75_V7_20260918`

状态：**已闭环；L1 为正式无 prior 胜者，但未达到 0.30pp 投入回报门槛；不追加 G2。**

平台结果：L0=69.0632%，L1=**69.2794%**。L1 比 G0 高 0.0520pp、比 L0 高 0.2162pp，按预注册规则保留 L1；但 L1−max(G0,L0)=0.0520pp，未达到 0.30pp 投入回报门槛，因此本轮关闭，不追加 G2 或调参。按用户要求另生成的侧包 G0+prior0.9（legacy test-batch balanced-prior，不属于预注册 L0/L1 无 prior 对照）平台 **72.4677%**，仍为绝对最高，距 75% 差 2.5323pp。

完整命令、指标、哈希、提交验证与最终判定见 [`results/prelim75_v7_execution_20260919.md`](../results/prelim75_v7_execution_20260919.md) 和 [`results/prelim75_v7_final_20260919.json`](../results/prelim75_v7_final_20260919.json)。

## 起点

- 权重父模型：v6 G0，平台 69.2274%
- 父 checkpoint SHA-256：`fb164dcac4ce3aae9f66129cf9ad5ba5160fabd26960742b5e00d2a8bdb43bea`
- 原有回退包：v6 G0 ZIP，SHA-256 `efccb9f2b601a8b24830ef601739b166c214b8316b441bb9bc7d7aaf6bd4b634`
- 不重建已删除缓存，不使用测试数据，不引入 teacher/伪标签/恢复掩码/定位网络。

## 唯一正式对照

| 项目 | L0 | L1 |
|---|---|---|
| 父模型 | v6 G0 | v6 G0 |
| 训练轮数 | 3 | 3 |
| sample epochs | 16/17/18 | 16/17/18 |
| cosine horizon | 18 epoch（H=58,068） | 3 epoch（H=9,678） |
| floor | 0.01 | 0.01 |
| warmup | 无 | 无 |
| 其他损失/监督/框/推理 | v6 G0 | 相同 |

实际训练更新次数 K=3×M=3×3,226=9,678；L0 三轮结束 schedule multiplier 为 `0.9336825748732972`，L1 为 floor `0.01`。sample epoch 只控制视图选择，不传入 LambdaLR。

## 实现要点

- `ScheduleSpec`、`build_fresh_scheduler`、`audit_position`、`step_and_audit` 固定在 `prelim75_cooldown.py`。
- 训练循环在 `optimizer.step()` 前记录 `used_lr`，在 `scheduler.step()` 后记录 `next_lr`，逐 update 审计并在 `lr_trace.csv` 留痕。
- G0 global wrapper 原样复用；local 训练框仍为冻结 V1 框；loss 复用 `fusion_gce_terms`。
- 共同 smoke 选择 32 或 16×2；两组共享 microbatch、有效 batch 分母与视图序列。
- 队列：read-only preflight → prepare → CPU scheduler check → common smoke → L0 train/eval/deliver → L1 train/eval/deliver → final pending feedback。
- 预算上限 28,800 GPU 秒，最多两个平台候选；不自动上传、不 push、不自动开 G2、不扫描 horizon/floor/LR/epoch。

## 代码落点

- `configs/prelim75_v7.yaml`
- `reproducibility/aegis_f1/aegis_clip/prelim75_cooldown.py`
- `reproducibility/aegis_f1/aegis_clip/cli/train_prelim75_v7.py`
- `scripts/run_prelim75_v7_queue.py`
- `reproducibility/aegis_f1/tests/test_prelim75_v7.py`

## 决策规则

- L0/L1 均 ≤ G0：保留 G0，关闭收尾调度配方。
- L0 最高且 > G0：保留 L0，不宣称退火有效。
- L1 严格高于 L0 且 > G0：保留 L1，报告 L1−L0。
- 同分：保留 G0；L0/L1 同分且均 > G0 时保留 L0。
- `L1 − max(G0,L0) ≥ 0.30pp`：记为达到投入门槛，但本轮仍结束。
- 无真实平台分时不宣称相应方向胜出；精确正确数/上传时间未提供则保持 null。
