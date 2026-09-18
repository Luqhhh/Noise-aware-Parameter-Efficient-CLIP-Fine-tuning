# PRELIM75 v6 执行方案：当前学生在线训练几何

计划 ID：`PRELIM75_V6_20260918`

权重父模型：`outputs/prelim75_v5_20260917/F1/candidate.pt`（SHA-256 `44e64a9528653f1d76db6a12b56a22c3ad23851665e8cef45d06f7ece9f52e1b`）。

## 当前状态

2026-09-18 已完成真实执行与平台闭环。D0 过门槛（1080/2048，阈值 205），共同 smoke 通过，G0/G1 各完成 3 epoch、重叠诊断和提交包。用户回传平台成绩：G0 `69.2274%`、G1 `69.1993%`；G0 比 F1 高 0.0881pp，比 G1 高 0.0281pp。G1−max(F1,G0) 为 −0.0281pp，未达到 0.30pp 投入回报门槛，故保留 G0、关闭在线几何配方，不追加 G2，不支持在线框平台收益结论。完整记录见 `results/prelim75_v6_execution_20260918.md`。

## 起点与投入门槛

v6 不重新做历史复现，也不续训旧模型。先执行 D0 固定投入诊断：从官方训练行中仅取 `w > 0` 的样本，每个内容组保留规范路径字典序最小的一张，按 `SHA256("prelim75-v6-d0/42/" + canonical_path)` 排序取前 2,048 组。冻结 F1，对原图与 tensor Flip 各做一次原生 224 global forward，读取最后一层 CLS-to-patch attention，按现有四尺度裁剪规则推导八个框，与 V1 缓存框比较。

当且仅当至少 `ceil(0.10 * 2048) = 205` 组在八个框里出现至少一个中心距离 `>= 8` 原生像素时，才允许启动 G0/G1。D0 不按准确率、类别难度、预测或新旧框差异选样；不足 2,048 组或未达门槛时本轮结束，不降低门槛，也不单独启动 G0。

## G0/G1 固定对照

| 项目 | G0 | G1 |
|---|---|---|
| 父模型 | v5 F1 | v5 F1 |
| global wrapper | 同一份训练可用最后层 attention replay wrapper | 相同 |
| local 训练框 | 冻结 V1 缓存 | 当前学生本次 global forward 的 detached attention 在线产生 |
| 原监督 w/q | 不变 | 不变 |
| 损失 | v5 F1 融合 GCE，T=1.5，blend=0.5 | 相同 |
| final inference | 各自单 checkpoint，原生 224、四尺度、Flip、无 prior | 相同 |

两组各训练 3 个 sample epoch（13/14/15），fresh AdamW、fresh cosine（18 epoch horizon、无 warmup）、effective batch 32、clip grad 1.0、FP32、seed 42，固定末轮，不按诊断指标选 epoch。两组共用一次独立 smoke 决定 32 或 16×2 microbatch。

## 梯度边界

训练 wrapper 在最后一个 visual residual block 注册临时 pre-hook，只保存输入 token 的 `detach().clone()`；随后正常执行 `model(images=..., return_features=True)`，保留 global logits/features 的 autograd 图。正常 forward 结束后在 finally 中移除 hook，再在 `torch.no_grad()` 内重放 `ln_1` 与 attention（`need_weights=True, average_attn_weights=False`），返回停止梯度的 CLS-to-patch attention。G1 的框为 `Box(stopgrad(Aθ(x)))`，梯度不经过 top-k、取整或框坐标；local 分类梯度照常反传共享 visual、shared head、O3/PTA。

只支持 eager、顺序执行、当前 full-finetune 模型。最后层 attention dropout 必须为 0，replay 模块不得更新状态、消耗正式 RNG、隐藏 BatchNorm running stats 或改变 train/eval 与冻结许可。

## 代码落点

- `configs/prelim75_v6.yaml`
- `reproducibility/aegis_f1/aegis_clip/prelim75_online_geometry.py`
- `reproducibility/aegis_f1/aegis_clip/cli/train_prelim75_v6.py`
- `scripts/run_prelim75_v6_queue.py`
- `reproducibility/aegis_f1/tests/test_prelim75_v6.py`

队列阶段：只读 preflight → prepare → D0 → gate → 独立共同 smoke → G0 → G1 → 固定诊断 → 打包。默认不加 `--execute` 只做只读 preflight。执行需提交后在 `main` 上运行；不自动上传、不 push、不追加 G2、不扫描 prior/温度/top-k/尺度/融合比例。

执行命令：

```bash
PYTHONPATH=reproducibility/aegis_f1 \
  python3 -u scripts/run_prelim75_v6_queue.py \
  --config configs/prelim75_v6.yaml --execute
```

GPU 累计预算上限 28,800 秒，包含检查、D0、共同 smoke、两组训练、诊断、推理和失败尝试；最多两个新平台候选。当前平台分数均保持 `null`。
