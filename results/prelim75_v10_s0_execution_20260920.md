# PRELIM75 v10 S0 执行检查点（2026-09-20）

计划 ID：`PRELIM75_V10_SAM_20260920`

状态：S0 普通 AdamW 控制组已完成训练、重叠诊断和固定测试推理，提交包已通过独立校验并复制到 C 盘桌面；S1 标准 SAM 未启动，v10 尚未形成优化方法对照结论。

## 执行边界

- 执行源码提交：`cddeaac827773f093fd188a33ca5bc14b9b04191`。
- 固定配置：`configs/prelim75_v10_sam.yaml`。
- 正式队列命令：`PYTHONPATH=reproducibility/aegis_f1 python3 -u scripts/run_prelim75_v10_sam_queue.py --config configs/prelim75_v10_sam.yaml --execute`。
- 用户在 S0 训练期间要求“S0 完成后暂停”。队列父进程先被挂起，使 S0 子进程在完整 optimizer update 边界完成并保存；S0 完成后结束旧自动队列控制器，未启动 S1。
- 后续仅执行 S0：
  - `PYTHONPATH=reproducibility/aegis_f1 python3 -u -m aegis_clip.cli.train_prelim75_v10_sam --config configs/prelim75_v10_sam.yaml --action evaluate --candidate S0`
  - `PYTHONPATH=reproducibility/aegis_f1 python3 -u -m aegis_clip.cli.train_prelim75_v10_sam --config configs/prelim75_v10_sam.yaml --action deliver --candidate S0`
- 没有平台自动上传，没有运行 S1，没有使用 prior 或 v9 source recrop。

## 门禁与 S0 训练

修复后的 A0 通过：原损失 `0.047027505934238434`，扰动损失 `0.23755624890327454`，增加 `0.1905287504196167`；g1/g2 范数分别为 `5.009396076202393` / `15.540825843811035`，实际 FP32 扰动半径为 `0.04999999329447746`。许可参数逐位恢复、冻结参数、buffer 和两遍 RNG 回放检查均通过。

共同 smoke 确认 S0/S1 首批索引、图像、Flip/尺度、目标、权重、reference、冻结框、global/local logits、完整 loss 和初始 LR 对齐。正式 S0 从 L1 独立加载，完成 sample epoch 19/20/21，每轮 3,226 次更新，共 9,678 次 AdamW 更新：

| sample epoch | 平均完整 loss | 平均 classification | 平均 anchor | epoch 秒数 |
|---:|---:|---:|---:|---:|
| 19 | 0.1188994664 | 0.0893999166 | 0.0147497749 | 1786.0060 |
| 20 | 0.1130333854 | 0.0847536049 | 0.0141398902 | 1450.1666 |
| 21 | 0.1087577641 | 0.0814544137 | 0.0136516752 | 1466.8262 |

S0 checkpoint：

- 路径：`outputs/prelim75_v10_sam_20260920/S0/candidate.pt`
- SHA-256：`923e358c68d29aa60f7d92daef3cd97f117ec8732cbb441f97094368d469b143`
- 状态：固定最后一轮，9,678 / 9,678 更新完成。
- 训练动作耗时：`4730.736696118001` 秒；峰值 CUDA allocated memory：`3064135680` bytes。

## 重叠诊断

10,316 行固定重叠诊断完成，未触发相对 L1 下降超过 2pp 的工程止损：

| 指标 | S0 | 相对 L1 |
|---|---:|---:|
| raw micro | 84.344709% | +0.126022pp |
| clean-core micro | 97.121811% | +0.136405pp |

`engineering_stop=false`。该诊断与模型训练数据重叠，不是独立泛化证据，也不能替代平台结果。

## 测试推理与提交包

最终推理使用归档 L1 数值上下文：matmul TF32 关闭、cuDNN TF32 开启、float32 matmul precision 为 `highest`，batch size 64；单 checkpoint、原十视图、温度 1.5、无 prior。

- 行数：24,967；basename 唯一且完整覆盖测试集。
- 标签：四位 `0000`–`0499`；自然预测覆盖 500 类，未强制类别覆盖。
- CSV SHA-256：`424258a941c2c10b36c9626a85db6b6d7b64d2ec3876193d73c05c97b64d15c8`。
- ZIP SHA-256：`98c2a51bb720ec127d7f0d52cbba200a250460917a3dcd80ba2aca65c7fb17a6`。
- ZIP 只含 `pred_results.csv`，且包内 CSV 与外部 CSV 逐字节一致。
- 仓库包：`outputs/prelim75_v10_sam_20260920/S0/submission/submission.zip`。
- 桌面包：`C:\Users\lqh22\Desktop\PRELIM75_V10_SAM_S0_20260920.zip`；复制后 SHA-256 与仓库包一致。
- `scripts/check_submission.py` 的全部检查通过。
- 用户回填平台分数：`69.3195%`；相对 L1 无 prior `69.2794%` 为 `+0.0401pp`。用户未提供精确正确数和实际上传时间，二者保持 `null`。

## 当前决定

S0 平台 `69.3195%` 已超过 L1 `0.0401pp`，成为当前无 prior 新高，但没有达到 `+0.30pp` 工程投入门槛。用户随后明确要求继续 S1；S1 已按原固定方案启动。在 S1 平台反馈前，仍不能判断标准 SAM 是否优于普通 AdamW，也不能执行 v10 的最终停止规则。
