# PRELIM75 v10：标准 SAM 对照实施补充

计划 ID：`PRELIM75_V10_SAM_20260920`

日期：2026-09-20

审阅基线：`5045e71e69f4ee7de67af6410bbb336c60ecb171`

状态：**实施、合成 CPU 测试与真实资产只读 preflight 已完成；尚未运行 GPU A0、正式 S0/S1 或平台推理。本文件补充原预注册，不建立第二条 v10 路线，也不改变原参数。**

## 已有结果与唯一问题

v9 R0/R1 平台分别为 69.1633%/69.2033%，均低于 L1 无 prior 69.2794%，故原图回采样已关闭。G0 + legacy test-batch prior0.9 的 72.4677% 继续作为不同来源协议的单列记录，其偏置、测试边际和校准收益不进入 v10。

v10 只回答：在相同 L1 初始化、官方训练数据、原 w/Q、V1 冻结框、损失、学习率调度、三轮和 9,678 次更新下，标准非自适应 SAM + AdamW 是否优于普通 AdamW，并超过 L1。L1 距 75% 为 5.7206pp；72.4677% 距 75% 为 2.5323pp，二者不可互换，也没有证据保证 SAM 补齐差距。

## 固定 S0/S1 合同

| 项目 | S0 | S1 |
|---|---|---|
| 完整 ID | `PRELIM75_V10_SAM_S0` | `PRELIM75_V10_SAM_S1` |
| 父模型 | L1 独立重载 | 同一 L1 独立重载 |
| sample epoch | 19/20/21 | 19/20/21 |
| optimizer | fresh AdamW | 相同 AdamW + 标准 SAM |
| SAM | 无 | 全许可参数共享 L2，`rho=0.05`，非自适应 |
| 更新数 | 3×3,226=9,678 | 相同 |
| FP32/effective batch | FP32/32 | 相同；16+16 两遍累积 |
| LR/WD | visual `1e-6/0`；head `1.5e-7/1e-4`；O3/PTA `3e-6/0` | 相同 |
| scheduler | 3-epoch cosine，floor 0.01 | 相同 |
| loss | L1 probability-fusion GCE + 2.0 anchor | 两遍均使用完整同一目标 |
| 几何/选择/推理 | V1 冻结框；固定末轮；原十视图无 prior | 相同 |

每轮使用全部 103,218 行，保留 18 行末批。两候选共 19,356 次正式 optimizer 更新。S1 每次更新做两遍完整有效 batch，因此控制的是数据、初始化、目标和更新次数，不声称 FLOPs 或耗时相同。

分类分母使用整个有效 batch 的 `sum(w)`；anchor 分母使用实际有效 batch 大小。零分类权重行仍保留原 anchor，不删除、不恢复标签。两遍不能分别按 microbatch 求平均，也不能在第一遍漏掉 anchor。

## 资产与只读 preflight

固定父 checkpoint 为 `outputs/prelim75_v7_20260918/L1/candidate.pt`，SHA-256 为 `cd485c7beed2781ac13d08fbd33d63b8f1c46971be008966f7280c67cec1fcee`。回退 CSV/ZIP SHA-256 分别为 `5277b2c33f63b951ea2236ae2e50b8c574ef9f8d2f989e8239fcce7b065c0171` 与 `55d25f1fea9c57ba3431164d53d21f34d20baa2a49c805ad87cca188b5634c5a`。

实施机实际重新核对并通过：L1 完整 visual/shared head/O3/PTA、103,218 行训练清单、10,316 行诊断清单、24,967 行回退推理 manifest、500 类映射、trust、content groups，以及以下冻结几何/监督哈希：

- geometry origin：`a52548511f5971836dbd6ffc07f62685050e2dc9b4318818573c53e9a7c5a696`
- boxes：`9041665389e51b8da6af21b457a6d8a69eb365495d17558e9a5dedb4a7ed2ff1`
- manifest：`93e35940e19bbc371a30e67f0c12a429ecd48e88ca784be0983d4c8eacfc86e4`
- paths：`3a9bbfe4e3049aff1f081962c798be713dd17bef354612ad86321327dcf3c50c`
- trust：`6868041cc7b995a3e8e557ae925d1d25160acf23af09202f46911ce92125b30f`
- 原监督语义：`7fc6dcf0efb87e0a402114a7ff7ef9421e7f1c8fedcc06db10ba6d6485c606c4`

本地/远端 refs 与进程检查未发现另一份 v10 SAM 实施或运行。输出固定为 `outputs/prelim75_v10_sam_20260920/`，存在时拒绝覆盖或自动续跑。

## 标准 SAM 更新顺序

S1 的一次更新严格为：完整 batch 累积 `g1`；在 clip 和 AdamW weight decay 之前计算所有许可参数共享的全局 L2 范数；完整复制原 FP32 参数；一次性添加 `0.05*g1/(||g1||+1e-12)`；清梯度；在同一批准备好的图像、Flip、scale、框、w/Q 和 anchor reference 上计算 `g2`；在 `finally` 中用备份 `copy_` 精确恢复；逐位验证；仅对 `g2` 做 `clip_grad_norm_=1.0`；在原参数上执行一次 AdamW step；最后推进一次 scheduler。

不使用 `p -= e` 恢复，不平均 g1/g2，不对扰动求导，不做 ASAM/SANER/NCSAM/LookSAM，也不把 decoupled weight decay 加入扰动目标。若第二遍异常或中断，先恢复扰动参数再失败；若 optimizer step 自身失败，不假定参数或状态原子回滚，只允许从完整边界重启。

训练每一更新记录 base/perturbed loss、g1/g2 范数、实际 FP32 半径、恢复状态、裁剪状态和前后 LR。正式 loss 增量不作为选 epoch 或调 rho 的依据。

## A0 与 smoke 门禁

A0 使用 sample epoch 19 的首个有效 batch，不创建 optimizer、不读 test、不做持久更新。必须同时满足：所有数值有限；无冻结梯度；参数逐位恢复；实际 FP32 位移范数在 `0.05±5e-6`；扰动损失严格增加；无活跃 dropout/可变 BN buffer；两遍不改变 RNG 或模型 buffers；microbatch 16 不 OOM。

共同 smoke 在两个隔离临时实例中分别执行真实 S0/S1 一次 AdamW update，验证 optimizer 状态、两遍峰值和 scheduler；随后丢弃实例。S0/S1 第一遍的 batch、图像、Flip/scale、targets、weights、reference、固定框、global/local logits、loss 与初始 LR 必须一致。首更新后参数不要求相同。

资源估计结合本机 L1 历史训练耗时和同步 smoke 的 S1/S0 比率，乘 1.25 不确定性并预留 4,000 秒诊断/推理；S1 估计超过 8 小时或全队列估计超过 28,800 秒即关闭。

## 推理数值边界

训练继承原训练数值设置；诊断和最终推理在独立进程显式使用：matmul TF32=False、cuDNN TF32=True、float32 matmul precision=highest、batch 64。该设置继承 v9 已完成的 24,967 行 L1 零差异重放，禁止直接用训练 `gpu_setup()` 初始化推理。

候选推理只加载自身单 checkpoint，使用原 native local 张量裁剪、四尺度、Flip、T=1.5、无 prior；不携带 v9 recrop、训练框、anchor teacher、监督或校准。

## 预算、停止与交付

GPU 动作累计上限 28,800 秒，包含 A0、smoke、失败尝试、S0/S1、诊断和推理。不能缩减训练/测试清单或扩充预算。重叠诊断仅在 raw/clean-core 相对 L1 下降超过 2pp 时拒绝明显失败，不挑参数。

平台顺序固定 S0、S1。S1 不高于 S0 即关闭 SAM/SANER/NCSAM；S1 高于 S0 但不超过 L1 则保留 L1；S1 超过二者则保留 S1，但相对 `max(L1,S0)` 不足 0.30pp 只记录小收益并关闭；达到 0.30pp 只允许另行预注册后续研究。同分优先 L1，其次较简单 S0。

每个未触发工程停止的候选生成一个 24,967 行 ZIP，经内置和独立提交检查后复制到 C 盘桌面；不自动上传平台。用户只回填百分比时，不反推精确正确数或上传时间。

## 实现入口与验证

- `configs/prelim75_v10_sam.yaml`
- `reproducibility/aegis_f1/aegis_clip/prelim75_sam.py`
- `reproducibility/aegis_f1/aegis_clip/cli/train_prelim75_v10_sam.py`
- `reproducibility/aegis_f1/aegis_clip/cli/infer_prelim75_v10_sam.py`
- `scripts/run_prelim75_v10_sam_queue.py`
- `reproducibility/aegis_f1/tests/test_prelim75_v10_sam.py`

只读入口：

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -u scripts/run_prelim75_v10_sam_queue.py --config configs/prelim75_v10_sam.yaml
```

正式入口只在实现提交并推送、main 与 origin/main 完全一致、工作区干净、输出不存在时接受 `--execute`。当前合成 SAM 测试为 34 passed；v7/v9/v10 定向测试为 76 passed；完整 Aegis 测试为 609 passed。它们验证数值合同和集成守卫，不替代真实 GPU A0 或平台证据。
