# S_MIXUP_A04 结果报告

**实验**: S_MIXUP_A04 — MixUp alpha=0.4, probability=0.2（强混合强度消融）
**假设**: 更强的标签/输入平滑进一步抑制噪声记忆，风险是破坏细粒度结构。
**运行**: commit `7ef3fd2`，seed42，独立从头训练（无 init checkpoint），单 GPU RTX 4060。
**训练**: early stop @ epoch 47/50（patience 10），best epoch 37，总耗时 2h 19m 44s。
**验收**: 2026-07-16 训练结束附近 WSL 断开，2026-07-17 完成中断恢复审计
（详见 `outputs/mixup/alpha_04/recovery_audit_20260717.md`）：训练完整正常结束，
checkpoint/optimizer 全 tensor finite，产物齐全。

## 官方本地指标（AMP 口径 — 与训练时 checkpoint 选择路径一致）

| 指标 | A04 | 父实验 w1_gce05_mixup (α=0.2, p=0.2) | delta |
|---|---|---|---|
| micro | **0.70424583** (7265/10316) | 0.71161303 (7341/10316) | **−0.74 pp** |
| macro | 0.70369768 | 0.71115273 | −0.75 pp |
| bottom-10% | 0.31570005 | 0.32027438 | −0.46 pp |
| median per-class | 0.75 | 0.75 | 0 |
| best epoch | 37 | 46 | — |

## 诊断口径（FP32 prediction records — 配对分析基准）

| 指标 | A04 | 父实验 | delta |
|---|---|---|---|
| micro | 0.70434277 (7266/10316) | 0.71151609 (7340/10316) | −0.72 pp |
| macro | 0.70378813 | 0.71105752 | −0.73 pp |
| bottom-10% | 0.31470008 | 0.32027440 | −0.56 pp |

两口径结论一致。父实验的 FP32 口径来源：其 `prediction_records.csv` 由与 A04
相同的无 autocast 记录管线生成（代码段在 commit `1f830de` 与当前 HEAD 逐行相同），
其 AMP 口径来源于 `eval_results.json`（训练管线 post-eval，train.amp=true）。
无混用；两条腿均有同精度父基线，无 pending 项。

## AMP/FP32 精度对账

- AMP 7265 vs FP32 7266：14 个样本预测翻转（fp16 下 wrong→correct 2 个、
  correct→wrong 3 个，净 −1），全部为置信度 0.13–0.44 的边界样本。
- 属精度边界现象，**不是产物损坏**；两条路径各自确定性可复现
  （FP32 重生成与训练管线原件逐字节相同；AMP 复评逐 batch 轨迹与原 post-eval 相同）。
- 明细：`seed42/checkpoints/amp_fp32_flip_census.txt`；
  各产物 precision/n_correct/checkpoint SHA 见 `artifact_manifest.json`。

## 完整性检查

- best.pt (epoch 37, step 26418) / last.pt (epoch 47, step 33558)：
  model 154 tensors + optimizer 6 tensors（共 88.6M 元素）全 finite ✓
- best.pt SHA-256 `1000404492ec…` 与 manifest 一致 ✓
- train/val CSV、class mapping SHA 与批次 provenance 一致 ✓
- `train_log.csv` 的 `head_grad_norm` 列在 epoch 20,23,…,47 为 `nan`：
  记录为**非阻断日志异常**（E1/E2 健康运行同模式，loss/acc 全程有限，
  checkpoint finite；该列计算逻辑未逐行核查，暂不定性）。

## 结论

alpha 0.2→0.4（p=0.2 不变）在两个精度口径下均一致地劣化 micro/macro/bottom-10%
（约 −0.7 pp），支持"过强混合破坏细粒度结构"的假设方向。A04 判定为
**eliminated_no_gain**（本地口径）；不改变 S_MIXUP 批次计划，P04（α=0.2, p=0.4）照常执行。
