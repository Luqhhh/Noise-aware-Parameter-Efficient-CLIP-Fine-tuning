# S_MIXUP_A04 中断恢复审计（2026-07-17）

## 背景
2026-07-16 A04 训练结束附近 WSL 意外断开。本审计确认训练状态、产物完整性，并完成确定性复评。
**未重新训练；未启动 P04；未 commit/push。**

## 1. 进程 / GPU
- 无残留训练进程（仅 VS Code server 相关进程）。
- `nvidia-smi`：无 compute 进程，1550MiB 为 WSL 显示占用。A04 已停止。

## 2. Git 状态
- HEAD: `7ef3fd2fa413fbf98977a5c51b82eabd9bb453cc`（results: add S-PEFT E0-E2 verification）
- 与 `artifact_manifest.json.commit_sha`、`s_mixup_batch_provenance.json.start_commit` 完全一致 → 复评即在原运行 commit 上执行，无需切换。
- 工作区无 tracked 文件修改（`git diff` 为空），仅有 untracked 的 outputs/configs/scripts。
- 未执行 pull/rebase。

## 3. 训练完整性结论：**完整正常结束**
- `train_log.csv`：47 行（epoch 1–47），配置 epochs=50，patience=10。
- 日志：`Early stopping triggered at epoch 47 ... Best val acc: 0.7042 at epoch 37`，
  随后 post-training eval 全部完成，最后一行 `Prediction records saved`（22:33:31）
  与已验收的 E1/E2 完整运行的结束行格式一致 → 断开发生在全部产物写盘之后。
- 无 NaN loss / OOM / Traceback。`train_log.csv` 中 `head_grad_norm` 列在
  epoch 20,23,…,47 出现 `nan`，记录为**非阻断日志异常**：健康的 E1/E2 运行也出现同一模式，
  loss/acc 全程有限，且 best.pt/last.pt 全 tensor（含 optimizer state）finite 检查通过；
  该列的计算逻辑尚未逐行核查，暂不定性为日志伪影。

## 4. Checkpoint 验证
| 文件 | 加载 | epoch | global_step | SHA-256 |
|---|---|---|---|---|
| best.pt | OK | 37 | 26418 | `1000404492ec141d490d894cd20f6bc1a6748e5767ac9ead74cb06ee10c8b5f9`（与 manifest 一致）|
| last.pt | OK | 47 | 33558 | `5d0bb200a75f00d81f18838f0356ceb14401961e3b5ddc0e0a278351b63f05c7` |
| best_raw.pt | OK | 37 | 26418 | `e3f1ec169abb751f1328f1760238a0b6ddf5c4a33972dceb974c3f45a9b8a9da` |

train/val CSV SHA-256 与 provenance 记录一致。config SHA-256（`46beba06…`）与 provenance 一致。

## 5. 确定性复评（原 commit + best.pt，未重训）
### evaluate.py（AMP，与训练时验证路径相同）→ `reeval_best.json`
- micro: **0.70424583**（7265/10316）— 与训练时 best_val_acc 逐位一致，逐 batch loss/acc 轨迹与原 post-eval 完全相同
- macro: 0.703698 | median: 0.75 | bottom-10%: 0.315700 | micro−macro gap: +0.000548 | loss: 0.577838

### regenerate_records.py（fp32 路径）
- `prediction_records.csv` 重生成结果与训练管线原文件**逐字节相同**（10316 条）→ 确定性可复现
- micro(fp32): 0.704343（7266/10316）| macro: 0.703788 | bottom-10%: 0.314700

### 7265 vs 7266 差异定位（AMP fp16 vs fp32）
- 完整普查：14 个样本预测翻转，全部为低置信度边界样本（conf 0.13–0.44）
- 净效应：fp16 下 wrong→correct 2 个、correct→wrong 3 个 → 净 −1，完全解释差异
- 明细见 `seed42/checkpoints/amp_fp32_flip_census.txt`
- 结论：数值精度伪影，非产物损坏；两条路径各自均确定性可复现

## 6. 产物清单（seed42/checkpoints/）
- `best.pt` / `last.pt` / `best_raw.pt` — 已验证
- `eval_results.json`（训练管线原始产物，未改动；AMP 口径，官方本地指标）
- `reeval_best.json`（复评，AMP 官方 / FP32 诊断两块显式分离，均标注 precision）
- `prediction_records.csv`（fp32 口径；重生成结果与训练管线原件逐字节相同后，
  路径规范化为相对 `train/...` 形式以匹配 E0/E1 提交格式，内容等价 n=10316, correct=7266）
- `per_class_metrics.csv`（fp32 重生成版）；原 AMP 版备份为 `per_class_metrics.train_pipeline.csv`
- `prediction_records.train_pipeline.csv`（原件备份）、`reeval_best.evaluate_only.json`（纯 evaluate 版备份）
- `artifact_manifest.json`、`resolved_config.yaml`、`config_snapshot_20260716_201259.yaml`
- `amp_fp32_flip_census.txt`（本次新增）

## 7. A04 验收结果
- **micro 70.42%**（AMP 口径 0.70424583；fp32 口径 0.70434277）
- macro 70.37% / 70.38%，median 75.00%，bottom-10% 31.57% / 31.47%
- best epoch 37，early stop @ 47/50
- 对照：parent w1_gce05_mixup（alpha=0.2, p=0.2）、baseline b2_gce05 —— 平台指标另行对比

## 8. 遗留状态
- P04 未启动（等待确认）
- 未 commit / push / pull / rebase
