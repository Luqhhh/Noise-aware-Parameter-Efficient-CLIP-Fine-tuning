# L05 输入分辨率敏感性诊断（预注册）

日期：2026-09-26。分支：`codex/l05_resolution_sensitivity`。本轮只读复赛 `val_dev`，不读测试图、不训练、不改已发布 L05 包。

## 问题

RM-FT 的旧几何探针表明，将验证图缩至测试图的量级只损失 0.17–0.31pp；但现任平台最佳 L05 使用 384px 和 attention-local 训练，不能直接沿用 RM-FT 的敏感性结论。测量固定 L05 checkpoint 在相同验证图、相同 CLIP center-crop + flip 推理下，对额外低通缩小是否敏感。

## 固定实验

- 冻结 L05 `best.pt` SHA-256 `35d17c0c3b9f281e13353959d098c2713def6b7d508d334de5d3a7e769cd2397`，原始验证分支缓存 `l05_val_branch_logits.pt`，`val_dev.csv` SHA-256 `d91106df365b62f9bf56eff51474c222925301546642fa427f6778258cb833ab`。
- 对每张验证图，若长边 >500，先用 Pillow bicubic 按比例缩到长边 500；否则保留原样。之后执行 checkpoint 原生 384px CLIP transform。固定 center 与 horizontal-flip 两路；`mean_probabilities`、T=1.4。原始分支 logits 来自同 checkpoint 的既有缓存，不重新拟合 prior。
- 在前 32 张原图上重算分支 logits，要求两路 top-1 与缓存逐图一致，最大绝对 logit 差 ≤0.02；否则停止，不计算效应。该门只验证数值协议。
- 比较全体 14,880 张的 raw macro/micro、逐图配对纠正/破坏、被实际缩小者占比。仅用于判断该模型对输入尺寸变换的敏感性，不能当作测试域迁移估计。

## 下一步判据

若该固定变换使 macro 或 micro 下降 ≥1pp，开一个新实验段研究训练侧的随机低通缩小；若两者都下降 <0.5pp，则关闭该训练方向；中间地带需另行设计。此处不根据测试集预测或平台成绩改变阈值。

本诊断不产生新候选模型。完成时复核现役 L05 提交包的 CSV/ZIP 及 9 项提交校验，记录其路径，作为本轮可提交保底产物。该保底包的既有平台分不归因于本诊断。

## 执行结果（在上述方案冻结后回填）

完整命令：

```bash
python3 scripts/probe_l05_resolution_sensitivity.py \
  --checkpoint /home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/RM_V5_L05_CUDA_LOCAL/seed42/checkpoints/best.pt \
  --cache /home/lux1/noise/worktrees/rematch750_f05_focus/artifacts/f05_focus_l05/l05_val_branch_logits.pt \
  --val-csv /home/lux1/noise/artifacts/stages/repechage/20260921/val_dev.csv \
  --image-root /home/lux1/noise/train \
  --output outputs/codex/l05_resolution_sensitivity/full.json
```

原图 32 张双分支 logits 对缓存的最大绝对差为 **0**，两路 top-1 均 32/32 一致。全验证集 14,880 张，其中 **9,946** 张实际缩小。原图 flip-TTA T=1.4 的 macro/micro 为 **75.4271% / 76.4651%**；低通缩小后为 **75.2193% / 76.2970%**，差 **−0.2078pp / −0.1680pp**。逐图有 271 个预测改变、51 个纠正、76 个破坏（净少 25 张正确）。机器可读结果：[results/l05_resolution_sensitivity_20260926.json](../results/l05_resolution_sensitivity_20260926.json)。

按预注册「两项损失均 <0.5pp」分支，**关闭 L05 训练侧 500px 低通缩小方向，不启动训练**。这只排除该固定几何扰动是当前 L05 重大薄弱点；它不解释剩余约 9.7pp 的平台差，也不能证明其他来源差异不存在。本轮没有新模型、没有新平台成绩。

本轮可提交保底产物为现役 `L05_T14_P060`：`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/{pred_results.csv,submission.zip}`。2026-09-26 重跑 `python3 scripts/check_submission.py --test_dir /home/lux1/noise/test --class-mapping /home/lux1/noise/artifacts/stages/repechage/20260921/class_to_idx.json --csv <上述CSV> --zip <上述ZIP>`，**9/9 通过**，37,444 行；CSV SHA-256 `51e0efe7178528d23993a44351069d2829b0cd3669c195a77d8d879e66776a75`，ZIP SHA-256 `e788f07636b80abb61685335cd8df108803863ea36ffbde8b353f8e8317fcd7f`，与平台 66.94797564362783% 的登记包一致。该包是旧赢家保底，未在本实验生成或改变。
