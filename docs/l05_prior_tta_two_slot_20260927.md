# L05 prior/TTA 两次平台探针（2026-09-27）

用户明确将当前两次平台提交额度用于 prior/TTA 参数搜索。现役对照是已获平台
**66.94797564362783%** 的单 checkpoint `L05_T14_P060`；其 ZIP 已冻结，不能重复上传。
本轮只改变现役同一个 L05 checkpoint 的确定性推理参数，不训练、不使用测试图像做拟合或选择。
prior bias 每个温度只用当前阶段 14,880 张验证图拟合，测试集只执行冻结推理。

开始前已 fetch、检查远端全部分支与近期 main。发现
`codex/k05_submission_campaign_20260927` 也预留这两次额度用于 K05 epoch4/6，但尚无
经校验的 E04/E06 包或平台成绩；用户新指令指定 prior/TTA，本轮以此为准，K05 方案
不可同时占用这两次额度。DP01 随机深度已关闭，不用其较差 checkpoint 搜索推理参数。

## 预冻结的两个上传臂

| 包 | 与现役唯一差异 | 验证 macro / micro | 与现役逐张预测差异 | 用途 |
|---|---|---|---:|---|
| `L05_T14_P070_PROBE_20260927` | prior strength 0.60 → 0.70 | 75.8574% / 76.6129% | 127 / 14,880 | 量 prior 强度的边际迁移 |
| `L05_T11_P060_PROBE_20260927` | TTA 温度 1.4 → 1.1 | 75.7451% / 76.5255% | 218 / 14,880 | 量温度的边际迁移 |

两臂都以已上传现役包作共同对照，每臂仅改一个参数，平台结果可分别归因。
第二臂本地比现役低，属于用户授权的诊断探针，不声称预计提分。两个选择与顺序
在触碰测试集预测前固定；不使用测试集预测分布、两包间测试预测差异或平台回执
继续挑参数。本轮不派生第三个点。

固定血缘、配置和标签见
[机器可读预注册](../configs/l05_prior_tta_two_slot_20260927.json)。
推理使用 `focus/f05-four-lines` 提交 `f050ecb` 的
`scripts/build_l05_tta_prior_submission_final.py`，脚本 SHA-256
`238be39ab22970e7e31b88851c539c8dd7ed5249fddef9c261d53dd848565d93`。
`--allow-recipe-drift` 仅允许记录上述预冻结 T/P 数值偏离旧固定配方，不改 TTA 视图、
融合、数据、checkpoint 或 prior 拟合域。每包须有 37,444 行 CSV、唯一根目录文件 ZIP、
ZIP/CSV 字节一致、9/9 提交校验及 SHA-256。账号由用户掌握；上传方回填平台回执。

## 执行与结果

两包已生成并各自通过 37,444 行 / 9 项提交校验，ZIP 内 CSV 与外部文件逐字节相同。
所有推理都使用同一个 L05 checkpoint SHA-256
`35d17c0c3b9f281e13353959d098c2713def6b7d508d334de5d3a7e769cd2397`。
在 `/home/lux1/noise/worktrees/rematch750_f05_focus` 执行的确切命令：

```bash
python3 scripts/build_l05_tta_prior_submission_final.py \
  --checkpoint outputs/f05_focus_l05/RM_V5_L05_CUDA_LOCAL/seed42/checkpoints/best.pt \
  --config outputs/f05_focus_l05/_runtime_configs/L05_RM_V5_L05_CUDA_LOCAL_cuda_0_mb4.yaml \
  --val-branch-cache artifacts/f05_focus_l05/l05_val_branch_logits.pt \
  --temperature 1.4 --prior-strength 0.7 --fusion mean_probabilities --tta horizontal_flip \
  --allow-recipe-drift --tag L05_T14_P070_PROBE_20260927 \
  --output-root /home/lux1/noise/outputs/codex/l05_prior_tta_two_slot_20260927 \
  --skip-desktop-copy --device cuda:0 --batch-size 32 --num-workers 2

python3 scripts/build_l05_tta_prior_submission_final.py \
  --checkpoint outputs/f05_focus_l05/RM_V5_L05_CUDA_LOCAL/seed42/checkpoints/best.pt \
  --config outputs/f05_focus_l05/_runtime_configs/L05_RM_V5_L05_CUDA_LOCAL_cuda_0_mb4.yaml \
  --val-branch-cache artifacts/f05_focus_l05/l05_val_branch_logits.pt \
  --temperature 1.1 --prior-strength 0.6 --fusion mean_probabilities --tta horizontal_flip \
  --allow-recipe-drift --tag L05_T11_P060_PROBE_20260927 \
  --output-root /home/lux1/noise/outputs/codex/l05_prior_tta_two_slot_20260927 \
  --skip-desktop-copy --device cuda:0 --batch-size 32 --num-workers 2
```

| 上传顺序 | ZIP 路径 | ZIP SHA-256 | 平台结果 |
|---:|---|---|---|
| 1 | `/home/lux1/noise/outputs/codex/l05_prior_tta_two_slot_20260927/L05_T14_P070_PROBE_20260927/submission.zip` | `ee50dfe3bcf595169d5d1f016be53576ba6668a48837f8541f16faf766dcea93` | 未上传 |
| 2 | `/home/lux1/noise/outputs/codex/l05_prior_tta_two_slot_20260927/L05_T11_P060_PROBE_20260927/submission.zip` | `5569ef0041478f62b66f68f1cde26233f79efb323a1abeae12555ed4a5862c35` | 未上传 |

CSV SHA-256 分别为 `ac5677d65f661cbb82e8e4ab78624765cd17c454a8c08a30d71be849a147c13c`
和 `e95b34709695f25c69e47f8946203468d111816527342eff957a4ba4e89c7029`。
已将两包完整复制到 Windows 桌面：
`/mnt/c/Users/lqh22/Desktop/L05_T14_P070_PROBE_20260927/` 与
`/mnt/c/Users/lqh22/Desktop/L05_T11_P060_PROBE_20260927/`。
桌面 ZIP/CSV 的 SHA-256 与原包逐项相同，ZIP 内外 CSV 逐字节一致；上传各目录的
`submission.zip` 即可。
完整机器可读结果见[结果文件](../results/l05_prior_tta_two_slot_20260927.json)。
上传方按既有约定回填 `results/rematch_submission_registry.csv` 的提交 ID、时间、平台阶段及
本文件平台结果。本机没有平台账号或回执，不能预填分数；两次名额目前尚未实际消耗。
