# V13 策略规则复核与修正

实验标识 `P75_RESOLUTION_LADDER_RULES_REVIEW`，复核对象为 `P75_RESOLUTION_LADDER`。
依据当前 [CLAUDE.md](../CLAUDE.md) 和 [完整规则](../COMPETITION_RULES_AGENT.md)；
这是仓库规则审查，不声称已获得组委会对本实现的单独认证。

## 结论与实际修正

分辨率 / 学习率阶梯、强增强、soft CE、按当前训练类频次采样及最后全量续训符合项目边界。
复核发现原实现仍可选择 EMA 父权重，而规则将 EMA / SWA / 权重平均列为未确认灰区；
原 CSV 写法还缺少 CLAUDE.md 要求的逗号后空格。两处现已修正，CPU 验证通过。

| 检查项 | 当前实现及边界 |
|---|---|
| 模型与初始权重 | OpenAI 官方 CLIP ViT-B/32；SHA256 固定；分辨率插值不更换骨干 |
| 数据 | 仅当前复赛 148,695 张官方训练图；不加载作者任务权重、删除清单、改标签或外部数据 |
| 训练与阶段衔接 | 每段仅 raw 权重，按固定 holdout 指标选 epoch；不计算、保存、选择或传递 EMA / SWA / soup |
| 最后全量续训 | 纳入此前验证图是重用官方训练池，允许；此后不验证、不选模、不报告独立验证成绩，固定第 5 轮 raw last |
| 最终推理 | 单一最终 checkpoint，四个预先固定视图等权聚合；不拟合测试统计或更新参数 |
| TTA 依据 | [2026-08-31 项目记录](repechage_prep_20260831.md)记载用户已确认组委会允许单 checkpoint 的多尺度 + Flip TTA |
| 提交 | `文件名, 0001`、无表头；ZIP 仅含 `pred_results.csv`，与外部 CSV 字节一致 |
| 当前执行权限 | 只做代码和 CPU 检查，GPU / 实际项目训练 / 新预测 / 平台上传均未启动 |

原始 vendor YAML 及许可证是只读来源记录，含 EMA 参数不构成执行配置。
本地 prepare 删除 `ema_decay` 并设置 `weight_averaging=false`；prepare / verify 拒绝旧 EMA 策略，
父 checkpoint 必须仅有 raw 模型状态且 `chosen=raw`。旧准备目录仍是历史记录，不用于当前执行。

## 已验证命令与记录

方案分支 `codex/palm_v13_rules_20260929` 从 `origin/main` 的 `10dd4df98838320309a51b9ba3858153db9ddf6d` 创建；
独立目录 `/home/lux1/noise/worktrees/palm_v13_rules_20260929`。在该目录执行：

```bash
PYTHONPATH=reproducibility/aegis_f1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python3 -m pytest \
  tests/test_palm_v13_strategy.py reproducibility/aegis_f1/tests/test_model.py -q
PYTHONPATH=reproducibility/aegis_f1 CUDA_VISIBLE_DEVICES='' python3 -m palm_v13.plan prepare \
  --output outputs/codex/palm_v13_rules_20260929/prepared
PYTHONPATH=reproducibility/aegis_f1 CUDA_VISIBLE_DEVICES='' python3 -m palm_v13.plan verify \
  --plan outputs/codex/palm_v13_rules_20260929/prepared/plan.json
python3 scripts/check_submission.py --test_dir /home/lux1/noise/test --num-classes 750 \
  --csv /home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/pred_results.csv \
  --zip /home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/submission.zip
```

实测 **49 passed**（30 项策略检查 + 19 项既有模型检查），覆盖合成 CPU 阶段衔接、
raw 父权重、旧 EMA 策略拒绝、全量段无验证及 CSV 精确字节；没有使用项目图像训练。
新 plan 绑定 133,815 / 14,880 个内容组隔离划分，四段合计仍为 38,266 次逻辑更新。
现役 L05 包复核 **37,444 行、9/9 校验、ZIP/CSV 字节一致**，没有生成新候选包。
详细摘要、SHA256 和实现文件见 [核验 JSON](../results/palm_v13_rules_review_20260929.json)。

训练启动仍需新的用户 GPU 指令、可证伪的有限探索方案及本机完整成本预算；
完整训练还须有配对修正 / 退化和指定问题改善的证据。外部 76.5890%、CPU 测试通过或空闲算力不能替代该门禁。
方案验证后推送，在独立 main 集成目录以**自动模式** `git pull --rebase --autostash origin main`、
合并、复核、推送；不另行 `stash pop`，在工程检查点暂停。
