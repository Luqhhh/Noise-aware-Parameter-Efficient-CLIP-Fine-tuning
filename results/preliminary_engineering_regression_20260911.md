# 初赛工程回归修复（2026-09-11）

运行 ID：20260911T130341Z。范围为方案一 P0/P1/P2 的工程续段，不是提分实验。
源基线：279ef30b176fad7c95fbc4910920d82ecebac7e6，main。

## 问题与修复

- baseline 训练的在线、缓存及筛选后重建 DataLoader 使用固定 timeout=120，num_workers=0 时在迭代前失败。单进程现在使用 timeout=0，多进程保留 120 秒；未改变训练算法或配置预算。
- OOF 权重测试把 float32 转为 Python float 后要求十进制精确相等。改为与 float32 预期张量比较，rtol=atol=0，仍检查 dtype、形状和精确值。
- 原 smoke test 被上面的加载问题提前阻断；修复后暴露出调用提交检查器时缺少类别信息。现在传入合成数据实际生成的 class_to_idx.json，保留检查器的拒绝缺省行为。
- 新增 outputs/preliminary_no_sweep/ 忽略规则，避免将本地源码快照、环境信息或后续样本级产物误纳入 Git。

## 实际验证

在同一已安装环境下执行，未安装新依赖、未修改测试容差、未新增 skip/xfail：

```bash
# 仓库根目录
python3 -m pytest -q tests
# cwd: reproducibility/aegis_f1；PYTHONPATH=/home/lux1/noise/reproducibility/aegis_f1
python3 -m pytest -q tests
# 仓库根目录；官方历史包只做格式/覆盖检查
python3 scripts/check_submission.py --test_dir /home/lux1/noise/test --num-classes 500 --csv /home/lux1/noise/outputs/delivery/fullft_dual_pa0.9/pred_results.csv --zip /home/lux1/noise/outputs/delivery/fullft_dual_pa0.9/submission.zip
```

根套件 **410 passed**（12 warnings，36.48 s）；AEGIS **357 passed**（8 warnings，9.59 s）；退出码均为 0。根 smoke test 完成合成图像训练、推理、CSV/ZIP 生成与校验，不是官方数据正式实验。git diff --check 通过。

同步 279ef30 后，SCOPE 的两个资产测试曾因约定路径缺少 checkpoint/trust 文件失败。已验证本机历史原文件与配置中的完整 SHA-256 一致，然后建立本地相对符号链接到约定路径，未修改 SCOPE 配置、模型、门控或测试。链接及来源记录见本地 restored_asset_links.json，并由本地 .git/info/exclude 排除；本提交不携带权重，其他机器仍需按原协议准备真实资产。未重跑 SCOPE 研究流程。

历史 pa0.90 提交检查退出码 0，覆盖 24,967 张官方测试图，ZIP 仅含 pred_results.csv，ZIP 内外 CSV 字节一致。实际四项哈希与上一轮 lock 均一致：

- checkpoint：f72b0104257f49d2667fe335553a861dd1dea947753feebdc7301b8890b48765
- CSV：790fabcfb57ada355bfdb2f732da5ea1e16d3c505cdbf77c558bccfe7112b16d
- ZIP：3d684c07027d905c3edf88e4ce88c3ef9f32a01a6304385600d3d0ced7af5251
- manifest：67f42552e0c39f25f314a716349f4c9984d9980c1eedee9da781cee36da45570

历史提交产物保持在 outputs/delivery/fullft_dual_pa0.9/，状态仍为 historical_reference_only。70.352866% 是旧平台回填，不是本轮成绩；未生成官方新候选。

## 记录与边界

本地审计目录：outputs/preliminary_no_sweep/20260911T130341Z/。首次失败日志保存在 test_reports/first_attempt/，最终日志在 test_reports/，精确测试 argv 和退出码在 commands.jsonl。上一轮 20260911T125732Z 原记录保留，同步前另存 /tmp 压缩备份。

Git 使用自动 stash 模式 git pull --rebase --autostash origin main，实际为从 273d361 快进到 279ef30，没有手动 stash/pop。该段只做本地提交，用户负责 push。

P2 回归阻塞解除。P3 仍不能验收：缺少 02_repechage_semifinal_readiness_plan.md 和完整教师/筛选/Adapter/校准器影响谱系；原验证样本全部进入父模型训练的事实未改变。effective_number 的已知错误属于方案二 R1，本段未改动或启用该算法。D0 approved=false；P4–P7 仍 blocked_on_user_decision，未选择机制、对照、阈值、预算或提交候选。

改动文件：.gitignore、experiments/baseline/train.py、tests/test_oof_soft_targets.py、tests/test_integration.py，以及本报告。仅修复代码与合成测试、汇总报告适合代码共享；本地审计快照和资产链接不纳入提交。
