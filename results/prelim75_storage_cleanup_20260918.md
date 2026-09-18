# 初赛本地存储清理记录（2026-09-18）

用户明确要求更新过时文档、删除桌面旧提交包和后续不需要的大文件。清理前工作树为干净的 `main`，与 `origin/main` 同步于 `be4f94581986117c2eec674f03446e08d6a4f189`。本次只删除已关闭路线、明确落败候选或可重建中间缓存；不删除训练/测试数据，不改平台记录。

## 已删除

| 类别 | 路径或范围 | 字节数 | 原因 |
|---|---|---:|---|
| 旧完整恢复 | `outputs/stage_readiness/20260912_full_rebuild_r1/` | 34,563,599,325 | 61.4010% 负结果；完整复现已移出当前计划 |
| 合成复现演练 | `outputs/stage_readiness/20260911_reproduction_dependencies_r1/` | 1,054,713,716 | 非正式候选；结果和哈希已写入报告 |
| v2 特征缓存 | `outputs/prelim75_v2_20260912/cache/` | 4,038,761,552 | 旧 head 路线缓存；当前 F1 推理不依赖 |
| 落败 checkpoint | v2 `H0/H1/V0`、v3 `S1`、v4 `C1`、v5 `F0` 的 `candidate.pt` | 4,867,194,818 | 均有真实平台负/次优结论；保留小型报告和提交包 |
| v3 失败现场 | `outputs/prelim75_v3_20260915_failed_epoch0_fusion_20260916/` | 443,777,474 | epoch0 失败输出，不是候选 |
| v5 暂停现场 | 两个 `outputs/prelim75_v5_20260917_paused_*` 目录 | 3,987,356 | 未生成候选 checkpoint，闭环后不再需要 |
| v5 F1 首步失败现场 | `outputs/prelim75_v5_20260917/F1_failed_keyerror/` | 41,848 | optimizer 更新前失败；修复提交与登记保留 |
| 已关闭恢复监督大张量 | v3 teacher 的 `probabilities.npy`、`supervision.pt` | 428,671,809 | S1 路线关闭；保留 `boxes.npy`、paths 和 manifest |
| 范数对齐落败 checkpoint | `outputs/preliminary_no_sweep/20260911_normalign_desktop_r1/model/candidate.pt` | 352,916,263 | 63.3676% 负结果；登记和小型提交证据保留 |
| 桌面 v5 副本 | `submission_prelim75_v5_F0_20260918.zip`、`submission_prelim75_v5_F1_20260918.zip` | 1,070,323 | 平台分数已回填；仓库内提交 ZIP 保留 |
| **合计** |  | **45,754,734,484** | **约 45.755 GB / 42.612 GiB** |

## 明确保留

- 晋级权重链：v2 V1、v3 S0、v4 C0、v5 F1。
- 当前胜者 F1 checkpoint：`outputs/prelim75_v5_20260917/F1/candidate.pt`，SHA-256 `44e64a9528653f1d76db6a12b56a22c3ad23851665e8cef45d06f7ece9f52e1b`。
- 当前胜者仓库提交包：`outputs/prelim75_v5_20260917/F1/submission/submission.zip`，SHA-256 `0863b93c785042692ec928107429d679815b50d2b1971b4a984cc1bceb9a705d`。
- F0/F1 小型 CSV/ZIP、训练结果、诊断、配置和源码快照，用于核对平台回填。
- v3 `teacher/boxes.npy`、paths 和 manifest，保留训练几何来源；只删除已关闭恢复分类监督的大张量。
- `reproducibility/aegis_f1/outputs/F1_FLAT_FULL_FT_R3MS/` 历史 P 模型资产及原 trust/官方清单。
- 桌面 `Luqhhh_bf_tap_predict_prelim_V30A_OOB_BOTH_TARGETS.zip`、`Luqhhh_bf_tap_predict_prelim_V30B_OOB_TIME_1024.zip` 和其他非本项目文件；它们不属于本轮旧包清理范围。

## 可恢复性

上述 `outputs/` 和桌面字节未被 Git 跟踪，删除后不能通过 `git checkout` 恢复。历史命令、哈希、平台成绩和小型提交证据仍在；如确需被删 checkpoint 或缓存，只能按相应历史流程重新生成。当前 F1 可独立完成最终推理，不依赖被删的旧 checkpoint、教师概率、恢复监督或 v2 特征缓存。

## 删除后验证

- 所有登记删除路径均不存在；`outputs/` 从 `51,102,414,471` bytes 降至 `5,348,750,310` bytes，差值与仓库内删除项 `45,753,664,161` bytes 精确一致；加上桌面两包 `1,070,323` bytes 后得到总清理量。
- V1、S0、C0、F1 四个晋级 checkpoint，F1 ZIP 和 v3 boxes 的 SHA-256 均与原登记一致。
- F1 `pred_results.csv` 仍为 24,967 行，与 ZIP 内 CSV 字节一致；重新运行 `scripts/check_submission.py` 后全部格式、覆盖、标签范围和 ZIP 内容检查通过。
- `results/prelim75_v5_final_20260918.json` 可由 `python3 -m json.tool` 解析，文档差异通过 `git diff --check`。
