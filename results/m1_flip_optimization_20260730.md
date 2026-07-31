# F1 REBUILD R1：M1 + Flip 优化（2026-07-30）

## 结论

在最近平台实测 `62.9791%` 的 F1 REBUILD R1 + M1 上，增加确定性的水平翻转
视图，并在概率空间进行两级融合：

1. 原图/翻转图分别生成 global 与 attention crop local 概率；
2. 每个分支按 `flip_weight` 融合原图与翻转图；
3. 再按 `local_weight` 融合 global 与 local。

固定 `crop160/top5/temperature1.0`，有界扫描
`local_weight ∈ {0.25,0.30,0.35,0.40}` 与
`flip_weight ∈ {0.25,0.50}`。选择
`local_weight=0.40, flip_weight=0.50`：它取得最高 raw micro，且相对当前平台包
对应的 M1 weight 0.35，trusted、proxy、clean-core 守护指标全部同向改善。

该候选已在平台实测为 **63.7802%**，成为新的审计完整平台最佳，离线增益经平台
验证为正。相对 F1 REBUILD R1 + M1 weight 0.35（`62.9791%`）`+0.8011pp`，
相对已报告原 F1 + M1（`63.3276%`）`+0.4526pp`，且后者仅有 `reported_unverified`
证据，本候选则有完整 checkpoint/prediction/ZIP 哈希审计。距离 70 分
`6.2198pp`；本地 raw micro `72.1307%` 到平台的 gap 为 `8.3505pp`。

## 选中候选

| Metric | M1 weight 0.35 | M1 + Flip 0.40/0.50 | Delta |
|---|---:|---:|---:|
| raw micro | 71.8011% | **72.1307%** | **+0.3296pp** |
| raw macro | 71.7716% | **72.1080%** | **+0.3364pp** |
| trusted micro | 81.0360% | **81.3223%** | **+0.2864pp** |
| trusted macro | 81.3955% | **81.5475%** | **+0.1520pp** |
| proxy micro | 79.7006% | **79.9852%** | **+0.2845pp** |
| proxy macro | 80.1684% | **80.3024%** | **+0.1340pp** |
| clean-core micro | 82.2398% | **82.5126%** | **+0.2728pp** |
| clean-core macro | 82.9285% | **83.0216%** | **+0.0931pp** |
| predicted classes | 500 | 500 | 0 |

相对相同 `local_weight=0.40` 的无 Flip M1，选中候选 raw micro
`+0.2811pp`、trusted macro `+0.1698pp`、proxy macro `+0.2207pp`、
clean-core micro `+0.2182pp`、clean-core macro `+0.1857pp`。因此增益并非仅由
M1 权重从 0.35 调至 0.40 造成。

完整 8 点结果见 `results/m1_flip_optimization_20260730.csv`。扫描原始 JSON
SHA-256：
`de95b3ee9c0af32f760d18bf99a4c8325dddfc12be9549eccc6ec1238b6710c9`。

## 平台包审计

- checkpoint SHA-256：
  `805e0df7d2ee7dfb2481a08664b71eb5f7ed600f27f1d5af726c926f3e6fb281`
- inference mode：
  `attention_crop_flip:topk=5:crop=160:local_weight=0.4:flip_weight=0.5:t=1`
- prediction count：24,967
- predicted classes：500
- corrupt images：0
- prediction CSV SHA-256：
  `c2e570c57763e190c8b970dcfe40cf12edc383a460edda14d4b83b7728156265`
- submission ZIP SHA-256：
  `67f4eda57291e34096edcb0545b142fd0a3114fb1c76eb1e17996afe87d692e0`
- ZIP 内容：仅一个 `pred_results.csv`
- `aegis-audit-submission --allow-tta`：passed
- 相对 62.9791 包：1,936 / 24,967 个预测改变，比例 7.7542%
- 状态：`platform_valid_promoted`（平台实测 **63.7802%**，新审计完整平台最佳）

路径：

`reproducibility/aegis_f1/outputs/F1_VISUAL_LORA_CLEAN_CORE_REBUILD_R1/seed42/submissions/m1_flip_l040_f050/submission.zip`

该协议每张图执行原图/翻转图各一次 global 与 local 前向，共四次模型前向；约为
原 M1 两次前向的 2 倍推理计算。它是带 Flip TTA 的单 checkpoint 包，不包含跨模型
融合。

平台实测 **63.7802%** 已按 checkpoint/prediction/ZIP SHA-256 精确登记到
`results/current_platform_summary.csv` 与 `results/submission_registry.csv`。

## 平台结果（2026-07-31 回填）

- 平台实测：**63.7802%**；
- 相对 F1 REBUILD R1 + M1 weight 0.35（62.9791%）：`+0.8011pp`；
- 相对已报告原 F1 + M1（63.3276%）：`+0.4526pp`，且本候选为 `audited`、
  原报告仅为 `reported_unverified`；
- 相对 A2 STRICT + M1（62.6870%）：`+1.0932pp`；
- 距离 70 分：`6.2198pp`；
- local raw micro `72.1307%` 到平台的 gap：`8.3505pp`；
- 平台状态：`platform_valid_promoted`，新的审计完整平台最佳；
- registry 行：`F1_REBUILD_R1_M1_FLIP_L040_F050_20260730`。

## 验证

- localization/TTA/submission 相关测试：21 passed
- 生成后提交审计：passed
- 所有融合点保持 500 类覆盖
