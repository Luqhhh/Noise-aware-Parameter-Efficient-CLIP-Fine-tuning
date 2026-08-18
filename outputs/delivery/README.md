> **本文件为 2026-08-03 快照，已被 [README_delivery_status.md](README_delivery_status.md) 取代。**
> 当前平台最佳：**70.352866%**（全微调 R3MS 父模型 + 双 Adapter prior 0.90，
> 17,565/24,967，2026-08-06）。FLAT 67.7014% 仅为 08-03 当时的最佳。

# 最终提交包交付（2026-08-03）

## 提交包列表（均审计通过）

### 1. `flat_child_l040_f050/` —— 本轮最佳（平台验证过）
- 模型：FLAT（A12_CORR 配方 + schedule_epochs 16 平 LR，on E2 父模型）
- 推理：M1+flip（crop160/top5/l040/f050）+ temp1.5 + prior0.85
- **平台 67.7014%**（+0.0161pp vs A12_CORR，新审计完整平台最佳）
- ZIP SHA-256：`db2f42f9925081a2ce788a1a826d0b83af7d960c02c44f481b3748956f7974ba`
- checkpoint：`6543d93b…b54`

### 2. `current_best_a12corr/` —— 前一最佳
- 模型：A12_CORR（LoRA 全 12 block + 伪标签修正，on E2 父模型）
- 推理：M1 attention-crop + flip 四视图（crop160/top5/l040/f050）+ temp1.5 + prior0.85
- **平台 67.6853%**（已实测，前平台最佳）
- ZIP SHA-256：`b2b1924a98ad9179c17d2421c3584a910070e0c611713707079c5ac58f3e0585`
- checkpoint：`ec6948c9…a3238`

## 备注
- 桌面 `submission.zip` = FLAT 包（已上传，平台 67.7014%）。
- 两个 checkpoint 本地 M1+flip 持平（0.8249 vs 0.8251），但 FLAT 裸表征
  +0.1pp 且平台转正——平台是唯一裁判。

## 平台要求
- CSV 格式 `image_name.jpg, 0001`（comma+space，4 位补零标签）
- 24,967 条预测，500 类全覆盖

## 完整实验记录
`/home/lux1/noise/results/75p_round2_20260803.md`
