# V5X O2L + M1 无 prior 提交交付

状态：`platform_valid_not_promoted_closed`

V5X 使用此前未提交的 O2L 噪声鲁棒检查点，并以 M1 注意力局部—全局概率融合完成确定性推理。它不使用类别 prior、输出拉平、平台分布反推、外部数据、第二模型或测试时训练。

## 固定配置

- 单一 checkpoint：`O2L_F1_ATTENTION_LOCAL_TRAINING/seed42/checkpoints/best.pt`
- checkpoint SHA-256：`32c463df68fa71cac20768162bae16105137865478e41677faa5965bbdb1fc6d`
- 全局视图：CLIP 224 center crop
- 局部视图：最后一层 attention，12 heads 均值，top-5 patches，crop 160
- 融合：全局/局部概率各 0.5
- 温度：1.0
- 测试图像：24,967；损坏图像：0

## 本地证据与风险

- 相对已平台评测的旧 F1+M1，验证 clean-core micro、trusted macro、raw micro 分别变化 `+0.1758pp`、`+0.2252pp`、`+0.2811pp`。
- 相对旧 F1+M1 测试预测，V5X 改变 2,249 / 24,967 行（`9.0079%`），不是重复提交。
- O2L center clean-core 相对旧 F1 center 为 `-0.7437pp`，因此平台增益尚未得到证明；不得把本地变化冒称为平台提升。

## 平台结果

- V5X：**64.0806%**（用户回传）
- 相对旧 F1+M1 63.3276%：`+0.7530pp`
- 相对当前无 prior 最优 G0 69.2274%：`-5.1468pp`
- 判定：提交有效，O2L 相对旧基线的正收益得到平台支持，但明显低于最新主线；不晋级、不继续扫描 O2L 推理参数。

## 交付资产

- ZIP：`outputs/delivery/v5x_o2l_m1_no_prior_20260919/submission.zip`
- ZIP SHA-256：`55e037ccb3891b456c1c58cc690527b075562a626d2c9211b9fa1aa0b910e7d4`
- CSV SHA-256：`54da4fd410fbb38521225521bbe8740452b7297ec0070fbe66d27617e5b2b0c9`
- 双重格式审计：通过
- 平台成绩：`64.0806%`

该包已作为隔离探针完成平台评测，没有与 prior 平衡输出混合。该路线现已关闭。
