# 存储清理（2026-08-05）

## 结果

按用户要求删除已淘汰且可重训/再生成的大文件。清理前：

- 根目录 `outputs/`：40 GiB
- `reproducibility/aegis_f1/outputs/`：135 GiB
- 未启用的 `reproducibility/aegis_f1/.venv/`：4.9 GiB

清理后：

- 根目录 `outputs/`：833 MiB
- `reproducibility/aegis_f1/outputs/`：860 MiB
- `.venv/`：已删除（项目命令使用已验证的系统 Python 环境）

累计释放约 178 GiB。删除内容不可直接恢复，但实验配置、结果报告和 Git 记录仍在，
对应模型与缓存可重训或再生成。

## 删除范围

- 根目录 outputs 下 111 个历史 checkpoint `.pt`（约 38.7 GiB）。
- AEGIS outputs 下除当前最佳 R2 epoch 3 外的 392 个历史 checkpoint `.pt`
  （约 130.3 GiB）。
- 已淘汰的 Phase-4 structural head/SWA checkpoint。
- 历史 local-adapter、CVRG、OOF 与 train-feature 缓存。
- 已淘汰模型和 crop176 的测试 logits。
- 未启用且可重建的本地 `.venv`。

## 保留并复核

- 当前最佳/下一轮父 checkpoint：
  `F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt`
  SHA-256 `67efab2bf954139b59df074ccf00c0113cbc6ff96163d6e8d66ffbe553b910a4`
- R2 教师原图/翻转 logits 缓存：
  SHA-256 `ce834c802a8e02d27e8878a7539c0239d70682750e2e1f8272a0fe00151840bc`
- 当前最佳多尺度 fused logits：
  SHA-256 `9bb1e6542781e2a0def12eb1c080bd475bac2267ac2ba02010fa391bd452b8c5`
- 当前最佳 ZIP 与桌面副本逐字节一致：
  SHA-256 `eaf93105260340d073ab74aa91058acffcece4a28d2e49ac2d5317297762233c`

清理后项目 outputs 中超过 50 MiB 的可变文件仅剩当前 R2 checkpoint 与教师 logits
缓存；多尺度 logits 小于该阈值但也显式保留。
