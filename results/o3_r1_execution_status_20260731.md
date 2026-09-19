# O3-R1 执行状态与团队边界（2026-07-31）

状态：**ADAPTER_GATE_FAILED（训练正常完成，但准确率门禁未通过；禁止生成提交）**

## 1. 分支与血缘

- 专用分支：`agent/o3-r1-local-adapter`
- 起点：`origin/main@04c4111`
- 实验：`O3-R1_F1_LOCAL_ONLY_FEATURE_ADAPTER`
- 准备方：x28639 / Codex
- 实际运行人：待团队群内确认，当前未分配给队长或 JJT

开始准备前已执行 `git fetch origin main --prune`，并检查远端分支、开放 PR、实验总表与
O3/O3-R1 协议。未发现正在执行或已产出结果的 O3-R1；最新 `main` 正在推进的 288px
高分辨率 F1 与本实验不共享方法变量或输出目录。

## 2. 实验边界

O3-R1 只验证一个已经预注册的问题：在保持 F1 全局路径和唯一线性分类头冻结的条件下，
能否用一个仅作用于 attention-local 特征的 `512→32→512` 零初始化残差 Adapter，保留
O2 已观察到的局部判别增益，同时避免共享 LoRA/分类头导致的全局退化。

本分支不改变原 O3-R1 的模型、数据、损失、优化器、局部裁剪、融合权重或门槛。唯一允许
的修订仍是原协议规定的 validation cache `batch_size=128` 数值复现修复。

权威协议：

- `reproducibility/aegis_f1/docs/O3_F1_LOCAL_ONLY_FEATURE_ADAPTER_PROTOCOL_2026-07-22.md`
- `reproducibility/aegis_f1/docs/O3_R1_BATCH_REPRODUCIBILITY_AMENDMENT_2026-07-22.md`

## 3. 合规与团队隔离

- 只使用官方 train、OpenAI CLIP ViT-B/32 和单一最终模型；不使用外部数据。
- 完全不使用 prior、输出拉平、prior 强度扫描或平台分数反推测试分布。
- 不读取测试标签，不进行测试时训练；本地门控通过前不生成测试预测或提交包。
- 输出只能写入 O3-R1 专属目录，禁止覆盖团队正在运行的输出。
- 本分支只承载 O3-R1；后续结果无论成功或失败均立即追加并单独推送，不与其他实验合并提交。

## 4. 执行授权与停止边界

预执行版本提交时，下列操作均未执行；截至本次更新，前三项已按顺序完成：

1. GPU validation cache 重建；
2. GPU train cache 生成；
3. CPU Adapter 训练；
4. 测试集推理、提交包生成或平台评测。

用户于 2026-07-31 补充持续授权：后续项目内缓存与训练阶段不再逐次请求，只要启动前重新
同步最新 `main`、确认没有重复实验、规则风险或团队任务占用，即可按预注册顺序直接执行。
平台上传仍不自动进行；若发现团队进程、协议不兼容或合规风险，必须立即停止并报告。

## 5. Validation cache 执行结果

用户于 2026-07-31 明确授权 `O3-R1 batch-128 validation cache`。启动前重新同步
`origin/main@04c4111`，开放 PR 只有本实验的草稿 PR，未发现重复 O3-R1 分支或运行进程；
父 F1 checkpoint SHA-256 与预注册值完全一致：
`7da95e427b959e85cbbf37c99d47d9909b941032e836fc219aaea8e690d72cc4`。

缓存命令固定使用 CUDA、AMP、`batch_size=128`、`crop_size=160`、`top_patches=5`、
`num_workers=4`。正式进程从 `2026-07-31T20:57:37+08:00` 运行至
`2026-07-31T20:58:18+08:00`，耗时约 41 秒，退出码为 0。PIL 报告了已有图片的 EXIF 与
透明调色板元数据警告，但没有图片读取失败、非有限张量或样本缺失。

产物：

- 路径：`/home/x28639/projects/AegisCLIP-F6-A2LoRA/outputs/O3_R1_F1_LOCAL_ONLY_FEATURE_ADAPTER/seed42/cache/validation_bs128.pt`
- 大小：`63,168,621` bytes
- SHA-256：`eb9c362f1c59646e30d0d20e9c02bbb8966bddbfcf542da1b5ce04e1dbf9a1d5`
- 样本：`10,316`
- 环境：RTX 4060 Laptop GPU；PyTorch `2.13.0+cu130`；CUDA `13.0`；AMP enabled

严格复现门禁：

| 检查 | 实际结果 | 门槛 | 判定 |
|---|---:|---:|---|
| center 最大绝对 logit 差 | `0.0` | `0.0` | PASS |
| center prediction agreement | `1.0` | `1.0` | PASS |
| M1 最大绝对 logit 差 | `3.814697265625e-06` | `<=4e-6` | PASS |
| M1 prediction agreement | `1.0` | `1.0` | PASS |

结论：**O3-R1 validation cache 复现门禁通过。** 原 O3 的训练前失败可归因于 batch-64
AMP 数值布局改变少量 attention top-5 选择；batch-128 已恢复到平台 F1+M1 的冻结数值条件。
该结论只授权进入下一道独立门禁，不代表 Adapter 会提升准确率。

## 6. Train cache 执行结果

开始前再次同步 `origin/main@04c4111`；O3-R1 分支相对 `main` 仅包含本实验的两个记录
commit，工作区干净，开放 PR 只有本实验 PR #10，未发现重复实验或团队训练进程。训练 CSV
的 SHA-256 为
`a4a47bcc54bdbf1afce6713815d6c39c2d9b34a905f06553b80b4d21f5e6c6bb`；含
`65,473` 个唯一训练路径、500 类、每类 26–185 张，与 `10,316` 张验证集零重叠。

正式缓存一次运行成功，命令固定使用 CUDA、AMP、`batch_size=64`、`crop_size=160`、
`top_patches=5`、`num_workers=4`，耗时约 201 秒，退出码为 0。仅出现两条透明调色板
元数据提示，没有图片读取失败或非有限张量。

产物：

- 路径：`/home/x28639/projects/AegisCLIP-F6-A2LoRA/outputs/O3_R1_F1_LOCAL_ONLY_FEATURE_ADAPTER/seed42/cache/train.pt`
- 大小：`400,895,006` bytes
- SHA-256：`0869a2e788153afee198ae274ee609d300dbbd9b8604ab759dc4b209b02ed45f`
- 样本：`65,473`；唯一路径：`65,473`；类别：`500`
- 最低 clean probability：`0.7000105381011963`
- 父 checkpoint SHA-256：`7da95e427b959e85cbbf37c99d47d9909b941032e836fc219aaea8e690d72cc4`
- 环境：RTX 4060 Laptop GPU；PyTorch `2.13.0+cu130`；CUDA `13.0`；AMP enabled

完整性审计的 14 项检查全部通过：样本数、唯一路径、类别覆盖及每类范围、clean threshold、
训练/验证零重叠、父 checkpoint、源 CSV、batch size、worker 数、crop size、top-patch 数、
CUDA 与 AMP 均与预注册一致；缓存加载器同时完成所有张量的形状、对齐与有限值检查。

结论：**O3-R1 train cache 完整性门禁通过。** 该结果只允许进入固定 CPU Adapter 训练，
不代表模型提升，也不授权平台上传。

## 7. 固定 CPU Adapter 训练结果

训练前第三次同步 `origin/main@04c4111`，未发现新提交、重复实验或团队进程；32 个逻辑 CPU
的系统负载约 `0.46`，O3-R1 专属 checkpoint 目录不存在。训练严格使用预注册参数：seed 42、
`512→32→512`、residual scale `0.25`、dropout `0.1`、AdamW、lr `5e-4`、weight decay
`1e-4`、batch `1024`、最多 20 epochs、patience 5、GCE q `0.5`、local loss weight
`0.25`、feature anchor weight `2.0`，并固定在 CPU 上运行。

正式训练一次运行成功，约 30 秒完成，在 epoch 13 早停；最佳安全 epoch 为 8。权威门禁结果：

| 检查 | 实际结果 | 门槛 | 判定 |
|---|---:|---:|---|
| clean-core micro 相对 F1+M1 | `+0.1487255 pp` | `>=+0.20 pp` | **FAIL** |
| trusted macro 相对 F1+M1 | `+0.1994252 pp` | `>=0.00 pp` | PASS |
| raw micro 相对 F1+M1 | `+0.1938701 pp` | `>=-0.10 pp` | PASS |
| local feature drift | `0.00180197`（约 `0.1802%`） | `<=0.01` | PASS |
| epoch-0 Adapter / M1 复现 | bit-exact / reproduced | 必须通过 | PASS |
| F1 global path | bit-exact | 必须通过 | PASS |

最佳 epoch 8 的 clean-core micro 为 `82.395887%`，基线为 `82.247162%`；raw micro 从
`71.636295%` 升至 `71.830165%`，trusted macro 从 `81.273472%` 升至 `81.472898%`。
所有安全 epoch 均覆盖 500 类，无空类别。局部分支本身的 raw micro 从 `66.867000%` 升至
`68.233812%`，说明局部 Adapter 确实学到互补信息，但融合后的 clean-core 增量不足以跨过
预注册门槛。

产物：

- `gate.json`：334 bytes；SHA-256 `26fefc3b0468b47e4bbdb8a097c7c24053679df1a4470f3e60b1c296ab7d28f8`
- `history.json`：22,654 bytes；SHA-256 `bc592b81cf379bc677727bce21d8da1d32d02bdef68892b1ec8f8579ed78338f`
- `best_adapter.pt`：143,969 bytes；SHA-256 `11e5cd8dd7d844e25a8310c02ba268dd729c3ab125cf17a105dbac1b03d968c8`
- `best.pt`：356,505,716 bytes；SHA-256 `bcb04ceeb56555ac3dfe54216d9fdb1137beacf1ee371f8492496f06efd06fb5`

结论：**O3-R1 正式门禁失败。** 唯一失败项是 clean-core micro 增量距离门槛仍差
`0.0512745 pp`。不得事后放宽门槛、不得生成测试预测或平台提交；该失败仍保留并立即推送，
作为后续正交实验的排除证据。

## 8. 后续固定顺序

1. 再次同步 `main` 并复核重复项；
2. batch-128 validation cache 与逐位复现门禁已完成并通过；
3. train cache 已生成一次，完整性门禁已完成并通过；
4. 固定 CPU Adapter 已执行一次，权威准确率门禁失败；
5. 本失败结果立即补充并单独推送；
6. O3-R1 到此停止，不生成测试预测或提交包；后续新假设必须重新预注册并独立建分支。
