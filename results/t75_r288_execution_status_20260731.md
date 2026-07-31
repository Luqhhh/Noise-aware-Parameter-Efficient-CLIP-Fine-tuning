# T75-R288 执行状态与预注册（2026-07-31）

状态：**TRAINING_GATE_FAILED（训练完成；禁止 localization、测试推理与提交）**

## 1. 分支、血缘与查重

- 专用分支：`agent/t75-r288-training`
- 起点：`origin/main@04c4111`；启动前同步至 `origin/main@6abf640`
- 实验 ID：`T75_R288_ARCHIVED_E2_W060`
- 执行人：x28639 / Codex
- 父 checkpoint：`E2_MIXUP_CE5_REPLICA` epoch 44
- 父 checkpoint SHA-256：`50e05d09921a0f9bf852589cae848b926c61892e6952f1082dfa25daae2e3ff6`

建立分支前已同步最新 `main`，检查远端分支、全部 PR、本地输出目录和 GPU 进程。没有
T75-R288 结果、重复分支或正在执行的训练。团队 `main` 中的通用 W060-R288 配置依赖
本机已缺失的 `E2_MIXUP_CE5_REBUILD`（预期 SHA-256 `1e4cd952...`），因此本实验不冒充
该血缘，而是明确使用原始 F1 实际使用且仍可审计的 archived E2 父模型，另设实验编号。

## 2. 单一科学变量与动机

相对 224px W060 F1，只改变视觉输入分辨率为 288px，并通过位置编码插值加载 224px E2
父模型；LoRA 覆盖、rank、优化器、噪声权重、训练 split 和验证 split 均保持固定。此前把
224px checkpoint 直接提高推理分辨率会退化，不能否定“在目标分辨率上重新训练 LoRA”；
本实验专门检验后者能否利用细粒度纹理。

配置：`reproducibility/aegis_f1/configs/t75_r288_archived_e2_w060.yaml`。

## 3. 固定训练协议

- 官方 train 103,218 张；固定 seed42 train 92,902 / val 10,316；无外部数据。
- OpenAI CLIP ViT-B/32；单模型；最后 4 个视觉 block 的 Q/V/out LoRA，rank 8、alpha 8。
- 288×288 weak RRC + horizontal flip；W060：trust threshold 0.60，最低样本权重 0.60。
- GCE q=0.5；不启用 pseudo-label correction、prior、输出拉平、MixUp 或测试时训练。
- 8 epochs；batch 64；AMP；head lr 5e-5；backbone lr 2e-5；无早停。
- 由于 288px 与 224px 父特征不在同一输入空间，feature distillation 固定关闭。
- 同理，288px 相对 224px 冻结特征的 drift 不再进入 selector；`drift_penalty=0`，仅以
  宽松的 `15%` 上限排除失控模型，而不把分辨率改变本身误判成训练退化。
- 输出仅写入本工作区 `reproducibility/aegis_f1/outputs/T75_R288_ARCHIVED_E2_W060/`。

唯一正式命令：

```bash
cd /home/x28639/projects/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning-t75-r288/reproducibility/aegis_f1
PYTHONPATH=$PWD /home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
  -m aegis_clip.cli.train \
  --config configs/t75_r288_archived_e2_w060.yaml
```

只允许一次正式运行，不自动重试。已知 224px 同类训练约 35 分钟，因此本实验硬超时预注册为
120 分钟；进程或 GPU 停滞时保留现场并停止，不通过修改 batch 或超参恢复同一运行。

## 4. 双层门禁

训练器内置门禁（必须全部通过）：

1. clean-core selector 相对 epoch 0 至少 `+0.20pp`；
2. raw micro 相对 epoch 0 不低于 `-0.10pp`；
3. mean feature drift `<=15%`；该阈值只作失控保护，不参与 selector；
4. 预测覆盖 500 类；
5. 父子 train/val 路径与标签 lineage 审计通过。

相对 224px W060 的突破门禁（决定是否值得另开 288px 局部推理实验）：

- global raw micro 至少 `70.8797%`（W060 70.5797% + 0.30pp）；
- clean-core micro 至少 `81.5259%`（W060 81.3259% + 0.20pp）；
- 两项必须同时满足，且内置 promotion 必须通过。

若任一门禁失败，本实验如实记录并停止，不做测试集推理、不生成提交包、不事后放宽门槛。
若全部通过，也只允许进入一个新的、单独预注册的 288px localization 实验；平台上传仍不
自动执行。

## 5. 启动前停止条件

正式启动前再次 fetch `main` 并查重；若 `main` 出现同实验结果、团队有 GPU 进程、父模型
或输入哈希不匹配、配置测试失败、比赛合规边界变化，则停止。T75 全程与 prior 路线正交。

第一次启动检查时，`main` 从 `04c4111` 前进到 `6abf640`，说明 288px 相对 224px 父特征
天然会产生约 4%–10% drift；原 1% drift budget 与惩罚会错误压低 clean-core selector。
训练尚未启动，故先合并团队修复，并把本实验同样改为 `drift_budget=0.15`、
`drift_penalty=0.0`。这只修正跨分辨率评估口径，不改变训练梯度、数据或模型容量。

## 6. 正式运行结果

第二次启动门禁确认 `origin/main@6abf640` 未再变化，分支干净、无重复 PR、无团队训练或
GPU 计算进程、专属输出目录不存在；父 checkpoint、train/val CSV 和 trust bundle 哈希
均与预注册一致。正式命令只运行一次，从约 `21:30:45` 至 `22:26:41`，耗时
`3356.5s`（约 55分56秒），退出码 0。PIL 只报告既有图片的 EXIF/透明调色板元数据提示，
没有图片读取失败、非有限损失或样本丢失。

Lineage 审计通过：父子 train 均为 92,902 张、val 均为 10,316 张，train/val 交叉重叠
为 0，标签不一致为 0；父 checkpoint SHA-256 为
`50e05d09921a0f9bf852589cae848b926c61892e6952f1082dfa25daae2e3ff6`。

epoch 0（未适应 288px）为 raw `70.046532%`、clean-core `80.070311%`、drift
`3.976548%`。训练器按 clean-core 选择 epoch 4：

| 指标 | epoch 0 | 最佳 epoch 4 | 变化 |
|---|---:|---:|---:|
| raw micro | `70.046532%` | `70.880187%` | `+0.833654pp` |
| clean-core micro | `80.070311%` | `80.949163%` | `+0.878853pp` |
| trusted macro | `79.281437%` | `80.137515%` | `+0.856078pp` |
| proxy macro | `78.289396%` | `79.016405%` | `+0.727009pp` |
| mean feature drift | `3.976548%` | `21.290892%` | `+17.314344pp` |
| predicted classes | `500` | `500` | 不变 |

内置 promotion：

| 检查 | 结果 | 判定 |
|---|---:|---|
| selector gain `>=+0.20pp` | `+0.878853pp` | PASS |
| raw floor `>=-0.10pp` | `+0.833654pp` | PASS |
| class coverage | `500` | PASS |
| trained epoch selected | epoch 4 | PASS |
| drift budget `<=15%` | `21.290892%` | **FAIL** |

相对 224px W060 的突破门禁：raw `70.880187%` 刚超过 `70.8797%` 门槛约
`0.000487pp`，但 clean-core `80.949163%` 低于 `81.5259%` 门槛 `0.576737pp`；
其本身也比 224px W060 clean-core `81.3259%` 低 `0.376737pp`。因此双层门禁均失败。

产物：

- `best.pt`：356,368,714 bytes；SHA-256
  `2692afbe733263fee3820ed47f2b62e6253da7b927821932955070b18be80e22`
- `promotion.json`：289 bytes；SHA-256
  `d2c44cfa663ecf7912ceb8cff0641d039847ee3359debfada62c8ac149e2363d`
- `split_lineage_audit.json`：1,481 bytes；SHA-256
  `335b812520f1b968cd7759c838e7a4a8d0cd2532e9f741d2cdc3235f1d61afd2`
- `artifact_manifest.json`：4,848 bytes；SHA-256
  `6c8fb35b9f3ec531b6c45def7f43b75dda7d9cf505f27056928b36f08b1f6850`
- `metrics.csv`：4,458 bytes；SHA-256
  `70b59474cf24c9a73d7ad29f3955d9a993cb92ca6536df201e61dd4562f43725`

结论：**T75-R288 正式失败并停止。** 288px 训练能把 noisy-val raw 相对 224px W060
抬高约 `0.3005pp`，但 clean-core 同时下降且视觉漂移远超保护上限，说明当前无跨分辨率
锚定的 LoRA 在拟合高分辨率统计时破坏了可靠结构。不得据 raw 单项生成 288px localization、
测试预测或平台提交，不得事后放宽 drift/clean-core 门槛。后续若继续高分辨率方向，必须作为
新的独立假设解决“跨分辨率结构保持”，而不是扫描当前学习率或 epoch。
