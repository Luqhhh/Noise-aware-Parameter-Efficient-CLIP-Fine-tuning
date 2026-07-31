# T76 LoRA Depth-8 Rank-4 执行状态与预注册（2026-07-31）

状态：**TRAINING_PASSED_M1_PAIR_PREREGISTERED_NOT_STARTED**

## 1. 分支、查重与血缘

- 专用分支：`agent/t76-depth8-rank4`
- 起点：`origin/main@f1fe860`
- 实验 ID：`T76_F1_DEPTH8_RANK4_ARCHIVED_E2`
- 配置：`reproducibility/aegis_f1/configs/t76_f1_depth8_rank4_archived_e2.yaml`
- 父 checkpoint：`E2_MIXUP_CE5_REPLICA` epoch 44，SHA-256 `50e05d09921a0f9bf852589cae848b926c61892e6952f1082dfa25daae2e3ff6`
- 严格对照：原始 `F1_VISUAL_LORA_CLEAN_CORE`，SHA-256 `7da95e427b959e85cbbf37c99d47d9909b941032e836fc219aaea8e690d72cc4`

启动前已同步最新 `main`，检查远端分支、开放 PR、配置、结果和本地输出；没有 depth-8/rank-4 或同义实验。团队现有 LoRA 配置最多只覆盖最后 4 个视觉 block。重建版 E2/W050 大 checkpoint 未纳入 Git 且本机不存在，因此本实验不冒充 W050 子实验，明确使用仍可完整核验的原始 E2/F1 归档链做 matched control。

## 2. 单一变量与研究依据

原 F1 在最后 4 个视觉 block 使用 rank 8、alpha 8；T76 仅改为最后 8 个 block 使用 rank 4、alpha 4。`blocks × rank` 均为 32，且 `alpha/rank` 均为 1，因此目标是在近似相同 LoRA 参数预算和相同残差尺度下，把容量从“后层集中”改为“跨层分布”。数据、父模型、增强、损失、学习率、训练轮数、trust 策略和选择门槛全部固定。

项目内 T70-03 已证明在最后 4 层把 rank 8 提高到 rank 16 会同时损害 M1+flip；这否定的是“同位置增加容量”，并未检验“固定预算扩大层覆盖”。近期视觉 PEFT 原始研究也指出，更低 rank 能让适配更均匀覆盖多层，并且不同层不应机械采用相同容量：

- 1LoRA（WACV 2026）：https://openaccess.thecvf.com/content/WACV2026/html/Quercia_1LoRA_Summation_Compression_for_Very_Low-Rank_Adaptation_WACV_2026_paper.html
- SR-LoRA（ICCV 2025）：https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Beyond_Low-Rank_Tuning_Model_Prior-Guided_Rank_Allocation_for_Effective_Transfer_ICCV_2025_paper.html

## 3. 固定训练协议

- 官方 train 103,218 张；固定 seed42 train 92,902 / validation 10,316；无外部数据。
- OpenAI CLIP ViT-B/32，单模型；只训练视觉 LoRA 与唯一线性分类头。
- 最后 8 个视觉 block 的 Q/V/out LoRA；rank 4、alpha 4。
- trust threshold 0.70；低于阈值样本不参与分类监督；GCE q=0.5；feature distillation weight=2.0。
- 224×224 weak RRC + horizontal flip；6 epochs；batch64；AMP；head lr 5e-5；visual lr 2e-5。
- 不启用 pseudo-label correction、prior、输出拉平、MixUp、模型融合或测试时训练。
- test 在全部本地门禁通过前禁止读取；平台上传不自动执行。

唯一正式命令：

```bash
cd /home/x28639/projects/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning-t76-d8r4/reproducibility/aegis_f1
PYTHONPATH=$PWD /home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
  -m aegis_clip.cli.train \
  --config configs/t76_f1_depth8_rank4_archived_e2.yaml
```

只允许一次正式训练，不自动重试；硬超时 120 分钟。正式启动前必须再次 fetch `main`、查重、检查团队 GPU、核验输入哈希与参数预算，并确认专属输出目录不存在。

## 4. 双层门禁

训练器内置门禁必须全部通过：clean-core selector 相对 epoch 0 至少 `+0.20pp`、raw micro 不低于 `-0.10pp`、mean feature drift `<=1%`、覆盖 500 类、父子 split/标签 lineage 完整一致。

相对原始 F1 的结构突破门禁：

- 原 F1 最佳 epoch 4：raw micro `70.676619%`、clean-core micro `81.530559%`、trusted macro `80.607790%`、drift `0.408131%`。
- T76 clean-core micro 必须至少 `81.730559%`（原 F1 +0.20pp）。
- T76 raw micro 必须至少 `70.576619%`（原 F1 -0.10pp）。
- trusted macro 不得低于 `80.607790%`；drift `<=1%`；覆盖 500 类。
- 只有内置门禁与结构突破门禁同时通过，才允许另行执行固定的 validation M1+flip 对照；否则立即关闭，不读取 test、不生成提交包。

不允许事后放宽门槛或围绕本次结果扫描 rank/depth。若结果为正，再由团队使用同一结构在本机缺失的 W050 重建父模型上独立复现。

## 5. 训练前静态审计

- `tests/test_config.py` 与 `tests/test_model.py`：36 项全部通过。
- 原 F1 对照 `last4 × rank8 × alpha8`：LoRA 参数 147,456；分类头参数 256,500；可训练总参数 403,956。
- T76 `last8 × rank4 × alpha4`：LoRA 参数 147,456；分类头参数 256,500；可训练总参数 403,956。
- LoRA 参数差与可训练总参数差均为 0；残差缩放 `alpha/rank` 均为 1.0，matched-budget 假设精确成立。
- train CSV SHA-256：`a726b8a3ca8bc5857136106aca80f01d557104d3661ef92ccedfb2c0ea087875`。
- validation CSV SHA-256：`54a790b35f836cfba4c19cbb5fe38c4b1b37aab62cc9d477f9285496b2d5568e`。
- trust bundle SHA-256：`52e59a991a5eb3c57abdfabee5647423726f51fbdd3da2ce377467664d173608`。

静态门禁通过，但训练尚未启动。配置与预注册必须先独立提交、推送并建立草稿 PR；随后再次同步 `main`、查重并检查团队进程。

## 6. 正式训练结果

启动前再次确认 `origin/main@f1fe860` 未变化、分支干净、只有本实验 PR #13、无团队 GPU 计算进程、专属输出不存在；正式命令只执行一次。运行从约 23:00:31 至 23:40:19，墙钟时间 2,388.2 秒（39 分 48 秒），退出码 0。图像库仅报告既有 EXIF/透明调色板元数据提示，没有读取失败、非有限损失、数据超时或样本丢失。

Lineage 审计通过：parent/child train 均为 92,902，validation 均为 10,316；双向交叉重叠为 0，标签不一致为 0，父 checkpoint SHA-256 精确匹配预注册值。

训练器按 clean-core selector 选择 epoch 4：

| 指标 | epoch 0 | T76 最佳 epoch 4 | 相对 epoch 0 |
|---|---:|---:|---:|
| raw micro | 70.230709% | 70.928657% | +0.697947pp |
| clean-core micro | 80.759868% | 81.787455% | +1.027584pp |
| trusted macro | 79.920453% | 80.827421% | +0.906968pp |
| proxy macro | 78.824830% | 79.648054% | +0.823224pp |
| mean feature drift | 0.000083% | 0.487214% | +0.487131pp |
| predicted classes | 500 | 500 | 不变 |

训练器内置 promotion 五项全部通过：selector gain、raw floor、drift budget、500 类覆盖和训练 epoch 选择均为 PASS；权威 `promotion.json passed=true`。

相对原始 F1 的 matched-control 结构门：

| 指标 | 原 F1 epoch 4 | T76 epoch 4 | 差值 | 门槛 |
|---|---:|---:|---:|---:|
| raw micro | 70.676619% | 70.928657% | +0.252038pp | ≥-0.10pp |
| clean-core micro | 81.530559% | 81.787455% | +0.256896pp | ≥+0.20pp |
| trusted macro | 80.607790% | 80.827421% | +0.219631pp | ≥0pp |
| proxy macro | 79.447788% | 79.648054% | +0.200266pp | 观察项 |
| mean feature drift | 0.408131% | 0.487214% | +0.079083pp | T76 ≤1% |

结构门全部通过。该结果支持“固定参数预算，把 LoRA 从后 4 层 rank8 分布到后 8 层 rank4”优于同父、同数据、同训练协议的后层集中方案；它也与 T70-03 的 rank16 负结果形成正交对照。

核心产物：

- `best.pt`（epoch 4）SHA-256：`996857951856d743001d837dbd55d7259ed08f1db1479ca91fff5aafd9dd3a23`。
- `promotion.json` SHA-256：`afd43776cee39c526339eab44723e49f70581f7e1641167bdbfe78b2ecf931ab`。
- `best_evaluation.json` SHA-256：`b32f01fc298770049780f3ea4128caf308dc9e9489fda6f433548ece4ff3e58e`。
- `artifact_manifest.json` SHA-256：`f9c9b05e49e55cd383e202e0a54387e911b7058d5b690e41a58d64e43e61c76a`。
- `split_lineage_audit.json` SHA-256：`335b812520f1b968cd7759c838e7a4a8d0cd2532e9f741d2cdc3235f1d61afd2`。
- `metrics.csv` SHA-256：`be9a99998a4eba0762e46420298a4ce5590f256d1718ba8248a990e0a2ccc254`。

按预注册规则，本阶段允许进入一次固定 validation M1+flip 对照，但仍未读取 test、未生成提交包、未进行 prior 或输出拉平。训练结果必须先作为独立提交推送，随后重新同步 `main`、查重与检查团队 GPU。

## 7. 固定 validation M1+flip 成对对照预注册

为避免把 T76 与不同父模型的团队重建版 F1 混为一谈，M1+flip 阶段固定为同一 archived-E2 血缘的两次确定性 validation 推理：原始 F1 对照与 T76 候选。两者都使用同一 T76 配置提供 validation/trust 元数据，只改变 checkpoint；不扫描 localization 参数。

固定参数：attention top-5、crop160、local weight 0.40、水平翻转、flip weight 0.50、temperature 1.0、batch128。

原始 F1 对照命令：

```bash
cd /home/x28639/projects/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning-t76-d8r4/reproducibility/aegis_f1
PYTHONPATH=$PWD /home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
  -m aegis_clip.cli.sweep_localization \
  --checkpoint /home/x28639/projects/AegisCLIP-Noise-Robust/outputs/F1_VISUAL_LORA_CLEAN_CORE/seed42/checkpoints/best.pt \
  --config configs/t76_f1_depth8_rank4_archived_e2.yaml \
  --output outputs/T76_F1_DEPTH8_RANK4_ARCHIVED_E2/seed42/localization/m1_flip_control_original_f1_l040_f050.json \
  --crop-sizes 160 --top-ks 5 --local-weights 0.4 \
  --include-horizontal-flip --flip-weights 0.5 \
  --temperature 1.0 --batch-size 128
```

T76 候选命令与上式完全相同，只把 checkpoint 换为本实验 `best.pt`，输出换为 `m1_flip_candidate_t76_l040_f050.json`。两次命令各只执行一次，不使用 `--overwrite`。

M1+flip 晋级门：

- T76 raw micro 至少高于原始 F1 `+0.15pp`；
- T76 clean-core micro 至少高于原始 F1 `+0.15pp`；
- trusted macro 与 proxy macro 均不得回退；
- 两者均覆盖 500 类，checkpoint SHA-256 与预注册值一致；
- 四项同时通过才允许另行考虑 test 推理；否则保留全局训练正结果，但关闭当前 T76 的测试与提交路径。

执行前必须再次 fetch `main`、查重并检查团队 GPU。先运行原始 F1 对照并审计/记录，再运行 T76 候选；任一命令失败不自动重试。
