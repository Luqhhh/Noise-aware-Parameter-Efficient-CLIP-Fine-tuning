# T76 LoRA Depth-8 Rank-4 执行状态与预注册（2026-07-31）

状态：**PREREGISTERED_PREFLIGHT_PASSED_NOT_STARTED**

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
