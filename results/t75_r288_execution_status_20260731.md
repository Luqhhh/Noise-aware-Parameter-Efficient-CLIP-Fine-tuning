# T75-R288 执行状态与预注册（2026-07-31）

状态：**PREREGISTERED_NOT_STARTED**

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
