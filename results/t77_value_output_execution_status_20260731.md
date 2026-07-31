# T77 Value/Output LoRA 执行状态与预注册（2026-07-31）

状态：**USER_STOPPED_AFTER_EPOCH2_INVALID_FOR_PROMOTION**

## 1. 分支、查重与边界

- 专用分支：`agent/t77-value-output`
- 起点：`origin/main@0107f73`
- 实验 ID：`T77_F1_DEPTH8_RANK6_VALUE_OUT_ARCHIVED_E2`
- 配置：`reproducibility/aegis_f1/configs/t77_f1_depth8_rank6_value_out_archived_e2.yaml`
- 父 checkpoint：archived E2 epoch 44，SHA-256 `50e05d09921a0f9bf852589cae848b926c61892e6952f1082dfa25daae2e3ff6`
- matched control：原始 F1，SHA-256 `7da95e427b959e85cbbf37c99d47d9909b941032e836fc219aaea8e690d72cc4`

启动前已同步最新 `main`，检查远端分支、开放 PR、团队配置/结果和独立研发归档。现有 J0/J1 是最后 4 层 Q/V/out rank8，T70-03 是最后 4 层 Q/V/out rank16，T76 是最后 8 层 Q/V/out rank4；没有最后 8 层 V/out-only rank6 或同义实验。当前本机仍不存在团队 W050 重建 checkpoint，因此 T77 与 T76 一样只在可完整核验的 archived-E2/F1 同血缘上做结构消融，不冒充 W050 结果。

本实验完全避开 prior、输出拉平、平台分布推断、外部数据、模型融合和测试时训练；只使用官方 train。test 在所有本地硬门通过前禁止读取，平台上传不自动执行。

## 2. 机制假设与单一结构变量

T76 已在完全相同 LoRA 预算下证明，把适配从最后 4 层 rank8 扩展到最后 8 层 rank4，可使全局 raw、clean-core、trusted macro 与 proxy macro 同向提高约 0.20–0.26pp；但固定 M1+flip 的 raw 仅剩 +0.0097pp。局部-only 略有改善而融合不再获益，说明主要问题可能不是容量不足，而是 Q-LoRA 直接改写注意力权重后，与依赖注意力图选 crop 的 M1 发生接口失配。

T77 因而只改变 LoRA 注入位置：

| 结构 | blocks | targets | rank | alpha | LoRA 参数 | alpha/rank |
|---|---:|---|---:|---:|---:|---:|
| 原始 F1 | 4 | Q/V/out | 8 | 8 | 147,456 | 1.0 |
| T76 | 8 | Q/V/out | 4 | 4 | 147,456 | 1.0 |
| **T77** | **8** | **V/out（Q 冻结）** | **6** | **6** | **147,456** | **1.0** |

三者分类头均为 256,500 参数，可训练总参数均为 403,956。T77 保留跨层覆盖与完全相同预算，但不直接更新 Query 投影，目标是在保留全局适配收益的同时减少 attention localization 漂移。MTLoRA 对注意力子模块的功能拆分指出，Q/K/V 适配会改变 attention mechanism，而 output projection 负责把注意力结果映射到任务特征空间；1LoRA 则报告更低 rank 可把适配更均匀地扩展到多层。这些工作只构成结构动机，胜负仍由本项目预注册门禁决定：

- MTLoRA（CVPR 2024）：https://openaccess.thecvf.com/content/CVPR2024/papers/Agiza_MTLoRA_Low-Rank_Adaptation_Approach_for_Efficient_Multi-Task_Learning_CVPR_2024_paper.pdf
- 1LoRA（WACV 2026）：https://openaccess.thecvf.com/content/WACV2026/html/Quercia_1LoRA_Summation_Compression_for_Very_Low-Rank_Adaptation_WACV_2026_paper.html

除 targets/rank 的预算重分配外，父模型、split、trust、增强、损失、学习率、轮数、batch、selector 与推理协议全部锁定为原始 F1/T76 协议。

## 3. 最小实现与静态审计

为精确表达 V/out-only，模型代码增加独立 `lora_adapt_q` 与 `lora_adapt_v` 开关；未显式设置时仍继承原有 `lora_adapt_qv`，旧配置和旧 checkpoint 参数命名保持兼容。新实现仍在零初始化时与冻结 CLIP 精确恒等，并只注册被选择的低秩参数。

- 全套 `reproducibility/aegis_f1` 测试：229 passed。
- 真实 ViT-B/32 静态实例化：可训练视觉参数 147,456；可训练总参数 403,956。
- Q-LoRA trainable tensors：0；V-LoRA blocks：8；out-LoRA blocks：8。
- 配置 SHA-256：`993d56cc45777341b48db913ff0c569008134f8863d8b9a09626f12599d4f253`。
- train CSV SHA-256：`a726b8a3ca8bc5857136106aca80f01d557104d3661ef92ccedfb2c0ea087875`。
- validation CSV SHA-256：`54a790b35f836cfba4c19cbb5fe38c4b1b37aab62cc9d477f9285496b2d5568e`。
- trust bundle SHA-256：`52e59a991a5eb3c57abdfabee5647423726f51fbdd3da2ce377467664d173608`。

## 4. 固定训练协议

- 官方 train 103,218 张；固定 seed42 train 92,902 / validation 10,316；无外部数据。
- OpenAI CLIP ViT-B/32，单模型；只训练最后 8 层 V/out LoRA 与唯一线性分类头。
- rank6、alpha6；trust threshold 0.70；拒绝样本分类权重 0；GCE q=0.5；feature distillation weight=2.0。
- 224×224 weak RRC + horizontal flip；6 epochs；batch64；AMP；head lr 5e-5；visual lr 2e-5。
- 不启用 pseudo-label correction、prior、输出拉平、MixUp、模型融合或 test-time training。

唯一正式命令：

```bash
cd /home/x28639/projects/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning-t77-vo/reproducibility/aegis_f1
PYTHONPATH=$PWD /home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
  -m aegis_clip.cli.train \
  --config configs/t77_f1_depth8_rank6_value_out_archived_e2.yaml
```

只允许一次正式训练，不自动重试；硬超时 120 分钟。正式启动前必须再次 fetch `main`、查重、检查团队 GPU/进程、复核输入哈希，并确认专属输出目录不存在。

## 5. 双层晋级门

训练器内置 promotion 五项必须全部通过：clean-core 相对 epoch0 至少 +0.20pp、raw 不低于 epoch0 -0.10pp、mean feature drift <=1%、500 类覆盖、split/label lineage 完整一致。

相对原始 F1 epoch4 的结构门：

- raw micro 至少 `70.826619%`（原始 F1 +0.15pp）；
- clean-core micro 至少 `81.680559%`（原始 F1 +0.15pp）；
- trusted macro 不低于 `80.607790%`；
- proxy macro 不低于 `79.447788%`；
- drift <=1%，覆盖 500 类。

只有内置门和结构门同时通过，才允许执行一次固定 validation M1+flip：attention top5、crop160、local weight0.40、flip weight0.50、temperature1、batch128。复用已审计原始 F1 control（JSON SHA-256 `54ee60d8b9c9389b172a119e05701fa436cf6c5834eb2907a1295fe924b85f55`）作为严格对照。

M1+flip 最终门：

- raw micro 至少高于原始 F1 control +0.15pp；
- clean-core micro 至少高于原始 F1 control +0.15pp；
- trusted macro 与 proxy macro 均不得回退；
- checkpoint SHA 精确匹配并覆盖 500 类。

任一项失败即关闭 T77：不读 test、不生成提交包、不做 localization/rank/depth/target 扫描。若全部通过，才允许另行进行一次固定 test 推理与合规审计。任何 W050 复现或 flip0.60 重调都必须作为新的独立预注册实验，不能作为 T77 的事后补救。

## 6. 用户停止记录（2026-08-01）

正式命令按预注册只启动一次。用户在 epoch2 完整评估结束、epoch3 训练过程中要求停止，进程组已收到 TERM 并退出，GPU 已释放。该运行没有完成 6 epochs，也没有生成最终 promotion/manifest，因此所有 checkpoint 均标记为**中断产物，不具备晋级、test 或提交资格**；不自动续跑、不从中断点恢复。

截至停止时的完整可审计中间指标：

| 指标 | epoch0 | epoch1 | epoch2 | epoch2 vs epoch0 |
|---|---:|---:|---:|---:|
| raw micro | 70.230711% | 70.715392% | 70.851105% | +0.620394pp |
| clean-core micro | 80.759871% | 81.341267% | 81.584638% | +0.824767pp |
| trusted macro | 79.920471% | 80.427212% | 80.660397% | +0.739926pp |
| proxy macro | 78.824788% | 79.350168% | 79.462588% | +0.637800pp |
| flip agreement | 88.571149% | 88.968593% | 89.143080% | +0.571931pp |
| mean feature drift | 0.000083% | 0.297827% | 0.418127% | +0.418043pp |

epoch2 已超过预注册 raw 结构阈值 70.826619%，但 clean-core 81.584638% 仍低于 81.680559% 约 0.095921pp；训练曲线仍在上升，然而未完成协议，不能据此宣布 T77 通过或失败。该正向趋势只作为未来重新预注册时的研究依据。
