# REMATCH750 首轮执行（2026-09-21）

## 状态与边界

当前阶段 `repechage`，数据版本 `20260921`，seed 42。平台由用户上传并回传成绩；本地只生成、校验和登记提交包。RM-FULL 等 RM-FT 与 RM-LT 实测成绩齐全并确定胜出配方后执行，不因 pending 重复上传。

初赛权重与二进制缓存已删除。只迁移代码与超参数经验，模型从官方 OpenAI CLIP ViT-B/32 权重（本地文件指纹必须匹配官方 CLIP 包的模型 URL SHA-256）及全新有 bias 的 `Linear(512,750)` 开始。单模型、224px global center crop、无 TTA/prior、无 trust/伪标签/额外数据。

## 已测数据准备结果

全部 148,695 张训练图和 37,444 张测试图已通过 Pillow 严格像素解码，失败 0；不允许 truncated-image 宽容读取或占位图预测。部分图有 EXIF/透明度告警，但 RGB 解码成功。

训练侧按解码 RGB 像素及尺寸 SHA-256 分组，共 146,381 个独立内容组，其中 1,882 个跨标签冲突组。未改标签、未删除样本、未做测试内容分组。seed 42 的约 90/10 划分实际为 133,815 / 14,880 张，训练和验证内容组交集为空，验证覆盖 750 类。稀少组类别保留至少 min(4, 总组数) 个训练组。验证标签仍含噪，结果不等于干净测试准确率。

| train_dev 支持段 | 类别数 | 训练图片 | 验证图片 |
|---|---:|---:|---:|
| 少样本 `<20` | 2 | 16 | 2 |
| 中等 `20–99` | 12 | 675 | 74 |
| 多样本 `>=100` | 736 | 133,124 | 14,804 |

因此不能仅依据原始最大/最小计数比推断多数类别都处于尾部；本次两个少样本类的验证总共只有两张图片，波动必须谨慎解释。

资产目录：`artifacts/stages/repechage/20260921/`。包含 `dataset_manifest.json`、解码报告、两份映射、内容分组、full_train/train_dev/val_dev/test_manifest CSV、class_support 和 features。数据 manifest 记录清单哈希与逐图内容指纹；特征另绑定路径顺序、映射、官方权重、预处理与 fp32 数值精度。在线训练和推理逐图检查内容是否仍与审计一致。

## 固定配方

| 项目 | RM-LP | RM-FT | RM-LT |
|---|---|---|---|
| 配置 | `configs/rematch750_lp.yaml` | `configs/rematch750_ft.yaml` | `configs/rematch750_lt.yaml` |
| 初始化 | 官方权重＋新分类头 | 同一个 RM-LP best | 同一个 RM-LP best |
| 训练 | 缓存归一化特征、冻结视觉 | full_finetune，冻结 conv1/位置嵌入 | 与 FT 相同 |
| Epoch / cosine horizon | 20 / 20 | 8 / 8 | 8 / 8 |
| Batch | 256 | 32 | 32 |
| head / visual LR | 0.005 / 0 | 1e-4 / 3e-6 | 与 FT 相同 |
| 损失 | CE | 2 epoch CE，随后 GCE q=.5 | 与 FT 相同 |
| Anchor | 0 | 官方复赛参考特征，权重 2.0 | 与 FT 相同 |
| 采样 | shuffle | shuffle | sqrt_class_balanced |
| 增强 | 官方 center crop | weak_rrc_flip | 与 FT 相同 |

AdamW、weight decay 1e-4、AMP；沿用 Aegis cosine 的 1% LR floor，按实际每轮 batch 数计算更新总数。关闭 MixUp、SAM 和 loss reweighting。每两轮验证；只按 noisy raw macro、同值时 raw micro 选一个 checkpoint，drift penalty 为 0。记录每轮实际训练曝光；最终报告合并独立组数、支持样本、曝光次数、验证支持和召回。只保留 last/best 恢复权重，避免每轮复制全模型。

## 可重放命令（项目根目录）

```bash
# 仅首次生成；已有目录拒绝覆盖，不重复数据导入。
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.rematch prepare
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.rematch verify
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.rematch cache

# 依次执行，每个完成的实验段必须生成并验证提交包。
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.rematch train --config configs/rematch750_lp.yaml
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.rematch infer --config configs/rematch750_lp.yaml
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.rematch train --config configs/rematch750_ft.yaml
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.rematch infer --config configs/rematch750_ft.yaml
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.rematch train --config configs/rematch750_lt.yaml
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.rematch infer --config configs/rematch750_lt.yaml
```

配置要求 CUDA；沙箱中 GPU 不可用时需用获准的 GPU 执行环境，禁止静默回落 CPU。旧 v10 任务不自动启动。特征只编码本阶段训练图，验证侧仅作无标签编码，不参与噪声打分、原型或伪标签拟合。

输出为 `outputs/rematch750/<candidate>/seed42/`。提交必须覆盖全部 37,444 张测试图，每张一次，映射合法，ZIP 与 CSV 字节一致，任何解码或哈希失败均拒绝发布。不要求预测覆盖全部 750 个类别。

## 平台登记与全量训练

登记表 `results/rematch_submission_registry.csv` 中 `ready` 表示包已通过本地验证、尚未上传；`pending`/`valid`/`failed` 根据用户提供的真实提交回执更新。每日最多两次按用户提供的平台 reset-period 标识计数，不推测时区或重置时间。同一预测已提交则拒绝重复登记；最高分包与后续默认方案分开管理。

```bash
# 占位字段必须替换成真实平台回执；pending 无 --score。
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.rematch record \
  --candidate RM_FT --submission-id PLATFORM_ID \
  --submitted-at ISO_TIMESTAMP_WITH_TIMEZONE --platform-period PLATFORM_RESET_PERIOD \
  --status valid --score MEASURED_SCORE

# 双候选成绩齐全、选定 winner 后生成配置，拒绝提前执行。
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.rematch prepare-full --candidate RM_FT
```

生成的 `rematch750_rm_full_lp.yaml` 从官方权重重建全量分类头，`rematch750_rm_full.yaml` 从这份全量 LP 初始化视觉训练。使用开发阶段选定 epoch 数，保留原 LP 20 / FT或LT 8 的 cosine horizon，不将 horizon 缩到选中轮数。全量模式采用 last_epoch，原验证集仅称重叠诊断，不再用于挑选 checkpoint。按上述 train/infer 入口顺序执行两份生成配置，RM_FULL 的提交仍需独立平台测量。

首轮后如需噪声处理，另行建立仅训练侧的 CVT/OOF 连续降权实验；不自动启用旧 trust、伪标签、prior、SAM 或多尺度扫描。

## 工程验证与交付检查点

2026-09-21 完整 Aegis 测试初跑：618 passed、1 skipped、2 failed。两个失败均为历史 `test_scope_protocol.py` 直接依赖已按用户要求删除的初赛 checkpoint，不是复赛功能失败。新增平台登记、全量 horizon 和训练循环集成测试后，排除这两个历史资产集成测试的结果为 **622 passed、1 skipped、2 deselected**；skip 为沙箱无 CUDA 的采样器测试。复赛专项包含 13 项测试。

真实数据的解码报告、数据 manifest 和逐类支持量已同步到 `results/rematch750_20260921_*`。首段交付边界为数据准备＋RM-LP，已完成训练与提交校验后提交推送；后续执行结果见文末各段记录。

## RM-LP 实测交付

固定 20 轮训练完成，选中 epoch 20（原 20 轮 cosine horizon 不变）。独立验证 **macro 62.636375%、micro 63.635755%（9,469 / 14,880）**，750 类全覆盖。少样本段 macro 0%（2 类 / 2 张）、中等段 macro 10.115740%（12 类 / 74 张）、多样本段 macro 63.662893%（736 类 / 14,804 张）。不将本地 noisy validation 当作平台成绩。

checkpoint 的 152 个视觉张量逐项与官方 OpenAI 权重相同；只有 384,750 个分类头参数被训练。全部模型张量有限。特征缓存为 `[148695,512]` fp32，归一化最大误差 `2.980232238769531e-07`；编码后再次核对全部训练图片的文件 SHA-256，与审计清单一致。

提交文件：

- `outputs/rematch750/RM_LP/seed42/submission/pred_results.csv`
- `outputs/rematch750/RM_LP/seed42/submission/submission.zip`
- 共享分类头初始化 checkpoint：`outputs/rematch750/RM_LP/seed42/checkpoints/best.pt`

37,444 条预测全部通过文件名、数量、唯一覆盖、映射、四位编号及 ZIP 内容检查；使用逗号＋空格格式，ZIP 内外 CSV 字节一致，坏图及占位图预测为 0。提交登记状态 `ready`，尚未上传，平台分数未知。

模型、配置、映射、代码、CSV/ZIP 哈希绑定及精确命令见 [RM-LP 最终记录](../results/rematch750_20260921_rm_lp_final.json)、[提交登记表](../results/rematch_submission_registry.csv)。逐类独立组数、样本支持、实际曝光和召回见 [逐类报告](../results/rematch750_20260921_rm_lp_per_class.csv)。训练/推理代码快照逐文件一致。

GPU 补测原先跳过的数值测试：1 passed；最后复赛与提交专项复核：17 passed。历史日志中的 `head/medium/tail` 等分三段属于兼容诊断，本轮正式报告采用 `support_head/support_middle/support_tail` 的固定计数边界。

RM-LP 交付检查点已完成并提交推送；该检查点当时 FT/LT 尚未启动，后续进展见下文。

## RM-FT 执行数值修正（2026-09-21）

首次 RM-FT 在首 batch 检出 FP16 梯度溢出：默认 GradScaler 初始值 65,536，视觉梯度为 inf，优化器跳过更新，首步审计随即停止；未生成训练 checkpoint。现场保留于 `outputs/rematch750/RM_FT/seed42_failed_initial_amp_65536/`。

FT/LT 同步设置 `amp_initial_scale=128`、`amp_growth_interval=1000000000`（本轮内不自动增大），新增 `require_finite_gradients=true`：在优化器与调度器更新之前检查有限损失/梯度，失败立即停止，不静默漏掉更新。每 200 步记录实际进度和 AMP scale。学习率、数据、损失、增强、8 轮 cosine horizon 与 FT/LT 单变量采样对照不变。数值修正及训练/恢复定向测试 25 项通过，包含非有限梯度不得调用 optimizer 的集成检查。

## RM-FT 实测交付（当前检查点）

从同一复赛 RM-LP epoch 20 独立初始化，完成普通 shuffle 的 8 轮视觉微调；第 1–2 轮 CE、第 3–8 轮 GCE q=.5，feature anchor 2.0。选择 epoch 8，原 8 轮 cosine horizon 不变。

| 验证轮次 | 750 类 macro | micro |
|---|---:|---:|
| 2 | 68.255639% | 69.348121% |
| 4 | 70.634145% | 71.653223% |
| 6 | 71.662009% | 72.728497% |
| 8（选中） | **71.836197%** | **72.903228%** |

最终独立验证正确 10,848 / 14,880；相对 RM-LP，macro +9.199822pp、micro +9.267473pp。固定支持段 macro：少样本 0%（2 类 / 2 张验证图）、中等 15.324074%（12 类 / 74 张）、多样本 72.952801%（736 类 / 14,804 张）。中少样本仍明显落后，不能只凭整体分数认定长尾问题已解决；尾段仅两张验证图，不作稳定统计结论。

最终审计通过：全部优化器参数的 step、global_step、scheduler step 均为 33,456（4,182 × 8），无 AMP 跳步；scale 始终 128。conv1 和位置嵌入与父模型逐项相同，150 个视觉张量已更新，所有模型张量有限。FT/LT 除实验 ID 和采样方式外的配置逐项相同，两者绑定同一 RM-LP checkpoint。

交付文件：

- `outputs/rematch750/RM_FT/seed42/checkpoints/best.pt`
- `outputs/rematch750/RM_FT/seed42/submission/pred_results.csv`
- `outputs/rematch750/RM_FT/seed42/submission/submission.zip`

37,444 条测试预测全部通过唯一覆盖、四位标签映射、ZIP 内容及内外 CSV 字节一致性检查；解码失败与占位图预测均为 0。单 checkpoint、固定 global、无 prior/TTA。登记为 `ready`，用户尚未回传平台成绩。

精确命令、配置、代码版本、指标和哈希见 [RM-FT 最终记录](../results/rematch750_20260921_rm_ft_final.json)；逐类支持、独立组数、训练曝光与召回见 [逐类报告](../results/rematch750_20260921_rm_ft_selected_per_class.csv)。训练/推理源码快照逐文件一致，25 项数值修正定向测试通过。

本段完成 RM-FT 的训练和提交就绪闭环，按仓库协作约定提交推送后暂停。**RM-LT 尚未训练，RM-FULL 尚未定版**；下一段从原 RM-LP（不是 RM-FT）初始化 RM-LT。


## RM-LT 实测交付与桌面副本（2026-09-21）

使用 `configs/rematch750_lt.yaml`，按上文 train / infer 命令完成 8 轮，选中 epoch 8，保留 8 轮 cosine horizon。从与 FT 相同的 RM-LP checkpoint 独立初始化；配置比对除实验名称与平方根采样外完全一致。训练代码 commit 为 `a63f7ad6d61c52400da963a819bfde564fb54dc8`，训练/推理代码逐文件哈希一致。每十分钟监控，无非有限梯度中断；33,456 次优化器更新与调度器步数一致，AMP scale 128，conv1/位置嵌入保持冻结，150 个视觉张量发生更新且全部模型张量有限。

| 候选 | 选中轮次 | 750 类 macro | micro | 正确 / 验证图片 |
|---|---:|---:|---:|---:|
| RM-FT | 8 | 71.836197% | 72.903228% | 10,848 / 14,880 |
| RM-LT | 8 | 71.865541% | 72.694892% | 10,817 / 14,880 |

LT macro 比 FT 高 0.029343 个百分点，micro 低 0.208336 个百分点；本地 noisy validation 差距很小，不等同于平台胜负。LT 少样本段 macro 0%（2 类 / 2 张），中等段 23.101853%（12 类 / 74 张），多样本段 72.855908%（736 类 / 14,804 张）。FT 对应中等段 15.324074%，多样本段 72.952801%；尾类样本过少，不能据此宣称已经解决长尾噪声问题。逐类报告记录了 1,070,520 次实际训练曝光。

LT 提交为 `outputs/rematch750/RM_LT/seed42/submission/submission.zip`，含恰好 37,444 条预测，覆盖与标签映射合法、四位编号和逗号空格格式正确，ZIP 内外 CSV 字节一致，坏图或占位图预测为 0。模型、配置、映射及 CSV/ZIP 哈希绑定见 [LT 最终记录](../results/rematch750_20260921_rm_lt_final.json)；逐类结果见 [LT 逐类报告](../results/rematch750_20260921_rm_lt_selected_per_class.csv)。本段没有修改训练代码，因此不重复既有工程测试，以实际完整训练、参数审计和全量提交校验作为验证证据。

应用户要求，以下文件已复制到 `C:\Users\lqh22\Desktop\submission`，源与目标 SHA-256 均一致，目录原有文件保留：

- `RM_LP_20260921.zip`：回退候选，可跳过平台提交。
- `RM_FT_20260921.zip`：普通采样 baseline。
- `RM_LT_20260921.zip`：平方根采样对照。

[桌面交付记录](../results/rematch750_20260921_desktop_delivery.json)保存源、目标与哈希。三个候选登记仍为 `ready`，尚无平台回执。由用户上传并回传成绩；本段交付后等待 FT/LT 平台比较，RM-FULL 尚未启动。
