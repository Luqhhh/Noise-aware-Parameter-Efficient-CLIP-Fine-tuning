# 最新发现（2026-07-21）

## A2_AEGIS_PARENT_SWAP：Split-lineage 协议修复与最终结论

- **原始 A2 parent swap 本地 79.22% raw_micro 是假信号**：A2 parent 使用 d3_strict split (91,195/10,322)，AEGIS child 使用 prepare split (92,902/10,316)。val 样本泄漏到 parent 训练集，导致本地准确率假胀 8.5pp。
- **协议修复**：新增 `canonical_sample_path`（统一 train/train_dedup）、fail-closed lineage audit、epoch-0 baseline evaluation、promotion gate
- **严格复跑 epoch-0 = 69.43%**：精确匹配 A2 本地准确率，确认 lineage 修复正确
- **LoRA 真实增益 +0.19~0.39pp**（vs 假 +8.5pp），双 seed promotion 通过
- **平台 Bare = 60.65%（+0.14pp vs F1 E2），TTA = 61.15%（+0.05pp vs F1 E2）**
- **结论：A2 parent swap 确认成立但收益边际**，不进入 P3/P4 参数搜索
- **教训**：parent-child split 必须完全相同（SHA-256 级别验证），epoch-0 evaluation 是必不或缺的 parent swap gate

# 发现（2026-07-20）

## A2 多 seed 稳定性

- A2 seed=42 vs seed=3407 本地 paired delta = −0.07pp（−7 张图），McNemar p=0.457，完全不显著
- 但平台 TTA 差 0.90pp（61.21% vs 60.31%），本地完全不可见
- 结论：**单 seed 平台结果不可靠**，所有候选必须双 seed 验证

## A1/A3 均有害

- A3（5-signal consensus relabel 100，0.1%）：TTA 59.89%（−0.42pp vs A0）
- A1（CL classwise drop 8680，9.5%）：TTA 59.55%（−0.76pp vs A0）
- A2（三方共识 delete 991，1.1%）：TTA 61.21%（+0.90pp vs A0）
- 结论：**精度 > 覆盖面**，删除 > 重标

## Purification 天花板已触达

- A0→A2 本地 paired delta 仅 +17 张图（0.165pp, p=0.196）
- 冻结 CLIP + GCE + MixUp 框架下，数据筛选层的边际增益已饱和
- 唯一的出路是 visual LoRA PEFT

## 已关闭方向

- OOF relabel / pseudo-label（A3 5-signal 共识仍有害）
- Classwise CL-only drop（A1 −0.76pp）
- NR_COMBINED_CLEAN_CORE（Layer 2/3 负信号）
- Rejected 半监督回收（OOF 准确率 ~69% 不足以支撑可靠回收）
- ELR、PEFT LN-tune、EMA loss、prototype weighting（旧证据充分）

---

# OOF 执行发现

- 2026-07-15：用户决定继续使用 Windows Codex 环境，不切换到 Linux agent runtime。
- 当前没有 `cache_features.py`、`analysis.oof.run_oof` 或 pipeline 进程。
- `cache/preliminary/clip_vit_b32_openai/` 只有 `class_to_idx.json`、`idx_to_class.json` 和 `fingerprints.json`，尚无 `features.pt`。
- 已生成的 fold 基础设施与 `outputs/phase3/` 仍在；需要检查完整性后续跑，无需重新设计折分。
- 工作区包含用户原有修改和大量未跟踪实验产物，必须保持隔离，不能做清理或覆盖。
- 从 Windows 端直接运行单条 `wsl.exe -d Ubuntu --cd ... <command>` 可用；嵌套长 shell 字符串存在引号风险。
- 两个启动脚本语法检查通过；pipeline 会先建特征缓存，再跑固定 50 epoch、q=0.5 的 3-fold OOF。
- PyTorch CUDA 正常识别 RTX 4060 Laptop GPU（8 GB）。
- 8 个 OOF 单元测试全部通过。
- 缓存数据集遇到坏图会回退为零张量；翻转一致性数据集目前直接 `Image.open`，严格清洗集预计可用，但这是后续阶段的潜在异常点。
- OOF 已完整结束：91,195 个样本全部得到唯一且有限的 OOF logits，整体准确率 69.4479%。
- 三折 holdout accuracy 分别为 69.3246%、69.7085%、69.2907%；均固定训练 50 epoch，未用 holdout 选 epoch。
- soft weight 范围为 0.302199–1.0；类别 65、338、407 中低于 0.5 权重的样本比例超过 30%。
- Wave 1 配置文件已存在：`w1_gce09.yaml`、`w1_ce5_gce07.yaml`、`w1_gce07_mixup.yaml`，需要先审计内容和输出状态，避免重复训练。
- 结果登记文件已存在：`results/phase3_experiments.csv` 与 `results/submission_registry.csv`。
- 工作区仍包含用户原有修改和未跟踪实验产物；所有新增工作必须保持隔离，不能清理或覆盖。
- 第一次训练接口 `rg` 检索因正则竖线被 Windows/WSL 参数层拆分而失败；后续改用多个 `-e` 参数。
- 公共代码已经支持 ScheduledLoss、MixUp、ManifestLoader 和多类 SampleWeightProvider，剩余实验大概率只需配置、审计与少量分析脚本。
- 现有 `w1_gce09.yaml`、`w1_ce5_gce07.yaml`、`w1_gce07_mixup.yaml` 均指向 `outputs/ref/seed42`，与 B 计划要求的 strict split 存在冲突。
- Wave 1 配置当前 `use_cached_features: false`；必须先与实际 `b2_gce05` 基线对齐，不能自行改变训练路径。
- `b2_gce05.yaml` 确认使用 `outputs/d3_strict/seed42`；旧 `gce_q07.yaml` 与现有 Wave 1 配置仍使用 `outputs/ref/seed42`。
- 旧 `w1_ce5_gce07/seed42` 和 `w1_gce07_mixup/seed42` 已有完整评估/推理产物，但不满足当前 strict split 契约，不能直接作为正式 B 结果。
- `w1_gce09/seed42` 目前只有日志和 resolved config，没有完整 checkpoint/评估产物，需要检查是否中断。
- `b2_gce07` 已有 seed42、3407、2026；`b2_gce05` 目前只有 seed42。
- 本地 q=0.9 日志停在 epoch 40/50，最后记录 val acc 58.91%，没有完整 checkpoint/评估产物，不能作为正式完成结果。
- 用户提示 q=0.9 可能由队友完成但本地 main 未更新；必须先 fetch 远端核实，禁止直接重复训练。
- 旧 ref-split CE5→GCE07 的 micro/macro 为 69.7751%/69.7199%，MixUp 为 69.6103%/69.5609%；因 split 不同只能作线索，不能与 strict B2 正式比较。
- q=0.7 的 seed3407/2026 strict 结果约为 69.56%/69.54%，多 seed 产物已存在。
- 远端 main 已从本地 `719ee18` 前进到 `cb786a8`。
- 远端没有新增 q=0.9 完整产物；本地 q=0.9 仍需后续补完或重跑。
- 远端已完成 strict 的 `W1_CE5_GCE05` 与 `W1_GCE05_MIXUP`，并更新 Phase 3 基线和训练代码。
- 远端新增路径与本地 OOF 主要位于不同子目录；可用 fast-forward 同步，保留本地 dirty worktree。
- 更正：远端 `W1_CE5_GCE05` 和 `W1_GCE05_MIXUP` 的 resolved config 都使用 `outputs/ref/seed42`，不是 strict split。
- 更新后的团队计划仍因平台结果将 CE5 q=0.5 设为训练基线（本地 73.14%），将 q=0.5 MixUp + TTA 设为提交基线（平台 60.3637%）。
- CE5 q=0.5 的 micro/macro/bottom10 为 73.1388%/73.0940%/33.6710%；MixUp q=0.5 为 71.1613%/71.1153%/32.0274%。
- OOF weight 实验必须使用 `d3_strict` 并与同源 `B2_GCE05` strict 基线比较；跨 split 平台基线只作提交参考，Protocol Audit 要显式标注。
- 最新训练侧已有 `oof_manifest` 与 `relabel_manifest` provider，不需新增训练循环。
- 当前 OOF 权重 CSV 列为 `sample_id,image_path,original_label,quality,weight`，不满足 ManifestLoader 要求的 `training_label,sample_weight,quality_score`，直接训练会 fail-closed。
- 必须先规范化 manifest schema、做 100% 覆盖/标签/范围审计并生成 weight audit；类别 65/338/407 的低权重告警可能按计划关闭训练分支。
- OOF/Relabel provider 以 `image_path` 精确匹配，strict CSV 与 OOF 都使用 `train_dedup/...`，路径规范一致。
- `quality.py` 已有完整质量与权重计算，但没有生成训练侧 canonical schema 的函数。
- 现有测试只验证概率/权重范围，没有覆盖 ManifestLoader schema；应先添加失败测试，再实现规范化与审计。
- `run_oof.py` 当前显式写出旧版 `quality/weight` schema；现有 `sample_quality.csv` 可作为无损重建 canonical manifest 的来源，无需重跑 OOF。
- 离散权重按类别三等分，天然约 33% 权重低于 0.5；30% 停止条件应作为 soft 主 gate，离散对照仅在 soft gate 通过后开放，否则协议自相矛盾。
- canonical manifest、低权重 gate 与 fail-closed 测试 3/3 通过；原 OOF 回归测试 8/8 通过。
- 全量 canonical soft/discrete manifest 均覆盖 91,195/91,195，标签一致、无 relabel、权重和质量分数全部有限。
- soft 权重范围 0.302199–1.0、均值 0.677837；类别 65、338、407 的低于 0.5 权重比例超过 30%。
- `weight_audit.json` 决策为 `stop_before_weight_training`；OOF soft/discrete 训练按协议关闭。
- relabel 的前置条件“weighting 有明确正收益”无法满足，因此 relabel 分支按 gate 跳过并关闭。
- `results/phase3_experiments.csv` 已登记 CE5 q=0.5 与 MixUp q=0.5 为 platform_best，但 trusted 指标、split/checkpoint hash 和多 seed 行仍为空。
- q=0.9 未在结果表登记，远端也无完整产物；若要求本地可复现交付，仍需补跑或取得队友 artifact。
- 旧 CE5→GCE07 与 q=0.7 MixUp 有完整 ref-split 本地产物，但尚未登记到 Phase 3 结果表。
- 更新后的 Phase 3 文档仍保留 q=0.9、CE5→GCE07、q=0.7 MixUp 为正式 Wave 1 项；当前最多保留的两个候选实际是 CE5 q=0.5 与 MixUp q=0.5。
- `common/trusted_subset.py` 已实现 V1 trusted subset、连续 trust weight 与 class-balanced trusted accuracy。
- 仓库未发现现成的 B 候选 trusted validation 整合脚本；需要先读取函数输入契约，再补统一报告层。
- 固定验证集的模型无关信号已存在于 `outputs/analysis/d3_vs_b2_seed42/sample_metrics.csv`，并已有 trusted manifests 与 protocol audits。
- trusted 报告无需重算特征；可将候选 `prediction_records.csv` 与固定 sample metrics 按样本键合并，再调用现有 V1/V2 函数。
- 固定 sample metrics 含 10,316 个验证样本、500 类，并有 kNN/prototype/flip/duplicate conflict 全部模型无关信号。
- V1 trusted 仅 2,073 样本（20.095%）且覆盖 336/500 类，不能单独作为 gate；必须同时报告 V2 trust-weighted 与 class-balanced top-K。
- sample metrics 的绝对路径来自队友机器 `/home/lux1/...`，候选预测路径必须规范化为稳定的 `class_name/file_name` 键后合并。
- 候选 prediction records 统一为 `image_path,true_label,pred_label,pred_conf`；CE5/MixUp 与 sample metrics 同源，B2 仅绝对路径前缀不同。
- 以路径最后两段 `class_name/file_name` 作为 join key 可跨机器稳定匹配，同时必须审计唯一性、100% 覆盖和标签一致性。
- 更正上一条关于 B2 的初步判断：全量覆盖审计证明 B2 与固定 sample metrics 只重合约 1,000 条，缺 9,309、额外 9,315，并非只有路径前缀不同。
- CE5 与 MixUp 均完整覆盖固定 10,316 样本；因此二者可在同源验证集上比较，但不能把不同 split 的 B2 当作逐样本父基线。
- CE5 seed42：raw micro/macro/bottom10=73.1679%/73.1221%/33.6710%，V1 trusted micro=99.9518%，V2 weighted=97.5528%，class-balanced top5=93.12%。
- MixUp seed42：raw micro/macro/bottom10=71.1516%/71.1058%/32.0274%，V1 trusted micro=99.9035%，V2 weighted=97.2217%，class-balanced top5=92.44%。
- CE5 在本地 raw 与全部 trusted 指标领先；MixUp 的优势仅体现在平台分数（60.3637% 对 CE5+TTA 60.25%），两者都保留进入多 seed。
- trusted V1 仍只有 20.095%/336 类，最终判断必须继续同时报告 V2 weighted 与 500 类 class-balanced 指标。
- 多 seed 固定 `split_dir=outputs/ref/seed42`、`split_seed=42`，仅改变 `train_seed`；不能使用 CLI 的 `--seed-override`，因为它会同时更换 split。
- CE5 与 MixUp 都补 seed3407；平台主候选 MixUp 另补 seed2026，满足 Phase 3 双 seed/三 seed策略。
- 当前正式候选使用在线 CLIP 编码；尽管 A0+冻结 backbone 可使用缓存，配对确认不更换数据路径，避免引入额外数值变量。
- CE5 seed3407 第 1/2 epoch 验证准确率为 36.40%/53.87%，耗时 5m50s/6m58s，训练正常。
- 本地接力脚本会在 CE5 完成后串行运行 MixUp seed3407、MixUp seed2026，并为三者生成统一 trusted report。
- 提交前确认本地 main 与 origin/main 同为 `cb786a8`，没有远端领先提交。
- 仓库不存在 `.gitattributes`，尚未配置 Git LFS。
- `outputs/phase3/oof` 约 482MB，`outputs/w1_ce5_gce05/seed3407` 约 1.0GB。
- 需要按历史提交惯例筛选可推送的配置、代码、指标、审计与轻量结果；普通 Git 不能直接推送 339MB checkpoint。
- `.gitignore` 已忽略 `outputs/**/*.pt`，所以 CE5 三个约 339MB checkpoint 与 OOF logits/feature tensors 不会被加入 Git。
- 历史 CE5 seed42 提交只包含 config snapshot、metrics、prediction records、logs 与 submission，不含 checkpoint。
- OOF 的 `sample_quality.csv`、soft/discrete manifest 分别约 33MB/22MB/21MB，单文件均低于 GitHub 100MB 限制。
- 本次沿用历史策略：提交轻量可审计产物，权重继续留在本地 ignored 路径。
- CE5 seed3407 完整 trusted report：raw micro/macro/bottom10=70.2404%/70.1875%/30.7608%，V2 trust-weighted=96.9305%。
- OOF、manifest finalize 与 trusted report 共 14 个提交前测试全部通过。
- CE5 artifact manifest 登记 checkpoint/train/val SHA-256，最佳 epoch=38；训练代码基线 commit=`cb786a8`。
- 结果表新增 OOF gate-closed 行和 CE5 seed3407 local-confirmed 行，均为 33 列 schema。

## 2026-07-17 S_OOF_DISCRETE runtime finding

- Root cause of the epoch-1 fail-fast: OOFManifestProvider resolved manifest symlinks, but TrainImageDataset only made CSV paths absolute and preserved the train_dedup alias.
- Normalizing incoming provider lookup paths fixes the string-key mismatch without changing any weight value or experiment protocol.

## 2026-07-18 S_ELR_BASE runtime plan

- `configs/s_elr_base.yaml` uses strict `d3_strict/seed42`, CUDA, 50 epochs, batch size 128, online ViT-B/32 with frozen backbone, GCE q=0.5, MixUp, and ELR.
- ELR state is checkpoint-serializable in `common/elr.py`; no extra per-epoch external query is needed.
- Long-run monitoring will be hourly via a persistent heartbeat; user-facing updates should report only the newest completed epoch and best validation metric.
- Preserve existing dirty/untracked files and avoid staging or pushing this run unless the user later asks.

## 2026-07-19: S_OOF_ZERO_0001_FF Final Fit 验证

S_OOF_ZERO_0001_FF 是 S_OOF_ZERO_0001 的 final_fit 变体：
- 相同 OOF manifest（p<0.001, 7% 排除）
- 训练模式从 dev（train+val split）切换为 final_fit（全量训练集）
- 全量样本数更多（train+val），无验证集

平台结果：
- Bare: 60.29%（+0.33pp vs dev mode 59.96%）— 首个突破 60% bare 的方法
- TTA: 60.51%（+0.23pp vs dev mode 60.28%）— 首个突破 60.5% TTA 的方法

结论：
- final_fit 带来的全量训练升幅确认有效
- OOF binary zero p<0.001 是当前最优噪声处理策略
- 后续实验应优先考虑 final_fit 模式以获取完整数据利用
