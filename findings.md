# 最新发现（2026-08-03）

## A12_CORR（A12 + 伪标签修正）平台 67.6853%，新平台最佳，距 70 分 2.31pp

- **A12 + 伪标签软修正平台实测 67.6853%**，新的审计完整平台最佳，相对 A12
  （67.6173%）`+0.0681pp`。全局 raw 0.7121 / clean-core 0.8183，全面超过 A12。
- **关键机制发现：伪标签修正与广度 LoRA 兼容，但与深度 LoRA 不兼容**。W060
  （最后 4 block LoRA）上 CORR 曾因 M1+flip 下降而不叠加（组合包 67.1807%
  回归）；A12（全 12 block）上 CORR 从 epoch 4 起持续正向（epoch2 略落后 →
  epoch3 追平 → epoch4 反超）。广度 LoRA 为伪标签修正提供了更稳定的表征。
- 连续第二个"全局更强、本地 M1+flip 略降、平台转正"的 checkpoint：A12（M1+flip
  −0.19pp vs W060，平台 +0.04pp）→ A12_CORR（M1+flip 0.7233 介于，平台
  +0.07pp vs A12）。本地 M1+flip ±0.2pp 以内不能可靠预测平台方向。
- 更长训练方向关闭：A12_E10（epoch 6→10）best=epoch9 仅比 A12 原版 +0.01pp
  clean-core，低于可靠阈值，不生成包。A12 最优 epoch 已在其原始 6-epoch 运行内。
- promotion PASS（selector +0.0095、raw +0.0079）；best=epoch 8。
- ZIP SHA-256：`b2b1924a98ad9179c17d2421c3584a910070e0c611713707079c5ac58f3e0585`。

# 历史发现（2026-08-02 之前）

## A12（LoRA 全 12 block）平台 67.6173%，新平台最佳，距 70 分 2.38pp

- **LoRA 广度扩展（4→12 block）平台实测 67.6173%**，新的审计完整平台最佳，
  相对 W060 + temp1.5 + prior0.85（67.5812%）`+0.0361pp`。
- **关键现象：本地 M1+flip 融合指标与平台方向相反**。A12 全局更强（raw
  70.58→71.07，+0.49pp；clean-core 81.33→81.69，+0.37pp），但本地
  **M1+flip 融合略低于 W060**（raw 0.7222 vs 0.7241、clean-core 0.8251 vs
  0.8270，任意融合点全负）；平台却更高。第三次"容量/覆盖面增加"实测
  （R16、高分辨率、A12），唯独 A12 在平台转正。
- 与 R16（rank 8→16）、高分辨率（256/288px）同一失败模式的不同结局：三者
  全局提升、本地 M1+flip 下降；A12 下降幅度最小（−0.19pp vs R16 −0.38pp、
  288px −0.54pp），且平台最高。
- **结论**：本地 M1+flip ±0.2pp 以内不能可靠预测平台方向；A12 提供了一条
  可行的训练侧改进路径（广度 LoRA），值得继续沿此方向扩展（如更大 rank 的
  广度、或 A12 + 更多 block + 重调 M1）。
- 训练 6 epoch，promotion gate PASS（selector +0.0086、raw +0.0078）；
  best=epoch 6。推理用与最佳包相同协议（M1+flip temp1.5 prior0.85）。
- ZIP SHA-256：`a4e14f56ce2990220804cf7d78a1d124b39272b14e0ab533449c2432d07854d5`。

# 历史发现（2026-08-01 之前）

## Balanced-prior 校准：strength 0.8 平台 67.3329%，距 70 分仅 2.67pp

- **强度曲线实测 0.6/0.75/0.8/1.0，strength 0.8 当前最优**：W060+0.8 =
  **67.3329%** > W050+0.75 = 67.2848% > W060+1.0 = 67.2007% > W060+0.6 =
  66.9404%。曲线 0.8 仍上升、1.0 回落——**峰值在 0.8~1.0 之间**，0.85/0.9
  待测。
- 关键发现：**1.0 过矫正**（测试集非完美均衡 467×50 + 33×49），峰值在中间
  强度；用户坚持测中间值被平台反复证实正确。
- 累计校准收益：local→platform gap 由 8.35pp 压缩到 ~4.9pp，距离 70 分
  `2.67pp`。
- 模型测试预测严重不均衡（校准前最差类 2 个、最好类 ~180 个预测），平台测试
  类别均衡 —— 两个假设都被平台分数验证。
- 校准只调整推理 logits，可叠加任何新 checkpoint。
- 完整报告：`results/prior_alignment_20260731.md` 与
  `results/70p_campaign_20260731.md`。

## M1 + Flip 四视图融合：平台 63.7802%，新的审计完整平台最佳

- 对 F1 REBUILD R1 的原图/翻转图分别提取 global 与 attention-local 概率，在单
  checkpoint 内进行确定性四视图融合。
- 有界 8 点扫描的最佳 raw 点为 `local_weight=0.40, flip_weight=0.50`；相对
  62.9791 包对应的 M1 weight 0.35，raw `+0.3296pp`、trusted macro
  `+0.1520pp`、proxy macro `+0.1340pp`、clean-core `+0.2728pp`、
  clean-core macro `+0.0931pp`，类别覆盖维持 500。
- 新候选改变测试集 1,936 个预测（7.7542%），ZIP SHA-256 为
  `67f4eda57291e34096edcb0545b142fd0a3114fb1c76eb1e17996afe87d692e0`，
  审计通过。
- 平台实测 **63.7802%**：相对 F1 REBUILD R1 M1 weight 0.35（62.9791%）
  `+0.8011pp`；相对已报告原 F1 + M1（63.3276%）`+0.4526pp`。这是新的审计
  完整平台最佳，验证了离线四视图融合增益可转移到平台。
- 距离 70 分 `6.2198pp`；本地 raw micro `72.1307%` 到平台的 gap
  `8.3505pp`。
- 完整报告：`results/m1_flip_optimization_20260730.md`。

## E2→F1 严格重建成功，M1 平台 62.9791%

- 原 F1 checkpoint 虽缺失，但同 split 重建的 E2 父模型 raw `70.3470%`，
  超 archived 父模型 `+0.1163pp`；lineage 审计无泄漏。
- F1 LoRA 相对 epoch 0 raw `+0.2230pp`、clean-core `+0.4092pp`，漂移仅
  `0.3982%`，promotion 通过。
- M1 weight 0.35 进一步带来 raw `+1.2602pp`、clean-core `+0.9548pp`，
  trusted/proxy 同向；这是比局部 Adapter 更大且更一致的本地机制增益。
- 重建 F1 absolute clean-core 仍比 archived F1 低 `0.2456pp`，所以不能把原
  F1+M1 平台 63.3276% 直接赋给新包；必须真实平台回填。
- 新包 ZIP SHA-256：
  `02d37906accdf6b49e40733b4e675220f0177b5d71c5662984de68df5e781bb6`。
- 平台最终为 **62.9791%**：比 A2 STRICT + M1 高 `0.2921pp`，说明更强
  checkpoint 表征部分转移到平台；但比已报告原 F1 + M1 低 `0.3485pp`，
  状态 `platform_valid_not_promoted`。

## A2 STRICT 局部特征 Adapter：有效但未晋级

- attention-local 数值路径修复后，局部专用 34,336 参数残差 Adapter 可稳定改善
  M1；最佳点 raw `+0.0872pp`、trusted macro `+0.1219pp`、clean-core
  `+0.1821pp`。
- 预注册 clean-core 门槛为 `+0.20pp`，最佳点仍差约 2 个 clean-core 样本；
  不降低门槛、不生成平台包。
- 更低 anchor、更高 trust、64 维瓶颈以及 local loss 0.75/1.00 均无更好结果，
  说明继续扩大强度/容量不是下一步。
- 原 F1 checkpoint 缺失，但 E2 配方、同源 split 和官方数据仍在；下一步严格重建
  E2→F1，不用 split 不同的主工程 CE5 checkpoint 进行有泄漏风险的选模。

## A2 STRICT + M1 weight 0.35：平台 62.6870%，机制有效但未晋级

- 最新更正后的有效平台分数为 **62.6870%**。
- 相对同 checkpoint Bare 提升 +2.0349pp，attention-local 的因果方向继续成立。
- 与已报告 A2 + M1 62.6747% 几乎持平；本地双 seed 显示的 weight 0.35 优势不能可靠预测平台增益。
- 比 F1 + M1 63.3276% 低 0.6406pp，说明当前主要瓶颈仍是 checkpoint 表征，不是 M1 融合权重。
- 决策：关闭 A2 STRICT 上的 M1 权重/裁剪小搜索，下一轮回到更强 F1 表征或真正的新机制。

## M1 attention-local：固定定位成立，较低局部权重双 seed 更稳

- 已在本仓库复现最后 block / mean-12-head / top-5 / crop160 的 M1 推理；A2 STRICT seed42 的固定 1:1 融合相对 global raw +0.8525pp，与外部平台 M1 正增益方向一致。
- `crop ∈ {144,160,176}`、`top-k ∈ {3,5,9}` 网格没有找到比原 `crop160/top5` 更稳的定位参数。
- 将局部分支权重从 0.50 降至 0.35 后，seed42/3407 raw 分别 +1.0657/+1.0076pp，clean-core +0.8123/+0.7843pp；trusted/proxy 指标也全部同向，500 类覆盖不变。
- 144/160/176 三尺度局部概率平均的 M2 只提高 seed42 raw，proxy 与 clean-core 均轻微回退，未过全指标同向门槛；不补第二 seed。
- A2 STRICT seed42 + M1 weight 0.35 已生成审计通过的 24,967 条预测平台包；平台最终 62.6870%，状态为 `platform_valid_not_promoted`。
- 完整报告：`results/m1_localization_optimization_20260730.md`。

# 发现（2026-07-22，2026-07-30 文档核对）

## Phase 4 P0–P4：机制实验全部关闭

- **P0 结构化 Head**：多原型 proxy macro 虽提升约 0.26pp，但 raw 净破坏 44 个预测；LDA 同样以 raw 回退换 proxy 增益；Ridge 最优仍是原 head。
- **P1 Checkpoint Averaging**：SWA-1/2/3 的 clean-core 均低于 epoch 6；greedy soup 只保留 epoch 6，未形成有效平均。
- **P2 Clean Routing**：Hard gate 与现有样本选择阈值重合；Soft gate clean-core 仅 +0.042pp，同时 raw −0.039pp、proxy macro −0.083pp，未达到 +0.20pp gate。
- **P3 Prototype-Contrastive**：clean-core −0.084pp，预测类别数 499，promotion 失败。
- **P4 Dynamic Trust**：epoch 2 刷新后 clean-core 全部下降；最佳点仍在刷新前。
- **总判断**：Phase 4 没有平台候选，普通 LoRA/routing/trust 参数搜索停止。完整数据见 `docs/phase4_results.md` 和 `results/phase4_experiments.csv`。
- **推理侧单独进展**：独立研发侧报告 F1 + M1 63.3276%，高于本仓库已审计 Flip TTA；因本地缺少 ZIP SHA-256，仅作为 `reported_unverified` 锚点。

## A2 LoRA 平台测试结果（2026-07-22）

- A2_LORA_MIN：裸推理 **61.1167%**；horizontal_flip TTA **61.6574%**。
- A2_LORA_FULL：裸推理 **61.5733%**；horizontal_flip TTA **62.1781%**。
- 当前 A2 LoRA 消融最高为 **A2_LORA_FULL + TTA 62.1781%**；详见 [A2 LoRA 平台结果](docs/a2_lora_platform_results_2026-07-22.md)。

# 发现（2026-07-21）

## A2_AEGIS_PARENT_SWAP：Split-lineage 协议修复与最终结论

- **原始 A2 parent swap 本地 79.22% raw_micro 是假信号**：A2 parent 使用 d3_strict split (91,195/10,322)，AEGIS child 使用 prepare split (92,902/10,316)。val 样本泄漏到 parent 训练集，导致本地准确率假胀 8.5pp。
- **协议修复**：新增 `canonical_sample_path`（统一 train/train_dedup）、fail-closed lineage audit、epoch-0 baseline evaluation、promotion gate
- **严格复跑 epoch-0 = 69.43%**：精确匹配 A2 本地准确率，确认 lineage 修复正确
- **LoRA 真实增益 +0.19~0.39pp**（vs 假 +8.5pp），双 seed promotion 通过
- **平台 Bare = 60.65%（+0.14pp vs F1 E2），TTA = 61.15%（+0.05pp vs F1 E2）**
- **A2 STRICT seed=3407 平台 Bare = 60.64%（vs seed=42 Bare 60.65%，双 seed 仅差 0.01pp，LoRA 增益高度稳定）**
- **结论：A2 parent swap 确认成立但收益边际**，不进入 P3/P4 参数网格；后续最小机制验证已完成并全部关闭
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

## Purification 天花板已触达（当时判断，后续 Phase 4 已验证）

- A0→A2 本地 paired delta 仅 +17 张图（0.165pp, p=0.196）
- 冻结 CLIP + GCE + MixUp 框架下，数据筛选层的边际增益已饱和
- 当时判断唯一上行方向是 visual LoRA PEFT；后续 A2 STRICT 证实小幅正收益，Phase 4 则证实其扩展机制没有达到晋级门槛

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
