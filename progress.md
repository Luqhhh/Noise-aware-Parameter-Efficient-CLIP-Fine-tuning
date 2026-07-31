## A2 LoRA 平台测试结果（2026-07-22）

- A2_LORA_MIN：裸推理 **61.1167%**；horizontal_flip TTA **61.6574%**。
- A2_LORA_FULL：裸推理 **61.5733%**；horizontal_flip TTA **62.1781%**。
- 当前 A2 LoRA 消融最高为 **A2_LORA_FULL + TTA 62.1781%**；详见 [A2 LoRA 平台结果](docs/a2_lora_platform_results_2026-07-22.md)。

# 执行进度

## 2026-07-31 M1 + Flip 平台结果：63.7802%，新平台最佳

- M1 + Flip `local0.40/flip0.50` 平台实测 **63.7802%**，成为新的审计完整平台
  最佳（原 F1 + M1 63.3276% 仅为 `reported_unverified` 锚点）。
- 相对 F1 REBUILD R1 M1 weight 0.35（62.9791%）`+0.8011pp`；相对已报告原
  F1 + M1（63.3276%）`+0.4526pp`；相对 A2 STRICT + M1（62.6870%）
  `+1.0932pp`。
- 距离 70 分 `6.2198pp`；本地 raw micro `72.1307%` 到平台的 gap `8.3505pp`。
- 已按 checkpoint/prediction/ZIP SHA-256 精确登记到
  `results/current_platform_summary.csv` 与 `results/submission_registry.csv`，
  状态 `platform_valid_promoted`。

## 2026-07-30 F1 REBUILD R1：M1 + Flip 新候选

- 已实现单 checkpoint 的原图/水平翻转 × global/attention-local 四视图概率融合，
  并保留显式 TTA 风险确认与 fail-closed 参数校验。
- 固定 `crop160/top5`，扫描 4 个 local weight × 2 个 flip weight；选择
  `local=0.40, flip=0.50`。
- 相对当前 62.9791 包对应的 M1 weight 0.35，raw `+0.3296pp`、trusted macro
  `+0.1520pp`、proxy macro `+0.1340pp`、clean-core `+0.2728pp`，500 类覆盖
  不变。
- 新包审计通过：24,967 条、500 类、0 损坏图；ZIP SHA-256
  `67f4eda57291e34096edcb0545b142fd0a3114fb1c76eb1e17996afe87d692e0`。
- 该包平台实测 63.7802%，新的审计完整平台最佳；精确回填见上方 2026-07-31
  条目。
- 完整报告见 `results/m1_flip_optimization_20260730.md`。

## 2026-07-30 E2→F1 严格重建与 M1 新候选

- 同源 `preliminary/seed42` split 重建 E2 父模型：epoch 39 raw
  `70.3470%`，比 archived 父模型高 `+0.1163pp`，500 类覆盖。
- F1 parent/child lineage 审计通过：train/val SHA 完全一致，交叉泄漏 0，标签错配 0。
- F1 visual LoRA 最佳 epoch 4：相对 epoch 0 raw `+0.2230pp`、
  clean-core `+0.4092pp`、漂移 `0.3982%`，promotion PASS。
- 固定 M1 `crop160/top5` 后，weight 0.35 相对 global raw `+1.2602pp`、
  clean-core `+0.9548pp`，trusted/proxy 同向，覆盖 500 类。
- weight 0.40 只在 raw/clean-core micro 略高，却损失 trusted/proxy/clean-core
  macro；按多指标规则保留 0.35。
- 新平台包已审计：24,967 条、500 类、0 损坏图；ZIP SHA-256
  `02d37906accdf6b49e40733b4e675220f0177b5d71c5662984de68df5e781bb6`，
  平台 **62.9791%**，状态 `platform_valid_not_promoted`。
- 相对 A2 STRICT + M1 62.6870% 提升 `+0.2921pp`；相对已报告原 F1 + M1
  63.3276% 仍低 `0.3485pp`。当时历史最高仍为 63.3276%（2026-07-31 已被
  M1 + Flip 63.7802% 超越）。
- 相关测试 29 passed；Aegis 子工程全量 75 passed、1 个既有 `p4_ablation`
  stage 校验失败；两个新配置独立加载通过。
- 完整报告见 `results/f1_rebuild_20260730.md`。

## 2026-07-30 A2 STRICT 局部特征 Adapter

- 已修复 O3 的 attention-local 数值路径，在 A2 STRICT 上完成局部专用
  `512→32→512` 残差 Adapter 首轮与六个有界消融。
- 最佳 `local_loss_weight=0.50`：raw `+0.0872pp`、trusted macro
  `+0.1219pp`、clean-core `+0.1821pp`、漂移 `0.6538%`、覆盖 500 类。
- 因 clean-core 未达到预注册 `+0.20pp` 门槛，全部候选
  `gate_failed/best_not_promoted`；不生成测试提交，关闭该分支。
- 本机保留与原 F1 父模型同配方的 CE5+MixUp 训练资料；已启动同 split 的 E2
  重建，若复现父基线再进入 F1 LoRA，不使用 split 不同的现成 checkpoint 冒充父模型。
- 完整报告见 `results/local_adapter_a2_strict_20260730.md`。

## 2026-07-30 A2 STRICT + M1 weight 0.35 平台结果

- 用户先回报 62.8670%，随后更正为 **62.6870%**；结果表以最新值为准。
- 相对 A2 STRICT Bare 60.6521% 提升 +2.0349pp，确认 attention-local 的真实平台收益。
- 相对已报告 A2 + M1 62.6747% 仅 +0.0123pp，且 checkpoint 不同，不能归因为 weight 0.35。
- 低于 F1 + M1 63.3276% 0.6406pp，状态改为 `platform_valid_not_promoted`。
- M1 融合权重搜索关闭；下一轮优先恢复/重建更强的 F1 表征，而不是继续围绕 A2 STRICT 调推理参数。

## 2026-07-30 M1 attention-local 推理优化

- 在当前仓库实现最后视觉 block 的 CLS→patch attention 提取、top-k 加权中心裁剪、global/local 概率融合和 fail-closed 推理入口。
- A2 STRICT seed 42 复现固定 M1 (`crop160/top5/weight0.50`)：raw +0.8525pp、clean-core +0.3501pp。
- 小范围定位网格确认 `crop160/top5` 仍是最稳组合；不改定位机制，只将局部分支权重降至 0.35。
- M1 weight 0.35 双 seed 同向：raw +1.0657/+1.0076pp，clean-core +0.8123/+0.7843pp，均覆盖 500 类。
- 后续多尺度 M2 在 seed42 虽将 raw 推至 70.8777%，但 proxy macro 和 clean-core 分别比单尺度候选低 0.0156/0.0140pp；未满足全指标同向门槛，停止且不补 seed3407。
- 已生成并审计 A2 STRICT seed42 + M1 weight 0.35 平台包；ZIP SHA-256 为 `ddbbf0b9e408e9fbcd4fc7d00c8c16e647a872634c61625d9c9e9c935d549e66`。
- 平台最终为 62.6870%，状态 `platform_valid_not_promoted`；70 分目标尚未达成。
- Aegis 子工程相关测试 25 passed；新增入口关闭 local view 后，Bare 预测与既有 24,967 条提交逐字节一致。全量 71 passed、1 failed；唯一失败是既有 Phase 4 配置使用 `stage: p4_ablation`，而当前 config validator 只接受三个比赛 stage，与本轮改动无关。

## 2026-07-30 文档与结果索引同步

- 已核对截至 `7c8b966` 的提交历史和 2026-07-22 之前的全部 Phase 4 本地产物；没有发现更新提交之后的新实验运行。
- 原 `docs/phase4_plan.md` 已确认有意删除，由 `docs/phase4_results.md` 和 `results/phase4_experiments.csv` 取代。
- README、CURRENT_STAGE_ACCEPTANCE、findings、outputs 索引和结果表已统一到 Phase 4 最终状态。
- F1 + M1 63.3276%、A2 + M1 62.6747%、A2 + M3 62.0259% 作为独立研发侧已报告锚点登记；因本仓库缺少提交 ZIP 哈希，状态标为 `reported_unverified`。
- F2/O1/N3 截至现有证据仍是“提交包已审计、待平台”；独立研发仓库不在当前机器，无法推断后续平台结果。
- `pytest -q tests` 实跑：402 passed、1 skipped、2 failed；失败为 CPU DataLoader 非零 timeout 与 float32 精确相等断言，已进入当前验收未决项。

## 2026-07-22 Phase 4 突破实验完成

- **P0 结构化分类头**：多原型 (MP-1~6) 和 LDA/Ridge (SH-1~6) 全部未通过晋级 gate。raw_fixed < raw_broken 始终成立，关闭方向。
- **P1 Checkpoint Averaging**：6 epoch 完整结束；SWA-1/2/3 均未超过 epoch 6 clean-core，greedy soup 只保留 epoch 6，关闭方向。
- **P2 Clean-Routed LoRA**：CR-0/CR-1/CR-2 均完成。Hard gate 与现有 selection threshold 重合；Soft gate clean-core 仅 +0.042pp 且 raw/proxy 下降，关闭方向。
- **P3 Trusted Prototype-Contrastive**：最佳 clean-core 比 CR-0 低 0.084pp，且只覆盖 499 类，promotion 失败，关闭方向。
- **P4 Dynamic Trust Refresh**：epoch 2 刷新后所有 epoch 的 clean-core 均低于刷新前，关闭方向。
- **最终结论**：P0–P4 无平台候选；当前 CLIP ViT-B/32 方法族在本仓库框架内到顶，不继续普通参数搜索。
- **A2 STRICT seed=3407 平台 Bare = 60.64%**（vs seed=42 Bare 60.65%，双 seed 仅差 0.01pp），LoRA 增益高度稳定。
- **增强特征缓存**：horizontal_flip 缓存已生成并审计通过（103,218 samples, 512-dim）。

## 2026-07-21 A2_AEGIS_PARENT_SWAP 协议修复与严格复跑

- 发现原 A2 parent swap 本地 79.22% 是 split lineage 泄漏（parent d3_strict vs child AEGIS prepare）
- 实现 lineage audit + epoch-0 baseline + promotion gate + canonical sample path fix
- 严格复跑 (F1_VISUAL_LORA_CLEAN_CORE_A2_PARENT_STRICT) 双 seed 均通过
- epoch-0 = 69.43%（精确匹配 A2 本地），LoRA 真实增益 +0.19~0.39pp
- 平台 Bare = 60.65%（+0.14pp vs F1），TTA = 61.15%（+0.05pp vs F1）
- A2 STRICT seed=3407 平台 Bare = 60.64%，双 seed Bare 差 0.01pp → LoRA 增益确认稳定
- 结论：A2 parent swap 确认正收益但边际，P3/P4 不追
- 分支 fix/a2-aegis-parent-lineage 已合并 main，协议修复已归档

## 2026-07-15 OOF 执行（Windows 续跑）

- 已确认继续使用 Windows 端控制 WSL。
- 已读取文件化计划技能并恢复任务上下文。
- 已检查仓库：无活跃 OOF 任务；特征缓存尚未完成。
- 已记录 Windows 沙盒刷新和命令引号两类错误，后续使用直接 `wsl.exe` 调用。
- 下一步：检查脚本、fold 产物、GPU 和单元测试，然后启动缓存阶段。
- 已完成脚本语法、fold 审计、GPU 和 8 个 OOF 单元测试检查：全部通过。
- 下一步：启动特征缓存；缓存完成后自动进入 3-fold OOF。
- 21:47 已启动完整 pipeline；缓存共 807 批，采用 batch 128 / 0 worker 稳定配置，已确认持续编码。
- 2026-07-16：pipeline 状态 0 完整结束，缓存、3-fold 训练、OOF 推理和质量清单均完成。
- 三个最终 CSV 均为 91,196 行（含表头），审计确认原始验证集未使用、无 holdout epoch 选择、logits 全部有限。

## 2026-07-16 B 剩余任务

- 用户要求继续完成 B 的全部剩余部分，包括此前暂缓的多 seed。
- 已恢复计划、进度和发现记录；下一步盘点现有实验、训练接口与缺失产物。
- 已确认训练框架原生支持 schedule、MixUp 和 manifest 样本权重。
- 发现 Wave 1 配置 split 与 strict 契约冲突；正在对照 B2 基线配置和已有输出。
- 已确认旧 CE5→GCE07/MixUp 结果使用 ref split，不计入当前 strict 正式结果。
- 正在检查 q=0.9 中断原因、现有指标和正式 strict 重跑集合。
- 已确认本地 q=0.9 停在 epoch40；用户提示队友可能已在远端完成。
- 下一步只 fetch 远端 main 并检查差异，不覆盖当前 dirty worktree。
- 已 fetch：远端 main 为 `cb786a8`，无 q=0.9 完整结果，但有 CE5 q=0.5、MixUp q=0.5 完整结果。
- 已检查变更路径，准备 fast-forward 同步远端，同时保留本地 OOF 和用户修改。
- 已 fast-forward 到 `cb786a8`，本地 OOF 和用户修改均保留。
- 已核对新基线 lineage：CE5/MixUp 使用 ref split；OOF downstream 将继续严格使用 d3_strict。
- 已确认训练侧 OOF/relabel provider 可用，但本地 OOF manifest schema 不兼容。
- 下一步规范化 manifest 并执行 fail-closed weight audit，再按 gate 决定是否训练。
- 已确认路径键可直接匹配；问题仅在 manifest 列 schema。
- 下一步为 canonical manifest 与 gate audit 补测试和实现。
- 已确定从 `sample_quality.csv` 无损重建 canonical manifest，不重跑三折 OOF。
- soft manifest 作为主 gate；discrete 只在 soft gate 通过后运行，并在审计中记录该依赖。
- 已添加 `analysis/oof/finalize_manifests.py` 和 3 个单元测试。
- 新增 3 个测试与原有 8 个 OOF 测试全部通过；准备运行全量 manifest 审计。
- 已生成 canonical soft/discrete manifest 与 `weight_audit.json`。
- 全量审计 100% 覆盖且无 schema 错误，但 soft gate 被类别 65/338/407 触发。
- 已按协议关闭 OOF weight 训练、discrete 对照和 relabel，不人工修改 manifest。
- 已核对结果登记表：两个当前候选缺 trusted/hash/多 seed；q=0.9 与旧 q=0.7 补项未登记。
- 下一步盘点 trusted 评估实现，再创建最小多 seed/q=0.9 执行集合。
- 已找到 trusted subset/weight/class-balanced 核心函数。
- 正在确认输入信号来源和 prediction_records schema，准备补统一可信度报告。
- 已找到固定验证集 `sample_metrics.csv` 和 trusted manifests，可直接复用。
- 下一步核对列名和样本键，随后实现候选统一 trusted report。
- 已确认固定信号覆盖 10,316/500 类；V1 trusted 仅覆盖 336 类。
- trusted report 将使用稳定 class/file 键，并同时产出 V1、V2 weighted、V2 class-balanced。
- 已确认 prediction records schema 和跨机器稳定 join key。
- 开始实现统一 trusted report，并将先评估 B2、CE5、MixUp 三个 seed42 候选。
- 已添加统一 trusted report、边界测试与跨机器样本键测试；新增 3 个测试全部通过。
- 运行全量 fail-closed 覆盖审计：B2 与固定信号严重不匹配，拒绝出分；CE5/MixUp 均 100% 匹配。
- 由此确认 B2 与 Phase 3 两个候选使用不同验证 split，禁止再宣称其本地精度差是同源增益。
- 已为 CE5 与 MixUp 生成 raw、V1、V2 weighted、V2 class-balanced、bottom10 和协议审计文件。
- CE5 在所有同源本地/可信指标领先，MixUp 保留平台分数优势；两者进入 seed3407。
- 下一步生成隔离的 seed 配置，校验 resolved split 后启动多 seed 训练。
- 已新增 CE5 seed3407、MixUp seed3407 和 MixUp seed2026 配置，固定 ref/seed42 split，仅改变 train_seed。
- 首次 CE5 启动在 epoch 前因配置末尾字段被补丁行数截断而 fail-fast；已补齐并成功重启，无 checkpoint 污染。
- CE5 seed3407 已正常运行，前两轮 val acc 36.40%→53.87%。
- 已新增并通过语法检查的本地串行接力脚本。
- 接力持久会话已在 04:12:48 启动并写入独立日志；当前等待 CE5 prediction_records。
- CE5 完成后将自动运行可信报告、MixUp seed3407、可信报告、MixUp seed2026、可信报告并写完成标记。

## 2026-07-16 OOF 与 CE5 seed3407 提交

- 用户要求把当前已完成的 OOF 与 CE5 seed3407 一起提交并推送到 main。
- 提交范围明确排除未完成的 MixUp seed 配置/接力脚本，以及用户原有 splits CSV 改动。
- 将先审计远端 main、Git LFS/大文件规则、产物完整性和待提交路径，再 commit/push。
- 已确认 main 与 origin/main 都在 `cb786a8`。
- 已确认仓库无 Git LFS 配置，OOF/CE5 完整目录含大量大文件。
- 下一步对照历史实验提交，确定大张量排除范围并验证轻量交付完整性。
- 已确认 `.gitignore` 会排除全部 `.pt`，无需引入 Git LFS。
- 已确认历史 CE5 提交不包含 checkpoint，本次保持一致。
- OOF 三个较大 CSV 可正常提交；下一步生成完整 CE5 trusted report 并精确列出待 stage 文件。
- 已用完整 10,316 条预测生成 CE5 seed3407 trusted report。
- 相关 14 个单元测试全部通过；下一步精确 stage 并审查暂存内容。
- 已向 `results/phase3_experiments.csv` 追加 OOF 与 CE5 seed3407 两条完整记录。
- 下一步精确 stage 指定路径，并验证暂存文件列表、大小与 CSV schema。
- 结果表经 Python CSV 解析确认共 13 行、所有行均为 33 列。
- `git diff --check` 通过；准备按白名单路径暂存。
- 已按白名单暂存 64 个 OOF/CE5/可信报告/测试/结果登记文件，共 671,524 行新增。
- 所有 `.pt`、MixUp 配置、接力脚本、过时工作计划及用户原有改动均未暂存。
- 下一步 fetch 远端确认 main 未前进，然后检查作者配置并创建提交。

## 2026-07-17 S_OOF_DISCRETE

- Confirmed main/origin main at a4202b1, no active duplicate trainer, no checkpoint, and an idle GPU.
- First launch failed before optimizer step 1 because manifest keys resolved symlinks while DataLoader paths did not.
- Added a failing symlink-path regression test, then normalized provider lookup paths with Path.resolve().
- Related 12 tests pass; all 91,195 real strict-train paths now load weights (min 0.3, max 1.0, mean 0.634606).
- Next: restart training from epoch 1 and continue through evaluation and both submission variants.

## 2026-07-18 S_ELR_BASE 平台结果

- S_ELR_BASE TTA = 59.14%，本地 68.20%。
- 比 OOF zero-weight（60.28%）低 1.14pp——ELR 的 EMA 引导不如直接按 OOF 置信度剔除噪声。
- registry、phase3_experiments.csv 已更新。

## 2026-07-18 S_OOF_ZERO_001 平台结果

- S_OOF_ZERO_001 bare = 59.38%（-0.58pp vs 0.001 threshold 59.96%）
- S_OOF_ZERO_001 TTA = 59.92%（-0.36pp vs 0.001 threshold 60.28%）
- 0.01 阈值排除 12% 样本 vs 0.001 阈值排除 7%——更宽阈值在 bare 和 TTA 上均更差。
- registry、phase3_experiments.csv、README 已更新。

## 2026-07-19 S_OOF_ZERO_0001_FF 平台结果

- S_OOF_ZERO_0001_FF bare = **60.29%**（+0.33pp over dev mode 59.96%，**NEW BEST**，首次突破 60%）
- S_OOF_ZERO_0001_FF TTA = **60.51%**（+0.23pp over dev mode TTA 60.28%，**NEW BEST**，首次突破 60.5%）
- final_fit 全量训练（无验证集）带来显著升幅，验证了 full-data 策略
- Bare-TTA gap 仅 0.22pp（dev mode 0.32pp），全量训练后模型更稳定
- OOF binary zero p<0.001 路线确认为当前最强策略


## 2026-07-19 ROBUST_LORA TTA 平台结果

- LoRA (rank=8, alpha=16, last_block_lora) TTA = **60.24%**
- freeze_clip=false, lora_lr=1e-5, 训练 6 epoch（early stop at epoch 6）
- 首次 LoRA PEFT 平台测试结果
- local micro=69.40%, best epoch=1

## 2026-07-19 A0 (NR_CTRL_FIXED) TTA 平台结果

- A0 TTA = **60.30%**（2-view Flip TTA）
- ⚠️ 此结果为 pre-fix 版本：rejected 样本（7%）通过 MixUp 污染 clean 样本
- 需要 rerun with `reject_policy: drop`

## 2026-07-19 A0 fixed (reject_policy=drop) 平台结果

- A0 bare = **59.90%**, TTA = **60.31%**（FIXED: reject_policy=drop, 6354 rejected 物理删除）
- local micro = 69.33%, epoch = 50
- vs pre-fix: bare 59.90% vs 旧版 invalid
- reject_policy=drop 修复后，TTA 60.31% 验证了 MixUp 污染的修复

## 2026-07-19 A2 (NR_CL_KNN_DROP) TTA 平台结果

- A2 TTA = **61.21%**（NEW BEST, +0.11pp over AEGIS F1 61.10%）
- CL+kNN consensus drop: 991 rejected (1.1%), reject_policy=drop
- kNN other-fold-only, confident-joint + OOF/kNN agreement
- 首次突破 61.2%
