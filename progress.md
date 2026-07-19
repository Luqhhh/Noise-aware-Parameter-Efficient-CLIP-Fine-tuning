# OOF 执行进度

## 2026-07-15 Windows 续跑

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
