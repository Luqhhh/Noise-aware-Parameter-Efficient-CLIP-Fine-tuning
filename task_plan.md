# Phase 3 B：完整剩余工作执行计划

## 目标

在 Windows Codex 环境中控制 WSL，完成 B 负责的 GCE 精调补项、OOF 加权对照、条件重标注、多 seed 验证、Protocol Audit 与 B→A 交付。原始 `val.csv` 不参与 OOF 折分或 OOF epoch 选择。

## 阶段

- [x] W3-0：生成 duplicate-aware 3-fold，并审计重复组泄漏与原验证集重叠
- [x] 实现并测试 OOF 训练、推理、质量特征与权重生成代码
- [x] 检查 Windows→WSL 启动链、脚本语法、GPU 与当前缓存状态
- [x] 生成全量 CLIP 特征缓存
- [x] 固定 50 epoch 依次训练并推理 3 个 fold
- [x] 合并 OOF logits，生成 sample quality、soft/discrete manifest 和审计报告
- [x] 验证行数、有限值、无泄漏、权重范围和测试结果
- [x] 盘点现有 B 实验、公共训练接口、结果与缺口
- [ ] 完成 Wave 1 补项：CE5→GCE07、低强度 MixUp 与统一比较
- [x] 执行 OOF soft/discrete weight 对照并应用停止条件（soft gate 关闭训练）
- [x] 若 weighting gate 通过，执行保守自动重标注及 weight-only/hard 对照（未执行：gate 未通过）
- [ ] 对通过 gate 的候选补 seed3407，并为平台候选补 seed2026（进行中）
- [ ] 生成 trusted/raw/macro/bottom10 指标与完整 Protocol Audit（seed42 候选指标已生成）
- [ ] 整理 B→A 交付包、复现命令、结果登记字段并做最终验证

## 约束与决策

- 基线：GCE q=0.5，固定 50 epoch，不用 holdout early stopping。
- 严格训练集：`outputs/d3_strict/seed42/train.csv`。
- 原验证集：`outputs/d3_strict/seed42/val.csv`，不进入 OOF。
- 保留用户现有未提交改动；未经明确要求不 commit、不 push。
- 继续使用 Windows 端 Codex，通过 `wsl.exe` 调度 Linux 训练。
- 用户 2026-07-16 要求完成 B 的全部剩余部分，覆盖此前暂缓的多 seed。
- relabel 仍是条件任务：只有 OOF weighting 有明确正收益且满足协议时才执行；gate 未通过即以审计关闭分支。

## 错误记录

| 错误 | 处理 |
|---|---|
| Windows 沙盒刷新 UNC 工作区时 `helper_unknown_error` | 通过本地 Windows 工作目录和已批准的 `wsl.exe -d Ubuntu` 调用绕开 |
| 首次组合检查命令的嵌套引号被 `wsl.exe` 错误解析 | 改为每条 Linux 命令直接调用，不再嵌套长 `bash -lc` 字符串 |
| 之前缓存任务约 1 分钟后因 WSL 被卸载而中止 | 本轮先验证启动链，保持可观测会话并分阶段执行 |
| Windows 原生 `apply_patch` 无法刷新 UNC 工作区 | 使用 WSL 内 `git apply -` 的标准差异输入作为受控后备 |
| 组合窄 hunk 标准补丁两次匹配失败 | 读取准确行号后改为整文件差异，避免再次尝试同一路径 |
| 训练接口 `rg` 正则中的竖线被 WSL 参数层拆分 | 改用 `rg -e pattern1 -e pattern2`，不在参数中使用 shell 管道字符 |
| 新增 seed 配置的首个补丁声明行数不足，末尾优化器字段被截断 | 训练在 epoch 1 前 fail-fast；补齐字段并核对实际文件尾部后重启 |
| `setsid -f` 未在桌面 WSL 调用中保留接力子进程 | 改用持久执行会话并通过独立日志确认接力已启动 |

## 2026-07-17 S_OOF_DISCRETE continuation

- [ ] Train seed42, evaluate best checkpoint, generate bare/TTA submissions, and validate both archives.
- Decision: user explicitly overrides the prior soft-gate stop for this discrete-weight run; preserve the original 0.3/0.6/1.0 manifest and record the warning.

## 2026-07-18 S_ELR_BASE

- Latest `main` verified at `873d959`; do not overwrite unrelated dirty or untracked worktree files.
- Run `configs/s_elr_base.yaml` with strict split `outputs/d3_strict/seed42`, seed42, CUDA, 50 epochs.
- Method: GCE q=0.5 + MixUp (alpha 0.2, p 0.2) + ELR (momentum 0.9, target weight 1.0, warmup 10, ramp 10).
- Start as a persistent WSL user service and preserve logs/checkpoints under `outputs/s_elr_base/seed42/`.
- Monitoring rule: check service/log state once per hour, not after every epoch; do not commit or push unless explicitly requested.
- After training, verify best checkpoint, run normal and horizontal-flip TTA inference/submission generation, and run `check_submission` on both archives.
