# 2026-09-12 缺失资产重建与工程实施记录

> 当前执行范围已由用户调整：复现任务移出本轮计划。以下保留为历史事实与未完成项记录，不再据此自动启动复现。当前待办见 [current_execution_plan.md](../docs/current_execution_plan.md)。

运行目录：`outputs/stage_readiness/20260912_full_rebuild_r1/`。
源基线：`83c66b8`。用户授权重建缺失产物、整体推进两方案、清理无用大文件。
本记录为执行中截面，不是两方案完成声明。当前不推送、不上传平台。

## 已实际执行

- 从已校验的 OpenAI ViT-B/32 初始化重新编码 103,218 张图，得到 103218×512 fp32 特征。样本顺序与历史记录一致，全部有限；旧特征张量缺失，不能认证历史数值等价。特征 SHA-256：`118f658c1837ee93611ae04f2a4e4630439a02e5ef921de0466d06beedfb6dfe`。
- 按历史 R2 多尺度教师的登记参数重建 logits/trust，完整命令及退出日志在运行目录。历史统计项一致；12 个 trust 数据张量中，仅 pseudo_confidence 的 81 个元素不同，最大绝对差 `1.1920928955078125e-7`。其余数据相等。不据此宣称位级相同或擅自通过数值验收。
- 新 trust SHA-256：`e990efc1679307a88bce2b97e3ab0db2f26a1a009d4d261d19602a887c2c157d`；新 logits 缓存 SHA-256：`7e0af4ac9ea2fd19ab732f586183bfb5d4ae13e166074345906d5483a751dde8`（实际文件哈希以 `teacher_validation.json` 为准）。
- 只删除 9 个已被后续演练替代的合成 checkpoint，共 3,163,629,675 bytes。逐文件路径、哈希和理由在 `cleanup_manifest.json`；原运行目录保留日志与 `retention_event.json`，这些旧节点已不能恢复。真实模型、官方数据、历史最佳包、无校准对照包保留且复核哈希。
- 新监督日志开关的合成训练完成，154 个模型张量与既有关闭开关的合成对照完全相等。生成 5 类/5 图 CSV、ZIP，并通过格式与包内外一致性检查；产物 `diagnostic_smoke2/submission/` 仅限软件演练。
- 官方当前复赛 750 类、半决赛 500 类结构演练完成。生成中文技术报告 PDF 初稿，逐页检查渲染；不代表正式材料验收。

## 模型恢复完成（22:34）

`recovery_plan.json` 登记 7 个顺序节点，最多 73 个 epoch（E2 原有早停政策保留）。
只改变运行路径、来源标签和可选观察日志，不改变历史方法或扫描参数。
E2 从官方初始化训练；其后按原配置逐级恢复 visual LoRA、MLP LoRA 的三轮、fullfit 和 selftrain R1。
这些是重新训练的新产物，不能要求与丢失的历史序列化文件同哈希。

```bash
python3 scripts/run_historical_recovery_queue.py --run-dir outputs/stage_readiness/20260912_full_rebuild_r1
python3 scripts/run_historical_recovery_queue.py --run-dir outputs/stage_readiness/20260912_full_rebuild_r1 --execute
```

第一个命令只审计。第二个已完成七节点恢复，最终队列状态 checks_passed。E2 于第 49 轮按原策略早停，其余六个节点均按登记轮数完成。
队列核对冻结源码、配置、预算、原始方法、父 checkpoint 选择和完成产物哈希；已有输出或失败会停下，不隐式 overwrite/resume。
源码固定在 `training_source/`；日志为 `e2_recovery.log`、各节点日志和 `recovery_queue_status.json`。
此入口专用于本次登记恢复，不是完整正式阶段复现认证。现存 cvt/trust 更早上游仍需闭合。
验证参与 checkpoint 选择；历史全量训练链的原验证属于重叠诊断，不能作为独立泛化证据。

## 实现与验证

新增内容组切分可行性诊断、开发 OOF 作用域过滤、实际监督量账本、教师特征缓存显式重定位，以及版本 2 冻结 prior 的来源/模型/映射/协议检查。
正式 infer 拒绝测试批内 prior 拟合；旧版本 prior 不能直接通过新入口。未实现可信拟合审计生产链，当前使用无校准路径。
发布目录加入三个阶段说明/待定配方模板、环境版本清单、官方权重字节核验记录、证据索引及技术报告源文件。

最终根套件 422 passed / 1 skipped，包含新增队列完成绑定测试 5 项。
最终 Aegis 回归 433 passed / 1 skipped；CUDA 单测在沙箱内不可用而跳过，实际 GPU 特征/教师生成已成功，七节点模型恢复已完成。
测试日志与退出码保存在运行目录。`git diff --check` 通过。

## 尚未完成

| 工作包 | 实际状态 |
|---|---|
| P0–P3 / R0 资产及来源 | 已盘点并部分补齐，仍有上游谱系未闭合 |
| P4–P7 初赛机制 | 范数对齐 63.3676%，匹配无校准对照 66.7681%，差 -3.4005 pp，已拒绝；本次恢复尚非新候选 |
| R1 明确错误与回归 | 已修复并通过当前回归 |
| R2 完整真实训练链 | 7 节点恢复完成；现存更早 trust/cvt 来源及全部最终分支仍未完整重建 |
| R3 开发隔离 | 图审计、切分及 OOF 检查已有；开发 trust 行过滤已接线并测试；完整父模型来源真实性认证未完成 |
| R4 长尾监督诊断 | 实际目标量日志已有；原始频次/内容组/可信覆盖全字段尚缺 |
| R5 校准 | 应用绑定检查已有；可信拟合审计与正式生产流程未闭合 |
| R6 演练 | 合成训练/推理/打包及当前阶段结构检查完成；真实完整规模资源账本未完成 |
| R7 材料 | 可维护源文件与 PDF 初稿已有；最终模型材料包未完成 |
| R8 干净环境 | 正式全过程、结果比较及用户复核未完成 |

不能将本次工程回归、恢复中的训练或 PDF 初稿写成两方案全部完成。历史最佳 70.352866% 不转移到新缓存、模型或校准协议。

## 恢复模型交付检查

最终 Selftrain R1 第 3 轮原验证 raw_micro = 72.1501%，仅为重叠诊断。七个 best checkpoint 已再次按清单核验实际 SHA-256。

在源码 19fa2ee 下按单 checkpoint、原单视图、无校准路径执行 infer（`--tta none --local-view none`），退出 0。完整命令绑定在运行目录 recovered_inference_registration.json。生成 24,967 行预测，经 check_submission.py 检查通过，ZIP 内 CSV 与外部文件逐字节相同。

产物：`outputs/stage_readiness/20260912_full_rebuild_r1/recovered_selftrain_submission/submission.zip`。ZIP SHA-256：`e56b3129c0027ac37fc136d825e69c29de4bc0d1a65060d1763830ce7c5cb479`。未上传、未替换桌面包；没有平台成绩，也未选为历史最佳替代。上游 trust/cvt 真实性闭合仍是完整复现的未完成项。

新增 report_class_support.py 与 class_support 模块，实际生成 500 类统计，绑定原始/拟合/验证 CSV、内容组和类别映射文件哈希；10,316 个验证样本的内容组均已进入当前拟合范围。缺少的逐类准确率和频次分组留空，不能填零或从已有总分推算。冲突组数是原标签冲突统计，不是真值噪声率。新模块反例与手算测试 2 项通过。

## 桌面待测登记与资源账本

用户要求将恢复包放到桌面并等待回传分数。已复制到 `submission_recovered_selftrain_r1.zip`，核对与本地 ZIP 哈希一致。登记 ID：`RECOVERED_SELFTRAIN_R1_BARE_NOCAL_20260912`，平台分数和实际上传时间保持空白，不覆盖现有历史最佳或匹配对照记录。该模型及单视图协议与此前双 Adapter 四尺度+Flip 流程不同，未来分差不能全部归因于单一训练机制。

`report_recovery_resources.py` 对照逐轮监督账本与训练日志：实际共 72 epoch（E2 49，其他共 23），92,682 个观察到的训练 batch，6,750,840 次样本抽样；各节点日志跨度合计 16,839.922 秒。观察到 batch 不证明 AMP 从未跳过更新。当前七节点目录文件总量 32,855,241,512 bytes，不是磁盘峰值。峰值显存/内存/磁盘未完整采集，保持 null，不能从当前数值倒推。报告位于本地运行目录 resource_report.json，包含日志与每轮账本哈希。

统计工具的手算例及缺失账本反例测试共 2 项通过。待测包保持冻结；仍需补齐的上游来源、开发 trust 接线、可信校准生产流程和干净环境复现见前述状态表。

开发 pipeline 增量：组集合哈希绑定、训练/验证组隔离、OOF assignments 精确匹配、trust 学习前子集过滤、开发链禁止最终合并。反例覆盖验证特征混入、旧全量 OOF、组文件篡改和缓存标签错配。保留正式来源认证未完成状态，没有新正式训练。

本段验证：Aegis 全套 439 passed / 1 skipped（CUDA 不可用）；随后加入开发输出保护并运行相关套件 5 passed。资源报告测试 2 passed。桌面包哈希不受工程代码修改影响。
