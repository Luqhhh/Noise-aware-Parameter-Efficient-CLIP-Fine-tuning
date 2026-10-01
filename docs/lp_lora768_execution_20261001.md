# A机 LP_LORA768_20261001 本机执行记录

成员clairvoyanttt，独立方案分支`clairvoyanttt/lp_lora768_20261001`。实现基于2026-10-01最新main（启动前735b2eb），父资产与交接包只读，共享环境未改动。本记录当前为正式训练在途，尚未交付新提交包、尚无平台结果。

## 固定协议与运行

协议仍为`configs/team_exploration_20261001/machine_a.json`，普通DEV768末轮头父，119,074条正权重监督。head_only仅更新cosine头；lora_and_head更新同一父的全12块rank32/alpha64 LoRA及头。每臂4轮，每轮3,722个逻辑更新，合计每臂14,888次；逻辑32/micro8/worker2/seed42，其余配方完全沿队长任务单。没有balanced监督、best轮选择、第二seed、超时门或full延伸。

主输出为同一轨迹EMA第2/3/4轮的算术平均；末轮raw和EMA只作诊断。全部导出均包含98项LoRA及头参数，control冻结LoRA也必须保存。两臂逐批记录增强图像、索引、Mixup置换/系数及可靠质量证明。两份主checkpoint分别固定448/512/576+flip求和解码，无bias，生成37,444行CSV/ZIP及9项提交检查后才算本段完成。

独立工作目录（WSL可访问）：
`/mnt/c/Users/clairvoyant/.codex/worktrees/clairvoyanttt-lp-lora768-20261001/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning`

本机位置文件：`.planning/2026-10-01-clairvoyanttt-lp-lora768/locations.json`。迁移机器只改这个文件，不改原sidecar。

```bash
PYTHONPATH=reproducibility/aegis_f1 /home/clairvoyant/.venvs/noise-clip/bin/python -u -m lp_lora768 \
  --locations .planning/2026-10-01-clairvoyanttt-lp-lora768/locations.json \
  --output /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/outputs/clairvoyanttt/lp_lora768_20261001/20261001T222332_retry1 \
  --execute --allow-local-parent-baseline
```

首次复现父预测应先用同一入口`--action replay --execute`与一个全新输出目录；若严格逐图一致，正式运行无需`--allow-local-parent-baseline`。不覆盖既有运行。

## 严格源重放偏离（必须保留）

26份交接资产、图片字节、官方权重、全部train/val有序路径与标签、mapping、原监督绑定均已核对。父模型本机14,880张448中心预测与源有3张不同，故**队长的严格逐图源重放门没有通过**。原源macro75.602691%/micro76.518817%；本机macro75.611025%/micro76.525538%，11,387张正确。目标组仍由训练前源预测冻结，417张/源错误173张，不按候选结果重选。

batch32、头CPU/FP64、禁TF32、math SDPA及CPU FP32检查均保留这些本机选择；根因尚未定位。用户在被告知这3张差异并询问源环境版本后回复“这些不影响我本地实际运行啊”。据其本地执行优先的指导，本段继续固定训练，以本机实际父预测作为两臂共同配对基线，另保留源预测列。此处是本机执行偏离记录，不把它写成严格源重放通过，也不声称队长批准了门禁例外。

证据：`results/lp_lora768_20261001/parent_replay.json`、`replay_diagnostics.json`、`backend_diagnostics.json`、`local_baseline_ruling.json`与`current_inputs_verified.json`。全部源文件未改写。

## 已验证的实现与成本

真实资产测试4项通过：首128张父预测、冻结control完整导出、两臂实际更新和独立RNG配对、control冷加载。首5逻辑更新探针通过；探针权重丢弃，正式两臂从原父重新加载。直接复用v1 weighted_mixup_loss核对真实首批的微批可靠质量归一化，损失等价通过。完整导出冷加载logits逐项精确一致。

head_only探针5步2.9067秒/峰值724,022,784字节；lora_and_head为3.6375秒/2,197,156,352字节。128张初始中心评估分别1.7331/1.5374秒。完整父验证123.521秒。每臂四轮预计8,654.94/10,830.91秒，只记录估计，不据此提前停止；启动与DataLoader开销使估计不等于实际完成时间。详细成本及提交推理估计见`cost_probe.json`。

Windows Git曾将历史跟踪文本转成CRLF，使历史SHA测试失败；在确认独立工作目录跟踪文件没有内容改动后恢复原LF，相关68项测试全部通过。最新main完整回归为762 passed/13 skipped/2 failed；两项失败均为test_scope_protocol.py的ScopePreflightError（仓库已知上一阶段资产缺失）。真实资产测试另行显式启用，4 passed。不修改历史冻结摘要迎合本机换行。

## 实际在途与交付边界

当前新轨迹head_only已在本机4060 Laptop启动（2026-10-01 22:31左右），随后顺序执行lora_and_head。首次故障记录见末节。动态状态与日志：
`outputs/clairvoyanttt/lp_lora768_20261001/20261001T222332_retry1/progress.json`、`execution.log`。

本段完成后独立复算全量/目标/尾部/其余/重叠的修正退化，固定净+75、目标净+25及比率1.25判据只支持supports_review；否则关闭固定配方。四轮与两份提交包未完成前，不报告训练段完成。本段不上传平台，不自动启动full。收尾立即推送方案分支，在main集成目录按自动模式pull --rebase --autostash、合并、重新核验并push。
## 首次运行设备异常与完整状态恢复

首目录`20261001T211600`在head_only第2轮batch1268（4,990个尝试batch）发生CUDA unknown error。仅完成1轮；最后完整进度为4,934次更新，日志记录失败尝试。该不完整轨迹不计为四轮结果，原产物与错误日志保留。根因尚未建立；新进程CUDA矩阵运算及精确重建的同一增强/Mixup batch原父更新通过，不根据这个结果声称已定位硬件问题。证据`first_run_failure.json`。

新目录`20261001T222332_retry1`从原父权重重新开始两臂固定4轮。零更新父预测与首次运行的logits全部逐元素一致（最大差0），仍是3张源差异。重放入口报告重复rows字段已修正；第二次完整预测已算完、NPZ保留，用该完整真实输出补写报告，明确保留入口错误记录。

新增完整训练状态：每1,000次成功更新及轮边界保存raw权重、AdamW、OneCycle、GradScaler、EMA和EMA2–4平均累计状态，另记录成功batch前缀SHA。`--resume`仅在输入/代码绑定一致时恢复完整状态，保留失败尾部，再从同一采样位置继续；不会拿EMA冒充raw起点，不改变更新总数。真实父权重两臂测试证明恢复后下一步的loss/权重/EMA/调度/AMP逐项精确一致，恢复后的实际增强图像与索引一致、成功日志前缀一致，失败尾部保留。

5项真实资产/恢复测试通过；额外恢复采样检查1passed。独立交付核验脚本为`python scripts/verify_lp_lora768_delivery.py --output <run_dir> --assets <handoff_dir>`，须在两份包生成后实际执行。

当前动态状态以新目录`progress.json`、`execution.log`、`head_only/recovery/latest.json`为准；首目录的进度是故障前记录。四轮与新包尚未完成。
## 2026-10-02 对照臂完成与首次完整状态续跑

对照臂恰好完成4轮/14,888次成功更新，固定EMA2–4主结果micro76.4314516%、macro75.5113731%（11,373张正确）。相对本机父修正184、退化198，净−14；末轮EMA第4轮为micro76.3776882%、macro75.4590670%，仍只作诊断。三份完整导出均有98项参数，冻结LoRA逐项等于原父；四轮日志计数独立核对通过。真实报告见`results/lp_lora768_20261001/retry1/head_only/`。这些不是两臂比较或提交包完成声明。

LoRA臂第1轮batch556再次报CUDA unknown error（香港时间00:05:14），原556条尝试记录、堆栈和失败报告保留。新进程CUDA矩阵运算正常；设备异常根因未知。按既有`--resume`从该臂step0完整状态恢复，失败尾部归档，已完成的对照臂直接复用；没有从EMA替代raw，也没有重做对照臂。恢复后首批loss与原首批完全一致，已越过故障位置并保存第1,000步完整恢复点。续跑在上方确切命令末尾加`--resume`，同一输出目录追加execution.log。故障证据见`retry1/cuda_failure_before_resume1.json`。

主main实现及协议绑定与实际运行一致；最新整合回归787 passed/12 skipped/2 failed，两项失败均为仓库已知ScopePreflightError。实际两臂比较、37,444行CSV/ZIP和完整独立交付审计仍待LoRA臂完成，不启动额外轮次。
