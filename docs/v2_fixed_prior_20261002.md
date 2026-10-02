# V2_FIXED_PRIOR_20261002

目标仍为平台80%。现役用户报告74.41512658903963%，距80%为5.58487341096037pp；
按37,444张整体准确率换算，至少29,956张正确才达到80%，比当前推算27,864张多2,092张。
没有测试逐图真值，2,092不是已定位错误清单。

## 输出前冻结的单次方案

复用已经训练的v2 full第5轮raw单checkpoint，只在本机执行一次固定推理与均衡bias校正。
依据是同一768 full模型平台原始71.53883132144003%→固定bias74.41512658903963%，
净+2.87629526759960pp（推算1,077张）；这支持在独立v2上检验相同先验机制，不能保证迁移。
A、B四轮配对均未过原门，不继续其full或强度搜索；不等待空闲资源就启动新训练。

v2原解码是四视图**softmax概率平均**，不是v1六视图logits求和。
本方案保留`resize576/center/flip/resize806.4`、576输入、BF16前向、float32 softmax求和，
以`log(mean_view_probability)`作为bias输入。此变换保持无bias argmax；不切换TTA或温度。
固定200次、damping=1、strength=1、目标1/750，不扫描、无测试标签、无模型参数更新。
为零概率设置固定1e-30下限，仅用于取log；报告实际零数并核对原始argmax不变。
soft概率质量均衡不是准确率改善，也不要求每类恰好50个argmax。

配置：[v2_fixed_prior_20261002.json](../configs/v2_fixed_prior_20261002.json)。
训练成本为0；完整计算上限为37,444张×原四视图、一次200步bias及一次独立复算，
先用64张当前阶段val图检查同架构S1权重的本机CUDA成本，不用其分数选参数。
不设置墙钟截止，不做自动重训/重试。其他CUDA任务存在时拒绝启动。

## 输入与交付门

原固定v2交付任务独立运行，本段不连接远端、不修改其推理、下载或关机服务。
只有本地`delivery_receipt.json`确认下载校验，且checkpoint、metadata归档、原始CSV/ZIP
逐文件SHA一致时才执行。读取原plan、binding、stage状态和配置，核对完整5轮/9,290更新、
750类、当前阶段manifest、官方模型摘要、原四视图与原raw策略，不改写原metadata。
配置仅映射本机官方权重与数据位置。缺文件、部分下载、绑定错配即停止。

同一份本机缓存生成raw和bias两包；均为37,444行并经过9项正式提交检查。
独立NumPy/float64复算bias、逐图预测、CSV/ZIP字节及输入摘要。
本机raw与服务器原CSV全量比较；跨GPU/Torch可能存在数值差异，原包保留，
如有差异记录数量及对应top1间隔，不能声称逐图相同；bias配对只比较同一份本机缓存。
实际full输出前进一步固定数值迁移门：差异不超过0.1%（全量最多37张），且差异图本机
top1−top2概率间隔不超过0.01。两门均通过才出包，否则保存原始概率缓存并诊断，
不重复推理或放宽门限。这是工程一致性界限，不是本地/平台识别收益门；
依据此前A源/本机3/14,880张差异记录为数值迁移保留小幅容差。
没有独立val选模或full训练内指标，没有平台分前不替换现役或声明达到80%。

## 当前检查点

2026-10-02准备时，本机已有12:11 CST监控快照：v2训练结束、controller在推理；
本地完整交付目录尚不存在。该快照不是本段重新连接服务器的实时证据。
实现验证和实际推理结果将在本记录追加；不能把准备完成称为候选交付。

独立分支`codex/v2_fixed_prior_20261002`，目录
`/home/lux1/noise/worktrees/v2_fixed_prior_20261002`，起点`7448855`。
每段完成后方案push，main按自动`pull --rebase --autostash`、合并、复核、push后停在检查点。

## 规则合规性

- Backbone: CLIP ViT-B/32；Pretrained weights: OpenAI official。
- External data: No；Cross-stage data/checkpoint reuse: No。
- Test data used for training/adaptation: 仅官方确认范围内的固定均衡bias统计拟合，无梯度或模型参数更新。
- Multi-model ensemble: No；四视图来自同一个full raw checkpoint。
- Manual cleaning required: No；Fully reproducible: Yes。
- Rule risk: 固定均衡bias依据用户转述官方答复，见[原确认](v1_test_prior_authorization_20261001.md)。

现役包保留：
`/home/lux1/noise-worktrees/v1_768_full_test_bias_20261001/outputs/codex/v1_768_full_test_bias_20261001/submission/submission.zip`，
SHA256 `0f83a2458199d524d3c70f19744fc7043497a531ba968d740ab6f15ebfaf4fca`。
37,444行、9项及包血缘核验见[现役验证](../results/v1_768_bias_platform_20261002/validation.json)。

## 已完成实现与本机成本检查

新增独立入口`scripts/run_v2_fixed_prior.py`、独立复算`scripts/verify_v2_fixed_prior.py`及固定JSON。
不修改原v2训练/推理实现。入口核对原冻结模型/预处理代码逐字节相等，原plan和checkpoint绑定，
当前阶段train/val/test清单与映射，下载SHA、raw终点，以及实际测试图逐文件SHA。
归档只读取指定普通文件，拒绝重复/软链接metadata；没有解压任意路径或远端连接。
回执在本机复制冻结，关机状态后续变化不改写此推理的源身份。

14项新增测试及74项相关现有回归合计**88 passed**，见[原日志](../results/v2_fixed_prior_20261002/tests.log)。
覆盖概率解码保持、独立float64拟合、破损/缺失交付在CUDA前拒绝、错映射/像素拒绝、
metadata重复和软链接、合成端到端双包与独立重放、禁止覆盖/强度改动及源raw数值迁移门。
测试的合成包不构成当前阶段候选交付。

真实S1完整checkpoint SHA核对后，以相同V2架构执行64张当前val图×原四视图，
4.330452876秒完成，CUDA峰值allocated546,075,648 bytes；逐图像素SHA也核对64/64通过。
粗线性估算完整测试四视图2,533.586秒（42.23分钟），未包含下载、完整像素校验与独立复算，
小批次数据加载开销/温度/图像尺寸不同会改变实际时间。未计算或使用S1探针准确率。
首次沙箱命令在读取nvidia-smi时退出、未创建探针目录；随后获准访问本机驱动完成以上一次探针。
见[实测成本](../results/v2_fixed_prior_20261002/cost.json)和[准备记录](../results/v2_fixed_prior_20261002/preparation.json)。

准备检查点完整下载回执仍不存在；完整full权重、全量原始重放、两份新包及平台效果均**未验证**。
本段没有新训练或test推理；本机已有监控快照作为上游状态来源存于
[快照](../results/v2_fixed_prior_20261002/upstream_monitor_snapshot.json)，不是本段实时远端查询。

以下命令在本方案worktree执行。`check`只检查本地已下载文件，缺少回执明确返回waiting；
`run`在全部输入门通过后才访问本机CUDA，只执行一次，已有output时拒绝覆盖。

```bash
export PYTHONPATH=reproducibility/aegis_f1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python3 scripts/run_v2_fixed_prior.py \
  --config configs/v2_fixed_prior_20261002.json \
  --output outputs/codex/v2_fixed_prior_20261002/candidate --action check

python3 -u scripts/run_v2_fixed_prior.py \
  --config configs/v2_fixed_prior_20261002.json \
  --output outputs/codex/v2_fixed_prior_20261002/candidate --action run

python3 -u scripts/run_v2_fixed_prior.py \
  --config configs/v2_fixed_prior_20261002.json \
  --output outputs/codex/v2_fixed_prior_20261002/cost_probe --action probe \
  --probe-parent /home/lux1/noise/worktrees/v2_continuation_20261001/outputs/codex/v2_continuation_20261001/handoff/v2_s1_384_bundle_20261001/run/best.pt \
  --probe-parent-sha256 3643b9a10a8cf3b330db6537a4b1d9c4cc630a78649099fce9f81b7f1538b817

python3 -m pytest tests/test_v2_fixed_prior.py tests/test_v2.py \
  tests/test_v2_swa.py tests/test_v2_continuation.py tests/test_v2_completion_delivery.py \
  reproducibility/aegis_f1/tests/test_v1_preprojection_test_bias.py -q
git diff --check
```

上方probe已执行，原输出保留，不重复运行同目录；正式`run`尚未执行。
执行成功后两包位于`candidate/submission_{raw,bias}/`，
`candidate/independent_verification.json`应为passed，`candidate/report.json`应为completed_verified_delivery；
这些路径目前是预期输出，不能作为已有产物引用。后续独立重放使用：

```bash
python3 scripts/verify_v2_fixed_prior.py --config configs/v2_fixed_prior_20261002.json \
  --output /home/lux1/noise/worktrees/v2_fixed_prior_20261002/outputs/codex/v2_fixed_prior_20261002/candidate \
  --report /tmp/v2_fixed_prior_independent_recheck.json
```
