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
