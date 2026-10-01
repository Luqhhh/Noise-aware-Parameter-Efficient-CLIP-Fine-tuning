# 三台本地机器的优化探索分工

用户已确认三台本地机器各自拉代码执行，租用的4090作为第四台继续跑现有v2；今晚两个平台名额暂不分配。本轮分工是两条固定四轮的新配对，以及一条已有权重的低成本出包任务。机器A/B的具体GPU与执行人尚未回填；较快的空闲GPU执行A，另一台执行B，本机4070 Laptop执行C。

用户随后明确要求不设计时间上限：本轮不配置墙钟截止、推理超时或按预计时长关闭路线；固定轮数、目标问题和质量止损保持。

本段是方案与交接准备。A/B尚需各自本地agent实现入口、预检后执行，不能把任务单当成已实现的训练命令或已跑结果。C已完成591.98秒推理与提交校验，实际结果记录在[LR512固定平均出包](lr512_swa_package_20261001.md)。所有新平台收益未知。

## 为什么优先探索这三项

现役full v1 SWA为70.98600576861446%，到75%尚差约1,503张净修正。这个数字来自总分换算，没有测试逐图真值，不能把验证集提升按比例加到平台分上。证据见[当前入口](current_execution_plan.md)和[误差预算纪律](p75_error_budget_policy_20260929.md)。

| 机器 | 固定任务 | 现有证据与本轮问题 | 固定工作量 |
|---|---|---|---|
| A，较快的本地GPU | 768头拟合后继续LoRA，与只训练头配对 | 官方冻结特征768 teacher净+247；当前DEV普通768头净+60。已经完成768 full，仍缺少共享监督下继续视觉学习的独立证据 | 两臂各4轮及完整评估、出包；不设时间上限 |
| B，另一台本地GPU | v1 LoRA弱增强与固定强增强配对 | WFT DEV净+87却平台低0.5502pp，尚不知原因。检验强增强是否改善低训练相似度代理与主要混淆，而非假定来源漂移已经成立 | 两臂各4轮及分组、评估、出包；不设时间上限 |
| C，本机4070 Laptop | LR512 DEV既定EMA2–4单checkpoint推理 | 已有固定平均权重，DEV修正324/退化214/净110；现有交付主要是末轮EMA。先检验这份已有候选，训练成本为0 | 一次推理与完整校验；已完成 |

A的思路受到[LP后再微调的研究](https://arxiv.org/abs/2202.10054)启发：先拟合分类头可能减少同时改变读出与视觉特征造成的不稳定。这是研究假设；论文没有证明我们这份LoRA、噪声数据或平台会受益。它也不是重复已完成的全量768轨迹。

B保持现有监督、架构、采样与学习率，只比较两个预先固定的增强配方。它与已关闭的L05 MixStyle/Fourier/RSC、已失败的WFT全参续训、v3监督替换不同。本轮不改标签、不撤销额外监督、不做CutMix/强度组合扫描，不依赖v2结束。

C交付后停在检查点，不为了让三台GPU都训练而派生第三条高成本路线。LR512_FULL仍未启动，原门禁不变；NDCW、SAM、监督阶梯与旧TTA/先验扫描保持当前冻结状态。

## 各机如何接手

各机先fetch、检查全部分支和最新main，再从最新origin/main建独立worktree。将下面对应任务单交给该机Codex或Claude执行即可；任务单已包含本轮用户授权、固定对照、实现要求和停止条件。

- 机器A：[768头后继续LoRA任务单](team_exploration_20261001/machine_a.md)，协议[JSON](../configs/team_exploration_20261001/machine_a.json)。
- 机器B：[强增强配对任务单](team_exploration_20261001/machine_b.md)，协议[JSON](../configs/team_exploration_20261001/machine_b.json)。
- 机器C：[LR512固定平均出包任务单](team_exploration_20261001/machine_c.md)，协议[JSON](../configs/team_exploration_20261001/machine_c.json)。

JSON是给实现者使用的冻结协议，**不是现有v1/v2 CLI能直接接收的配置**。A/B不能以修改旧入口的固定门禁来开新路线；新增独立入口，复用已验证组件。8GB显存为默认试算情境，micro8、逻辑batch32、worker2；实际成本以各机真实探针为准。不能根据未确认的GPU型号承诺耗时。

## 权重交接

Git只交付方案与代码，不会带上大部分checkpoint。已准备一个白名单交接包，包含DEV512父权重、普通DEV768头父权重、train_dev目标、LR512固定平均权重、原sidecar、相关配置及历史代码。包不含图片、官方CLIP权重、冻结特征大缓存或任何连接凭据；各机复用自己已核验的20260921官方数据和官方权重。

源清单：[asset_sources.json](../configs/team_exploration_20261001/asset_sources.json)。在本机重新导出及验包：

```bash
python3 scripts/export_team_exploration_assets.py export --output /absolute/new/path/team_exploration_assets.zip
python3 scripts/export_team_exploration_assets.py verify --bundle /absolute/new/path/team_exploration_assets.zip
```

队友将已验ZIP放进自己的共享目录后，先运行verify再解压到独立目录。路径迁移只建立新location map，保留原checkpoint/plan/config字节与摘要；读取旧artifact时使用它自己的binding，另外绑定本机数据、官方初始化和新实验。不能改写旧sidecar或跳过SHA检查来适配路径。A/B完成前向重放后才能开训。原历史源码只用于重建父模型或复现，不能恢复旧任务。

ZIP的实际路径、摘要、资产核验与交付检查记录在[本段核验JSON](../results/three_machine_exploration_20261001/validation.json)。由用户自行转给另两台机器；本会话没有向队友发送消息或在他们的机器启动进程。

## 统一评价和投入边界

A/B只训练133,815张train_dev，共享14,880张内容组隔离val_dev。禁止使用已经看过val的full768/full v1作为DEV父模型。固定完整验证集448中心视图、无bias，报告macro/micro、总体与目标组修正/退化、尾75类和非目标组。额外报告同一checkpoint的固定448/512/576及flip六视图；不要用两种视图挑选训练轮次。主候选在开训前固定为EMA2–4算术平均，末轮raw/EMA只作轨迹诊断。

两条路线均先落盘零更新父预测，固定样本顺序和独立随机数流，进行5次真实逻辑更新的探针；记录显存、吞吐，以及完整验证、存盘和两臂最终预测/校验的预计成本，供比较投入使用，不因预计或实际运行时长中止任务。探针权重丢弃，正式两臂重新从同一原父权重开始。

下面是本次**待复核门**，不是平台提分承诺或完整训练启动许可：主候选相对配对对照及未训练父模型均净+75张以上，预注册目标组相对对照净+25以上，全量修正/退化比至少1.25，目标组确有足够错误且退化没有集中在其他大组。轨迹、零更新复现和完整产物都必须通过。达到条件记录supports_review，未达关闭固定配方；不改阈值补跑。完整训练需要结合目标规模、实际成本和平台迁移证据重新判断，不自动排入队列。

训练中有数值错误、资产错配或全量micro相对父模型下降超过2pp则停止并交付失败证据；中止段不冒充完成候选。正常四轮完成的两臂各交付单checkpoint、CSV/ZIP、9项校验和逐图比较，不能融合两臂。完整训练收益与平台收益都未知。

## 今晚平台名额

用户明确今晚再决定两个名额归属，本计划不替用户上传或占号。优先把已交付的768 full raw、768 full固定测试均衡bias、512原模型固定bias、CRT768与新LR512固定平均包列成候选清单，保留各自训练人口和解码差异。若今晚重点回答768 full或测试bias哪个因素有效，应选择能回答该问题的配对；不能把DEV包与FULL包的分差归因于单一因素，不能依据预测类别数更均衡就优先上传。

现役包继续为`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/submission.zip`，SHA256 `1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`；37,444行、9项通过的既有依据见[现役校验](../results/v1_full_swa_platform_20261001/submission_check.log)。本规划段不声称产生A/B训练结果。
