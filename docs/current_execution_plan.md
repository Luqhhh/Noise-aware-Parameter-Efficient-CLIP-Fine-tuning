# 当前执行入口（2026-10-01）

当前最高用户报告平台分为 **full v1 SWA 70.98600576861446%**；v3固定两臂已完成并交付，
本地结果不支持推进监督阶梯。WFT448数值修复已核验并重启串行队列，LR512待串行，v2尚未启动训练。
本页只维护现行状态、下一步和证据入口；各段命令、配置、原始指标及暂停/恢复经过保留在执行记录。

## 最新授权本机串行执行：V1_CONTINUATIONS_SERIAL_20261001

用户明确要求“在本机串行进行剩下几个不依赖v2的实验”。队列为WFT448_DEV → LR512_DEV，
随后仅在各自完整DEV总体/分组/轨迹共同支持时执行对应固定4轮FULL。
最初服务在WFT第75个batch遇AMP缩放溢出停止；150项测试及128次真实更新已核验数值修复。
2026-10-01 02:37:08 CST修复后新服务已启动，重新从原v1 SWA执行4轮，保留旧基线/失败记录；见[恢复记录](continuations_serial_recovery_20261001.md)。
独立方案分支与worktree均已建立；使用当前v1 DEV/FULL SWA和各自监督，不依赖v2。
本机CUDA预检通过，固定micro8、逻辑batch32，FP32评估batch32；不改变监督、LR或固定导出窗口。
预先记录保守FULL投入条件；混合证据保留DEV包待复核，不以小幅原标签下降自动否定路线。
持久串行入口逐段完成预测包/9项校验、独立配对复算和Git集成，然后进入下一独立段。
已有计算任务时等本机资源，失败不自动重启；不启动v2、SWA窗口搜索、监督阶梯或平台上传。
当前最高用户报告平台分仍为70.9860%；本启动记录没有新识别分。
配置、精确命令、资源测量、FULL支持边界及实时状态路径见
[串行记录](continuations_serial_20261001.md)与[实际服务启动](continuations_serial_start_20261001.md)。各段最终结果以上方已交付检查点和controller为准；本节描述启动沿革，不能当作最终训练成绩。


## 当前阶段与平台结果

当前为 `20260921` 复赛数据：750类，148,695张官方训练图、37,444张测试图；
固定内容组隔离划分为133,815张train_dev / 14,880张val_dev。
数据说明见 [数据集记录](rematch_dataset_20260921.md)，硬约束见
[比赛规则](../COMPETITION_RULES_AGENT.md)。跨阶段只迁移代码与公式，重新构建任务资产。

| 候选 | 用户报告平台分 | 状态与证据 |
|---|---:|---|
| full v1 SWA4–12 EMA | **70.98600576861446%** | 当前最高；[平台反馈与包复核](v1_full_swa_20260930.md#用户平台反馈) |
| train_dev v1 SWA4–12 EMA | 70.4091% | 已交付；[固定SWA尝试](v1_swa_trial_20260930.md) |
| 原v1末轮EMA | 69.23405619057793% | 已交付；[v1训练](v1_training_20260929.md) |
| L05_T14_P060 | 66.94797564362783% | 历史同阶段参照；[平台探针记录](l05_prior_tta_two_slot_20260927.md) |

full v1 SWA比train_dev SWA高0.5769pp，比原v1高1.7519pp，比L05高4.0380pp。
按同一37,444张整体准确率推算兼容26,580张正确；75%需28,083张，仍差
**4.0140pp / 约1,503张净修正**。计数由报告分数换算，平台提交ID和实际上传包绑定未独立核验。
精确来源与未知项见 [平台复核JSON](../results/v1_full_swa_platform_20261001/artifact_verification.json)。

full训练纳入val_dev，其86–88%级本地指标仅为训练内诊断；不可与train_dev候选的独立留出分数
作泛化比较。平台增益只依据平台报告，本地排序不能替代平台排序。

## 当前方案状态与下一步

以下状态来自已经入库的核验结果；工作目录、分支存在或历史授权本身不构成新的训练结果。

| 方案 | 已验证状态 | 当前决定与执行记录 |
|---|---|---|
| v1 / train_dev SWA / full SWA | 均已训练、出包及校验 | 保留full SWA现役包；[v1策略](v1.md)、[full执行](v1_full_swa_20260930.md) |
| v2分辨率/LR阶梯 | `engineering_ready / not_started` | 尚无GPU训练结果；[v2策略与入口](v2.md) |
| V2_FULL_LAST3_SWA | 已实现full_576 RAW3–5固定CPU平均导出 | 原第5轮raw默认提交不变；缺快照拒绝导出、不重训补齐；[工程记录](v1_continuations_20261001.md#v2_full_last3_swa) |
| v3固定384、每臂2轮 | `completed_delivered`；平台分未知 | `no_support_for_ladder`，保持末轮EMA，不自动追加训练或改选raw；[最终交付](v3_final_delivery_20261001.md) |
| WFT448_DEV / LR512_DEV | 本机串行队列已启动；WFT正式训练通过原第75批失败点 | 固定4轮，逐段最终状态/指标以置顶交付检查点为准；[执行核验](continuations_serial_recovery_20261001.md) |
| WFT448_FULL / LR512_FULL | 条件队列；逐段状态见置顶交付检查点/实时controller | 各自须有完整DEV证据及支持分析，不能用已入训val_dev选模或拟合bias |
| SUPERVISION_LADDER | 条件候选 | 本轮v3没有配对推进信号，不自动执行 |
| NDCW_A0 | 审计完成；规模门关闭 | full影响248张、dev196张，小于744/670门槛，不启动N1；[A0实测](ndcw_a0_20260930.md) |
| P75_MASK_E6_EXTENSION A | 已完成两臂E6；净−11张 | `close_recipe_no_full_training`；[执行记录](p75_mask_e6_execution_20260929.md) |
| P75_SUPERVISION_PIPELINE B | 已获原授权，但缺绑定的验证flip特征，未启动 | 不重编码补缺，不把实现准备称为实测结果；[流水线记录](p75_supervision_pipeline_20260929.md) |
| P75_FULL_SAM | `stopped_by_user`；无完整候选验证/平台结果 | 保留产物，不自动重启或续训；[历史启动记录](p75_full_sam_execution_20260928.md) |

WFT448合并全部LoRA后全视觉塔低LR续训，LR512保留LoRA并使用实际512输入；
两条DEV路线均固定4轮、逻辑batch32、末轮EMA及固定EMA2–4导出，先落盘未训练父模型基线。
142项CPU测试、真实父权重/目标预检通过；WFT合并零更新FP32 logits误差0，LR实际257 tokens。
这属于工程证据，没有新识别成绩；详见 [工程核验](../results/v1_continuations_20261001/validation.json)。

最新用户授权已覆盖本机WFT448_DEV→LR512_DEV串行执行及有支持证据的固定FULL。
串行执行遵守原固定协议和资源边界，每段完成后出包、核验并封存；未经新证据支持不扩配方。
成熟包的原优先顺序及每日名额见新增策略记录；
v2尚未开训、v3已结束，不再作为等待中的在途任务。

## v3最终配对结果

2026-10-01 01:26:03 CST完成。original和v1_supervision各2,786次更新，初始化一致，
完整共享轨迹逐字节相等。固定末轮EMA在全部14,880张val_dev上的独立复算结果：

| 臂 | macro | micro |
|---|---:|---:|
| original | 53.0370% | 54.0995% |
| v1_supervision | 48.8111% | 49.8925% |
| 监督臂减对照 | −4.2259pp | −4.2070pp |

修正424、退化1,050、净−626；改判3,848张。尾部75类macro−3.8563pp、净−44张。
监督臂末轮raw为55.8867% / 56.8414%，只作诊断，冻结交付仍为EMA。
这次短配对没有全量或尾部正信号；原标签含噪且无平台回执，不能推断平台变化。
37,444行CSV/ZIP通过9项正式复核，ZIP内外CSV字节一致。
见 [最终执行及命令](v3_final_delivery_20261001.md)、
[配对结果](../results/v3_final_delivery_20261001/comparison.json)及
[独立校验](../results/v3_final_delivery_20261001/final_validation.json)。

此前第2轮后暂停已被用户“继续”覆盖；磁盘恢复只补齐已完成对照EMA导出，随后完成原监督臂。
[接力启动](v3_after_v1_20260930.md)、[暂停记录](v3_pause_epoch2_20260930.md)、
[恢复记录](v3_resume_checkpoint_20260930.md)保留当时经过，不再表示当前运行或暂停状态。
历史 `REMATCH750_V3` 与本v3分别管理，不恢复旧winner或NPU任务。

## 搜索纪律与冻结项

先读 [长期实验搜索原则](../CLAUDE.md#实验搜索与投入原则)和
[当前误差预算纪律](p75_error_budget_policy_20260929.md)。有限探索须有可观察问题、
可证伪假设、固定评估和成本/停止边界；完整训练须由目标问题与全量修正/退化证据支持。
不要求探索前已证明数千张净修正，不把低置信度、训练筛除量或普通local微涨当作机制证据。
取消疑似无目标监督时保留全量指标，并使用候选产生前冻结的内容代理解释；代理不是真值。

- L05邻域patch pooling/readout、dropout、regularization、TTA weight、temperature、prior、crop及同量级扫描继续冻结。
- P75_SUPPORTED_CE、旧三通道R1和R2未获得新的启动依据；旧R1撤销103,407张不确定样本监督的做法不作为默认方案。
- P75_TEXT_PAGE_MASK命中0，门禁关闭；语义代理命中1,246张，E4净0、后续E6净−11，均不支持自动完整训练。
- NDCW不降阈值凑数量，不派生Drop/Relabel；监督阶梯不因代码就绪自动启动。
- 已关闭旧轮次不自动补OOF、多seed、网格、续训或组合；历史预算、授权和待跑清单不自动继承。

P0支持代理联证、缓存证据拆分、200张有限视觉核验和语义污染三项统计已交付。
核验保留无法判断的标签状态，未证明污染主导平台差距；不继续拆同一缓存或扩人工样本。
证据见 [P0/旧重建记录](p75_supervision_rebuild_20260929.md)、
[缓存复核](p75_evidence_recheck_20260929.md)、[视觉核验](p75_visual_audit_20260929.md)、
[内容污染统计](p75_content_contamination_analysis_20260929.md)。

## 计算资源与执行边界

后续仅用本机CPU/CUDA，NPU已停用。不连接、探测、排队或恢复历史远端资源。
诊断优先已有缓存/CPU；训练和推理前只读核对本机GPU及已有任务，不抢占。
WFT448/LR512入口发现CUDA在用进程会拒绝启动；最新授权串行controller会等待本机资源，失败不自动重启。
硬约束仍为官方OpenAI CLIP ViT-B/32、当前阶段官方数据、单checkpoint和确定性推理；
测试集只读，不用测试预测分布调参。平台上传与真实结果回填由用户决定。

开工先fetch，检查全部分支、近期main与重叠方案；每条方案使用独立分支、worktree和输出目录。
已验证文档/方案立即提交并推送，在main集成目录pull、合并、复核后push，随后停在检查点。
详细协作流程见 [CLAUDE.md](../CLAUDE.md#协作约定)。

## 交付与历史入口

现役full v1 SWA包：
`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/{pred_results.csv,submission.zip}`。
桌面副本：`/mnt/c/Users/lqh22/Desktop/v1_full_swa_submission.zip`。
37,444行、9项通过、ZIP内外CSV一致；源ZIP/桌面ZIP SHA-256均为
`1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`。
见 [full最终验证](../results/v1_full_swa_20260930/final_validation.json)、
[最近正式校验](../results/v1_full_swa_platform_20261001/submission_check.log)及
[包摘要复核](../results/v1_full_swa_platform_20261001/artifact_verification.json)。

v3新包位于
`/home/lux1/noise/worktrees/v3_after_v1_20260930/outputs/codex/v3_after_v1_20260930/prepared_exif_recovery/submission/`；
路径、单EMA摘要与校验见 [v3最终交付](v3_final_delivery_20261001.md)。平台分未知，未替换现役full包。
纯文档/诊断段引用既有校验，不为凑新包启动训练。

- [本次文档更新与核验](documentation_refresh_20261001.md)：修改范围、证据、README不变与校验命令。
- [整理前完整入口快照](history/execution_plan_before_docs_refresh_20261001.md)：保留多轮暂停/恢复和授权沿革。
- [本机资源切换前的执行沿革](history/execution_plan_before_local_only_20260929.md)：早期初赛/复赛记录。
- [历史文档索引](history/README.md)：旧交接、方案与NPU实测；历史“当前”“待运行”不构成执行授权。
