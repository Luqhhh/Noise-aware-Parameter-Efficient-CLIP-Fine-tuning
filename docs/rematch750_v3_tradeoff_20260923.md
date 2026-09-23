# REMATCH750_V3：精度优先的耗时权衡

状态：**用户已于2026-09-23明确要求加入batch 1024并开始实施扫描**；先前的暂停执行指令已解除。精度目标仍为同NPU B32基准macro下降≤0.1pp；仍不使用NPU 0，只选当时空闲的设备。实施分支 `codex/rematch750_v3_scan`，独立目录 `/home/lux1/noise/worktrees/rematch750_v3_scan`。

实施进度：方案已增加batch1024补偿点、16轮动态审计和设备/数据绑定脚本。NPU1显存仅余约12.6/65.45GB，未选用；NPU2/6/7在一次只读探针中各余约65.07GB。正式训练和结果须以运行日志及审计为准；下表中的新增点位不是实测结果。

参考为V2同后端NPU batch32八轮：macro0.7190471887588501、micro0.7298387289047241、纯训练2670.459578s、完整CLI2870.871812s。精度可接受下界raw_macro=0.7180471887588501（71.80471887588501%）。一次达界只表示本次独立验证满足约束，不代表统计等价。

## 有界比较

完整运行 `configs/rematch750_v3_b64.yaml`（8/8轮）、`configs/rematch750_v3_b128_e16.yaml`（16/16轮）和新增 `configs/rematch750_v3_b1024_e16_lr4.yaml`（16/16轮）。B1024相对已完成的八轮点延长到16轮，同时head/visual LR各乘4（4e-4 / 1.2e-5）；这只是有界精度补偿假设，不保证收敛或精度等价。旧B1024八轮结果只作参照，不重跑。只有B64与B128均未达精度约束时，最多补一次B64_LR2：batch64、head LR2e-4、visual LR6e-6，其余不变。最多四次新完整训练；不扫描workers/prefetch，不恢复LoRA或关闭路线。

共同设置：910B2单卡、只选择当时空闲的非0号设备；workers16、prefetch2、pinned、fused AdamW、foreach norm、OMP4。CPU绑定须按所选卡实际拓扑确定，不沿用NPU0的144–191编号。同NPU LP SHA `d5cb8f5265754fd900d3efde23e24fefbcf616c747f2cab13e4dc2201fdd689b`，133815/14880同内容组划分；B64为8 epochs/cosine horizon8，B128与B1024补偿点为16/16；原head1e-4/visual3e-6（补偿点例外）、CE2轮后GCE q=.5（B64六轮，16轮点十四轮）、anchor2.0、shuffle、无LR warmup。每20步及轮末记录成功更新，每两轮验证（B64：2/4/6/8；16轮点：2/4/6/8/10/12/14/16），raw_macro主选模、同值raw_micro。

B64每轮2091步、八轮16728步；B128每轮1046步、十六轮16736步；B1024每轮131步、十六轮2096步。各自完整跑满预先确定的轮数，不能只用CE段或重建optimizer续训冒充。B1024仍只有B32成功更新数的1/16；它以较高学习率和更多轮数测试能否弥补，不把更新预算称为等价。各自独立目录，共享训练/验证数据、特征和初始化只读，Python依赖不修改。

## 选择与交付

执行、optimizer有限性、成功更新、checkpoint及逐类重载审计先通过；将已验证B32也纳入候选集合，在macro满足阈值的配置中选**完整CLI耗时最短**者，另列纯训练秒数、macro/micro差、固定尾部75类及其余675类。若新点均不满足精度约束，或完整耗时未优于基线，则保留已验证B32。不用吞吐短窗口估算代替完整耗时。即使16轮run选中更早checkpoint，仍计实际跑完整个run的成本；不能事后按所选epoch倒算节时。

用户的效率筛选与+0.20pp平台精度晋级门分开：满足≤0.1pp精度损失的最快新配置可生成工程交付CSV/ZIP并独立校验，标记为效率候选，不自动声称平台更优、不自动上传。若无新点满足约束，保留并复核原FT交付包。报告包含所有失败点，不只报告胜者。

## 隔离与重放

本机方案worktree `/home/lux1/noise-worktrees/rematch750_v3_tradeoff` 与实施worktree `/home/lux1/noise/worktrees/rematch750_v3_scan` 分开；远端实施源码目录 `/workspace/noise-worktrees/rematch750_v3_scan`。绑定脚本 `scripts/bind_rematch750_v3.py` 将本阶段原始数据/特征绝对路径指向 `/workspace/noise` 并校验同哈希NPU LP；输出在实施目录的 `outputs/codex/rematch750_v3_tradeoff/<candidate>/seed42/`，不覆盖原运行。

模板仍以 `train.device: npu:UNASSIGNED` 表示未绑定设备，不能直接运行。选定空闲物理卡后，单卡可见环境使用 `ASCEND_RT_VISIBLE_DEVICES=<物理卡号>`、配置内逻辑 `npu:0`；绑定脚本拒绝物理0号和错误的可见设备环境变量。执行前还须复核CPU拓扑和设备空闲状态。

`scripts/audit_rematch750_v2.py` 已改为按配置epochs和interval生成验证点与成功更新数；保留checkpoint有限性、optimizer步数、重载精确一致等审计标准。

训练不串接平台上传。阶段结束推送方案分支，再在main集成目录以自动模式 `git pull --rebase --autostash origin main` 同步、合并、重新校验、推送；不执行手动stash pop。

修订原则：不同epoch可以胜出；按真实完整耗时评价，不要求相同样本遍历预算。B64×8与B128×16的预期更新数接近（16728 vs16736），但后者图像遍历翻倍、CE/GCE更新分配不同，不能称优化过程等价。

## 方案摘要

| 点位 | batch | epochs / cosine | head / visual LR | CE / GCE轮数 | 预期总更新 | 定位 |
|---|---:|---:|---|---:|---:|---|
| 既有B32 | 32 | 8 / 8 | 1e-4 / 3e-6 | 2 / 6 | 33456 | 已测基准，不重跑 |
| B64 | 64 | 8 / 8 | 1e-4 / 3e-6 | 2 / 6 | 16728 | 中间batch参照 |
| B128_E16 | 128 | 16 / 16 | 1e-4 / 3e-6 | 2 / 14 | 16736 | 用更多epoch补偿较大batch |
| B1024_E16_LR4 | 1024 | 16 / 16 | 4e-4 / 1.2e-5 | 2 / 14 | 2096 | 大batch延长训练并提高LR的补偿点 |
| 条件B64_LR2 | 64 | 8 / 8 | 2e-4 / 6e-6 | 2 / 6 | 16728 | 仅前两点均精度不达标时的一个补偿点 |

后续可研究的“大batch前期、小batch收尾”暂不纳入本轮：它需要新增调度与恢复逻辑，也会扩大比较范围。本轮1024只测一个预先固定的延长训练与LR补偿点。

本轮已获执行授权。只有完整训练、审计、推理及提交校验完成后，才报告新增点位的实测精度与效率；配置解析通过不等于实验通过。
