# P75_SEMANTIC_MASK_PAIR：固定配对试跑

> **后续状态**：原E4在旧预算内关闭并交付诊断包；新授权E5/E6配对已完成，
> E6修正24、退化35、净−11，固定配方关闭，不晋级完整训练。
> 见[后续执行](p75_mask_e6_execution_20260929.md)。本文保留原E4协议与结果，
> 当前平台分、现役包和执行边界见[当前执行入口](current_execution_plan.md)。


本段承接已验收的[语义内容阶段A](p75_semantic_content_probe_20260929.md)，用户已明确确认具体CLIP文本用途合规并要求继续。唯一干预：自动命中的1,246张train_dev图全局及局部分类损失置零；原始标签、样本顺序、损失分母、feature anchor、其余图的监督均不变。没有新检测器、阈值或人工名单。

## 启动前冻结

- 两臂均由同阶段RM-LP父权重开始，SHA256 `d5cb8f5265754fd900d3efde23e24fefbcf616c747f2cab13e4dc2201fdd689b`。该分类头已经接触旧监督，不能将本轮称为全学习过程去除污染。
- 自动表SHA256 `39b2cd9e8d94d1fd114c7f0a7581297528ecae49a82d8d4ddf6f5e3d06bbf5de`；训练命中1,246，其余132,569；验证固定代理132，其余14,748。其余组不是干净真值。
- 官方OpenAI ViT-B/32，384px，seed42，batch4×累积256，有效batch1024；原16轮学习率日程截在6轮。前2轮CE，之后GCE(q=0.5)，第5轮开始原局部分支。保留原feature anchor权重2。
- 原L05训练源码锁定 `f050ecb59e0a885da0b43cdc6c8c316953fb5e7d`，在本段独立输出目录物化。复用现有分类屏蔽，仅加固定停止点和样本顺序摘要，不修改其他工作目录。
- 两臂依次到第4轮，然后分别从last.pt恢复至第6轮。恢复含模型、优化器、调度器、AMP、数据生成器及主进程RNG；DataLoader工作进程重启，因此不声称与不中断的旧L05逐位一致。两臂采用相同进程边界，逐轮核对样本顺序、实际更新数与学习率。
- **第6轮为主要判断，第4轮只作观察。** 不以早期学生与L05最终模型的差距判失败；L05只作为另外列出的现役参考。
- 从GPU任务开始累计子任务墙钟时间，训练/验证/出包合计上限21,600秒。预留1,800秒做固定验证与一个诊断提交包；每个两臂训练阶段第一臂最多用剩余训练时间的一半。超时终止本段自有进程，不碰其他任务，不追加预算；若仅两臂第4轮完成，明确标记主要实验未完成，不解释成机制无效。
- 解码固定为center+flip，mean_probabilities，T=1.4，prior strength=0.6；每个checkpoint各自从验证缓存拟合原流程prior，不用测试集适配。
- 为满足训练段可提交产物约定，预定输出masked最后共同完成轮次的**单checkpoint诊断包**，不以此自动占平台名额或宣布晋级。没有模型/多checkpoint集成。

## 判读

报告两臂同轮次全量原标签macro/micro、逐张修正/退化；按冻结命中及其余组分解，同时保留相对L05的描述性结果。训练期逐组记录实际视图py、sqrt(py)、pmax及屏蔽前分类损失，不能用最终推理快照代替历史。

仅命中组置信度下降不支持扩训；其余组出现可辨识的净改善才提供后续投入信号。全量含噪指标下降不自动否决，须检查下降是否集中于命中代理；代理改善也不等于平台泛化改善。不自动追加完整训练、配方或筛选强度。

## 复现命令

工作目录 `/home/lux1/noise-worktrees/p75_semantic_mask_pair_20260929`，分支 `codex/p75_semantic_mask_pair_20260929`。prepare拒绝覆盖，run拒绝重置已启动预算。

```bash
python3 scripts/run_p75_semantic_pair.py prepare
python3 -m pytest tests/test_p75_text_page_runtime.py -q
python3 scripts/run_p75_semantic_pair.py verify
python3 scripts/run_p75_semantic_pair.py run
# 可独立等待上述已启动任务；不执行GPU任务，不重置预算。
python3 scripts/finalize_p75_semantic_pair.py --expected-head <启动收尾器时的方案提交SHA>
```

配置位于 `configs/p75_semantic_mask_pair_20260929/{control,masked}.json`；私有运行、日志及预算状态位于 `outputs/codex/p75_semantic_mask_pair_20260929/`。源码及输入哈希在该目录manifest.json，逐子任务命令、用时、返回码在status.json。

## 实测状态

启动前6项分类屏蔽测试通过，两臂当前阶段训练协议校验通过；另3项超时清理、失败状态、分组计数检查通过。2026-09-29 12:50本机启动control；首20次优化更新无跳步、训练233.24秒。此处尚无配对验证结果，不能将准备检查写成训练收益。

长任务收尾器只在已有训练/缓存/出包状态完成后进行CPU报告与校验，不启动额外GPU作业。逐张报告、源哈希、配对顺序/更新/学习率审计和9项提交校验通过后，自动提交推送方案，再在main集成目录使用`git pull --rebase --autostash origin main`同步、合并、重新校验并推送；不执行手动stash pop。工作区出现并发改动、校验失败或合并冲突时停止自动收尾，不强推、不重跑。最终状态记录在`delivery_status.json`。

## 已验证结果

四轮配对评估完成，六轮主判断未完成。预算计费累计16261.78秒；无预算追加。

# P75 semantic masking pair

Original-label accuracy; selected and remaining groups are content proxies, not clean truth.

| Epoch | Proxy | Rows | Control macro | Masked macro | Control micro | Masked micro | Fixed | Damaged | Net |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | all | 14880 | 72.941563% | 72.941270% | 73.696237% | 73.696237% | 24 | 24 | 0 |
| 4 | selected_proxy | 132 | 17.619048% | 10.000000% | 20.454545% | 14.393939% | 2 | 10 | -8 |
| 4 | remaining_proxy | 14748 | 73.330574% | 73.388627% | 74.172769% | 74.227014% | 22 | 14 | 8 |

Slice macro averages only classes present in that slice.

Content groups are automatic proxies, not clean labels.
Same used validation split, not an independent test.
Shared RM-LP parent head already saw original supervision.
No automatic full-training or platform promotion.

局部分支后的完整配对未完成，不能用四轮结果否定机制。其余代理净修正8张；该数字仍是含噪标签代理结果，不自动晋级完整训练或占用平台名额。

本次训练调度器中断后，用户明确要求只做四轮评估与诊断出包。恢复时保留原状态，补记masked任务开始至恢复时刻的保守墙钟耗时（含空闲中断间隔）；原退出码未知，未伪造成功退出，未重置预算、未续训第5–6轮。恢复入口：`python3 scripts/close_p75_semantic_pair_e4.py`；先冻结同轮次配对报告再出包。

诊断单学生包：`/home/lux1/noise-worktrees/p75_semantic_mask_pair_20260929/outputs/codex/p75_semantic_mask_pair_20260929/deliveries/P75_SEMANTIC_MASKED_E4/pred_results.csv`、`/home/lux1/noise-worktrees/p75_semantic_mask_pair_20260929/outputs/codex/p75_semantic_mask_pair_20260929/deliveries/P75_SEMANTIC_MASKED_E4/submission.zip`。9/9校验通过；未上传平台。逐张预测、分组训练期统计、顺序/更新数审计、执行命令与用时均归档。报告中的相对L05结果仅为现役参考，主对照为同轮次control。

复核：`python3 scripts/verify_p75_semantic_pair.py --archive results/p75_semantic_mask_pair_20260929`。方案结果提交、main合并并重新校验推送后暂停，不自动扩训。
