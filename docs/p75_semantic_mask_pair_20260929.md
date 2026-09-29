# P75_SEMANTIC_MASK_PAIR：固定配对试跑

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
```

配置位于 `configs/p75_semantic_mask_pair_20260929/{control,masked}.json`；私有运行、日志及预算状态位于 `outputs/codex/p75_semantic_mask_pair_20260929/`。源码及输入哈希在该目录manifest.json，逐子任务命令、用时、返回码在status.json。

## 实测状态

启动前6项分类屏蔽测试通过，两臂当前阶段训练协议校验通过。此处尚无配对训练结果；完成后填写实测与交付信息，不能将准备检查写成训练收益。
