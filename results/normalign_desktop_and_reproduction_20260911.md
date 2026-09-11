# 范数对齐桌面候选与复现工程续段（2026-09-11）

源基线：1444d0638daea55fa95e928b0ece54fa4f772ef8，main。用户明确要求以官网为准、把范数对齐生成提交包到桌面，并继续工程工作。

## 桌面交付

已经生成并复制到 `C:\Users\lqh22\Desktop\submission.zip`（WSL: `/mnt/c/Users/lqh22/Desktop/submission.zip`）。复制前该文件不存在；复制后重新核对 SHA-256。未上传平台。

- 生成目录：`outputs/preliminary_no_sweep/20260911_normalign_desktop_r1/submission/`
- checkpoint SHA-256：`f4f9eeabcd02b583b4895188d743652f02a3f97b2550a0ef9895409cbcffb610`
- CSV SHA-256：`6cf3e8540564b5a7be105b61f18db5a5068daa31c53b3a441c3f8ff1c34a8129`
- ZIP/桌面文件 SHA-256：`98239d20ca2fde68b628755d73ce806091f171fb4632c84bad49f6f3257bdb81`
- 24,967 行；测试图覆盖完整；ZIP 只有 pred_results.csv，ZIP 内外 CSV 字节一致；提交检查退出码 0。

机制保持共享分类头平均范数对齐，仅 classifier.weight 改变；视觉塔、偏置、两个 Adapter 逐张量重载核验不变。移除了旧 optimizer/RNG/训练指标等恢复状态，产物明确 inference_only。

固定协议：四尺度 112/128/144/160，权重 .20/.30/.40/.10，top-k=5，local=.40，Flip=.50，温度 1.5，双 Adapter，batch 64，fp32，**无 prior 校准**。此包不继承历史 70.352866% 平台分数。原重叠诊断 raw 少对 146 张、clean-core 少对 77 张的 rejected 结论保留；本包属于用户明确指定的待测候选，不标 promoted。已登记到 submission_registry.csv，online_accuracy 与实际平台提交时间留空。

已有 CLI 实际执行命令（cwd: reproducibility/aegis_f1）：

```bash
PYTHONPATH=. python3 -m aegis_clip.cli.materialize_classifier_norm --checkpoint outputs/F1_FLAT_FULL_FT_R3MS/seed42/dual_adapters/best.pt --authorization ../../outputs/preliminary_no_sweep/20260911_normalign_desktop_r1/authorization.json --output-dir ../../outputs/preliminary_no_sweep/20260911_normalign_desktop_r1/model
PYTHONPATH=. python3 -m aegis_clip.cli.infer --checkpoint ../../outputs/preliminary_no_sweep/20260911_normalign_desktop_r1/model/candidate.pt --output-dir ../../outputs/preliminary_no_sweep/20260911_normalign_desktop_r1/submission --local-view attention_multiscale --local-crop-sizes 112,128,144,160 --local-scale-weights 0.20,0.30,0.40,0.10 --local-top-k 5 --local-weight 0.40 --local-temperature 1.5 --adapt-local-features --adapt-part-token-features --tta horizontal_flip --tta-fusion mean_probabilities --tta-temperature 1.5 --tta-view-weight 0.50 --acknowledge-local-view-risk --acknowledge-tta-risk --batch-size 64
```

## 官网规模与工程进度

用户已裁决以[官网赛题页](https://www.aicomp.cn/tracks/tracks-1/3714.html)为准。当前规模落盘 `configs/official_stage_sizes.json` 并更新规则文档：复赛 750/148695/37444，半决赛 500/90197/24912（类别/训练/测试）。旧 PDF 数字和旧 1500 类演练文档保留为历史记录，新增说明不再作为当前默认规模。

R2 新增薄入口 `aegis_clip.cli.reproduce_stage`：默认 audit-only；显式执行要求带来源的协议和预算；argv 数组调用既有 CLI；检查输出归属和冲突；目录锁；节点原子状态；package 源码/配置/输入/输出绑定；只复用已完成且匹配的节点。正式数据执行明确阻塞，完整 R3 间接影响审计尚未实现；不能把 synthetic 标签或 approved=true 当作正式来源证明。

旧 stage_pipeline 的 final_train 兼容为 prepare_final_train_csv，日志明确 model_training_performed=false；现在每个已完成步骤即原子保存记录。未重写训练循环。

真实软件演练：5 类 20 张合成训练图，使用官方初始 CLIP 编码器缓存，再调用现有训练 CLI 完成 1 epoch 并实际保存 checkpoint；没有读取历史已训练父模型/trust/Adapter。最终节点链状态 checks_passed，总用时 13.23 秒。再次 --execute --resume 核验源码、配置、输入输出后复用节点，无再次训练。早期工程验证运行保留在 executed/、executed_codebound/；代码增加绑定与冲突审计后在新的 executed_final/ 验证，均为软件修复演练，没有正式参数扫描。

```bash
# 仓库根目录；默认审计，不启动模型
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.reproduce_stage --manifest outputs/stage_readiness/20260911_reproduce_r1/recipe_final.json
# 显式合成演练及已完成节点复用
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.reproduce_stage --manifest outputs/stage_readiness/20260911_reproduce_r1/recipe_final.json --execute
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.reproduce_stage --manifest outputs/stage_readiness/20260911_reproduce_r1/recipe_final.json --execute --resume
```

按官网类别数的结构演练也已完成：750 类合成 9,112 训练/1,500 测试；500 类合成 6,112 训练/1,000 测试。两者使用 train-min=12、train-max=80、test-per-class=2、seed=42；prepare_stage 使用合成 val_ratio=.10 和实际合成样本数，均检查通过，不冒用官方总数量或正式划分政策。

## 验证与限制

根测试 414 passed，AEGIS 392 passed；最后一次 AEGIS 392 passed 已包含新增审计冲突检查。退出码全部 0；git diff --check 通过。模型物化测试含保存重载、父对象未修改、只有指定参数改变和拒绝无来源授权。复现测试含真实 CSV 节点、默认审计无副作用、输入/输出变动拒绝、并发锁、可捕获中断不标完成、遗留 final_train 的真实动作。

R2 仅完成审计优先入口及合成真实节点验证。完整阶段配方/生产者链、R3 开发与最终数据作用域、R4 监督账本、R5 校准绑定、R6 完整正式规模资源账本、R7 技术 PDF/材料包、R8 干净环境正式训练仍未完成。恢复承诺限已完成节点；失败或强制中断节点不能自动覆盖重启，不承诺任意 step 精确恢复。说明见 `reproducibility/release/README_engineering.md`。

本段本地 commit，不自动 push。正式数据、权重、合成产物和本地环境清单不纳入 Git；只共享代码、测试、汇总报告与候选哈希登记。
