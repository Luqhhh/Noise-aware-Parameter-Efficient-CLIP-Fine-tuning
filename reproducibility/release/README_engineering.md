# 复现入口工程说明

> 当前执行范围已由用户调整：复现任务移出本轮计划。以下保留为历史事实与未完成项记录，不再据此自动启动复现。当前待办见 [current_execution_plan.md](../../docs/current_execution_plan.md)。

当前官方规模及来源在 `configs/official_stage_sizes.json`。2026-09-15 组委会通知为缓解赛程周期限制，将复赛/半决赛类别数分别减半；750/500 类是当前口径，旧 1500/1000 类合成测试只用于软件边界覆盖。

新增入口从仓库根目录运行：

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.reproduce_stage --manifest "$RECIPE"
```

默认仅审计，不创建训练输出或启动模型。每个 manifest 必须明确 `schema_version: 1`、stage、dataset_id、fit_scope、output_root、protocol 和有序 nodes。protocol 需要 decision_source 和 max_nodes/max_wall_seconds 预算，不能只有 approved=true。

节点使用 argv 数组调用现有 CLI，cwd 相对仓库解析；config_path/config_sha256 绑定配置，input_artifacts 绑定路径/阶段/作用域/哈希，output_artifacts 必须位于独立输出根，depends_on 引用前面的节点。示例结构以本地实际运行的 `outputs/stage_readiness/20260911_reproduce_r1/recipe_final.json` 为准，该文件包含运行时本地路径，不纳入公开包。

显式执行合成演练：

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.reproduce_stage --manifest "$RECIPE" --execute
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.reproduce_stage --manifest "$RECIPE" --execute --resume
```

当前执行只接受 synthetic_dryrun。正式数据执行拒绝运行，直到完整 R3 间接谱系审计和正式协议接线完成；审核字段不是数据来源真实性的充分证明。没有采用任何正式阶段训练配方。

支持的恢复粒度是**已完成节点**：重新核对 package 源码、配置、输入、输出哈希后复用；失败/中断节点要求新的运行目录，不承诺任意训练 step 或 epoch 的无缝恢复。目录锁拒绝并发执行；每节点开始、结束与可捕获异常原子写状态；强制终止可能留下锁和 running 状态，需要人工核对进程，不能自动删锁重跑。时间预算限制子命令执行，不宣称已证明所有外部工作进程的恢复能力。

旧 stage_pipeline 的 final_train 仍兼容读取，但其实际动作记录为 prepare_final_train_csv / model_training_performed=false。生成 CSV 不能代替真实 train 节点或最终模型。

实际合成演练已从新输出目录完成公开初始化编码器特征缓存与现有训练 CLI 的 1 epoch 训练，并完成节点复用验证。这仅验证接口与执行链；它不是完整历史最佳谱系、正式模型精度或容器复现。

发布包、环境锁、正式配方、完整作用域/校准绑定、技术 PDF 与 R8 干净环境正式训练仍未完成。预测提交包只能含 pred_results.csv，不能混入这些材料。


## 传递作用域审计（仅声明一致性）

新增 `aegis_clip.cli.audit_scope_graph --manifest <scope_manifest.json>`，从
`reproducibility/aegis_f1` 执行。入口只读输入，stdout 输出报告；阻塞退出码为 2。
该入口不会启动训练，也不会解除 `reproduce_stage` 对正式配方的限制。

版本 1 manifest 必须包含 `stage`、`dataset_id`、`class_mapping_sha256`、
`target_id`、`nodes`、`evaluation_role`（`development_evaluation` 或
`overlap_diagnostic`），以及 `evaluation_groups`、`official_test_groups`。
后两项使用 `{ "path": "relative-group-set.json", "sha256": "actual file hash" }`；
文件内容是唯一内容组 ID 的 JSON 数组。相对路径基于 manifest 所在目录。

节点字段：`artifact_id`、`parent_artifact_ids`、`stage`、`dataset_id`、
`class_mapping_sha256`、`scope`、`producer_record`，以及三个明确区分的字段：
`learned_from_groups`、`selected_using_groups`、`encoded_groups`。这三个字段可用
上述文件绑定对象或内联 ID 数组；空数组表示明确没有接触，缺失/null 表示来源未知。
所有祖先都参与检查；仅编码不构成拟合，但编码器祖先的训练接触会继续传递。

报告统计重叠组数，保留污染节点的依赖路径。父节点缺失、循环、测试拟合、跨阶段、
类别映射错配、合成产物进入正式链、来源未知均阻塞。`final_fit` 不能用于认证开发独立性。
显式 `overlap_diagnostic` 可以记录已知验证重叠，仍禁止测试拟合。

**边界：**文件哈希只绑定声明内容；不会证明生产命令、阶段标记或内容组清单的真实性。
`source_authenticity_verified` 和 `authorizes_formal_execution` 始终为 false。
这不是完整 R3/R5 来源验证，不能把本入口 checks_passed 当成正式协议批准。


## 复现节点依赖检查补充

节点 `operation` 必须等于实际执行的 Python CLI；同一个命令选项不能重复，
包括 `--flag value` 与 `--flag=value` 混用。读取先前节点产物必须在 `depends_on`
的传递祖先中包含其生产者，单凭节点顺序不算声明依赖。

启用初始化谱系检查的训练必须绑定 `lineage.parent_train_csv` 和
`lineage.parent_val_csv`。特征提取必须绑定配置；推理必须显式绑定 checkpoint，
使用 prior 时也必须将其声明为输入。复现入口拒绝测试批内 prior 拟合参数。
这些补充不是 prior 来源真实性证明；R5 仍未完成。

合成链曾在 `outputs/stage_readiness/20260911_reproduction_dependencies_r1/`
执行 features → train → infer，并生成经格式检查的 5 类、5 张预测包；该本地演练目录已在 2026-09-18 按存储清理要求删除，命令、哈希与结论仍见 `results/reproduction_dependencies_20260911.md`。
这是软件演练包，不能上传官方赛事或进入正式模型谱系。正式执行仍阻塞；
配置中隐式图像依赖、官方初始化权重绑定和完整生产链等检查仍需继续补齐。


## 历史教师生成记录核对

仓库根运行 `python3 scripts/audit_teacher_producer.py --audit <teacher_audit.json>
--base-dir <AEGIS根目录> --output-dir <新目录>`。该入口只核对记录指定的实际字节，
生成参数完整的非执行命令模板；不填补缺失阈值，不覆盖历史 trust/cache。
缺资产或哈希不符返回 2。模板保留新输出路径与运行条件占位符，不能直接用于训练。
`bytes_checks_passed` 只表示所列文件匹配历史记录；不证明生产来源或作用域真实性。
目前支持历史明确登记的 attention_multiscale 分支，其他分支不会套用默认值。


## 2026-09-12 实现增量与使用边界

* `prepare_stage` 输出 split_diagnostics.json；不可切分类、容量不足或实际缺类时停止，已有输出拒绝覆盖。
* `assign_oof_folds(..., fit_scope="development_fit", allowed_fit_groups=...)` 只对训练行分折；显式检查验证内容组和已登记拟合范围。旧调用仍按 final_fit 处理。开发 trust 行过滤已接线；仍不能认证全部父模型来源独立。
* 配置 `diagnostics: {longtail: {enabled: true}}` 开启实际监督账本，可选 frequency_segments 必须显式给出每类 head/middle/tail。未登记分组返回空值。记录每 batch 分类损失实际分母，不含辅助损失或真实梯度范数。暂拒绝 trust_subspace 的非标准目标。原始类频、内容组/可信覆盖等全量 R4 字段仍需补充。
* `build_teacher_trust --feature-cache-dir <new-cache>` 允许显式使用重建缓存，检查特征协议和历史样本顺序；不修改 checkpoint。读取路径重定位不是历史数值等价证明。

正式 `infer` 拒绝测试批内 prior 拟合；离线历史 `align_logits_to_prior` 函数仍保留。版本 1 prior 不再直接进入该推理入口。版本 2 绑定 checkpoint、数据集、类别映射、推理参数指纹、来源审计及样本/组/logits 文件，检查实际训练目录、样本顺序和 logits 来源元数据。跨模型 prior 暂不支持。

版本 2 检查仍依赖经核对的来源审计声明；它不是对任意伪造审计的密码学认证，也不自动证明完整父模型谱系。现有声明图报告 source_authenticity_verified=false，不能直接升级后拿来解锁校准。自动生成可信审计和正式 fit 发布流程尚未闭合；不得手工把 false 改 true。无校准推理可继续使用。

当前合成演练未生成正式 prior。技术报告的可维护源文件在 preliminary/technical_report.md，PDF 初稿在本地恢复目录；正式阶段材料模板保持 pending，不能当作完成配方。


## 已授权历史资产恢复队列

本次历史恢复执行时独立登记在 `outputs/stage_readiness/20260912_full_rebuild_r1/recovery_plan.json`，执行记录见 `results/stage_readiness_recovery_20260912.md`。该 61.4010% 负结果对应的本地大输出已在 2026-09-18 清理；以下命令仅说明历史入口，不能直接读取已删登记目录。仓库根运行 `python3 scripts/run_historical_recovery_queue.py --run-dir <登记目录>` 默认审计；显式 `--execute` 等待已启动 E2 的最终训练清单，再顺序调用冻结源码中的现有训练 CLI。此专用队列只接受已登记的七节点、最多 73 epoch 历史恢复，父模型选择沿用原配置。它不解除通用 reproduce_stage 的正式来源限制，也不认证尚未闭合的 trust/cvt 上游。


## 开发 pipeline 的拟合行隔离

manifest 显式选择 `fit_scope: development_fit`，并提供 `allowed_fit_groups: {path, sha256}`。组文件相对路径按 manifest 所在目录解析，内容为唯一组 ID 数组，必须精确对应开发训练 CSV 的组集合。已有 train/val/content_groups 文件相互检查，训练与验证不得共享内容组。分组政策须事先登记，入口不替用户生成正式比例或缺类回退。

开发 folds 只读取训练行；oof 在读取缓存前拒绝包含额外验证行或标签错配的旧 assignments。trust 可读取逐图编码的全量缓存，但在任何学习统计之前按训练 CSV 顺序过滤特征、标签、路径和组，拒绝重复、缺失或标签错配缓存。输出记录 fit_scope 与 fitting_samples。开发链拒绝 final_train/prepare_final_train_csv。旧 manifest 默认 final_fit 保持兼容。

来源报告仍保持 source_authenticity_verified=false：范围过滤不能证明缓存编码器、父模型和历史声明的真实性，不能解除通用正式复现入口的来源限制。本次只运行软件测试，未选择或执行新的正式开发训练配方。
