# 复现入口工程说明

当前官方规模及来源在 `configs/official_stage_sizes.json`。750/500 类是当前复赛/半决赛口径；旧 1500/1000 类合成测试只用于软件边界覆盖。

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

合成链已在 `outputs/stage_readiness/20260911_reproduction_dependencies_r1/`
执行 features → train → infer，并生成经格式检查的 5 类、5 张预测包。
这是软件演练包，不能上传官方赛事或进入正式模型谱系。正式执行仍阻塞；
配置中隐式图像依赖、官方初始化权重绑定和完整生产链等检查仍需继续补齐。
