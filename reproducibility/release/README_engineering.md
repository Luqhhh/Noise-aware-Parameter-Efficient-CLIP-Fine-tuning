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
