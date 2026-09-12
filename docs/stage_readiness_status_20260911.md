# 阶段工程准备进度（2026-09-11）

> 当前执行范围已由用户调整：复现任务移出本轮计划。以下保留为历史事实与未完成项记录，不再据此自动启动复现。当前待办见 [current_execution_plan.md](current_execution_plan.md)。

> 本日后续更新：用户已明确“以官网为准”。当前规模已落盘到
> `configs/official_stage_sizes.json` 并更新规则文件：复赛 750 类、半决赛 500 类。
> 以下冲突段落保留为发现过程，来源裁决已完成，不再等待用户选用哪个版本。
> 本日又新增 `aegis_clip.cli.reproduce_stage` 的审计优先入口，已完成实际合成
> 特征缓存/训练节点及已完成节点复用验证；正式数据执行仍被阻塞，完整 R2/R3 尚未验收。

方案二已由用户在本轮消息提供。用户同时授权 Agent 代选**初赛**机制，具体登记在 `.planning/2026-09-11-preliminary-norm/decision.json`；这不代表复赛/半决赛配方已选定，也不授权自动推送或上传。

本段实际落地：

- R0：新增 `scripts/audit_stage_assets.py`，读取声明清单，仅盘点实际本地字节。输出 `assets_inventory.json`、`missing_assets.csv`。缺失文件不复制文档哈希；编码、学习、选择三种 group 来源字段独立保留。存在文件但缺少生成命令时仍为 `missing_producer`，文件哈希通过不等于来源审核通过。
- R1：effective-number 使用等价公式 `(1-beta) / -expm1(counts * log(beta))`，修复 Python float 上调用 `.pow` 的错误；避免 beta 接近 1 时 float32 幂/相减失去精度。原有类均值和样本均值两种归一化语义均保持。测试比较 60 位 Decimal 数学参考，并覆盖非法 beta、非有限/零/负计数及 CUDA 设备一致性。没有启用任何长尾训练组合。
- 先前 DataLoader 和浮点断言修复已在 `c0d25cd` 中完成。

实际命令（从仓库根目录）：

```bash
python3 scripts/audit_stage_assets.py --manifest outputs/stage_readiness/20260911_r0_r1/declared_assets.json --output-dir outputs/stage_readiness/20260911_r0_r1/inventory
```

清单中的正式资产路径由运行时声明提供，脚本不下载权重、不修改输入、不重建软链接、不把未知来源补为已验证。输出目录必须新建。当前 13 项历史资产均能读取，但生产命令和完整间接 group 影响尚未重建；R0 的跨机器恢复清单仍不完整。环境和审计产物仅留本地忽略目录。

## 新发现的规则来源冲突

2026-09-11 实际读取的[官网赛题页](https://www.aicomp.cn/tracks/tracks-1/3714.html)列出复赛 750 类、148,695 训练、37,444 测试；半决赛 500 类、90,197 训练、24,912 测试。用户计划/PDF来源记录分别为 1500/297282/74896 与 1000/180274/49857。保留两者，暂不修改代码常量或宣称哪份数据已经到位；实际官方清单与当前通知需核实。PDF 字节本轮未独立读取，官网提交材料段落的第二次抓取超时，不能补写为已复核。

## 尚未完成

R2 真实节点复现与恢复、R3 间接谱系/开发隔离、R4 类级真实监督账本、R5 校准来源绑定、R6 1000/1500 类新演练、R7 完整技术材料/PDF/复现包、R8 干净环境完整训练均未在本段完成。当前数据规模冲突须先登记处理，不能把旧规模演练当成当前官方规模验收。

原验证集全部参与父模型训练，仍是重叠诊断。新的分类器实验两臂均不使用 prior，不继承历史 pa0.90 成绩。工程测试通过和初赛诊断结果不代表上述阶段工作包已验收。


## R3 传递声明审计增量

新增 `lineage.audit_declared_scope_graph` 与只读 CLI `audit_scope_graph`，沿祖先传播
训练和模型选择接触，单独记录固定编码；检查缺祖先、循环、阶段/数据集/类别映射、
合成或 final_fit 产物的角色。组集合文件绑定实际 SHA-256。仍只检查声明一致性，
不能证明 producer/source 真实性，不解除正式 reproduce_stage 的阻塞。

实际本地审计在 `outputs/stage_readiness/20260911_scope_graph_r1/`：原验证集
10,198 个独立字节内容组全部位于全量训练范围，开发独立性审计 blocked；
上游 R3 教师/trust 生产链与完整编码接触仍未知。这里的计数是内容组，
对应原验证清单 10,316 个样本。没有声称排除了所有近重复图。
官方测试图像只为来源交集审计计算文件哈希，没有学习参数或使用预测统计拟合。
清单和内容组数组仅本地保存，不进入公开仓库。


## R2 依赖执行检查增量

复现节点新增生产者依赖祖先、实际 operation、重复参数、父谱系清单及推理模型输入检查。
合成 5 类链完成 features → 实际 train → infer → 提交检查，验证完成节点复用。
完整正式训练链仍未完成。

本次读取 `configs/f1_flat_full_ft.yaml` 的 10 个输入字段，9 个字段对应文件存在，
但 `features.tensor_path` 指向的 `artifacts/stages/preliminary/features/clip_vit_b32_openai/features.pt`
缺失（路径相对 AEGIS 根）。重复字段可能指向同一文件，因此不称为 10 个独立资产。
路径清单和 manifest 存在不能代替张量文件；预期生成入口为 `aegis_clip.cli.cache_features`，
恢复需另建输出目录并核对实际官方初始化、样本顺序和预处理，不承诺恢复相同序列化哈希。
初始化父模型和 trust 包存在，但完整生产来源尚未验证。冻结特征参考与教师伪标签输入分开记录。
私有清单见 `outputs/stage_readiness/20260911_reproduction_dependencies_r1/fullft_input_audit.json`。


## 2026-09-12 教师生产记录追溯

已从历史 `teacher_trust_audit.json` 和 trust 元数据恢复依赖：
`cvt_v1 → fullfit_r1_teacher_v1 → selftrain_r1_teacher_v2_relaxed → selftrain_r2_teacher_v3_multiscale`。
这 4 个 trust 文件存在。3 个教师增强节点的 9 条输入引用中，7 条实际哈希匹配，
2 条 checkpoint 引用缺失：`F1_FLAT_MLP_LORA_SELFTRAIN_R1_FP32/seed42/checkpoints/epoch_3.pt`
和 `F1_FLAT_MLP_LORA_FULLFIT_R1_FP32/seed42/checkpoints/epoch_3.pt`（相对 AEGIS outputs）。
最末级 R2 checkpoint 存在且哈希匹配；最末级多尺度教师 logits 缓存缺失。
此前已确认原始 CLIP 特征张量缺失。这些是实际资产缺口，不用其他 seed/epoch 填补。

生成阈值与视图来自已有审计和历史报告，未重新选择或扫描。根 cvt 生产过程、各父模型训练
及缓存恢复仍未验证，完整 R2/R3 尚未完成。核对清单和来源记录仅本地保存于
`outputs/stage_readiness/20260912_teacher_provenance_r1/`。
