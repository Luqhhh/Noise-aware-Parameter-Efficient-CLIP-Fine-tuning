# 当前执行计划（2026-09-20）

## 当前入口：PRELIM75_V10_SAM_20260920（S0=69.3195%，S1 待平台）

v10 已按原预注册接入为 L1 上普通 AdamW 与标准非自适应 SAM 的严格配对。固定 `rho=0.05`、sample epoch 19/20/21、每候选 9,678 次更新、三组原 LR/WD、3-epoch cosine、原 w/Q、概率融合 GCE、2.0 anchor、V1 冻结框和最终十视图无 prior 推理；S1 两遍使用同一完整有效 batch，只用 g2 在精确恢复后的原参数上执行 AdamW。v9 平台结果没有改动协议。完整规格见 [v10 SAM 预注册](preliminary_75_v10_sam_preregistration_20260920.md) 与 [实施补充](preliminary_75_v10_sam_implementation_20260920.md)。

真实只读 preflight 已核对 L1 checkpoint、训练/诊断清单、回退 CSV/ZIP/manifest、500 类映射、trust、content groups、V1 boxes/paths/manifest 和原监督语义，103,218 行训练集对应每轮 3,226 次更新。首次 A0 在前向前发现 O3/PTA 训练态 dropout 并按守卫停止，未创建 optimizer 或读取 test；实施修复显式回放整批 CPU/CUDA RNG，使 SAM 两遍共享 dropout mask 且净消耗一次随机流，不关闭 dropout、不改 rho/loss/样本/更新数。修复后的 37 项 v10、79 项 v7/v9/v10 定向和完整 612 项 Aegis 测试通过；真实 A0、隔离 AdamW smoke 与共同输入检查随后通过。

用户要求在 S0 完成后暂停，因此正式 S0 普通 AdamW 控制组完成 3 轮、9,678 次更新后结束原自动队列。用户随后明确要求继续，S1 标准 SAM 从同一 L1 独立完成 3 轮、9,678 次更新；全部 SAM 步骤保持固定半径、精确参数恢复与 RNG 回放，S0/S1 common-control 通过。两候选均完成 10,316 行重叠诊断、无 prior 十视图测试推理和 24,967 行提交校验，桌面副本哈希一致。S0 ZIP SHA-256 为 `98c2a51bb720ec127d7f0d52cbba200a250460917a3dcd80ba2aca65c7fb17a6`；S1 ZIP SHA-256 为 `b92af8bfdbead9762b89f81e0bd54701e66885337777c0c0ab8c9b32823b6d01`，两者有 2,382 行预测不同。用户回填 S0 平台 **69.3195%**，比 L1 无 prior 69.2794% 高 **0.0401pp**，成为无 prior 新高但未达到 0.30pp 投入门槛。S1 平台结果待回填；此前不能判断 SAM 是否优于 S0，v10 尚未闭环。完整证据见 [v10 执行记录](../results/prelim75_v10_execution_20260920.md)。

## 已关闭：PRELIM75_V9_20260920

v9 固定冻结 L1 无 prior 单模型，只改变 attention native 框内 local 像素的取得方式。R0 从 native 224 PIL canvas 用 Pillow bilinear float-box 重采样，R1 将完全相同的框按真实 Resize(224)+CenterCrop(224) 整数几何逆映射到原始解码 RGB 后重采样；global tensor、global logits、attention、四个框、O3/PTA、四尺度及融合协议均不变。逐图 `source_gain<1.5` 时两候选严格回退原生 L1 local tensor，不拟合参数、不使用 prior、不读取测试批统计选择规则。

执行先做训练源 D0：按固定 SHA-256 顺序抽取 2,048 个唯一内容组的字典序 canonical 代表，仅解码尺寸；至少 205 组达到 `source_gain>=1.5` 才允许 GPU 阶段。随后依次执行冻结 smoke、10,316 行重叠工程诊断、完整 L1 基线重放和最多两个唯一预测集的 R0/R1 推理。完整测试重放必须与归档 L1 的 24,967 个 `(basename,label)` 全量一致，否则不交付新候选；结果不用于回改门槛、插值核、尺度或融合参数。代码入口为 `configs/prelim75_v9.yaml`、`aegis_clip/source_recrop.py`、`aegis_clip/prelim75_source_recrop.py`、`aegis_clip/cli/infer_prelim75_v9.py`、`scripts/run_prelim75_v9_queue.py` 和 `tests/test_prelim75_v9.py`。完整规格见 [v9 执行方案](preliminary_75_execution_v9_20260920.md)。

只读资产 preflight 已核对 L1 checkpoint、回退 CSV/ZIP/manifest、500 类映射、103,218 行训练清单、101,980 个内容组、10,316 行重叠诊断清单及 24,967 个测试 basename；L1 checkpoint 含完整 visual、shared head、O3、PTA，登记哈希匹配。D0 为 1,532/2,048，通过 205 门槛；冻结 smoke 通过。首轮完整重放因误用训练 `gpu_setup()` 关闭 cuDNN TF32，造成 32 行不一致并按守卫停止；固定 batch A/B 确认根因后，修复为归档通用 L1 数值协议。修复后完整 baseline 重放 24,967 行差异为 0，R0/R1 分别相对 L1 改变 423/1,783 行，两份唯一候选均通过独立提交校验并复制到 C 盘桌面。累计 GPU 动作按保守口径为 6,676.012 秒，低于 7,200 秒预算。

用户按 R0、R1 顺序回填平台 **69.1633% / 69.2033%**，分别比 L1 无 prior 69.2794% 低 **0.1161pp / 0.0761pp**；R1 比 R0 高 0.0400pp，但仍未超过 L1。按固定规则保留 L1 并关闭 v9，不派生门槛、核函数、尺度或融合扫描。执行记录见 [v9 执行记录](../results/prelim75_v9_execution_20260920.md)。

## 已关闭：PRELIM75_V8_20260919

用户给定 `PRELIM75_V8_20260919` 固定方案：停止自动短续训，固定 G0/L1，只从当前初赛训练图和原 w/Q 为各自拟合一个有界 500 维类别偏置；不读取测试批统计或旧测试拟合偏置。此前“复现暂时不考虑，从计划删除”的决定继续有效；本次来源核对不冒充完整上游谱系认证。

v8 固定 B0=G0、B1=L1，模型与 O3/PTA 全冻结。每个模型按既有四尺度＋Flip完整协议在 103,218 个训练样本上生成最终融合 FP32 `log(p)`，再以原监督构造类均衡拟合权重，用 CPU FP64 L-BFGS-B 拟合 `lambda=0.01`、边界 `±log(4)` 的类别常数，测试应用强度固定 0.90。两个模型各自拟合、各自绑定，不移植偏置、不融合预测、不扫描超参数。

代码、配置、队列和来源守卫已接入；修复后 38 项定向数值/绑定测试与全套 `540 passed, 1 skipped` 通过。真实来源审计、两次训练集冻结前向、两次 CPU 拟合与两个测试推理均已完成，B0/B1 提交包独立校验通过。用户回填 B0=**67.8816%**、B1=**62.0659%**：B0 比匹配 G0 无 prior 低 **1.3458pp**，B1 比匹配 L1 无 prior 低 **7.2135pp**。两候选均为负结果，故按预注册止损规则保留 L1 无 prior 69.2794% 作为默认非测试拟合路径并关闭 v8；不派生强度、正则、边界或求解预算扫描。原交付曾因 `repr(preprocess)` 的进程地址导致协议哈希失败，且在读取测试样本前停止；严格兼容入口确认除此地址外描述符完全一致，未修改原缓存或偏置。完整规格见 [v8 执行方案](preliminary_75_execution_v8_20260919.md)，数值与产物见 [v8 执行记录](../results/prelim75_v8_execution_20260919.md)。

已报告的 G0+legacy test-batch prior0.9 为 72.4677%，仍是不同来源协议的绝对最高；本轮不撤销该记录，也不把平台接受等同于规则许可。v8 的合规默认路径仅使用训练源冻结偏置，并继续保留正式推理的 test-fitted prior 拒绝检查。

## 当前入口：PRELIM75_V7_20260918（已闭环，保留 L1 无 prior 胜者）

v7 从 v6 G0（平台 69.2274%）独立初始化 L0/L1，只改变 fresh LambdaLR 的 cosine horizon：L0 使用原 18-epoch horizon，L1 在正式 3 epoch 内衰减到仓库原有 1% floor。其余全部沿用 v6 G0：原监督 w/q、融合 GCE（T=1.5、blend=0.5）、旧 V1 训练框、Anchor 2.0、三个参数组、effective batch/分母、sample epoch 16/17/18、固定末轮和最终四尺度＋Flip无 prior 单 checkpoint 推理。实际训练步数 K=3M=9,678；L0 的 H=58,068，L1 的 H=9,678；第一更新前 LR 相同，之后才因 horizon 不同而分离。

代码入口已接入：`configs/prelim75_v7.yaml`、`reproducibility/aegis_f1/aegis_clip/prelim75_cooldown.py`、`.../cli/train_prelim75_v7.py`、`scripts/run_prelim75_v7_queue.py`、`reproducibility/aegis_f1/tests/test_prelim75_v7.py`。CPU 调度对齐通过：L0 三轮边界 multiplier 为 1.0 / 0.9924798377 / 0.9701478473 / 0.9336825749；L1 为 1.0 / 0.7525 / 0.2575 / 0.01；floor 在各自 horizon 末端精确为 0.01。真实 G0 parent 的 metadata/diagnostic 只读 preflight 已通过，队列默认只读模式不创建 v7 输出。

2026-09-18/19 的真实执行已完成：L0/L1 各完成 3 epoch、诊断和 ZIP 包，common-control 通过。用户回传平台成绩 L0=69.0632%、L1=**69.2794%**；L1 比 G0 高 0.0520pp、比 L0 高 0.2162pp，按预注册规则成为当前无 prior 胜者。但 L1−max(G0,L0)=0.0520pp，未达到 0.30pp 投入回报门槛，因此 v7 关闭，不追加 G2、续训或参数扫描。另有一项按用户要求生成的侧包 G0+prior0.9（legacy test-batch balanced-prior，非预注册 L0/L1 无 prior 对照）获得平台 **72.4677%**，超过旧历史最高 70.352866% 2.114834pp，距 75% 还差 2.5323pp；该结果已单独登记，不能冒充 v7 正式无 prior 结论，也不自动派生新训练。

当前状态：v7 已闭环，正式无 prior 保留包为 L1，绝对最高包仍为协议不同的 G0+prior0.9；75% 目标未达到。完整闭环见 [执行记录](../results/prelim75_v7_execution_20260919.md) 与 [最终记录](../results/prelim75_v7_final_20260919.json)。本地提交、不 push。

## 上一轮：PRELIM75_V6_20260918 已完成闭环（保留 G0）

v6 固定从 v5 F1（平台 69.1393%）独立初始化 G0/G1，各训练 3 个 sample epoch：G0 使用 v3/V1 冻结训练框，G1 使用当前学生每次 global forward 最后一层 attention 生成的在线 local 训练框。两组共享同一个训练可用 global wrapper、原监督 w/q、v5 F1 的融合 GCE（T=1.5、blend=0.5）、Anchor 2.0、fresh AdamW/cosine 和最终四尺度＋Flip无 prior 单 checkpoint 推理。框选择本身停止梯度，local 分类梯度仍照常反传；未引入定位网络、教师概率、恢复掩码或测试数据。2026-09-18 实际执行已完成：D0 过门槛（1080/2048，阈值 205），共同 smoke 通过，G0/G1 各完成三轮固定末轮训练、诊断和提交包。用户回传平台成绩：G0 **69.2274%**、G1 **69.1993%**；G0 比 F1 高 0.0881pp，比 G1 高 0.0281pp。G1−max(F1,G0)=−0.0281pp，未达到 0.30pp 投入回报门槛，故保留 G0、关闭在线几何配方，不支持“当前学生在线框优于冻结 V1 框”的平台收益结论，不自动追加 G2。历史 prior0.90 的 70.352866% 仍是不同协议最高纪录，75% 目标未达到。

D0 固定检查 2,048 个内容组；本轮 `material_change_groups=1080`，达到 205 门槛。GPU 累计预算上限 28,800 秒；实际队列总耗时 9,333.45s。最多两个新平台候选，实际创建 G0/G1 两个包；未自动上传、push、追加 G2 或扫描参数。G0/G1 首 batch、Flip/尺度、soft targets、weights、global logits/features/anchor 的 common-control 全部对齐。完整平台反馈、诊断差值、哈希与决策见 [v6 执行与平台闭环](../results/prelim75_v6_execution_20260918.md) 和 [v6 final 记录](../results/prelim75_v6_final_20260918.json)。当前保留提交包为 G0：ZIP SHA-256 `efccb9f2b601a8b24830ef601739b166c214b8316b441bb9bc7d7aaf6bd4b634`。

实现入口保留在 `configs/prelim75_v6.yaml`、`reproducibility/aegis_f1/aegis_clip/prelim75_online_geometry.py`、`.../cli/train_prelim75_v6.py`、`scripts/run_prelim75_v6_queue.py`、`reproducibility/aegis_f1/tests/test_prelim75_v6.py`。CPU 合成集成测试覆盖 wrapper/attention/detach/hook/框几何/D0选样/空监督安全；完整 Aegis 测试通过。G0/G1 的 checkpoint、诊断、CSV/ZIP 和队列 final status 均保留在 `outputs/prelim75_v6_20260918/`。G1 桌面包仍保留，但当前胜者与保留提交包为 G0。

完整 v6 规格、梯度边界、预算与代码落点见 [v6 执行方案](preliminary_75_execution_v6_20260918.md)。

完整 v7 调度语义、代码落点和决策规则见 [v7 执行方案](preliminary_75_execution_v7_20260918.md)。

## 上一轮：PRELIM75_V5_20260917（已完成）

v5 固定从 v4 C0（平台 68.4544%）独立初始化 F0/F1，各训练 3 epoch。两组都将训练分类温度从旧 C0 的 1.0 固定为最终推理同源的 1.5；F0 保留 global/local 独立 GCE，F1 只增加 50% 的 global/local 概率融合 GCE。原监督、权重、V1 几何、优化器设置和最终四尺度＋Flip无prior推理保持一致。F1 与 F0 才是新增融合目标的同条件比较，F0 与 C0 的差值不能只归因于温度。

代码、配置、CLI 和固定队列已经接入。正式 GPU smoke、F0/F1 各三轮训练、重叠诊断及固定测试推理均已完成，两个无 prior 单模型提交包都通过 24,967 行格式检查。F0/F1 checkpoint SHA-256 分别为 `afd21e73442441fbb3b885c1081b34bde00306c910c3623c50e841ad3c7079f9`、`44e64a9528653f1d76db6a12b56a22c3ad23851665e8cef45d06f7ece9f52e1b`；ZIP SHA-256 分别为 `6bbe24731f2a83a5e4ba401e94fe8035944eb57d2bad04d23c5365c775a14a3a`、`0863b93c785042692ec928107429d679815b50d2b1971b4a984cc1bceb9a705d`。

F0 重叠 raw/clean-core 为 83.491665%/96.098757%，F1 为 83.559519%/96.166962%；两者均未触发 2pp 工程止损。两个包在 24,967 个测试预测中有 299 行不同。用户已回传真实平台成绩：F0 **69.0752%**、F1 **69.1393%**。F1 比同条件 F0 高 0.0641pp、比 C0 高 0.6849pp，故保留 F1 为当前无校准胜者；但相对 `max(C0,F0)` 只高 0.0641pp，未达到 0.30pp 投入回报门槛。F1 仍低于历史 prior0.90 最高 1.213566pp，距 75% 低 5.8607pp。v5 已关闭，不自动追加 F2、续训或推理扫描；精确正确数和上传时间保持 null。

执行中 F1 在首步 optimizer 更新前因梯度审计键名错误停止；提交 `2a96800` 将 `weighted["fusion"]` 修正为 `weighted["fusion_gce"]`，不改变损失或超参数。修复通过 18 项 v5、49 项 prelim75 和完整 489 项 Aegis 测试；F1 随后从 C0 独立重启并完成。两次用户暂停现场和失败现场在闭环前保留，未将半程权重用于候选；v5 关闭后已按用户要求清理。完整固定规格见 [v5 执行方案](preliminary_75_execution_v5_20260917.md)，实际命令、指标与哈希见 [v5 执行记录](../results/prelim75_v5_execution_20260918.md)。平台上传不是代理自动操作；v5 成绩闭环提交已按用户明确指令推送到 `origin/main`。不会自动启动 F2 或扫描 prior/温度/融合比例。

2026-09-18 存储清理已完成：删除旧完整恢复输出、合成复现输出、v2 特征缓存、明确落败 checkpoint、暂停/失败现场、已关闭恢复路线的大概率张量及桌面 v5 副本，共 `45,754,734,484` bytes。保留晋级链 V1→S0→C0→F1、当前 F1 checkpoint/仓库提交包、F0/F1 小型提交证据和 v3 几何框；未触碰桌面其他项目包。精确清单见 [存储清理记录](../results/prelim75_storage_cleanup_20260918.md)。

## 上一轮：PRELIM75_V4_20260916 已完成

v3 已完成两组真实平台回填：S0 为 **68.2982%**，S1 为 **67.2167%**。S0 比此前 V1 高 0.7050pp，S1 比同轮 S0 低 1.0815pp；因此保留 S0，关闭固定恢复蒸馏路线。完整数量、训练记录和最终决策见 [v3 执行记录](../results/prelim75_v3_execution_20260916.md) 与 [v3 最终记录](../results/prelim75_v3_final_20260916.json)。历史 prior0.90 最高仍为 70.352866%，没有被无校准 S0 刷新。

v4 已实现并完成 C0/C1 各 3 epoch 固定末轮训练与交付。C0 延续原 global/local GCE 与 global 同像素锚定；C1 只对原监督权重大于零、旧 clean_probability 至少 0.90、目标严格等于原标签一热、内容组无原标签冲突的样本，将 global 分类损失最多 25% 混为 CE。local 始终保持 GCE。CPU 支持检查得到 54,131 个唯一合格组、覆盖 497 类，故 C1 按原门槛运行。

权重父模型为 S0；训练框只读复用 v3 中由 V1 生成并登记哈希的几何，没有读取 v3 teacher 类别概率、恢复权重或恢复样本清单。两组均使用 sample epoch 7/8/9、fresh AdamW/cosine、batch 32、固定末轮。C0/C1 的重叠 raw 诊断分别为 83.2299%/83.2396%，clean-core 分别为 95.7441%/95.8123%，均未触发 2.0pp 工程止损；这些指标不作独立泛化结论。

两个无 prior 单模型包均覆盖 24,967 个测试文件并通过提交检查。用户回传 C0 **68.4544%**、C1 **68.4303%**；C0 比 S0 高 0.1562pp，比 C1 高 0.0241pp，因此保留 C0 为当前无校准胜者，关闭高信任 CE 变体。C1 虽比 S0 高 0.1321pp，但没有隔离出新增 CE 收益，也未达到 0.30pp 投入门槛。C0 仍比历史 prior0.90 最高低 1.898466pp，距 75% 低 6.5456pp；历史最高未刷新。本轮关闭，不启动 C2、β/阈值扫描或自动续训。完整记录见 [v4 执行记录](../results/prelim75_v4_execution_20260917.md) 与 [v4 最终记录](../results/prelim75_v4_final_20260917.json)。

## 已完成

- 共享分类头范数对齐固定实验完成：平台 63.3676%；匹配无校准父模型 66.7681%，该变体 rejected，不追加参数扫描。
- 缺失的七节点模型恢复完成，最终 Selftrain R1 单模型、单视图、无校准预测包生成并通过 24,967 图提交检查。
- 恢复包曾按要求放到桌面并完成平台回填；ZIP SHA-256 为 `e56b3129c0027ac37fc136d825e69c29de4bc0d1a65060d1763830ce7c5cb479`。该 61.4010% 负结果及哈希继续登记，桌面副本和旧完整恢复目录已在 2026-09-18 清理。

## 上一轮：已归档

用户已回传 RECOVERED_SELFTRAIN_R1_BARE_NOCAL_20260912 平台分数61.4010%，已登记，不晋级。低于无校准父模型66.7681%共5.3671pp，但模型组成和视图协议不同，不归因为单一机制。历史最高70.352866%及原包保留。

## 本轮：执行及四组平台回填完成，保留V1无校准胜者，75分未达到

用户已授权“按顺序执行”。main上完成固定H0/H1共享分类头重拟合和V0/V1视觉续训；完整方案见 [preliminary_75_execution_v2_20260912.md](preliminary_75_execution_v2_20260912.md)，实际命令、指标和哈希见 [执行记录](../results/prelim75_v2_execution_20260912.md)。103218行P十视图缓存初始logits重构差0。P/trust/清单/映射实际哈希匹配；历史两个包本轮复检通过。

评分同时保留两个参照：历史最高 **70.352866%**（完整P＋四尺度/Flip＋prior0.90）与同条件无校准直接对照 **66.7681%**。新训练从历史方案同一份完整checkpoint启动；超过无校准对照只记本轮工作协议改善，历史最好是否刷新另行比较。

| 候选 | 末轮loss | 重叠raw micro | 重叠clean-core micro | 实际平台 |
|---|---:|---:|---:|---:|
| H0 | 0.37411068 | 79.439706% | 92.333925% | 65.8629% |
| H1 | 0.37419679 | 79.652965% | 92.415768% | 66.1914% |
| V0 | 0.13968389 | 82.212096% | 94.816530% | 67.4090% |
| V1 | 0.18329918 | 81.785578% | 94.093573% | 67.5932% |

四组均固定末轮、使用相同原生224四尺度/Flip双Adapter无校准协议，实际候选attention重新计算；均完成24967行CSV/ZIP校验，包内外CSV字节相同。仓库提交证据位于 `outputs/prelim75_v2_20260912/{H0,H1,V0,V1}/submission/`；桌面同哈希副本只描述交付时状态，后续已清理。四组平台成绩均由用户真实回填，精确正确数与实际上传时间未提供；H0为65.8629%、H1为66.1914%，分别低于匹配P66.7681%共0.9052pp、0.5767pp，均不晋级，H路线关闭。旧10316行验证完全重叠，只作诊断，不能据表中本地指标确认平台提分或75目标。

H0/H1仅共享head训练，完整表示保持P不变；V0全局训练保留原局部分支，V1局部监督真实反传至视觉投影/O3/PTA，继续冻结conv1/位置编码，终态冻结参数检查通过。V0/V1均batch32，无OOM降档；global锚定来自同像素张量的冻结官方模型，local不使用整图锚定。源码、监督、配置和每组产物均记录哈希，完整上游谱系仍未认证。

已执行在线6epoch、缓存head20epoch，四个独立候选均交付并完成真实平台回填。C判定 `closed_below_combination_gate`，本轮取消：H0/H1均低于P，H路线关闭。V0为67.4090%，V1为67.5932%；V1比V0高0.1842pp，比同协议P高0.8251pp，选为本轮固定无校准协议的最终胜者。V1比历史70.352866%低2.759666pp，距75分仍差7.4068pp；历史最佳未刷新，75目标未达到。本轮按既定停止规则结束，不追加重拟合、续训、prior拟合或参数扫描。完整闭环见 [最终记录](../results/prelim75_v2_final_20260913.json)。只本地提交、不push，不操作平台账号。

## 已移出当前计划

完整上游谱系重建、为复现补齐的校准生产流程、独立环境重新训练及完整复现验收不再是本轮待办或初赛候选交付的阻塞项。它们未完成，未被认证通过；现有代码、来源限制和历史记录保留。未来重新启动需用户明确提出。

上一轮已选实验、交付和成绩回填完成；其候选未实现平台提分，不代表所有优化方向均已完成。H/V工程候选交付和四组平台回填已完成，H及C已关闭，V1为本轮无校准胜者；原验证重叠诊断与平台指标继续分开记录。
