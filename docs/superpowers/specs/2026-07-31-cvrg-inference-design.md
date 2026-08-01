# CVRG：跨视图可靠性门控推理设计

日期：2026-07-31
状态：设计已由用户逐节确认，等待书面规格复核

## 1. 摘要

CVRG（Cross-View Reliability Gating）用验证集监督学习一个轻量、测试时冻结的可靠性门控器，对单个 CLIP checkpoint 的四路推理结果进行逐样本动态融合。它复用现有原图全局、原图局部、翻转全局、翻转局部四次前向，不增加首版测试计算量。

CVRG 不学习新的 500 类分类头，也不在测试时更新参数。它只回答一个共享问题：给定当前图片的跨视图证据，某一路视图的 Top-1 预测有多大概率正确？

## 2. 目标

- 超越固定 M1+Flip 融合，而不是继续扫描 crop、top-k、temperature、local weight、flip weight 或 prior strength。
- 保持单 checkpoint、无模型集成、无测试时训练、无外部数据。
- 首版保持当前四次前向预算。
- 使用严格 OOF 交叉拟合证明增益不是验证集拟合假象。
- 与现有 balanced-prior strength 1.0 组合，保留当前 67.2007% 平台锚点。
- 产物可审计、可复现、失败时关闭而不生成测试包。

## 3. 非目标

- 不训练第二个图像分类模型。
- 不学习 500×500 混淆矩阵或类别专属门控器。
- 不使用测试集 kNN、图传播、伪标签或在线参数更新。
- 不重启 V1/V2 Sinkhorn transport，不调整已关闭方法的数值参数。
- 不在首版加入额外类别条件裁剪、遮挡或梯度反向传播。
- 不把 W060 与 W050 融合；每个候选始终只使用一个 checkpoint。

## 4. 基线与输入

主基线为同一 checkpoint 的固定四路 M1+Flip 概率融合，再应用 balanced-prior strength 1.0。

四路顺序固定为：

1. original_global
2. original_local
3. flipped_global
4. flipped_local

当前等效基线权重固定为：

w0 = [0.30, 0.20, 0.30, 0.20]

局部视图继续使用 crop160、top5；翻转权重继续为 0.50。CVRG 不改变这些视图的生成方式，只改变逐样本融合规则。

验证和测试缓存至少包含：

- 四路 float32 logits，形状 N×4×500；
- 四路归一化视觉特征，形状 N×4×D；
- 原图及翻转图的 final-block CLS-to-patch attention；
- 图片名、checkpoint SHA-256、split SHA-256、视图模式与特征模式；
- 验证缓存包含真实标签；测试缓存不得包含或推导标签。

## 5. 可靠性特征

特征必须类别无关、低维且顺序固定。首版允许以下特征。

### 5.1 单视图特征

每一路分别计算：

- 最大类别概率；
- 归一化预测熵；
- 归一化熵定义为 -sum(p log p) / log(500)；
- logit energy 定义为 -logsumexp(z)，使用未做 prior alignment 的当前视图 logits；
- Top-1 与 Top-2 概率差；
- Top-1 与 Top-5 累积概率；
- logit energy；
- logit L2 范数。

### 5.2 跨视图特征

- 每对视图的 Jensen-Shannon divergence；
- 每对视图的 Top-1 是否一致；
- 每对视图的 Top-5 Jaccard overlap；
- 四路一致预测数量；
- global/local 特征余弦相似度；
- original/flip 特征余弦相似度。

### 5.3 attention 几何特征

- mean-head attention entropy；
- top5 attention mass；
- 归一化裁剪中心 x、y；
- 原图与翻转图映射回同一坐标系后的中心距离；
- 裁剪是否接触图像边界。

### 5.4 明确禁止的特征

- 真实标签、类别 ID、图片路径或文件名编码；
- 原始 500 维 logits 直接作为门控输入；
- 测试集类别计数、测试邻居或测试伪标签；
- 平台分数派生特征。

连续特征只使用训练折统计量标准化。视图类型使用固定 one-hot 编码。
若训练折中的连续特征标准差小于 1e-12，则将缩放因子固定为 1，并在 manifest 标记该常量特征。

## 6. 门控模型与融合

对图片 x 的视图 v 定义二元目标：

t_v = 1[argmax p_v = y]

共享逻辑回归门控器输出：

r_v(x) = P(t_v = 1 | phi_v(x))

每个视图样本的特征包含本视图特征、跨视图上下文和视图类型。四路共享同一组系数，以避免 500 类和四个独立模型带来的过拟合。

动态权重为：

w_v(x) = softmax(log w0_v + logit r_v(x))

可靠性在进入 logit 前裁剪到 [1e-4, 1-1e-4]，仅用于数值稳定。最终概率采用算术混合：

p_fused(c | x) = sum_v w_v(x) p_v(c | x)

门控系数为零时必须精确恢复固定基线。融合之后应用现有 balanced-prior strength 1.0，再取 argmax。

首版不使用乘积专家、类别映射或第二阶段规则，以便把动态门控作为唯一因果变量。

## 7. 训练与交叉拟合

W060 和 W050 使用相同特征模式和流程，但分别拟合门控器，不共享系数，也不融合 checkpoint。

每个 checkpoint 执行确定性五折分层 OOF：

1. 按真实类别分层；同一图片的四个视图必须属于同一折。
2. 外层每轮用四折训练，一折预测。
3. 标准化统计只能由外层训练折计算。
4. L2 正则强度只允许在外层训练数据内部做三折分组交叉验证。
5. 固定候选为 C = 0.01、0.1、1.0，以 Brier score 选择；不得根据外层 OOF 结果修改候选。
若多个 C 的平均 Brier score 相同，则选择更小的 C，以偏向更强正则。
6. 逻辑回归使用未加类别重权的 binary log loss，保留正确概率的校准含义。
7. 聚合五个未见预测，生成覆盖全部验证集的 OOF 动态融合结果。
8. 对完整 OOF 融合 logits 统一应用 balanced-prior strength 1.0。
9. 与同 checkpoint、同缓存的固定融合 + prior 1.0 基线做配对比较。

只有方法门控通过后，才允许使用完整验证集重新计算标准化统计并拟合最终门控器。最终门控器在测试时完全冻结。
最终全量门控器的 C 使用完整验证集上的同一组确定性五折分组交叉验证选择，再在完整验证集上重拟合；候选集和平局规则保持不变。

## 8. 晋级门槛

W060 是主平台锚点，必须同时满足：

- OOF raw micro 至少 +0.20pp；
- OOF clean-core micro 至少 +0.20pp；
- 至少 4/5 个外层折的 raw micro 变化非负；
- 任一外层折 raw micro 不得低于 -0.10pp。

W050 是 checkpoint 鲁棒性控制，必须同时满足：

- OOF raw micro 变化为正；
- OOF clean-core micro 变化为正。

两个 checkpoint 都必须满足：

- trusted macro、proxy macro、raw macro、clean-core macro 均不得下降超过 0.05pp；
- raw 与 clean-core 的 wrong-to-correct 多于 correct-to-wrong；
- prior 后预测覆盖 500 类；
- 所有特征、可靠性、权重、概率和 logits 有限；
- 重复执行产生相同 OOF 预测与报告哈希。

任一门槛失败，CVRG 首版状态固定为 closed_no_test_inference。不得运行测试、生成 submission 或根据失败结果继续扫描特征、正则候选、融合公式或阈值。

## 9. 测试推理

只有门控通过后才执行：

1. 为通过审计的单 checkpoint 生成四路测试缓存。
2. 校验 checkpoint、特征模式、视图顺序和缓存 SHA-256。
3. 用最终验证集标准化统计生成可靠性特征。
4. 用冻结门控器得到逐样本四路权重。
5. 动态算术融合四路概率。
6. 应用 balanced-prior strength 1.0。
7. 生成 prediction CSV、submission ZIP 和 manifest。

W060 与 W050 若都生成候选，必须形成两个独立提交包和独立 manifest，不能进行跨 checkpoint 融合。

## 10. 失败保护

以下情况必须 fail closed：

- checkpoint SHA、split SHA、特征模式或视图顺序不匹配；
- 图片名缺失、重复或缓存行数不一致；
- 折间图片泄漏；
- 标准化统计来自验证全量或测试数据；
- 特征、系数、可靠性、权重、概率或 logits 出现 NaN/Inf；
- 权重为负或行和偏离 1；
- 门控器产物哈希不匹配；
- 测试阶段出现拟合、再标准化或结果依赖的参数修改。

不允许静默回退。只有显式关闭 CVRG 时才能运行原固定融合路径。

## 11. 组件边界

建议新增：

- aegis_clip/view_reliability.py：特征模式、可靠性门控、动态融合与序列化。
- aegis_clip/cli/cache_cvrg_views.py：生成验证/测试四路缓存。
- aegis_clip/cli/evaluate_cvrg_gate.py：五折 OOF、门槛判断和最终门控拟合。
- tests/test_view_reliability.py：核心单元与泄漏测试。

建议修改：

- aegis_clip/cli/infer.py：加载已审计门控器和缓存模式，写入扩展 manifest。
- pyproject.toml：注册缓存和评估命令。
- 结果文档与 submission registry：仅在门控通过后更新。

组件之间只通过显式缓存和序列化门控产物通信，不让训练代码直接依赖测试 loader。

## 12. 测试计划

### 12.1 单元测试

- 零系数门控逐字节复现固定融合；
- 权重非负且每行和为 1；
- 原图/翻转图交换时，对应特征、可靠性和权重同步交换；
- entropy、margin、JS divergence、Jaccard 和 attention 几何符合手算样例；
- 输入 NaN、Inf、错误形状和错误视图顺序全部拒绝；
- 特征模式中不存在标签、路径和类别 ID；
- 门控器保存后加载得到相同输出。

### 12.2 交叉拟合测试

- 同一图片四视图永不跨折；
- 每个 OOF 样本只由未见过它的门控器预测；
- 外层验证折不参与标准化和正则选择；
- 相同输入与 seed 产生相同折分配、系数和报告哈希；
- W060 与 W050 产物严格隔离。

### 12.3 集成与回归测试

- 缓存输出与在线四路提取逐元素一致；
- 关闭 CVRG 时现有推理输出不变；
- 动态融合后 prior alignment 与单独离线执行一致；
- manifest 正确记录所有哈希、模式和权重摘要；
- 最终提交通过 24,967 图片、500 类、无重复、ZIP 内容和标签格式审计。

## 13. 审计产物

每个 checkpoint 保存：

- view_cache_manifest.json；
- feature_schema.json；
- fold_assignment.csv；
- oof_predictions.pt；
- oof_evaluation.json；
- promotion_gate.json；
- final_gate.pt；
- final_gate_manifest.json。

manifest 至少记录：

- checkpoint、split、缓存、门控器和预测 SHA-256；
- 代码提交；
- 特征版本和视图顺序；
- 外层与内层折 seed；
- 正则选择结果；
- 权重均值、标准差、分位数和单视图最大权重；
- baseline 与 CVRG 的全部配对指标。

## 14. 风险与取舍

- 验证集与平台分布可能不同，因此使用两 checkpoint 方向一致、低维共享门控和严格 OOF。
- 四个视图高度相关，因此门控特征用于估计相对可靠性，不把它们宣称为独立证据。
- balanced-prior 仍是整批后处理；CVRG 本身保持逐样本、冻结、无测试自适应。
- 若动态门控只有极小 OOF 增益，按预注册门槛关闭，而不是靠平台反复试包。
- 若首版关闭，后续应另立规格研究候选类别反事实验证，不能在本规格内临时扩张范围。

## 15. 参考依据

- Shanmugam et al., Better Aggregation in Test-Time Augmentation, ICCV 2021.
- Le Coz et al., Confidence Calibration of Classifiers with Many Classes, NeurIPS 2024.
- Kull et al., Beyond Temperature Scaling: Dirichlet Calibration, NeurIPS 2019.
- Yang, SA-TTS: Stress-Aware Test-Time Scaling for Vision Models, CVPRW 2026.

## 16. 完成定义

本设计完成实施的条件是：

- 所有单元、交叉拟合、集成和回归测试通过；
- W060/W050 本地门控按预注册规则得到明确 pass 或 closed 结果；
- 只有 pass 才产生可审计测试包；
- 不改变现有固定推理的默认行为；
- 文档、结果表和 manifest 足以从固定缓存完整复现结论。
