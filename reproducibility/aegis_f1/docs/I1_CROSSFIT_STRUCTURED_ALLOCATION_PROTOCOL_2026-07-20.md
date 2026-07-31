# I1：交叉拟合全局边际 + 局部一致性课程分配

日期：2026-07-20

## 方法动机

此前 A2 只删除 991 张，KTA 严格改标只有 601 张；但 OOF 与噪声标签冲突达 30.55%，替代类别又呈高熵扩散，说明问题既不能由极少数阈值样本解释，也不适合拟合单一 500×500 噪声转移矩阵。

I1 借鉴 SoLar 的 Sinkhorn 类别边际约束与 CSOT 的全局/局部结构和课程预算，但针对本赛题做离线、交叉拟合适配，不声称逐字复现：

- 全局证据是 I0 复建的 91,195×500 严格 OOF logits；
- 类别边际使用官方训练集原始噪声标签计数，而非均匀先验；
- 40,649 个三视图一致原标签锚点和 601 个严格 KTA 改标锚点固定（共 41,250 个），不参加重新分配；基础 GMM trust 中偶然达到 1.0、但不满足显式多视图定义的样本不得固定；
- 剩余样本以 log-space Sinkhorn 在原始类别边际下分配；
- 只有分配标签得到 OOF、kNN、prototype 至少两票一致且 flip 稳定时，才进入课程候选；
- 按分配类别独立选取剩余容量的 30%，避免头部类吞掉课程预算；
- 若控制标签或结构化有效标签的某一类少于 10 张，只从该类原标签的未选样本中按可靠度回填至 10 张；回填样本禁止改标；
- 训练子集以 CSV 实体过滤，MixUp 只发生在已选样本之间，未选标签不会通过混合目标泄漏。

论文与作者代码：

- SoLar：<https://proceedings.neurips.cc/paper_files/paper/2022/hash/357a0a771bf65ee07926d6af41b75030-Abstract-Conference.html>
- SoLar 官方代码：<https://github.com/hbzju/SoLar>
- CSOT：<https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b0da24d136f46bfaee78e8da907127e-Abstract-Conference.html>
- CSOT 官方代码：<https://github.com/changwxx/CSOT-for-LNL>

## 资源与方法边界

CSOT 作者实现声明需要 24GB+ GPU，当前设备只有 8GB；因此 I1 不复制其 24GB batch-wise GCG 求解器，而以完整 OOF 证据做一次全局凸 Sinkhorn，再以局部一致性过滤。SoLar 强调估计类先验；CSOT 论文也说明其原实验未处理类不均衡。因此这里不用每类均匀分配，避免把训练集中低支持类别强行补到平均数。

## 严格配对

- I1C：同一个 Sinkhorn 选中子集，保留原标签；
- I1S：同一个选中子集，仅将局部一致的全局分配标签作为硬修正；
- 两者使用同一 A2 epoch48 初始检查点、同一随机种子、训练轮数、LR、GCE 与 MixUp；
- I1S−I1C 衡量“结构化改标”本身，I1C−A2 衡量“结构化筛选”本身。

## 预注册门槛

分配器门槛：

1. Sinkhorn 最大行误差不超过 `1e-4`，最大列绝对误差不超过 `0.05` 张；
2. 500 个类均有选中样本，每类至少 10 张；
3. 所有改标样本必须局部两票一致且 flip 稳定；
4. 不依据验证标签、测试图像或平台分数选择预算、温度或迭代次数。

模型门槛：

1. I1S clean-core micro 相对 I1C 至少 +0.50pp；
2. I1S 相对未继续训练的 A2 初始点至少 +1.00pp；
3. I1S trusted macro 不低于 A2 初始点，raw micro 不得下降超过 0.50pp；
4. 任一条件失败则不扫温度、预算或 Sinkhorn 轮数，不生成平台包；
5. 通过后才重训全量单模型并考察合规 TTA。

## 合规

分配器只使用官方训练图像的冻结 CLIP 特征、固定 OOF 模型、kNN、prototype 和 KTA；不读取测试图像，不使用外部数据，不用平台反馈选择超参数。最终模型仍是官方 OpenAI CLIP ViT-B/32 加单个线性头，推理为单模型单路径。

## 冻结分配结果

以下结果来自唯一一次预注册正式分配（`budget=0.30`、`temperature=1.0`、`Sinkhorn=100`、`minimum_class_support=10`），后续不依据模型或平台结果重选这些参数：

- OOF 去重训练样本：91,195；显式固定锚点：41,250，其中原标签锚点 40,649、严格改标锚点 601；
- 其余不确定样本：49,945；局部一致候选：22,592；按类课程选中：14,170（含 6 个仅保留原标签的类下限回填）；
- 最终训练子集：55,420，占 OOF 去重训练集 60.7709%；结构化硬修正：6,153，占所选子集 11.1025%；
- 修正来源覆盖 492 类、目标覆盖 498 类，全部修正均通过局部两票一致和 flip 稳定门槛；
- 控制标签每类 19--177 张，结构化标签每类 10--181 张；
- Sinkhorn 最大行误差 `4.8876e-6`，最大列绝对误差 `7.6294e-5`，全部分配器门槛通过。

冻结产物 SHA-256：

- `selected_train.csv`: `ac81fece9682c8db907f7f42bd7eb58e288096790943161cfa01b024d769ff41`
- `allocation.csv`: `801f925addfcbf169e4b6e75fa2c7df4e2eb3fa7d217907582090b793d2ee87c`
- `control_trust.pt`: `dcebbc8813a700bc67f608e2ea068236301c0345734e708c37efb8b181779615`
- `structured_trust.pt`: `06ec810281f2a4093f20fa5171d1107f908b2c68c2ec75ddde4053d1daaae57f`
- `allocation_audit.json`: `0e2744002a730bd217806ed9f4bc45c30d694118fe6c6150d383eb56846db2d6`

## 冻结训练配置

I1C 与 I1S 均从平台已验证的 A2 epoch48 检查点开始，使用中心裁剪冻结特征、单个线性头、GCE (`q=0.5`) 和特征空间 MixUp (`alpha=0.2`, `p=0.2`)。两组固定训练 20 epoch、batch size 256、AdamW、峰值 LR `5e-4`、1 epoch warmup、cosine 衰减、weight decay `1e-4`；随机种子均为 42。选择指标为 `clean_core_micro`，并同时冻结记录 raw micro、trusted macro 及初始 A2 指标。两组唯一允许差异是 trust bundle：I1C 保留原标签，I1S 对同一子集的 6,153 张样本启用硬修正。

### 验证集审计修正

首轮 I1C/I1S 在训练结束后的路径审计中发现，误用了独立工程的随机验证表，其中 5,568/10,316 张与 I1 训练子集重叠；该轮 A2 初始 raw micro 异常达到 79.3040%，明显偏离 A2 原始未见验证集的 69.4439%。因此该轮训练整体标记为无效，不用于方法判断，也不据此调整任何超参数。

修正轮 I1C2/I1S2 仅将验证表替换为 A2 训练时真正保留、与 I1 子集路径重叠为 0 的 `outputs/data/d3_strict/seed42/val.csv`；训练子集、trust bundle、初始化、随机种子和全部优化超参数保持不变。训练器同时加入 canonical-path 防泄漏断言，未显式声明 full-fit 诊断模式时，任何训练/验证路径重叠都会直接失败。

## 严格门控结果

修正后的未见验证集完整复现 A2 epoch48 的 raw micro `69.4439%`。I1C2 最佳轮次为 epoch14，clean-core micro `82.5617%`、raw micro `69.3373%`；I1S2 最佳轮次为 epoch1，clean-core micro `82.5755%`、raw micro `69.4439%`。因此 I1S2 相对 I1C2 仅 `+0.0138pp`，相对 A2 初始点 `+0.0000pp`，未达到 `+0.50pp` 与 `+1.00pp` 两项主门槛。I1 按预注册协议停止，不扫描温度、课程预算或 Sinkhorn 轮数，不生成平台提交包。

- I1C2 `best.pt` SHA-256: `a8e73a7a1c6b1dbaed108dcb62ce662695affcc31c8de51130dc75f724adb271`
- I1S2 `best.pt` SHA-256: `779d6b963afb11309eb21c1994d2da3901705613438b678a2cbf41643316d51d`

失败诊断显示，6,153 个修正中 5,552 个来自不确定分配，且其中 99.6% 的分配标签等于 OOF top1；A2 原先删除的 991 张中有 929 张被 I1 重新纳入并改标。OOF、prototype 与 kNN 均建立在同一 CLIP 表征和噪声标签上，不能视为三份独立证据。I1 的失败符合自确认闭环：代理标签拟合增强，但未见可信指标不升。
