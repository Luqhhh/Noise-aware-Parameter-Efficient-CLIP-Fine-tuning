# SCOPE-K2（Spatially Coherent Pairwise Evidence）可执行任务书

> **后续实施要求：** 使用 superpowers:executing-plans 按本任务书执行；任何实现判断与本任务书冲突时先停下并报告，不得用 outer fold、test 或平台结果补做选择。

**状态：** 设计冻结，尚未实施

**基线提交：** <code>830de947e9c07738ddb69b9a5d274f0c2a6269e3</code>

**工作分支：** 直接使用 <code>main</code>，不新建分支

**目标：** 在不改动父模型训练、候选生成、crop/fusion/prior 的前提下，用固定 7×7 四邻接图衡量 Top-1 与 runner-up 的空间连贯候选对证据；只有通过预注册 conditional grouped nested OOF 晋级门后才生成 SCOPE 测试 submission。

**技术栈：** Python 3、PyTorch、NumPy、scikit-learn、现有 Aegis/FULLFT_DUAL 推理与 submission 工具。

---

## 0. 本任务书的约束优先级

本文件同时是设计规范、实施计划和验收清单。后续对话可直接从“任务 0”执行，不需要重新设计。

以下约束不可放宽：

1. 暂停 PACE-K2 正式运行、端到端评估、final refit、test decision、submission 和平台测试。PACE 只作为 SCOPE 的同折离线消融。
2. 不重复普通 LoRA、OOF 重加权/重标注、全局 prototype、静态 crop/fusion/prior 调参、generic routing。
3. SCOPE 不使用 class ID、pair ID、prototype、kNN、检索库、每类参数或每候选对参数。类别索引只可用于读取已有分类头的两行权重。
4. 父 checkpoint、父推理协议、balanced-prior 0.90、候选对、六个证据视图、图结构、视图权重、阈值策略和晋级门都必须在 outer 结果前冻结。
5. 只有全部晋级门通过后才允许创建 SCOPE test cache 或 SCOPE test submission。
6. 失败时不得重跑父 test 推理冒充回退；必须使用字节级哈希匹配的归档父 CSV/ZIP。
7. 当前只新增本规划文档，不单独 commit。实际实验完成后，本文件与代码、报告及终态 submission 一起做一次本地 commit。
8. 只 commit，不 push。用户自行推送。

---

## 1. 2026-08-19 只读审计快照

### 1.1 Git 身份与工作树

已执行 <code>git fetch --all --prune</code>；只清理了过期的 <code>origin/pr/1</code> 至 <code>origin/pr/14</code>，未执行 pull、rebase、stash、commit 或 push。

- 当前分支：<code>main</code>
- <code>HEAD</code>：<code>830de947e9c07738ddb69b9a5d274f0c2a6269e3</code>
- <code>origin/main</code>：同上
- ahead/behind：<code>0/0</code>
- 写本文档之前的工作树：21 个 tracked unstaged 修改、57 个 porcelain <code>??</code> 路径项、0 staged、0 added/deleted
- 工作树内含尚未提交的 PACE-K2 实现和大量历史文件，全部属于既有状态，必须原样保留

已扫描所有本地/远端 refs、最近 main 历史、提交信息、文件名与文件内容。未发现 SCOPE、spatially coherent pairwise evidence、antisymmetric patch-set verifier 或 patch masking/re-forward 的已提交实现。脏工作树只有一处规划性文字提及 counterfactual re-forward family，不构成实现重叠。<code>origin/r1-part-token-execution</code> 是相邻 PartToken 方向，但不是 SCOPE。

### 1.2 PACE-K2 的真实状态

PACE-K2 不能被描述为已完成：

- 已提交：协议、exact duplicate-group 工件准备、部分配置与测试。
- 脏工作树已有但未形成正式闭环：classifier-space patch evidence、cache contract、Pass A parent cache、Pass B evidence cache、部分 grouped crossfit 代码与测试。
- 尚未完成：正式 smoke、5×3 nested crossfit 的完整验证、同折评估、final refit、test decision、submission、checker 和平台运行。
- 旧 PACE 协议绑定的是旧 R2 parent、三尺度和 prior 0.85；不得篡改旧协议来承载 FULLFT_DUAL。
- 当前脏代码中的 <code>PACE_VIEW_WEIGHTS=(0.225,0.250,0.025,0.225,0.250,0.025)</code> 属于旧协议。SCOPE 下的 matched PACE 消融必须使用本文件冻结的新 FULLFT_DUAL 六视图权重，不能静默复用该常量。

---

## 2. 最新父模型身份、恢复与硬校验

### 2.1 唯一合法父模型

| 字段 | 冻结值 |
|---|---|
| 身份 | F1_FLAT_FULL_FT_R3MS + dual adapters |
| 平台最佳 | 70.352866%，17,565 / 24,967 |
| checkpoint 原记录路径 | /home/lux1/noise/reproducibility/aegis_f1/outputs/F1_FLAT_FULL_FT_R3MS/seed42/dual_adapters/best.pt |
| 实施时恢复路径 | reproducibility/aegis_f1/artifacts/external/scope_k2/fullft_dual_pa090/best.pt |
| checkpoint SHA-256 | f72b0104257f49d2667fe335553a861dd1dea947753feebdc7301b8890b48765 |
| 主报告 | results/fullft_dual_adapters_20260806.md |
| 训练模式 | FULLFT，peft_mode=full_finetune |
| adapters | dual O3 BN32 + PartToken BN64 |
| PartToken | top_patches=8，temperature=0.07 |
| local crops | 112、128、144、160 |
| local scale weights | 0.2、0.3、0.4、0.1 |
| local_weight | 0.4 |
| original/flip fusion | flip_weight=0.5 |
| global/local temperature | 1.5 / 1.5 |
| parent top_k | 5 |
| prior | balanced-prior strength=0.90，max_iterations=50，tolerance=1e-6，damping=0.5 |

当前 WSL 中该 checkpoint 缺失。实施者必须从用户认可的原始资产或备份恢复到上述新路径；不得用文件名相同、近似权重或另一个 FULLFT checkpoint 代替。若原记录路径和已知备份都不可用，立即停止，资产门失败。

恢复后先只做哈希，不加载数据、不创建 cache：

    cd /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning
    sha256sum reproducibility/aegis_f1/artifacts/external/scope_k2/fullft_dual_pa090/best.pt

输出必须逐字符等于：

    f72b0104257f49d2667fe335553a861dd1dea947753feebdc7301b8890b48765  reproducibility/aegis_f1/artifacts/external/scope_k2/fullft_dual_pa090/best.pt

CLI 还必须在每次正式运行前自行计算并校验 SHA-256；不能只相信配置或 manifest 中写的值。

### 2.2 clean-core trust 资产

clean-core 使用冻结 trust bundle：

| 字段 | 冻结值 |
|---|---|
| 文件名 | selftrain_r2_teacher_v3_multiscale.pt |
| 实施时恢复路径 | reproducibility/aegis_f1/artifacts/external/scope_k2/fullft_dual_pa090/selftrain_r2_teacher_v3_multiscale.pt |
| SHA-256 | 6868041cc7b995a3e8e557ae925d1d25160acf23af09202f46911ce92125b30f |
| clean-core 条件 | clean_probability ≥ 0.70 |
| 权威记录 | results/f1_flat_mlp_lora_selftrain_r3_multiscale_prep_20260805.md |

    cd /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning
    sha256sum reproducibility/aegis_f1/artifacts/external/scope_k2/fullft_dual_pa090/selftrain_r2_teacher_v3_multiscale.pt

哈希不符或资产缺失时，clean-core 晋级门不可计算，本实验直接判为“阻塞/不晋级”；不得删掉 clean-core 门继续。

### 2.3 split、class map 与 duplicate groups

必须校验并记录以下冻结身份：

| 工件 | SHA-256 |
|---|---|
| outputs/data/master_splits/seed42/train.csv | a726b8a3ca8bc5857136106aca80f01d557104d3661ef92ccedfb2c0ea087875 |
| outputs/data/master_splits/seed42/val.csv | 54a790b35f836cfba4c19cbb5fe38c4b1b37aab62cc9d477f9285496b2d5568e |
| outputs/data/master_splits/seed42/class_to_idx.json | 0141916c3eab0b8e9471671e398acbdcbbd497fea5fbe81ec9c5bb9331ce6d65 |
| outputs/data/master_splits/seed42/idx_to_class.json | 8a7e030b5126124035177bd81e71c415dfb87f9b1ae6fba228b31c2cb7527913 |
| outputs/data/master_splits/seed42/split_manifest.json | d8cf3e129ed2c73ff72cbda120deee5285c5bb8e6f10f579c3d0d1c556d3587c |
| reproducibility/aegis_f1/protocol_artifacts/pace_k2_r2_parttoken/content_groups.json | 41e2668e0fa5e10291051c14a9c75fc96096a69ae7d36d84d0e774270a99bb87 |

duplicate-group 工件是与父模型无关的 exact-byte lineage，可直接复用，不得重建或重新切 split。其冻结审计是 103,218 行、101,980 个 group、1,238 个 duplicate extras、train/val group overlap 为 0。当前没有冻结的 near-duplicate 分组；报告必须把“只防 exact-byte duplicate，未防感知近重复”列为限制。

### 2.4 精确父回退 submission

失败终态只认以下归档文件：

| 文件 | SHA-256 |
|---|---|
| outputs/delivery/fullft_dual_pa0.9/pred_results.csv | 790fabcfb57ada355bfdb2f732da5ea1e16d3c505cdbf77c558bccfe7112b16d |
| outputs/delivery/fullft_dual_pa0.9/submission.zip | 3d684c07027d905c3edf88e4ce88c3ef9f32a01a6304385600d3d0ced7af5251 |
| outputs/delivery/fullft_dual_pa0.9/manifest.json | 67f42552e0c39f25f314a716349f4c9984d9980c1eedee9da781cee36da45570 |

这些哈希已在本次只读审计中重新计算并匹配。manifest 内保留旧绝对 checkpoint 路径，不得为“看起来更新”而改写归档 CSV/ZIP。ZIP 生成包含时间戳，重新打包通常不具备字节确定性，因此失败回退必须复制或直接引用归档字节，不能重新生成一个语义相同 ZIP。

---

## 3. SCOPE-K2 冻结数学定义

### 3.1 父模型八个 local views 与 SCOPE 六个 evidence views

最新父模型实际有 original/flip × 4 scales = 8 个 local views。用户指定的“六个父视图”在本协议中冻结解释为“从该父模型产生的六个证据视图”，不是删改父模型：

- 父 Top-2、margin、最终预测和 prior：严格使用完整 8-local-view FULLFT_DUAL。
- SCOPE、matched PACE 和 no-topology：只读取 <code>128、144、160 × original/flip</code> 六个 7×7 patch grids。
- crop112 继续参与父 local aggregation 和候选生成，但不进入候选对 patch evidence。
- 证据 view order 固定为：
  1. original_128
  2. original_144
  3. original_160
  4. flipped_128
  5. flipped_144
  6. flipped_160
- 从父四尺度权重抽取 <code>(0.3,0.4,0.1)</code>，仅对证据协议归一化为 <code>(0.375,0.5,0.125)</code>；original/flip 各占 0.5。
- 最终六个权重固定为：

\[
\alpha=(0.1875,\ 0.25,\ 0.0625,\ 0.1875,\ 0.25,\ 0.0625),\qquad \sum_v\alpha_v=1.
\]

不得在看到 inner/outer 结果后改用 V8，或在 V6/V8 中择优。V8 只能作为未来另行预注册的新实验。

### 3.2 候选对与 margin 符号

令父模型完成所有 frozen fusion 和 balanced-prior 0.90 后的 500 维对数分数为 \(L\)。exact tie 用较小 class index 获胜：

\[
a=\arg\max_c L_c,\qquad b=\operatorname{runnerup}(L).
\]

所有六个 evidence views 共用同一个候选对 \((a,b)\)；禁止逐视图重选 Top-2。

本协议把 parent margin 定义为：

\[
m=L_b-L_a\le 0.
\]

若实现拿到的是常见的 \(L_a-L_b\ge0\)，必须先取负号。缓存字段 <code>parent_margin</code> 和所有测试都按 runner-up minus Top-1 定义。

### 3.3 classifier direction 与 patch residual

设父分类头权重矩阵为 \(W\)，第 \(c\) 行是 \(w_c\)。对每个样本只计算一次：

\[
d=\lVert w_b-w_a\rVert_2.
\]

若 \(d<10^{-12}\)、任一值非有限、分类头不是单一线性头，样本为 evidence-ineligible 并保留父预测。否则：

\[
u=\frac{w_b-w_a}{d}.
\]

对 view \(v\) 的 row-major 7×7 patch \(p\)：

\[
r_{v,p}=u^\top(x_{v,p}-x_{v,\mathrm{CLS}}).
\]

空间约束：

- \(x_{v,p}\) 与 \(x_{v,\mathrm{CLS}}\) 必须位于分类头实际接收的同一基础空间：相同 final LayerNorm、visual projection、L2 normalization 和 dtype 路径。
- 用 \(x_{\mathrm{CLS}}\) 重建 base classifier logits，误差必须满足 <code>atol=1e-5, rtol=1e-5</code>。
- O3 和 PartToken 继续参与冻结父 local logits；SCOPE 不把单个 patch 送入 O3 或 PartToken，也不训练/新增 part token。
- 用 anchored identity 验证 dual adapted local logits 与现有父分支一致。
- classifier bias、balanced-prior bias 不进入 \(u\) 或 residual；它们已包含在 \(m\) 中。
- residual 与 evidence 的正式缓存/拟合使用 CPU float64；GPU 前向可用原冻结精度，落盘前显式转换并检查有限。

为保证精确反向一致，缓存或 helper 内先按 canonical unordered pair \((\min(a,b),\max(a,b))\) 计算一次，再按有向 \((a,b)\) 乘符号；不能依赖两次独立浮点计算恰好互为负数。

### 3.4 固定四邻接图与连贯能量

节点集 \(V\) 是 row-major 7×7 网格，共 49 个节点。边集只含水平/垂直相邻，不含对角线、环绕或可学习边：

\[
E_4=\{(p,q):\lVert p-q\rVert_1=1\},\qquad |E_4|=84.
\]

edge list 的顺序也冻结：先按 row-major 枚举 42 条水平边 \((i,j)\rightarrow(i,j+1)\)，其中 \(i=0..6,j=0..5\)；再按 row-major 枚举 42 条垂直边 \((i,j)\rightarrow(i+1,j)\)，其中 \(i=0..5,j=0..6\)。每条无向边只保存上述正方向一次。

令 \(z_p^+=\max(z_p,0)\)，定义无阈值、无 top-k 的固定 functional：

\[
H(z)=
\frac{1}{49}\sum_{p\in V}z_p^+
+
\frac{1}{84}\sum_{(p,q)\in E_4}\min(z_p^+,z_q^+).
\]

第一个 node term 保存总体方向证据；第二个 edge term 只奖励相邻节点共同拥有的同号证据，孤立极值没有 edge bonus。两项系数都固定为 1，不搜索 component threshold、component size、edge weight 或 mixing coefficient。

单视图 SCOPE 证据：

\[
E_v^{\mathrm{scope}}=H(r_v)-H(-r_v).
\]

六视图聚合：

\[
E^{\mathrm{scope}}=\sum_{v=1}^{6}\alpha_v E_v^{\mathrm{scope}}.
\]

### 3.5 no-topology 与 matched PACE

no-topology 只删掉 graph edge term，其他完全相同：

\[
H_0(z)=\frac{1}{49}\sum_pz_p^+,
\qquad
E_v^{\mathrm{nt}}=H_0(r_v)-H_0(-r_v)
=\frac{1}{49}\sum_pr_{v,p}.
\]

matched PACE 使用同一父模型、同一候选、同一六视图、同一权重和同一 folds，只替换统计量。tail size 固定为 7：

\[
E_v^{\mathrm{pace}}=
\frac12\left(
\operatorname{mean}(\operatorname{Top7}(r_v))
+
\operatorname{mean}(\operatorname{Bottom7}(r_v))
\right).
\]

PACE 对 patch 任意置换不敏感；SCOPE 明确依赖四邻接拓扑。不得把旧 R2 PACE 的父模型、尺度或 view weights 混入这项 matched ablation。

### 3.6 method-specific eligibility gate

父 cache 保存四个 constituent branch 的 top1：

1. original global
2. original local aggregate（完整四尺度）
3. flipped global
4. flipped local aggregate（完整四尺度）

这些 constituent 与现有父逻辑一致：各分支完成自身 temperature/adapter/softmax，但 branch conflict 在最终 balanced-prior 之前计算。若四个 branch top1 完全一致，任何 verifier 都不得切换。

对每种 evidence family \(q\in\{\mathrm{pace},\mathrm{nt},\mathrm{scope}\}\)，用该 family 的六个 \(E_v^q\) 独立计算：

- branch_conflict：四个 constituent top1 至少有一个不同。
- non_corrupt：Pass A 与 Pass B 都非 corrupt，且两者 corrupt mask 完全相同。
- weight_norm_valid：\(d\ge10^{-12}\)。
- support_count_positive：六视图中至少 4 个 \(E_v^q>0\)。
- total_positive：\(\sum_v\alpha_vE_v^q>0\)。
- orientation_positive：original 三尺度按 <code>(0.375,0.5,0.125)</code> 聚合后 >0，flip 也 >0。
- leave_one_scale_positive：依次同时移除 original/flip 的一个尺度，对剩余权重重新归一化；三个 leave-one-scale aggregate 全部 >0。

只有以上全部为真才是 family \(q\) 的 eligible。等于 0 一律视为不通过，不做 epsilon 放宽。

### 3.7 verifier score、abstention 与反对称

对使用 evidence magnitude 的方法：

\[
\eta=m+\beta E,\qquad \beta\ge0.
\]

\(\beta\) 是全局共享标量，无 intercept、class/pair/sample 参数。部署决策是：

\[
\widehat y=
\begin{cases}
b,& \mathrm{eligible}\ \land\ \eta>\gamma,\\
a,& \text{otherwise}.
\end{cases}
\]

阈值严格使用 <code>&gt;</code>；等于 \(\gamma\) 时 abstain 并返回父 Top-1。threshold 是 tagged union：

- <code>all_switch</code>：所有 eligible 行切换。
- <code>finite</code>：使用冻结的有限 \(\gamma\)。
- <code>no_switch</code>：没有行切换，并记录失败原因。

交换 \(a,b\) 时：

\[
u'=-u,\quad r'=-r,\quad E'=-E,\quad m'=-m,\quad \eta'=-\eta.
\]

这是必须逐样本满足的代数性质。canonical path 测试要求反向缓存值 bitwise 等于正向值取负；独立重算测试容差为 <code>atol=1e-6, rtol=0</code>。

---

## 4. 与既有方向的边界

| 方向 | 核心差异 |
|---|---|
| PACE-K2 | PACE 只取 top/bottom order statistics，patch 任意重排不改变结果；SCOPE 的新增信息只来自固定四邻接 edge term。 |
| PartToken | PartToken 是父模型中已有的 learned token/patch pooling；SCOPE 不新增 token/head，不训练 patch aggregator，只读 base 7×7 token。 |
| prototype / kNN | SCOPE 不构建 centroid、clean anchor、neighbor bank 或跨样本 memory。 |
| class/pair routing | SCOPE 不输入 class ID 或 pair ID，不拟合每类/每对参数；只有一个共享 \(\beta\) 和一个共享 threshold。 |
| crop/fusion/prior 搜参 | 父四尺度、八个 local views、dual adapters、fusion、temperature、balanced-prior 0.90 均冻结；证据六视图也在本文预注册。 |
| generic routing | SCOPE 只验证父 Top-1 与 runner-up，一次决策，不扩展 Top-3、不迭代翻转、不按样本类型训练路由器。 |

---

## 5. 两个备选方案

备选方案不与首轮 SCOPE 同时搜索；只有 SCOPE 失败并完成失败报告后，才可另开任务书。

### 5.1 学习式反对称候选对 patch-set verifier

用共享小网络 \(f_\theta\) 并由结构强制反对称：

\[
A_\theta(r,m)=\frac12\left[f_\theta(r,m)-f_\theta(-r,-m)\right].
\]

可选小型 DeepSets、固定图 GNN 或 Set Transformer。允许固定坐标，禁止 class ID、pair ID、prototype、per-class embedding 和检索特征。全部容量、正则、early stopping 和第三类样本处理必须在 inner fold 内冻结。

主要风险：

- 自由度和 label noise 使其更易过拟合。
- 只有 \(y\in\{a,b\}\) 的样本有明确 pairwise target。
- architecture/capacity 若看 outer 结果选择，会使 nested OOF 失效。

仅当固定 SCOPE 显示 topology edge term 在多数 outer folds 有正信号、但固定 \(H\) 表达不足时才优先此方向。

### 5.2 反事实 patch masking / re-forward

mask 必须在交换候选后保持不变。用 \(|r|\) 和固定规则选择 connected patch set \(S_k\)，使 \(S_k(-r)=S_k(r)\)。令：

\[
M(I)=L_b(I)-L_a(I),\qquad
E_{\mathrm{cf}}=M(I)-M(I\setminus S_k).
\]

交换候选后 \(M\) 变号而 mask 不变，因此 \(E_{\mathrm{cf}}\) 严格变号。mask 面积、connected selection、pixel fill 或 mask token、重跑哪些 views 都必须在新 outer 评估前预注册。

主要风险是每图多次 forward、mask OOD 伪影，以及大量可调 mask 自由度。它只作为 SCOPE 失败后的第二备选，不得用当前 outer/test 结果选择 mask 规则。

---

## 6. 为什么首选 SCOPE-K2，以及何时判失败

首选理由：

1. 它测量 PACE 完全忽略的空间拓扑信号，属于真正新增的信息源。
2. \(H\) 无训练参数、无 threshold/top-k/component 搜索，统计自由度远低于 learned verifier。
3. 候选对、patch residual、父 cache 和大部分审计可与 PACE 共享，计算量显著低于 re-forward。
4. 反对称由代数结构保证，不靠数据学出。
5. 同折 no-topology 与 PACE 消融可以直接判断收益究竟来自 graph edge 还是普通方向强度。

以下任一项发生即失败、阻塞或结果无效：

- 父 checkpoint 或 trust bundle 缺失/哈希不符。
- split/class/group 任何哈希不符。
- 无法严格复现父推理协议，或 parent cache 两次运行语义哈希不一致。
- token tap 不是 49 个 row-major patch，四邻接不是 84 条边，classifier-space identity 或 antisymmetry 测试失败。
- Pass A/B 行绑定、候选、crop box 或 corrupt mask 不一致。
- 需要查看 outer/test 才决定 \(H\)、V6/V8、view weight、gate、threshold policy 或候选范围。
- 任一晋级门失败，或 full SCOPE 未严格胜过 PACE/no-topology。
- 最优 \(\beta\) 无法 bracket、结果大量贴边/跨折失稳，且预注册 fail-closed 路径不能给出合格结果。
- 收益被少数 duplicate clusters 驱动，cluster bootstrap 下界不大于 0。
- 测试期需要改 prior、view、\(\beta\)、threshold、class balance 或任何 mask。
- submission checker 失败。

---

## 7. 防泄漏的 5×3 grouped nested OOF

### 7.1 必须如实称为 conditional nested OOF

正式验证 universe 是冻结 val 的 10,316 行，按 canonical path 排序。配置 <code>f1_flat_full_ft.yaml</code> 明确写有 <code>final_full_train.csv</code> 和 <code>validation_overlap_with_training: true</code>；因此该父 FULLFT 已结构性见过验证样本。

5×3 grouped nested OOF 能严格隔离的是：

- SCOPE/PACE/no-topology 的共享 \(\beta\)；
- eligibility 之后的 abstention threshold；
- 同一 exact duplicate group 的 scorer calibration 与 holdout；
- 消融、晋级和报告决策。

它不能把父模型本身变成 honest OOF parent。所有报告标题、表格和结论必须写“conditional grouped nested OOF given the frozen parent”，不得写“完全 held-out”“无泄漏绝对泛化”。若下一阶段要求 parent-level honest OOF，必须另行训练 outer/inner parents 与 dual adapters；这超出本任务书，不能临时扩项。

### 7.2 fold artifact

在任何 \(\beta\) 或 threshold 拟合前：

1. 读取并验证冻结 exact group map。
2. 按 canonical path 排序，生成 <code>formal_row_id=0..10315</code>。
3. outer 使用 <code>StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)</code>。
4. 每个 outer-train 独立生成 inner 3-fold，seed 固定为 <code>42 + 1000 + outer_fold_id</code>。
5. 同一 group 在 outer 或相应 inner 中绝不跨 fold。
6. 一次性写入 fold artifact；记录 artifact SHA、path/label/group binding、Python/PyTorch/NumPy/scikit-learn 版本、lockfile hash、代码提交、dirty diff hash、config/split/group/class/trust/checkpoint hashes。
7. 后续 run1/run2 和所有方法必须读取同一 fold artifact，禁止重新调用 splitter。

正式环境记录为审计值而非宽松依赖；当前已知 NumPy 2.5.1、scikit-learn 1.9.0，正式运行时必须记录实际 import version 并与 lock/config 一致。

### 7.3 \(\beta\) 拟合

只对 label 在候选对中的训练行拟合：

\[
\mathcal T_{ab}=\{i:y_i\in\{a_i,b_i\}\},\qquad t_i=\mathbb 1[y_i=b_i].
\]

按 <code>formal_row_id</code> 升序，在 CPU float64 上最小化：

\[
J(\beta)=
\frac1n\sum_i
\left[\operatorname{softplus}(m_i+\beta E_i)
-t_i(m_i+\beta E_i)\right]
+\frac{\beta^2}{2n},
\qquad \beta\ge0.
\]

无 intercept。deterministic solver 固定为：

- 若 \(\nabla J(0)\ge0\)，取 \(\beta=0\)。
- initial upper=1。
- 每次倍增，maximum upper=\(2^{20}\)。
- 最多 100 次 bisection。
- interval tolerance=\(10^{-12}\)。
- target 必须同时含 0 和 1；否则该 fit fail closed。
- 若 maximum upper 仍无法 bracket，记录失败并使用 <code>no_switch</code>，不得扩大范围重跑 outer。

### 7.4 threshold 选择与 refit 映射

每个 outer fold：

1. 三个 inner fit 各自只看 inner-train。
2. 合并三份 inner-held-out \(\eta\) 和 eligibility，形成 outer-train 的完整 inner OOF score。
3. threshold 候选仅为 <code>all_switch</code>、每个相邻 unique score 间对应的 finite cut、<code>no_switch</code>。
4. accuracy-changing precision 必须 ≥0.60，Wilson 95% lower 必须 ≥0.50。
5. 合格候选先最大化 corrections-regressions，再偏好更少 switches，再偏好更高 cut；无合格候选则 <code>no_switch</code>。
6. 保存选中 tag、\((k_{\mathrm{oof}},n_{\mathrm{oof}})\)、score hash 和失败原因。
7. 在完整 outer-train 重拟合 \(\beta\)。
8. finite threshold 不直接复用 inner OOF 数值 \(\gamma\)。用冻结 \(k_{\mathrm{oof}}/n_{\mathrm{oof}}\) 在 outer-train 的 refit eligible score 上做整数最近 rank 映射；tie-break 为更少 switches、再更高 cut。
9. 把得到的数值 threshold 固定后，只评估一次 outer holdout。outer holdout 的 label 或 score distribution 都不参与映射。

\(\beta\)、threshold 和所有 tie-break 只以 raw 原标签为训练/选择目标；clean-core 只在 outer holdout 上作为预注册晋级指标，禁止用于 inner 选择。

### 7.5 同折消融的精确定义

全部方法共享同一 parent cache、候选、formal rows、fold artifact、threshold policy 和 paired metrics：

| 方法 | eligibility | score | 可拟合参数 |
|---|---|---|---|
| Parent | 不适用 | 不适用 | 无；始终输出 \(a\) |
| margin-only | branch_conflict 且 non_corrupt | \(\eta=m\) | \(\beta=0\)，inner 选择 threshold |
| PACE-K2 | matched PACE family gate | \(\eta=m+\beta E^{pace}\) | 共享 \(\beta\) 与 threshold |
| SCOPE gate-only | full SCOPE family gate | \(\eta=m\) | \(\beta=0\)，inner 选择 threshold |
| SCOPE no-topology | no-topology family gate | \(\eta=m+\beta E^{nt}\) | 共享 \(\beta\) 与 threshold |
| SCOPE-K2 full | full SCOPE family gate | \(\eta=m+\beta E^{scope}\) | 共享 \(\beta\) 与 threshold |

这里 margin-only 不是 Parent 的别名：它在 branch conflict 行上用 margin 和 inner-selected threshold 测试“盲目切换近邻候选”。SCOPE gate-only 则隔离 full SCOPE sign/coherence gate 的价值，但不使用 evidence magnitude。

### 7.6 outer 指标与 cluster bootstrap

所有指标在完整 outer holdout 上计算，不得只在 switched rows 或 \(y\in\{a,b\}\) 子集上计算晋级结果。

必须报告：

- raw accuracy 与 paired delta；
- clean-core accuracy 与 paired delta；
- 每 fold correct count、delta、switch count；
- correction：Parent 错、方法对；
- regression：Parent 对、方法错；
- neutral：其余；
- switch precision、oracle availability；
- macro accuracy；父分数的 NLL/logloss、Brier、ECE 可作为不参与 gate 的父校准诊断，但 SCOPE 是 label-only verifier，不得为它事后发明 500 类概率来美化校准指标；
- McNemar paired table/p-value；
- 每 fold \(\beta\)、threshold tag、mapped threshold、eligible/switch counts、失败原因。

paired cluster bootstrap 固定为：

- cluster：冻结 exact duplicate group；
- strata：<code>(outer_fold_id, group-majority-label)</code>；
- draw：10,000；
- RNG：<code>numpy.random.Generator(PCG64(42))</code>；
- CI：95% percentile，NumPy quantile method=<code>linear</code>；
- 每次以 group 为单位有放回抽样，group 内所有 rows 一起进入；
- 统计量：SCOPE full minus Parent 的全 row accuracy delta。

### 7.7 晋级门：全部为 AND

SCOPE-K2 full 只有同时满足以下条件才晋级：

1. pooled outer raw accuracy 相对 Parent 至少 <code>+0.20pp</code>。
2. pooled outer clean-core accuracy 相对 Parent 至少 <code>+0.20pp</code>。
3. corrections - regressions ≥21。
4. 至少 4/5 outer folds 的 raw paired delta 非负。
5. 上述 10,000-draw paired cluster bootstrap 的 95% CI 下界严格 >0。
6. pooled outer raw correct count 严格大于 matched PACE-K2，且严格大于 SCOPE no-topology。
7. pooled outer clean-core correct count也严格大于 matched PACE-K2，且严格大于 SCOPE no-topology。
8. parent parity、cache binding、fold/group、49-node/84-edge、token-space、antisymmetry、run1/run2 semantic reproducibility 审计全部通过。

百分点比较使用未四舍五入的整数正确数/固定分母计算；“显示值相同但更多小数略高”不能替代严格 correct-count 门。

### 7.8 最终 refit 与测试期冻结

只有晋级后：

1. 在全 10,316 行上使用同一冻结 group map 做一次 3-fold OOF threshold calibration。
2. 由这份 full-val OOF 选择最终 threshold tag 和 \((k,n)\)。
3. 在全 val 的 \(y\in\{a,b\}\) 行拟合一个最终 \(\beta\)。
4. 用全 val refit eligible score 做 rank/count mapping，得到部署 \(\gamma\)。
5. 把 \(\beta\)、threshold tag/\(\gamma\)、view order/weights、graph edge hash、gate spec、parent/checkpoint/config/class/split/group/fold/code hashes写入只读 deployment payload。
6. final refit 不得改写已报告的 5×3 outer 指标。

测试期只允许父协议原有的 full-batch balanced-prior 0.90。禁止根据 test score/eligible 比例/class histogram 调整 \(\beta\)、\(\gamma\)、switch fraction、view weight、prior 或 gate；禁止平台反馈回流。

---

## 8. 预计文件与职责

### 8.1 新增文件

- <code>reproducibility/aegis_f1/aegis_clip/scope_protocol.py</code>
  冻结 parent identity、六视图、49/84 graph、solver、threshold、fold、bootstrap、promotion 和 asset hashes。
- <code>reproducibility/aegis_f1/aegis_clip/scope_evidence.py</code>
  residual grid、\(H\)、PACE/no-topology matched statistics、aggregation、eligibility、antisymmetry。
- <code>reproducibility/aegis_f1/aegis_clip/scope_cache.py</code>
  parent/evidence/fold/deployment schemas、binding 与 semantic hash 校验。仅当现有 pace_cache 可安全参数化时才抽成 method-neutral helper。
- <code>reproducibility/aegis_f1/aegis_clip/scope_crossfit.py</code>
  5×3 conditional grouped nested OOF、deterministic \(\beta\)、threshold/rank mapping、paired metrics、cluster bootstrap。
- <code>reproducibility/aegis_f1/aegis_clip/cli/cache_scope_parent.py</code>
- <code>reproducibility/aegis_f1/aegis_clip/cli/cache_scope_evidence.py</code>
- <code>reproducibility/aegis_f1/aegis_clip/cli/prepare_scope_folds.py</code>
- <code>reproducibility/aegis_f1/aegis_clip/cli/evaluate_scope_k2.py</code>
- <code>reproducibility/aegis_f1/aegis_clip/cli/infer_scope_submission.py</code>
- <code>reproducibility/aegis_f1/configs/scope_k2_fullft_dual_pa090.yaml</code>
- <code>reproducibility/aegis_f1/tests/test_scope_protocol.py</code>
- <code>reproducibility/aegis_f1/tests/test_scope_evidence.py</code>
- <code>reproducibility/aegis_f1/tests/test_scope_cache.py</code>
- <code>reproducibility/aegis_f1/tests/test_scope_crossfit.py</code>
- <code>reproducibility/aegis_f1/tests/test_scope_submission.py</code>
- <code>reproducibility/aegis_f1/protocol_artifacts/scope_k2_fullft_dual_pa090/nested_folds.pt</code> 及审计 manifest。
- <code>results/scope_k2_fullft_dual_pa090.md</code>
  正式 validation、消融、晋级/失败、终态 submission 和 checker 报告。

### 8.2 可能修改的文件

- <code>reproducibility/aegis_f1/aegis_clip/local_inference.py</code>
  仅当现有 opt-in path 不能返回 row-major 7×7 projected patches 时修改；默认返回 contract 必须不变。
- <code>reproducibility/aegis_f1/aegis_clip/candidate_patch_evidence.py</code>
  抽取 method-neutral pair residual helper；旧 PACE 输出必须回归不变。
- <code>reproducibility/aegis_f1/aegis_clip/pace_cache.py</code> 与 <code>pace_crossfit.py</code>
  只有在不破坏未完成 PACE 的前提下做小范围通用化；否则 SCOPE 使用独立模块。
- <code>reproducibility/aegis_f1/aegis_clip/checkpoint.py</code>
  仅晋级后嵌入 deployment payload；必须证明父 tensor、O3 和 PartToken payload 未变。
- <code>reproducibility/aegis_f1/pyproject.toml</code>
  增加上述 SCOPE CLI entry points。
- 对应 PACE、local inference、PartToken、prior 和 submission regression tests。

禁止把 SCOPE 逻辑塞进默认 <code>infer.py</code>；默认最佳父路径必须保持不变。

### 8.3 输出位置

- 中间 cache/评估：<code>reproducibility/aegis_f1/outputs/scope_k2/fullft_dual_pa090/</code>
- 晋级 submission：<code>outputs/delivery/scope_k2_fullft_dual_pa090/</code>
- 失败 submission：直接使用 <code>outputs/delivery/fullft_dual_pa0.9/</code> 的精确归档字节
- 报告：<code>results/scope_k2_fullft_dual_pa090.md</code>

---

## 9. CLI contract

所有 CLI 必须：

- 默认 fail closed；
- 支持 <code>--config</code>；
- 在运行开始和产物内记录 argv、cwd、代码/dirty diff hash、环境版本；
- 自行校验所有配置声明的 SHA；
- 使用 atomic write；
- 若目标已存在且 hash/manifest 不同则拒绝覆盖，要求新输出路径；
- test split 拒绝 label、clean_probability、pseudo_label、correction_alpha 等诊断字段。

### 9.1 cache_scope_parent

必需参数：

    --config
    --checkpoint
    --split {validation,test}
    --output
    --batch-size
    --num-workers
    --device

职责：完整复现八 local-view FULLFT_DUAL + prior 0.90，生成稳定 Top-2、margin、四 constituent、四尺度 crop boxes、row/path/corrupt binding 和 validation-only trust fields。

### 9.2 cache_scope_evidence

必需参数：

    --config
    --checkpoint
    --parent-cache
    --split {validation,test}
    --output
    --batch-size
    --num-workers
    --device

职责：严格复用 Pass A crop boxes，只对 128/144/160 original/flip 提取 base 7×7 tokens；一次生成 scope、matched PACE、no-topology 三组 evidence 与各自 gate audit。

### 9.3 prepare_scope_folds

必需参数：

    --config
    --parent-cache
    --group-artifact
    --output

只接受 validation parent cache；若输出已存在，只允许验证完全相同，不重新随机切分。

### 9.4 evaluate_scope_k2

必需参数：

    --config
    --parent-cache
    --evidence-cache
    --replicate-parent-cache
    --replicate-evidence-cache
    --fold-artifact
    --output-dir

先比较 run1/run2 的 row、candidate、score、prior、constituent、box、corrupt、evidence 和 gate semantic hashes；全部一致后才跑六方法同折评估、bootstrap、晋级门和 final deployment refit。输出 JSON、CSV、Markdown 和 machine-readable <code>decision.json</code>。

### 9.5 infer_scope_submission

必需参数：

    --config
    --checkpoint
    --parent-cache
    --evidence-cache
    --decision
    --deployment
    --output-dir

只接受 <code>decision.json</code> 中 <code>promoted=true</code>，且 decision、deployment 与所有上游绑定 hash 相同的 payload；否则拒绝运行。输出 raw decision audit、<code>pred_results.csv</code>、<code>submission.zip</code> 和 manifest。

---

## 10. Cache schema

### 10.1 Parent cache：scope_parent_cache_v1

至少包含：

- schema/protocol/config/code/dirty-diff/environment/checkpoint/class-map/split/trust/group hashes；
- <code>formal_row_id int64 [N]</code>、canonical paths、row binding hash；
- post-prior <code>candidate_indices int64 [N,2]</code>；
- <code>candidate_parent_log_scores float64 [N,2]</code>；
- <code>parent_margin float64 [N]</code>，且逐行等于 score[:,1]-score[:,0]；
- <code>parent_predictions int64 [N]</code>；
- <code>constituent_scores float32 [N,4,500]</code>、<code>constituent_top1 [N,4]</code>、order 与 hash；
- <code>crop_boxes int64 [N,2,4,4]</code>，顺序为 orientation × 112/128/144/160 × coordinates；
- prior bias、iterations、alignment report/hash；
- 完整 post-prior aligned log-score tensor 的 shape/dtype/semantic SHA-256；tensor 可在生成稳定 Top-2 后丢弃，但 hash 不得省略；
- corrupt mask 与 corrupt fallback 行的审计；
- validation-only：label、clean_probability、pseudo_label、correction_alpha；
- test cache：必须明确声明这些 validation-only fields 不存在。

不保存未使用的全分类中间 tensor；但 constituent 和最终 aligned score 必须足以审计父 parity。

### 10.2 Evidence cache：scope_evidence_cache_v1

至少包含：

- parent cache file/semantic hash；
- 完全相同的 formal_row_id/path/candidate/crop/corrupt binding；
- 固定 <code>view_order</code>、六权重、<code>grid_shape=[7,7]</code>；
- <code>adjacency=four_neighbor_row_major_v1</code>、canonical 84-edge list 与 hash；
- classifier weight hash、weight norm、token tap/classifier identity audit；
- <code>scope_view_evidence float64 [N,6]</code> 与 aggregate；
- <code>pace_view_evidence float64 [N,6]</code> 与 aggregate；
- <code>no_topology_view_evidence float64 [N,6]</code> 与 aggregate；
- 每个 family 的 positive count、orientation aggregate、三项 leave-one-scale aggregate 和最终 eligibility；
- canonical/reverse antisymmetry audit；
- Pass A/B crop box 与 corrupt equality；
- validation-only trust/label binding；test 中禁止出现。

禁止保存 <code>N×6×49×D</code> raw patch tensors。可保存 49/84 数量和固定 edge hash；不得把探索性 component 指标变成事后 scorer。

### 10.3 Fold/deployment schema

fold artifact 必须包含所有 outer/inner IDs、groups、labels、paths 和 provenance hashes。deployment payload 必须包含最终 \(\beta\)、tagged threshold、\(\gamma\) 或 no-switch reason、OOF/refit counts、method/gate/formula/version、view/edge hash，以及所有上游 binding。

任何一个 hash 不匹配都拒绝推理，不能 warning 后继续。

---

## 11. 单元、集成和回归测试

所有实现遵循 red-green：先写会失败的最小测试，再实现。

### 11.1 Protocol tests

- checkpoint、trust、split、class、group、fallback hashes 完整且不可变。
- full parent 是八 local views；evidence 恰好是冻结六视图。
- view order 与权重逐项相等，和为 1。
- node=49、edge=84，无 diagonal/wrap/duplicate/self-loop。
- parent margin 符号为 runner-up minus top1。
- test schema 排除 validation-only fields。

### 11.2 Evidence tests

- 手工 7×7 toy grid 验证 \(H\)、\(H_0\)、PACE tail7。
- patch permutation 保持 no-topology/PACE，但一般改变 SCOPE edge term。
- 相同 node values 下，连片正证据的 SCOPE 分数高于孤立正证据。
- 交换候选时 \(u,r,E,m,\eta\) 变号。
- canonical reverse 是 bitwise negative；独立重算误差 ≤1e-6。
- norm <1e-12、NaN/Inf、错误 grid、错误 classifier space 全部 fail closed。
- original/flip、support≥4、leave-one-scale 和严格 >0 边界正确。
- crop112 不进入 evidence，但仍进入 parent。

### 11.3 Parent/cache tests

- base CLS 重建 classifier logits。
- dual O3 + PartToken anchored local logits与父分支一致。
- 四 constituent 和最终八-view fusion与现有 local inference 一致。
- stable Top-2 tie-break 为较小 class index。
- prior 0.90 的 shared bias、iteration 和 score hash一致。
- Pass A/B row/path/candidate/box/corrupt mismatch 均拒绝。
- run1/run2 semantic hash determinism。
- test cache 注入 label/trust 字段时拒绝。

### 11.4 Crossfit tests

- exact group 不跨 outer/inner。
- outer labels不进入 inner fit/threshold/rank mapping。
- 六方法共用 fold IDs 和候选。
- \(\beta\ge0\)、无 intercept、row sort、boundary 0、bracket failure。
- threshold 的 all/finite/no-switch、strict >、precision/Wilson 和 tie-break。
- inner OOF count 到 outer/full refit rank mapping，不直接复用 numeric \(\gamma\)。
- third-class rows不参与 \(\beta\)，但参与 outer accuracy。
- cluster bootstrap以 group为抽样单位，seed/strata/quantile固定。
- 八个晋级门逐项 fail closed。

### 11.5 Submission/regression tests

- 未晋级时 infer CLI 拒绝创建 SCOPE test artifacts。
- 晋级 deployment 的任何 hash 改动都拒绝。
- CSV 格式、类范围、覆盖和 ZIP 内容通过现有九项 checker。
- 旧 PACE evidence/cache/crossfit 测试不回归。
- 默认 local inference、PartToken、prior、父 submission 路径不回归。

聚焦测试命令：

    cd /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/reproducibility/aegis_f1
    PYTHONPATH=$PWD pytest -q \
      tests/test_scope_protocol.py \
      tests/test_scope_evidence.py \
      tests/test_scope_cache.py \
      tests/test_scope_crossfit.py \
      tests/test_scope_submission.py \
      tests/test_candidate_patch_evidence.py \
      tests/test_pace_cache.py \
      tests/test_pace_crossfit.py \
      tests/test_local_inference.py \
      tests/test_part_token_adapter.py \
      tests/test_prior_alignment.py \
      tests/test_submission.py

---

## 12. 分任务实施顺序

### 任务 0：重新同步审计，不运行实验

- [ ] 保存当前 <code>git status --short --branch</code>。
- [ ] fetch/prune 并重新检查所有 refs 和 recent main。
- [ ] 若 <code>HEAD...origin/main</code> 仍为 0/0，不 pull。
- [ ] 若 origin/main 更新，在混合工作树使用 automatic stash 模式：

      git pull --rebase --autostash origin main

- [ ] Git 会自动恢复临时 stash；严禁再执行 <code>git stash pop</code>。
- [ ] 重做 SCOPE/verifier/masking 关键词扫描；若远端出现重叠，先 diff/协调，暂停实现。
- [ ] 确认仍在 main，不建分支。

建议命令：

    cd /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning
    git fetch --all --prune
    git branch --show-current
    git status --short --branch
    git rev-parse HEAD origin/main
    git rev-list --left-right --count HEAD...origin/main
    git for-each-ref --format='%(refname:short) %(objectname:short) %(subject)' refs/heads refs/remotes
    git log --oneline --decorate -n 20 main
    git log --all -i --grep='SCOPE\|spatially coherent\|pairwise evidence\|patch-set verifier\|counterfactual\|masking\|re-forward'
    for ref in $(git for-each-ref --format='%(refname)' refs/heads refs/remotes); do
      git grep -I -n -i -E 'SCOPE|spatially coherent|pairwise evidence|patch.set verifier|counterfactual|masking|re.forward' $ref -- . || true
    done
    rg -n -i 'SCOPE|spatially coherent|pairwise evidence|patch.set verifier|counterfactual|masking|re.forward' .

### 任务 1：资产恢复与父 parity 门

- [ ] 恢复 checkpoint 和 trust bundle 到 external 路径。
- [ ] 校验本文件列出的所有 SHA。
- [ ] 校验 fallback CSV/ZIP/manifest SHA。
- [ ] 新增 protocol/config 与失败测试。
- [ ] 加载 checkpoint 后核验 peft_mode、linear classifier、dual O3/PartToken payload。
- [ ] 只用 validation 小批次做 classifier/fusion/token parity；此处不得访问 test 或平台。

### 任务 2：通用 residual grid 与固定图 evidence

- [ ] 先写 49/84、toy grid、permutation、antisymmetry 和 gate tests。
- [ ] 抽取/实现 canonical pair residual。
- [ ] 实现 \(H\)、\(H_0\)、matched PACE tail7。
- [ ] 实现六视图聚合和 method-specific gate。
- [ ] 跑 evidence 与旧 PACE regression tests。

### 任务 3：Parent Pass A cache

- [ ] 先写 schema/binding/test-leakage 失败测试。
- [ ] 实现完整八 local-view parent cache。
- [ ] validation run1/run2 使用不同输出文件。
- [ ] 比较预测、aligned scores、prior、constituent、boxes、corrupt 和 semantic hashes。
- [ ] 任一不一致即停，不进入 Pass B。

正式命令：

    cd /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/reproducibility/aegis_f1
    PYTHONPATH=$PWD python3 -m aegis_clip.cli.cache_scope_parent \
      --config configs/scope_k2_fullft_dual_pa090.yaml \
      --checkpoint artifacts/external/scope_k2/fullft_dual_pa090/best.pt \
      --split validation \
      --output outputs/scope_k2/fullft_dual_pa090/cache/validation_parent_run1.pt \
      --batch-size 128 \
      --num-workers 4 \
      --device cuda

    PYTHONPATH=$PWD python3 -m aegis_clip.cli.cache_scope_parent \
      --config configs/scope_k2_fullft_dual_pa090.yaml \
      --checkpoint artifacts/external/scope_k2/fullft_dual_pa090/best.pt \
      --split validation \
      --output outputs/scope_k2/fullft_dual_pa090/cache/validation_parent_run2.pt \
      --batch-size 128 \
      --num-workers 4 \
      --device cuda

### 任务 4：Evidence Pass B cache

- [ ] 先写 Pass A/B mismatch 和 test-field rejection tests。
- [ ] 严格复用 Pass A crop boxes。
- [ ] 一次 forward path 产出三种 evidence family。
- [ ] validation run1/run2 与各自 parent cache 绑定。
- [ ] 比较 evidence/gate/token/antisymmetry semantic hashes。

    cd /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/reproducibility/aegis_f1
    PYTHONPATH=$PWD python3 -m aegis_clip.cli.cache_scope_evidence \
      --config configs/scope_k2_fullft_dual_pa090.yaml \
      --checkpoint artifacts/external/scope_k2/fullft_dual_pa090/best.pt \
      --parent-cache outputs/scope_k2/fullft_dual_pa090/cache/validation_parent_run1.pt \
      --split validation \
      --output outputs/scope_k2/fullft_dual_pa090/cache/validation_evidence_run1.pt \
      --batch-size 128 \
      --num-workers 4 \
      --device cuda

    PYTHONPATH=$PWD python3 -m aegis_clip.cli.cache_scope_evidence \
      --config configs/scope_k2_fullft_dual_pa090.yaml \
      --checkpoint artifacts/external/scope_k2/fullft_dual_pa090/best.pt \
      --parent-cache outputs/scope_k2/fullft_dual_pa090/cache/validation_parent_run2.pt \
      --split validation \
      --output outputs/scope_k2/fullft_dual_pa090/cache/validation_evidence_run2.pt \
      --batch-size 128 \
      --num-workers 4 \
      --device cuda

### 任务 5：冻结 fold artifact

- [ ] 只用 run1 parent cache、冻结 group map 和 protocol 生成一次。
- [ ] 审计 10,316 rows 全覆盖、每 group 单 fold、outer 5 折、每 outer-train inner 3 折。
- [ ] 第二次调用只能验证同一 SHA，不得覆盖或重新 split。

    cd /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/reproducibility/aegis_f1
    PYTHONPATH=$PWD python3 -m aegis_clip.cli.prepare_scope_folds \
      --config configs/scope_k2_fullft_dual_pa090.yaml \
      --parent-cache outputs/scope_k2/fullft_dual_pa090/cache/validation_parent_run1.pt \
      --group-artifact protocol_artifacts/pace_k2_r2_parttoken/content_groups.json \
      --output protocol_artifacts/scope_k2_fullft_dual_pa090/nested_folds.pt

### 任务 6：同折 conditional nested OOF、消融与晋级判定

- [ ] 先写 solver、threshold、fold leakage、bootstrap 和 promotion tests。
- [ ] evaluator 先强制比较 run1/run2 caches。
- [ ] 一次运行六个方法，禁止逐方法重切 folds。
- [ ] 输出 raw/clean-core、fold、paired、bootstrap 和消融报告。
- [ ] machine-readable decision 逐项列出八个 gate 的 pass/fail 和整数证据。
- [ ] 若失败，不创建 test cache，直接转任务 9 的父回退分支。

    cd /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/reproducibility/aegis_f1
    PYTHONPATH=$PWD python3 -m aegis_clip.cli.evaluate_scope_k2 \
      --config configs/scope_k2_fullft_dual_pa090.yaml \
      --parent-cache outputs/scope_k2/fullft_dual_pa090/cache/validation_parent_run1.pt \
      --evidence-cache outputs/scope_k2/fullft_dual_pa090/cache/validation_evidence_run1.pt \
      --replicate-parent-cache outputs/scope_k2/fullft_dual_pa090/cache/validation_parent_run2.pt \
      --replicate-evidence-cache outputs/scope_k2/fullft_dual_pa090/cache/validation_evidence_run2.pt \
      --fold-artifact protocol_artifacts/scope_k2_fullft_dual_pa090/nested_folds.pt \
      --output-dir outputs/scope_k2/fullft_dual_pa090/evaluation/final

### 任务 7：只有晋级后才做 test Pass A/B

- [ ] evaluator 的 <code>decision.json</code> 必须为 promoted=true。
- [ ] 使用同一 checkpoint/config 生成 test parent cache。
- [ ] test parent cache不得含 label/trust diagnostics。
- [ ] 生成 test evidence cache，禁止 test-time calibration。

    cd /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/reproducibility/aegis_f1
    PYTHONPATH=$PWD python3 -m aegis_clip.cli.cache_scope_parent \
      --config configs/scope_k2_fullft_dual_pa090.yaml \
      --checkpoint artifacts/external/scope_k2/fullft_dual_pa090/best.pt \
      --split test \
      --output outputs/scope_k2/fullft_dual_pa090/cache/test_parent.pt \
      --batch-size 128 \
      --num-workers 4 \
      --device cuda

    PYTHONPATH=$PWD python3 -m aegis_clip.cli.cache_scope_evidence \
      --config configs/scope_k2_fullft_dual_pa090.yaml \
      --checkpoint artifacts/external/scope_k2/fullft_dual_pa090/best.pt \
      --parent-cache outputs/scope_k2/fullft_dual_pa090/cache/test_parent.pt \
      --split test \
      --output outputs/scope_k2/fullft_dual_pa090/cache/test_evidence.pt \
      --batch-size 128 \
      --num-workers 4 \
      --device cuda

### 任务 8：晋级 submission

- [ ] 只读取冻结 deployment；不得重新拟合或 rank-map。
- [ ] 生成 SCOPE CSV/ZIP/manifest。
- [ ] 用相同 cache 再做一次 decision replay，要求 predictions 和 CSV byte hash一致；ZIP 因时间戳不要求跨生成 bitwise 相同。
- [ ] 运行 Aegis audit 与根九项 checker。

    cd /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/reproducibility/aegis_f1
    PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer_scope_submission \
      --config configs/scope_k2_fullft_dual_pa090.yaml \
      --checkpoint artifacts/external/scope_k2/fullft_dual_pa090/best.pt \
      --parent-cache outputs/scope_k2/fullft_dual_pa090/cache/test_parent.pt \
      --evidence-cache outputs/scope_k2/fullft_dual_pa090/cache/test_evidence.pt \
      --decision outputs/scope_k2/fullft_dual_pa090/evaluation/final/decision.json \
      --deployment outputs/scope_k2/fullft_dual_pa090/evaluation/final/deployment.pt \
      --output-dir /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/outputs/delivery/scope_k2_fullft_dual_pa090

    cd /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning
    PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.audit_submission \
      --config reproducibility/aegis_f1/configs/scope_k2_fullft_dual_pa090.yaml \
      --submission-dir outputs/delivery/scope_k2_fullft_dual_pa090 \
      --allow-tta

    python3 scripts/check_submission.py \
      --test_dir test \
      --csv outputs/delivery/scope_k2_fullft_dual_pa090/pred_results.csv \
      --zip outputs/delivery/scope_k2_fullft_dual_pa090/submission.zip

<code>--allow-tta</code> 只承认父协议已经冻结的 original/flip 与 crops，不授权新增测试时调参。

### 任务 9：失败时精确回退

- [ ] 本任务仅适用于已经产出完整 validation 判定、但结果 rejected 的实验段；checkpoint/trust 等前置资产缺失属于“尚未开始/阻塞”，不能用 fallback 冒充已完成 validation 段或提前 commit。
- [ ] 不创建任何 SCOPE test cache 或 SCOPE submission。
- [ ] 重新计算归档父 CSV/ZIP/manifest 的三个 SHA，必须匹配第 2.4 节。
- [ ] 直接对归档父 CSV/ZIP 运行 checker。
- [ ] 在 SCOPE 报告和 decision manifest 中记录 <code>terminal_submission=parent_fallback</code>、三个 SHA 和失败门。
- [ ] 若归档字节缺失，从已验证备份恢复精确字节；不能重打 ZIP。恢复失败则整个实验段仍未达到可暂停状态。

    cd /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning
    sha256sum \
      outputs/delivery/fullft_dual_pa0.9/pred_results.csv \
      outputs/delivery/fullft_dual_pa0.9/submission.zip \
      outputs/delivery/fullft_dual_pa0.9/manifest.json

    python3 scripts/check_submission.py \
      --test_dir test \
      --csv outputs/delivery/fullft_dual_pa0.9/pred_results.csv \
      --zip outputs/delivery/fullft_dual_pa0.9/submission.zip

### 任务 10：完整验证、报告与一次本地 commit

- [ ] 写完 <code>results/scope_k2_fullft_dual_pa090.md</code>，明确 conditional OOF 限制。
- [ ] 报告 parent、六个方法、每折、bootstrap、八个晋级门、终态 submission、所有关键 SHA 和 checker 原始结果。
- [ ] 跑聚焦测试、全量相关测试、<code>git diff --check</code>。
- [ ] 检查工作树，区分本实验文件与用户原有脏文件。
- [ ] 只有 validation 结果、晋级/失败判定、对应 CSV/ZIP、checker PASS、manifest 和报告全部齐全后才允许暂停。
- [ ] 用显式路径暂存本实验文件；禁止 <code>git add -A</code>，不得夹带无关历史文件。
- [ ] 把本任务书与实际代码、报告、可提交终态 artifacts 一起做一次本地 commit。
- [ ] 不 push。

建议最终验证：

    cd /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/reproducibility/aegis_f1
    PYTHONPATH=$PWD pytest -q

    cd /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning
    git diff --check
    git status --short --branch

暂存前必须逐项列路径并先检查：

    git diff --stat
    git diff --cached --stat
    git diff --cached --check

commit message 按终态二选一：

- 晋级：<code>feat(aegis): evaluate and promote SCOPE-K2 inference</code>
- 失败回退：<code>exp(aegis): evaluate and reject SCOPE-K2 inference</code>

提交后记录本地 commit SHA，并明确写“not pushed”。用户负责之后的 <code>git push origin main</code>。

---

## 13. 实验段暂停/交接的硬格式

不得在代码完成、单测通过或 validation cache 生成时提前暂停。一个 SCOPE 实验段只有同时拥有以下内容才算完成：

1. 实验 ID：<code>SCOPE-K2_FULLFT_DUAL_PA090</code>。
2. exact command/config/checkpoint 和全部 provenance hashes。
3. 5×3 conditional grouped nested OOF 结果与同折六方法消融。
4. 明确的 promoted/rejected 判定及每个晋级门证据。
5. promoted 时的 SCOPE CSV/ZIP；rejected 时的精确父 fallback CSV/ZIP。
6. root submission checker PASS；晋级时 Aegis audit 也 PASS。
7. 完整 result report 和 artifact manifest。
8. changed files 清单，明确哪些是实现变化、哪些是测量产物。
9. 一次本地 commit SHA。
10. 明确写“未 push，等待用户推送”。

若其中任一项缺失，后续执行者必须继续当前实验段，而不是停在中间等待新的算法方向。

---

## 14. 最终判定树

    checkpoint/trust/split/group/fallback hashes 全部通过？
      否 -> 阻塞；不运行正式 cache；不得收段或 commit；恢复正确资产后从本门重新开始
      是
       |
       v
    parent parity + run1/run2 + token/grid/antisymmetry 全部通过？
      否 -> rejected；不访问 test；使用精确父 submission
      是
       |
       v
    5×3 conditional grouped nested OOF 八个晋级门全部通过？
      否 -> rejected；不访问 test；使用精确父 submission
      是
       |
       v
    final full-val calibration/refit 冻结成功？
      否 -> rejected；不访问 test；使用精确父 submission
      是
       |
       v
    生成一次 test Pass A/B 与 SCOPE submission，audit/checker 全通过？
      否 -> rejected；终态切回精确父 submission，并对父 artifacts 跑 checker
      是 -> promoted；终态为 SCOPE CSV/ZIP
       |
       v
    写报告 + manifest + 显式暂存 + 一次本地 commit；绝不 push

本判定树没有“看平台分数再决定”“临时扩大搜索”“重新调 prior/view/threshold”分支。
