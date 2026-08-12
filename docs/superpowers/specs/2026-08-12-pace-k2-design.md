# PACE-K2：反对称候选补丁证据重排设计

日期：2026-08-12
状态：设计已由用户逐节确认，等待书面规格复核

## 1. 摘要

PACE-K2（Pairwise Antisymmetric Candidate Evidence, K=2）在当前最佳单 checkpoint 推理结果上，仅对父模型 Top-1 与 Top-2 做选择性重排。它不训练新的 500 类分类器，而是沿分类头中 challenger 与 incumbent 的权重差方向，检查六个局部视图的 patch 是否持续支持 runner-up。

PACE-K2 只学习一个全局非负证据系数 β，并用 PACE 层的嵌套交叉拟合确定一个全局弃权阈值 γ。任何一致性、统计或晋级门槛失败时都回退到父模型。测试阶段 PACE 参数完全冻结，不使用标签、邻居、伪标签或在线更新；唯一批次统计是逐字复现父协议原有的 balanced-prior bias。

## 2. 动机与假设

旧 CVRG checkpoint 上的严格 OOF 诊断显示：

- 10,322 张旧验证清单中，父预测错误 2,911 张；
- 错误样本中，真实类别仍在 Top-2 的有 753 张；
- 固定删除任一视图都使整体准确率下降；
- 视图一致样本准确率高，而视图冲突样本承担大部分错误。

这些数字只证明“候选条件、逐样本证据”值得研究，不作为 PACE 的正式效果估计。PACE 使用当前最佳 R2 PartToken checkpoint 对应的冻结验证清单，报告中该清单为 10,316 张。两个清单不得混用。

核心假设是：父模型的全局融合分数可能把 a 排在 b 前面，但局部 patch 在分类头的 b−a 方向上可能给出跨尺度、跨翻转稳定的反证。若这种证据在严格 OOF 中具有较高切换精度，则只重排 b 与 a；否则保持父预测。

## 3. 目标

- 在不调整 crop、融合权重、temperature、prior strength 或 PartToken 参数的前提下，研究新的候选条件重排机制。
- 相对当前平台最佳父模型，在 raw micro 与 clean-core micro 上都取得至少 +0.20pp 的 PACE 层 conditional nested-OOF 提升。
- 证明 patch 证据评分优于 margin-only 与共享完整 patch 资格门的 gate-only 对照。
- 保持单 checkpoint、无模型集成、无测试时训练、无外部数据。
- 对高置信一致样本逐项保持父预测。
- 形成可审计、可复现、可生成有效 submission 的独立实验。

## 4. 非目标

- 首版不扩展到 K=3、K=5 或 500 类 listwise 重排。
- 不搜索 K、tail 大小、视图支持数、稳定性规则或证据公式。
- 不训练类别专属、类别对专属或样本专属参数。
- 不把类别 ID、类别对 ID、图片名、路径编码或真实标签作为评分器输入。
- 不使用 prototype、kNN、测试集类别计数、图传播或伪标签。
- 不修改当前 PartToken pooling、Adapter 或父推理路径的语义。
- 不使用旧 CVRG gate 或旧 CVRG 缓存作为正式输入。
- 不在本规格内加入遮挡、mask 后重前向或反事实裁剪。

## 5. 固定父模型与正式样本清单

正式父模型为 2026-08-05 报告中的 F1 R2 crop112 Part-Token residual Adapter composite checkpoint：

- checkpoint SHA-256：
  26916fd3ec96311dcab7a637f416ad3455cf7c78087844d408a38958f168962a
- 报告路径：
  results/f1_flat_mlp_lora_selftrain_r2_part_token_adapter_crop112_20260805.md
- 平台结果：
  68.90295189650338%，即 17,203 / 24,967

父推理协议逐项固定：

- 原图与水平翻转；
- 原图全局与翻转全局；
- 128、144、160 三个局部尺度及各自翻转，共六个局部视图；
- local scale weights = 0.45、0.50、0.05；
- flip weight = 0.50；
- local weight = 0.40；
- global temperature = 1.5；
- local temperature = 1.5；
- balanced-prior strength = 0.85；
- local top-k = 5；
- 已训练 PartToken Adapter 及其 top-8、temperature 0.07 pooling 保持不变。

实施前必须完成以下先决条件：

- 从原始 `/home/lux1/noise` 正式运行目录、团队共享归档或用户提供的只读副本恢复 composite checkpoint，并核对 SHA-256；路径可以不同，哈希不能不同；
- 审计比赛“单模型”规则是否允许把一个共享标量 β 和阈值 γ 作为冻结 checkpoint 后处理；不允许时立即关闭；
- 正式 train CSV SHA-256 必须为 `a726b8a3ca8bc5857136106aca80f01d557104d3661ef92ccedfb2c0ea087875`，正式 val CSV 必须为 `54a790b35f836cfba4c19cbb5fe38c4b1b37aab62cc9d477f9285496b2d5568e`；
- class mapping 必须恢复并核对：`class_to_idx.json` SHA-256 `0141916c3eab0b8e9471671e398acbdcbbd497fea5fbe81ec9c5bb9331ce6d65`，`idx_to_class.json` SHA-256 `8a7e030b5126124035177bd81e71c415dfb87f9b1ae6fba228b31c2cb7527913`；
- split manifest SHA-256 必须为 `d8cf3e129ed2c73ff72cbda120deee5285c5bb8e6f10f579c3d0d1c556d3587c`，且其 train/val count、CSV hash、500 类、101,980 个 unique groups、1,238 个 duplicate images 与正式记录一致；
- 从同一正式运行目录、共享归档或用户提供副本恢复 R2 trust bundle `artifacts/trust/selftrain_r1_teacher_v2_relaxed.pt`，期望 SHA-256 为 `ff8688a818219d3737715cd7dc9d11d014e7a5e973fa68f966c0ce370a88a246`；
- 不得使用旧 A2/CVRG checkpoint、相邻 epoch 或重训近似物替代。

正式验证集以该 checkpoint 报告引用的冻结 split 为准。上述任一父 checkpoint、trust、split 或 class-map 工件无法恢复并匹配期望哈希时设置 `preflight_closed=true`，不得用新训练或近似重建替代，也不得进入 Pass A/B 或 OOF；只有已恢复的 hash-verified parent submission 可用于回退。

固定的 `d8cf...` 历史 manifest 只证明正式 CSV 身份、exact-SHA grouping 统计与零跨 split 结果；它不含 `content_groups_sha256`，历史生成器也未保存 path→SHA map，因此不得伪称存在可恢复的旧 group-map 哈希。PACE group map 是新的 protocol artifact：首次正式执行时读取上述 hash-verified train/val CSV，按 `canonical_sample_path(image_path)`（即 `class/filename`）升序遍历全部 103,218 行，对每个冻结官方图片文件的原始字节计算 SHA-256；key 为 canonical path，value 为 64 位小写 file-SHA256。随后验证 101,980 个 unique groups、1,238 个 duplicate extras、train/val 零 group overlap和路径全覆盖；禁止调用 `prepare_stage` 重分 split。调用 `atomic_json_dump(path_to_group, destination)` 写入固定可追踪路径 `reproducibility/aegis_f1/protocol_artifacts/pace_k2_r2_parttoken/content_groups.json`；该路径不在 `reproducibility/aegis_f1/.gitignore` 的 `artifacts/` 规则内，preflight 必须以 `git check-ignore` 确认未被忽略。该仓库函数内部固定使用 UTF-8、`ensure_ascii=False`、`indent=2`、`sort_keys=True` 并在末尾写换行。两次独立重建必须字节与 SHA 一致，随后把所得 SHA 作为显式常数写入 machine-readable PACE protocol config 并将 map/config/report 一起本地 commit；在该常数冻结前不得进入 Pass A/B 或任何拟合，后续运行必须 exact-match，否则 `preflight_closed`。任何计数、跨 split 或 tracked-path 检查不符都 `preflight_closed`，不得以结果导向方式调整算法。

首次成功冻结该新工件后必须进入一次性的 `group_artifact_frozen` 前置协调检查点：报告 map 路径/SHA、全覆盖与 duplicate/跨 split 检查结果、protocol config 路径、本地 commit SHA 和 `origin/main` 状态，然后停在 Pass A 之前供用户推送和团队同步。它是实验开始前的协议工件冻结，不是已经运行的实验段，因此此检查点不声称产生新 submission。恢复执行时必须重新 fetch/pull 并检查 main 是否出现同类工作，再核对已同步 config 中的 group-map SHA 后才能开始 Pass A；不得在同一次未同步的长运行中越过该检查点。

仓库没有已冻结的 near-duplicate 工件，本规格不临时引入 pHash/CLIP 相似度阈值，因此“未防感知近重复”必须写入限制。图片名、标签映射、trust bundle、group map 和 split 均记录 SHA-256。

诊断口径固定引用 `balanced_inference.py::prediction_metrics`：trusted 是按 clean probability 连续加权的 noisy-label 指标；proxy target 在 correction_alpha>0 时使用 pseudo label，权重为 max(clean_probability, correction_alpha)；clean-core 是 clean_probability≥0.70。训练 selection threshold 0.60 与 anchor threshold 0.80 不是 clean-core 定义。

## 6. 父分数与候选定义

PACE 的父 log-score `ℓ` 来自上述固定父推理的最终输出：

1. 八个原始视图按固定 temperature、flip、scale 和 local 权重融合；
2. 逐字复现父协议的 full-batch balanced-prior：uniform target、strength 0.85、max iterations 50、tolerance 1e-6、damping 0.5；
3. 对应用 prior bias 后的最终 logits 做稳定 `log_softmax`，得到 `ℓ`。

第 2 步是父协议唯一允许的无标签测试批次统计。PACE 不改变其 target、算法或参数，不做第二次 prior 对齐，也不根据 PACE 输出再校准。其 bias、迭代次数和报告必须缓存并记录哈希。若比赛规则禁止该固定父步骤，则父协议与 PACE 都 fail closed。

对每张图片：

- a = argmax `ℓ`，为 incumbent；
- b = `ℓ` 中严格排序第二的类别，为 challenger；
- 排序相同时按类别索引升序稳定破 ties；
- 候选始终是父模型最终 Top-2，真实类别不得参与候选生成或补入。

父 margin 定义为：

m_q = ℓ_b − ℓ_a

它等于 prior 后最终 logits 的 `s_b−s_a`，避免先 softmax 再裁剪造成 underflow。由于 b 是 runner-up，m_q 不大于 0。

旧 CVRG OOF 分数只作为结构诊断，绝不替代 `ℓ`，也不参与 PACE 拟合。

## 7. 两遍只读特征提取

balanced-prior 在整批父 logits 收集后才确定最终 Top-2，因此 PACE 使用两遍确定性、无梯度推理。两遍使用同一个 checkpoint，不构成模型集成或测试时训练。

### 7.1 Pass A：冻结父预测

第一遍完整复现父推理并缓存：

- canonical path 与 `formal_row_id:int64`；该 ID 在每个 split 内唯一且连续为 `0..N−1`：验证集按 hash-verified val CSV 原始冻结行顺序赋值，测试集按现有 `TestImageDataset` 的稳定排序顺序赋值；ID↔canonical path 绑定逐行缓存并哈希；
- 最终父 `ℓ` 的 Top-1/Top-2 类别和对应 log-score；
- 父预测；
- 四个聚合分支的完整 pre-prior score、Top-1 与各分支 score hash；
- 六个局部视图的固定 crop box；
- corrupt-image 标志；
- 父 prior bias 与完整 prior report；
- checkpoint、split、label map、推理协议与代码哈希。

四分支的唯一公式为：

- `global_orig = softmax(G_orig/1.5)`，`global_flip = softmax(G_flip/1.5)`；
- `local_orig = Σ_s α_s softmax(L_orig,s/1.5)`，`local_flip = Σ_s α_s softmax(L_flip,s/1.5)`，其中尺度顺序 128/144/160、`α=(0.45,0.50,0.05)`；
- 每个分支独立取 argmax，tie 按类别索引升序；均不含 full-batch prior bias、local_weight 0.4、跨 orientation 的 flip 融合或另一分支。

Pass A 固定使用正式父命令的 CUDA FP32 路径：`train.amp=false`、`--batch-size 128`、同一 device/runtime manifest；不得改 batch size。validation 若没有既有同协议 prediction hash，先运行两次 Pass A 并要求 prediction/log-score/prior-report/四分支 score hash 一致，再冻结为 validation baseline。

test prediction CSV 必须逐项复现 SHA-256 `c6ed3e6a7f63c49a9b821f0e09222a153d926702de6f4c42505781aa7ae89fdd` 的已审计父 CSV；父 ZIP SHA-256 为 `6333375eea0f0b7575b833de16daf89c897df521c9eaa3f64a71e546c5ec4dc6`，manifest SHA-256 为 `0ddff9c4e03b0bfefe3c8671d388679a4a33dbbcc547b9f88d1b440fdca1c06e`。任何差异都 fail closed。

### 7.2 Pass B：只提取候选证据

第二遍复用 Pass A 的固定 crop boxes，只重跑六个局部视图。每个视图只为已冻结的 a 与 b 计算 compact evidence。不得重新生成候选、重新执行 prior alignment 或根据标签改变 crop。

Pass B 不保存原始 N×6×49×512 patch tensor。每张图片只保存六个标量证据及必要的候选、分数和审计字段。

## 8. 分类空间一致性

首版只接受当前父 checkpoint 的 `peft_mode=visual_lora_mlp_lora`、`feature_adapter=Identity` 和 linear classifier；任一条件不符都 fail closed。分类空间计算图固定如下：

- `h_vCLS` 与 `h_vp` 分别是 visual ln_post 与 visual projection 后、尚未归一化的 CLS 和 patch；
- `x_vCLS = model.adapt_features(h_vCLS)`；
- `x_vp = model.adapt_features(h_vp)`，每个 patch 只调用一次；
- 当前 Identity feature adapter 下，两者都等价于各自一次 FP32 L2 normalize。

现有 `native_visual_forward_with_patch_features` 只暴露已归一化 patch，因此实施需额外暴露 `h_vp`，或在当前 Identity 条件下直接把返回的已归一化 patch 当作 `x_vp`；禁止再调用完整 `adapt_features` 做第二次归一化。

父 PartToken 路径与 PACE 只读路径必须分叉。现有已归一化 patch 原样进入父 top-8 pooling，再与 base `x_vCLS` 一起进入冻结 PartToken Adapter 一次，以复现父 local logits。单 patch 永不进入 PartToken Adapter；PACE 不用 PartToken-adapted CLS 替换 `x_vCLS`，也不把 `x_vp` 回写父 pooling。PartToken 只影响固定父 `ℓ`。

分类头方向固定使用 linear `model.classifier.weight`。实施需通过两个小批次数值门：

- FP32 下 `base_logits ≈ W x_vCLS + bias`，固定 `atol=1e-5`、`rtol=1e-5`；
- FP32 下 `adapted_logits ≈ base_logits + W(x_vPT−x_vCLS)`，同一容差，其中 `x_vPT` 是冻结 PartToken Adapter 输出。

## 9. 反对称 patch 证据

对候选 b 相对 a，定义单位分类方向：

先在 FP32 计算 `d = ∥w_b−w_a∥₂`。若 `d < 1e-12`，该图片 ineligible 并保持父预测；否则：

u_ba = (w_b − w_a) / d

对局部视图 v 的 patch p，定义相对 CLS 残差：

r_vp(b,a) = u_baᵀ (x_vp − x_vCLS)

分类器 bias 不进入该式，因为 patch−CLS 残差会使同一类别对的 bias 自动消去。权重差归一化避免类别对权重范数成为隐式 pair ID。

每个 CLIP ViT-B/32 局部视图含 P=49 个 patch。tail 数量不作为超参数扫描，而固定为：

r_tail = ceil(sqrt(P)) = 7

将 49 个 r_vp 稳定降序排序，等值时按 patch 索引升序破 ties。视图证据定义为：

E_v(b,a) =
0.5 ×
[mean(top-7 r_vp) + mean(bottom-7 r_vp)]

E_v 是上下尾均值的中点，用于对尾部 location 做显式反对称化；它不是 top-minus-bottom、离散度或“多数 patch 支持”的计数。在实数运算下：

E_v(a,b) = −E_v(b,a)

实现按 canonical 无序类别对只计算一个方向，反向证据直接取负；容差测试还需覆盖分别重算的方向。若任一 patch、特征、权重或证据出现 NaN/Inf，当前缓存整体 fail closed，不做逐样本静默修补。

## 10. 六视图聚合与严格弃权

六个局部视图的顺序固定为：

1. original_128
2. original_144
3. original_160
4. flipped_128
5. flipped_144
6. flipped_160

聚合权重复用父协议，不学习新可靠度：

α_orientation = 0.50、0.50

α_scale = 0.45、0.50、0.05

α_v 为两者乘积并归一化到和为 1。总证据为：

E = Σ_v α_v E_v

父模型的一致性门固定使用用户已确认的四个聚合分支，而不是六个原始局部视图：

1. original global；
2. flipped global；
3. original local：严格使用第 7.1 节 `Σ_s α_s softmax(L_orig,s/1.5)`；
4. flipped local：严格使用第 7.1 节 `Σ_s α_s softmax(L_flip,s/1.5)`。

若四个分支的 Top-1 彼此完全一致，该图片属于 unanimous subset；无论 post-prior 最终 incumbent 是否与该共同类别相同，PACE 都必须逐项保持父模型最终预测。其余图片属于 branch-conflict subset。Pass A 或 Pass B 标记为 corrupt 的图片强制不具备切换资格；两个 pass 的 corrupt 状态不一致时缓存整体 fail closed。

branch-conflict 图片只有同时满足以下固定条件才具备切换资格：

- 六个 E_v 中至少四个严格大于 0；
- E 严格大于 0；
- 仅用 original 三个局部视图聚合时证据大于 0；
- 仅用 flipped 三个局部视图聚合时证据大于 0；
- 分别删除 128、144、160 任一完整尺度后，剩余视图聚合证据仍大于 0。

上述支持数和 leave-one-scale-out 规则预先固定，不依据验证结果修改。

## 11. 共享评分器

PACE 主评分为：

η = m_q + βE

约束为：

- 父 margin 系数固定为 1；
- 不使用截距；
- β ≥ 0；
- 只有一个跨全部图片、类别和类别对共享的 β；
- 不使用标准化均值、类别 ID、pair ID、每类阈值或 embedding。

β 的拟合 population 固定为相应训练折内所有 finite、candidate-covered 且真实标签位于 {a,b} 的行；不依据部署资格 mask 过滤。每张原图等权，不做 class balance、pair balance、重采样或人为把 challenger 调成 50%。令：

- t=1，当真实标签等于 b；
- t=0，当真实标签等于 a。

设 z_i=m_i+βE_i，n 为上述训练行数，β 的完整目标固定为：

min_{β≥0} (1/n) Σ_i [softplus(z_i)−t_i z_i] + β²/(2n)

`m_q` 与 `E` 在缓存中保存 IEEE-754 float64；若上游来自 FP32，先分别转成 float64 再落盘。OOF、full-fit、测试和所有对照均按同一顺序计算 `product=β*E`，再计算 `η=m_q+product`，禁止使用 FMA contraction；阈值比较也只在 CPU float64 上执行。

上述目标在 CPU float64 中计算，严格凸且允许 β=0。导数固定为 `g(β)=[Σ_i E_i(stable_sigmoid(m_i+βE_i)−t_i)+β]/n`；样本始终按 `formal_row_id` 升序逐项累加，不使用并行归约或 FMA。stable sigmoid 分支为：`z≥0` 时 `1/(1+exp(−z))`，否则 `exp(z)/(1+exp(z))`。

求解器使用确定性的导数括区间与二分：若 β=0 处导数非负，直接取 0；否则从上界 1 开始逐次加倍，直到导数非负或上界达到 2²⁰。找到括区间后最多二分 100 次，或在区间宽度不大于 1e−12 时停止；并列取中点。若未能括住根、训练折没有两种 t、输入、目标或解非有限，则当前 fit 失败，并按第 13 节传播 no-switch。

真实标签不在 {a,b} 的样本不参与 β 拟合，但必须进入阈值效用和整体准确率评估。对这些样本，a→b 对 raw accuracy 的净效用为 0。

最终预测规则为：图片必须先通过第 10 节全部资格条件；随后 `all_switch` 输出 b，`finite` 仅在 η>γ 时输出 b，`no_switch` 保持 a。其他所有情况都保持 a。

## 12. γ 的选择

切换策略只由相应训练范围内、通过第 10 节 label-free 资格条件的 inner-OOF 预测选择。令 `n_eligible` 为 eligible 行数，将有限 η 的 distinct tie groups 稳定升序排列，tie group 作为不可拆分原子。若 `n_eligible>0`，finite cut 集合唯一规定为 `C_finite={v∈distinct(η): 0 < #{i:η_i>v} < n_eligible}`；它只包含产生非空且非全切决策集的 cut。OOF 决策候选使用显式 `threshold_mode ∈ {all_switch, finite, no_switch}`：

- `all_switch`：切换所有 eligible 行；
- `finite`：只对 `C_finite` 中的值令 γ 等于该值，并严格使用 η>γ；这会排除该 tie group 及其以下各组；
- `no_switch`：不切换任何行。

`all_switch`、`C_finite` 与 `no_switch` 对全部可区分切换集合各表示一次，不允许重复 mode。`C_finite` 为空仅表示没有 finite 候选，仍须评估 `all_switch`；只有 `n_eligible=0` 时才直接规范化为 `no_switch`。若 `all_switch` 不合格，则按第 291 行的统一 fallback 保存 `no_switch`；显式 mode 避免把 ±∞ 写入 checkpoint 或 JSON。每个 inner-holdout 的 η 必须由只在对应 inner-train 上拟合的 β 产生。不使用 outer holdout 或聚合 outer OOF 结果调阈值。

对一次 a→b 切换定义：

- W：父模型错误且 b 为真实标签，即 wrong-to-correct；
- L：父模型正确且 a 为真实标签，即 correct-to-wrong；
- N：a 与 b 都不是真实标签，raw accuracy 不变。

每个非 `no_switch` 切换候选（包括 `all_switch` 与 `finite`）必须满足：

- W/(W+L) ≥ 60%，该量命名为 accuracy-changing switch precision；
- 该比例的双侧 95% Wilson 区间下界大于 50%；
- W−L 严格为正。

Wilson 分母只含 W+L；W+L=0 时区间未定义且候选不合格。它在阈值选择中只是一项预注册的确定性保守启发式，不解释为考虑了多阈值选择与 duplicate-cluster 相关后的 95% 覆盖区间。另报告 W/(W+L+N)、N/(W+L+N) 和总切换数，避免把不改变准确率的切换隐藏掉。

在合格候选中，选择 inner-OOF 的 W−L 最大者；并列时选择总切换数更少者，再选择更高的 finite γ。任何实际零切换的候选因 W+L=0 不合格；`no_switch` 也不是合格候选。无合格候选、没有 eligible η 或 W+L=0 时，保存带明确 reason 的 `no_switch` tagged record，该 outer fold 或最终模型不切换；不得用 γ=0 表示“不适用”，因为 0 可以是合法 finite cut。

OOF 选中 `finite` 时，不把其 numeric γ 直接迁移到重新拟合的 β 尺度；冻结整数 `k_oof=W+L+N`、`n_oof=n_eligible` 与 OOF cut，`target_switch_fraction ρ=k_oof/n_oof` 只作为展示和审计字段。重新拟合 β 后，在相应 refit-train 的 label-free eligible 行上重算 η；令 refit eligible 数为 `n`，只枚举同一定义的 `C_finite`，每个 cut 的实际切换数为整数 `k`。选择精确最小化 `|k*n_oof-k_oof*n|` 的 cut；并列时选择更少切换，再选更高 γ，不用浮点比例决定最近候选。该映射不得读取 refit-train 标签、outer-test 或 test 分数。若 `C_finite` 为空或 `n=0`，则映射失败并规范化为 `no_switch`。

`all_switch` 与 `no_switch` 无尺度问题，按原 mode 直接保留。最终 payload 使用两个以 `mode` 为唯一权威的 tagged records：`oof_threshold` 与 `deployed_threshold`。所有字段始终存在，JSON“不适用”值必须为 `null`：

- `oof_threshold` 保存 `mode,oof_cut,k_oof,n_oof,rho,eligible_score_hash,no_switch_reason`；`deployed_threshold` 保存 `mode,refit_gamma,k_refit,n_refit,refit_fraction,eligible_score_hash,no_switch_reason`；
- `all_switch`：cut/gamma 与 reason 为 `null`，必须已有完整 eligible score array，且 `n>0,k=n`、比例为 1.0、score hash 非空；
- `finite`：reason 为 `null`，cut/gamma 必须为有限 float，必须已有完整 eligible score array，且 OOF 与 refit 都满足 `0<k<n`、score hash 非空，展示比例严格由对应整数对计算；
- score-complete 的 `no_switch`：cut/gamma 为 `null`，`k=0`，`n` 保存实际 eligible 数，比例在 `n>0` 时为 0.0、否则为 `null`；即使 `n=0`，也必须对 canonical empty float64 array 保存非空 score hash；
- score-unavailable 的 `no_switch`：若在完整 eligible score array 产生前失败或因最终 OOF no-switch 而预注册地跳过 refit，则 `k,n,ratio,eligible_score_hash` 全部为 `null`，不得伪装成有效零样本；
- `no_switch_reason` 只在 `mode=no_switch` 时为非空枚举，其他 mode 必须为 `null`。允许值固定为 `no_eligible`、`no_qualified_candidate`、`inner_fit_failed`、`full_refit_failed`、`finite_mapping_failed`、`final_oof_no_switch`；加载器还必须校验 reason 与 score-complete/score-unavailable 字段组合相符。

若 OOF `finite` 在 refit 映射失败，保留 `oof_threshold.mode=finite`，但 `deployed_threshold.mode=no_switch,no_switch_reason=finite_mapping_failed`。成功产生 refit scores 但 eligible 数为 0 时，对 canonical empty float64 array 计算 `eligible_score_hash`，并保存 `k_refit=n_refit=0,refit_fraction=null`；若 β fit 在产生 refit scores 前失败，则 `k_refit,n_refit,refit_fraction,eligible_score_hash` 全部为 `null` 并记录相应 fit-failure reason，不得伪装为空数组。加载器按 mode 与 reason 联合校验全部不变量并 fail closed；推理只在 `deployed_threshold.mode=finite` 时读取 `refit_gamma`，all/no 必须忽略该字段。若 tensor checkpoint 被迫保留浮点槽，可另存 `gamma_storage=0.0,gamma_active=false`，但规范 JSON/payload 的 gamma 仍为 `null`。

## 13. PACE 层的 conditional nested cross-fit

正式评估使用 5 个 outer folds，每个 outer-train 内使用 3 个 inner folds；全流程 seed 固定为 42，行与 group 先按 canonical path 稳定排序。

分组单位为原图及冻结 content_groups 工件中的 exact file-SHA duplicate group。同组图片不得跨折。分组工件必须在实验前冻结并记录哈希；不得根据 PACE 结果重建。本实验没有已冻结的 near-duplicate 工件，因此不临时发明感知哈希算法或阈值，并明确记录 near-duplicate 泄漏未被保护。

fold 生成器固定为仓库 lockfile 中的 scikit-learn 1.9.0，调用 `sklearn.model_selection.StratifiedGroupKFold(n_splits, shuffle=True, random_state=42)`；输入行按 canonical path 排序，`X` 为 `formal_row_id`、`y` 为逐图片 noisy label、`groups` 为 exact duplicate group ID。outer 用 n_splits=5；每个 outer-train 独立用 n_splits=3。Python、NumPy、scikit-learn 版本、lockfile hash、outer fold ID、每个 outer 对应的 inner fold ID 都必须在任何 β/γ 拟合前生成、落盘并记录哈希；实现以冻结 fold-ID 工件为权威，复跑不得重分配。

每个 outer fold 执行：

1. outer-test 完全封存；
2. outer-train 分成三个 inner folds；
3. 每个 inner-train 拟合 β，并预测对应 inner-holdout 的 η；
4. 汇总 inner-OOF η，只用它们构造 `oof_threshold`；finite 时冻结整数 `(k_oof,n_oof)`，ρ 只作展示；
5. 用完整 outer-train 重新拟合一个 β；
6. 构造 `deployed_threshold`：OOF mode 为 all/no 时直接复制 mode 且 `refit_gamma=null`；OOF mode 为 finite 时，才只用 outer-train 的 label-free eligible 分数和整数交叉乘法映射 numeric γ，映射失败则 deployed no-switch+reason；随后应用于 outer-test 一次；
7. 保存 outer-test prediction、score、资格原因和所有指标。

Pass A/B 特征来自冻结 checkpoint、固定父协议且不直接读取标签，因此可以预计算；fold 中只有 β 与 OOF 策略选择依赖 PACE 标签。任何标准化、阈值或统计若在实现中新增，都必须只来自相应训练折，否则 fail closed。finite 的覆盖率→γ 映射只允许读取相应训练范围的 eligible score 分布，不做 outer-test 再校准。

任一 inner-train β fit 失败时，不得丢弃该 holdout 或汇总不完整 inner-OOF；对应整个 outer procedure 保存 `oof_threshold.mode=no_switch` 与 `deployed_threshold.mode=no_switch`，cut/gamma 为 `null`，所有尚未生成的计数/hash 为 `null`，并分别记录 fit-failure reason。完整 outer-train β refit 失败时，保留已生成的 `oof_threshold`，将 `deployed_threshold` 设为 no-switch、未生成字段设 `null` 并记录原因。最终全验证 3-fold 流程中，任一 inner fit 或最终 full refit 失败都令 `conditional_local_gate_pass=false`，不生成 PACE 测试候选并走 parent fallback。

完成五折后，只聚合五个在 PACE β/γ 层未见过相应标签的 outer-test 预测。不得观察聚合 outer OOF 后再修改公式、tail、支持数、β 目标、γ 约束或晋级门槛。

必须明确：固定 R2 父配置标记 `validation_overlap_with_training: true`，且 PartToken Adapter 的 epoch 也曾在同一验证范围上选择。因此这里的 nested cross-fit 只能防止 PACE scorer 层直接泄漏，不能把整条父模型流水线变成端到端未见标签的无偏估计。下文指标是“条件于已冻结父模型的内部诊断”；只有未参与父训练/选择的冻结确认集，或一次不回流调参的平台提交，才能提供外部确认。

## 14. 三方法对照（含 parent baseline 共四组）

必须用相同 parent cache、candidate、outer/inner folds 和四分支 conflict 定义运行以下对照，每个对照独立按第 12 节 nested 选择自己的 γ，不强行匹配切换数。

### 14.1 Margin-only（共享 parent-conflict gate）

η_margin = m_q

该对照不读取 E、E_v、patch 支持票或 leave-one-scale 证据条件，只在 branch-conflict subset 上切换。它回答“提升是否只是低 margin 样本阈值切换”；由于共享父 conflict gate，不把它误称为字面上的 pure-q。

### 14.2 Gate-only（共享完整 patch eligibility）

η_gate = m_q

该对照使用与 PACE 完全相同的 patch-derived eligibility mask，但评分不加入 βE，并独立 nested 选择 γ_gate。它回答“patch 只作为资格门是否已经解释全部提升”。

### 14.3 PACE

PACE 必须在 raw micro 和 clean-core micro 的点估计上都严格优于 margin-only 与 gate-only，才可晋级；这只支持“主方法点估计胜过两项消融”，不单独构成统计显著的机制归因。可额外报告不使用 conflict gate 的 pure-q 全样本诊断，但它不作为主晋级对照。

另外报告 Top-2 oracle ceiling：若真实标签在 {a,b} 中就选真实标签，否则保持 a。oracle 只用于上界分析，绝不生成模型或 submission。

## 15. 晋级门槛

相对同缓存、同父协议的固定父预测，PACE 必须同时满足：

- outer-OOF raw micro 提升至少 +0.20pp；
- outer-OOF clean-core micro 提升至少 +0.20pp；
- raw 至少净增 `ceil(0.002×N)` 个正确预测；N=10,316 时为 21；
- raw micro 与 clean-core micro 点估计都分别严格高于 margin-only 与 gate-only；
- 至少 4/5 个 outer folds 的 raw micro delta 非负；
- 最差 outer fold 相对父模型的 raw micro delta 不低于 −0.10pp；
- 相对父模型的 raw macro、trusted macro、proxy macro、clean-core macro 任一 delta 不低于 −0.05 percentage points；
- unanimous subset prediction 与父模型逐项完全相同；
- 聚合 outer-OOF PACE switches 的 W 大于 L；
- 聚合 outer-OOF PACE switches 的 W/(W+L) 至少 60%，且 95% Wilson 下界大于 50%；
- 以 exact duplicate group 为重采样单位的配对 cluster bootstrap，raw accuracy delta 的 95% 区间下界大于 0；
- PACE 的验证预测类别覆盖不得少于父预测，500 类 label map 必须完整有效；
- 两次确定性运行产生完全相同的 fold、OOF prediction、β、两个 threshold tagged records 和报告哈希。

McNemar exact test、raw 与 clean-core 的 W/L/N、每折变化和置信区间全部报告，但 McNemar p 值不是替代上述门槛的通行证。Wilson 使用 `z=1.959963984540054` 的标准 score interval 公式，不做连续性校正；它只执行第 12 节预注册启发式门，不作为 post-selection 或 cluster-valid 正式推断。

cluster bootstrap 使用仓库锁定的 NumPy 2.5.1 与 `Generator(PCG64(42))`，固定 10,000 次，以 exact duplicate group 为抽样单位；按每组冻结 noisy label 众数分层，并列取最小类别索引，每个 stratum 有放回抽取与原 stratum 相同数量的 groups。每次重采样保留组内全部行，用图片数加权重算配对 delta。95% 区间取 percentile 2.5% 与 97.5%，使用 NumPy `quantile(method="linear")`。该 cluster bootstrap 是主置信区间；Wilson 与 McNemar 保留为样本级描述性统计。

clean-core、trusted 和 proxy 是冻结的噪声诊断代理，不宣称为真实干净标签。

## 16. 全量拟合与冻结 checkpoint

第 15 节全部通过只先标记 `outer_promotion_gate_pass=true`，尚不能标记 PACE 可部署。还必须完成以下冻结全量流程并通过最终 `final_deployment_gate`，才设置 `conditional_local_gate_pass=true` 并生成正式 PACE 测试候选。该标记不宣称端到端泛化已获无偏证明；一次不回流调参的平台结果只作为外部确认。

1. 在完整验证集上按同一 3-fold OOF 过程产生 η；
2. 只用这些 OOF 结果构造最终 `oof_threshold`；finite 时冻结整数 `(k_oof,n_oof)`，ρ 只作展示；
3. 若最终 `oof_threshold.mode=no_switch`，保留 OOF record 的原 reason，但构造 `deployed_threshold.mode=no_switch,no_switch_reason=final_oof_no_switch`，所有未生成的 refit 计数/hash 为 `null`，设置 `final_deployment_gate_pass=false`，跳过 full β/test evidence 并走 parent fallback；否则用完整验证集拟合最终 β；
4. full β fit 失败时部署记录为 `no_switch_reason=full_refit_failed` 且 final gate 失败；fit 成功时，OOF all-switch 直接部署 all-switch 且 gamma 为 `null`，OOF finite 才用完整验证集的 label-free eligible 分数与 `(k_oof,n_oof)` 的整数交叉乘法映射 numeric γ，映射失败则部署 `no_switch_reason=finite_mapping_failed` 且 final gate 失败；
5. `final_deployment_gate_pass=true` 的必要且充分条件是：full β fit 成功、两个 tagged records 通过 schema/hash 审计、`deployed_threshold.mode∈{all_switch,finite}`，并且 `n_refit>0`、`k_refit>0`。仅此时设置 `conditional_local_gate_pass=true`，将 evidence spec、β、两个 threshold tagged records、映射审计、资格规则、所有哈希和 promotion report 嵌入父 composite checkpoint 的 `pace_k2` payload；
6. 保存新的单一 composite checkpoint 并记录 SHA-256。final gate 失败时不得把与父预测等价的 no-switch checkpoint/submission 标为 PACE 成功，必须记录原因并走 parent fallback。

测试推理只加载这一个 composite checkpoint。不得在测试时重新拟合 β、选择 γ、改变父 prior 算法/参数、进行 PACE 特有再校准、修改规则或读取验证标签工件；第 6 节固定的父 full-batch prior bias 仍按原协议计算并审计。

## 17. 测试推理与 submission 回退

### 17.1 PACE 通过

只有 `preflight_closed=false`、`runtime_closed=false` 且 `conditional_local_gate_pass=true` 时，才允许开始生成 PACE 测试候选；此时仍是 `pace_candidate`，尚不能标记最终成功：

1. Pass A 完整复现父测试推理并冻结最终 a、b、log-score `ℓ` 与 crop boxes；
2. Pass B 只为 a、b 生成六视图 compact evidence；
3. 使用 checkpoint 内冻结的 β、`deployed_threshold` tagged record 和资格规则逐图片决定 a 或 b；
4. 生成 pred_results.csv、submission.zip 和 manifest；
5. 运行仓库九项 submission checker；
6. 复跑 prediction 决策并核对哈希。

只有以上六步全部成功，且最终确认 `runtime_closed=false`，才设置唯一终态 `experiment_outcome=pace_success`。submission 生成、九项 checker、ZIP/CSV 审计或确定性复跑哈希任一失败，都必须改设 `runtime_closed=true`，令 `experiment_outcome=parent_fallback` 或 `external_artifact_blocker`；此前为 true 的 gate 仅作为历史审计值保留，不能覆盖 runtime 失败。

### 17.2 PACE 关闭

若 preflight 通过但 `runtime_closed=true`、`outer_promotion_gate_pass=false` 或 `final_deployment_gate_pass=false`，则本关闭路径优先于任何已为 true 的 gate：

- 若尚未开始则不生成 PACE test evidence；若已部分生成，则立即停止并隔离/删除全部未审计的 PACE test cache/submission；
- 不把失败的 β、threshold tagged records 或切换规则应用于测试；
- 在 PACE 实验目录生成或复用 hash-verified 的父模型 prediction；
- 生成并验证父模型 fallback pred_results.csv 与 submission.zip；
- manifest 显式记录 fallback_parent=true、PACE 关闭原因和父 checkpoint SHA。

`runtime_closed` 表示通过 preflight 后，第 18 节任一 fail-closed 条件在 Pass A/B、cache/fold 审计、OOF、拟合或测试生成中触发。触发时必须原子地保存首个失败阶段与 reason，所有尚未产生的 gate 设为 `null` 而不是伪造 false，停止后续 PACE 步骤并丢弃未审计的 PACE cache/submission；只允许按本节复用 hash-verified parent prediction/submission，生成 failure report/manifest、运行 checker 并本地 commit。若 hash-verified parent submission 也不可恢复，则与 preflight 关闭相同，标记 external-artifact blocker，不能声称 submission-ready。

若 `preflight_closed=true`，不运行 Pass A/B、fold、OOF 或三方法对照。此时只有在已审计父 CSV/ZIP 可从正式运行目录、共享归档或用户副本恢复且分别精确匹配第 7.1 节哈希时，才复制到 PACE 实验目录并运行 checker；不得现场用缺失或近似父工件重新生成。若连 hash-verified parent submission 也无法恢复，则本段是明确的 external-artifact blocker，不能声称 submission-ready，并向用户报告所缺工件。

只有 hash-verified parent submission 可恢复时，本关闭路径才满足“每个实验分段具有可提交产物”的团队规则；否则按 external-artifact blocker 暂停并明确汇报，不能声称 submission-ready。任何情况下都不把验证失败的方法带到测试集。

## 18. Fail-closed 条件

终态唯一枚举为 `pace_success`、`parent_fallback` 或 `external_artifact_blocker`，必须且只能设置一个；`runtime_closed=true` 时禁止 `pace_success`。gate 描述算法阶段历史，不是终态。

以下任一情况在 preflight 前触发时设置 `preflight_closed=true`；在 preflight 通过后触发时设置 `runtime_closed=true` 并按第 17.2 节终止，不得继续到尚未产生的 gate：

- 父 checkpoint SHA、split SHA、label map、协议或代码版本不匹配；
- 正式验证清单与报告样本数不一致且没有先完成审计；
- Pass A 不能逐项复现父 prediction；
- Pass B 图片顺序、crop boxes、candidate 或 view order 与 Pass A 不一致；
- true class 被注入 Top-2；
- duplicate group 跨 outer/inner fold；
- outer holdout 参与 β、任一 threshold tagged record 或任何标签依赖统计；
- 除第 6 节固定父 prior 外，测试集参与拟合、标准化、阈值、邻居、类别计数或协议选择；
- linear classifier weight 无法精确取得，或父 checkpoint 不是规定的 linear/Identity feature-adapter 配置；
- patch 与 base CLS 未按第 8 节通过同一个基础 feature map 恰好一次；
- 重复应用 PartToken Adapter 或修改父 local logits；
- candidate、权重差、特征、证据、β、finite-mode γ、score 或报告出现 NaN/Inf；
- 缓存缺行、重复图片、类别越界、corrupt 状态跨 pass 不一致或非确定性复跑；
- 任一 tagged record 的 `mode` 或 `no_switch_reason` 不在预注册枚举中、mode/reason-specific null/计数/hash 不变量不成立，或 `finite` mode 没有有限 cut/gamma；
- 观察 OOF 或平台结果后修改预注册规则而未另立新实验规格；
- 竞赛规则审计认定冻结的共享 scorer 仍违反单模型限制。

不允许静默使用近似 checkpoint、近似 crop、缺失视图或默认参数继续。

## 19. 组件边界

建议新增独立组件，不修改现有平台最佳默认路径：

- aegis_clip/candidate_patch_evidence.py
  - linear classifier weight 与分类空间数值门；
  - patch/CLS 空间对齐；
  - 反对称 tail evidence；
  - 六视图资格与最终选择。
- aegis_clip/pace_crossfit.py
  - grouped outer/inner fold；
  - β 拟合；
  - γ 选择；
  - margin-only、gate-only、paired metrics 与 promotion gate。
- aegis_clip/cli/cache_pace_parent.py
  - Pass A parent cache 与 bit-exact audit。
- aegis_clip/cli/cache_pace_evidence.py
  - Pass B compact validation/test evidence。
- aegis_clip/cli/evaluate_pace_k2.py
  - conditional nested OOF、三方法对照、晋级判断与 final payload。
- aegis_clip/cli/infer_pace_submission.py
  - 只读冻结 payload 并生成 PACE 或 parent fallback submission。
- tests/test_candidate_patch_evidence.py
- tests/test_pace_crossfit.py
- tests/test_pace_submission.py

所有固定常数还必须落盘到 machine-readable protocol config，例如 `reproducibility/aegis_f1/configs/pace_k2_r2_parttoken.yaml`，并记录该配置哈希；实现不得从文档外补默认值。

新实验使用独立 output 目录，例如 outputs/pace_k2/r2_parttoken_crop112，不覆盖 outputs/cvrg 或现有 PartToken 资产。

## 20. 缓存契约

### 20.1 Parent cache

至少包含：

- schema_version；
- canonical paths、`formal_row_id:int64` 及逐行 path-binding hash；
- candidate_indices，形状 N×2；
- candidate_parent_log_scores，形状 N×2；
- parent_predictions；
- constituent_scores，形状 N×4×500；constituent_top1，形状 N×4，顺序与第 10 节四个聚合分支一致，并保存 score hash；
- local_crop_boxes，形状 N×2×3×4；
- corrupt flags，并要求 Pass A/B 一致；
- parent prior bias、迭代次数与 prior report hash；
- `oof_threshold` 与 `deployed_threshold` 只属于最终 payload，不从测试缓存估计；
- checkpoint、train/val split、class-map、trust bundle、exact-group-map、protocol、fold-ID 与 code hashes；
- validation-only 原始字段：`label:int64`、`clean_probability:float32`、`pseudo_label:int64`、`correction_alpha:float32`；由加载器按 `correction_alpha>0` 构造 corrected proxy target/weight，并按固定阈值 0.70 构造 clean-core mask。测试缓存不得包含这些字段。

### 20.2 Evidence cache

至少包含：

- 与 parent cache 相同的 paths、formal row IDs、candidate_indices、corrupt flags、split/class-map/trust/exact-group hashes；
- per_view_evidence，形状 N×6；
- aggregate evidence；
- support count；
- orientation 与 leave-one-scale stability flags；
- linear classifier mode、weight hash 与分类空间数值门结果；
- evidence spec、machine-readable protocol hash 与 parent-cache hash；
- validation-only 四个原始诊断字段逐行复制并核对 parent-cache hash；测试缓存不得含标签、trust 或 proxy 字段。

所有 tensor shape、dtype、finite、行对齐、候选范围、view order 和 hash 在加载时验证。`formal_row_id` 必须 int64、唯一、连续且与 canonical path 的冻结绑定一致；parent/evidence cache 必须按该 ID 逐行一致。compact cache 不保存 raw patch tokens。

## 21. 测试计划

### 21.1 数学与单元测试

- canonical 方向只计算一次且反向直接取负；分别重算方向时以 `atol=1e-6`、`rtol=1e-6` 验证 r、每视图 E 和总 E 的反对称；
- 给 classifier 所有 class weight 加共同向量时证据不变；
- classifier bias 改变不影响 patch−CLS evidence；
- base linear logits 与 PartToken anchored residual logits 都通过第 8 节数值门；
- patch 与 base CLS 通过相同基础 feature map 恰好一次，且父 pooling 支路未被替换；
- top/bottom tie 按 patch index 稳定；
- unanimous subset 永不切换；
- 少于 4/6 支持、任一 orientation 非正或任一 leave-one-scale 非正均回退；
- β=0 时主 score 等于 q margin；
- `deployed_threshold.mode=no_switch` 时逐项复现父预测；all-switch、`C_finite`、no-switch 与 tie-group 决策集合均有边界测试，且不存在零切换 finite cut 或重复 mode；单一 tie group 时分别覆盖 all-switch 合格与不合格两例；
- corrupt 图片强制不切换，Pass A/B corrupt 状态不一致时拒绝缓存；
- NaN/Inf、零权重差、错误 shape/view order 全部拒绝。

### 21.2 Cross-fit 与泄漏测试

- duplicate group 不跨 outer 或 inner fold；
- 每个 outer sample 只由未见其标签的 β/γ 预测；
- `oof_threshold` 与 `(k_oof,n_oof)` 只由 inner-OOF 选择，ρ 仅由该整数对展示；`deployed_threshold.refit_gamma` 只由训练范围的 label-free scores 映射；
- β refit 尺度漂移时 numeric OOF γ 不得直接复用；覆盖率映射用整数交叉乘法，等距候选的“少切换、再高 γ”并列规则精确回归；
- candidate 从父 log-score `ℓ` 生成且没有 true-label injection；
- 真实标签不在 candidate pair 的样本不参与 β，但进入阈值效用和总指标；
- margin-only 不读取任何 patch evidence 字段；
- gate-only 只读取资格 mask，不读取 E 作为 score；
- 相同 seed 和输入得到相同 fold、β、两个 threshold tagged records、OOF prediction 与报告 hash。

### 21.3 集成与回归测试

- 正式 train/val/class-map/manifest/trust/checkpoint 哈希、行数和路径全覆盖全部验证；
- PACE group map 在固定 `protocol_artifacts/pace_k2_r2_parttoken/content_groups.json` 路径两次独立重建字节/hash 一致，`git check-ignore` 返回未忽略，工件确实进入冻结 commit，并满足 duplicate 计数与 train/val 零 group overlap；首次 SHA 冻结进 protocol config 后，后续 exact-match，且流程未调用 `prepare_stage` 重分 split；
- `formal_row_id` 的 int64 dtype、唯一性、连续性、split 范围与 path-binding 在 parent/evidence cache round-trip 后保持一致；
- final deployment gate 覆盖最终 OOF no-switch、full refit failure、finite mapping failure、zero-switch record 四条父回退路径；只有 audited all/finite 且 `n_refit>0,k_refit>0` 才生成 PACE composite/submission；
- all/finite/no 两个 tagged records 的 JSON round-trip、hash、mode/reason 联合不变量、null 字段和推理忽略语义逐项回归；分别覆盖 score-complete 的空 eligible 数组使用 canonical empty-array hash，以及 inner/full fit failure 与 `final_oof_no_switch` 在未生成 score 时使用 null 计数/hash；
- Pass A 逐项复现父 prediction，四分支 score/Top-1 逐样本与现有 fusion 函数先 softmax 后加权的参考实现一致；
- 从缓存 crop box 重建的局部图与 Pass A 一致；
- compact evidence 与小批次保留 raw patch 后的参考计算一致；
- 第 18 节每个 fail-closed 阶段都能进入唯一的 preflight/runtime closed 终态；runtime failure 不要求伪造后续 gate，未审计 PACE 资产被丢弃，并只生成 hash-verified parent fallback 或明确 external-artifact blocker；
- 关闭 PACE 时现有 infer 默认输出不变；
- parent fallback CSV/ZIP 必须分别精确匹配第 7.1 节 SHA；Pass A 无法复现时只能复用 hash-verified 工件，不得生成近似 parent；
- pass/fail 两条 submission 路径都生成格式正确的 CSV/ZIP；
- 24,967 张测试图无缺失、无重复、标签四位且范围正确；
- ZIP 根目录只含 pred_results.csv；
- PACE 通过时 final composite 只需一个 checkpoint 文件完成推理。

## 22. 强制报告

每个 outer fold及聚合结果至少报告：

- raw、trusted、proxy、clean-core 的样本数、micro、macro 与 delta；
- parent Top-2 recall 与 oracle ceiling；
- unanimous/conflict/eligible/switch/fallback 数量和比例；
- 4/6、orientation、leave-one-scale 各过滤阶段保留率；
- `preflight_closed`、`runtime_closed`、`outer_promotion_gate_pass`、`final_deployment_gate_pass`、`conditional_local_gate_pass`、唯一 `experiment_outcome` 及首个关闭阶段/reason；尚未产生的 gate 必须为 `null`，任何 runtime failure 或 deployed no-switch 必须报告为 parent fallback/blocker，不能标作 PACE 成功；
- β、跨折 β 离散度、两个 threshold tagged records、`(k_oof,n_oof)`、展示 ρ、`(k_refit,n_refit)`、整数距离 `|k_refit*n_oof-k_oof*n_refit|`、finite 候选数、deployed γ、refit score/switch-fraction drift、映射失败/降级原因、η 和 E 的分布摘要；margin-only、gate-only、PACE 分别记录；
- candidate-covered binary scorer 的 Brier、log-loss、ECE；
- W、L、N、switch precision、Wilson interval；
- exact-duplicate cluster bootstrap interval 与 sample-level descriptive exact McNemar；
- predicted-class coverage 与每类变化分位数；
- parent、margin-only、gate-only、PACE 四组逐项比较；
- checkpoint、split、fold、cache、prediction、report、spec commit 和 code commit hashes。

结果文档必须明确区分：

- 旧 CVRG 诊断；
- 当前父模型正式 baseline；
- PACE 层 conditional nested-OOF 指标及父模型验证污染限制；
- margin-only、gate-only ablation；
- 测试 submission 审计；
- 是否真实晋级或仅 parent fallback。

## 23. 协作与 Git 边界

- 实施新实验段前，先在 main 检查 dirty worktree、全部 refs 与 origin/main 最新状态。
- mixed worktree 使用 git pull --rebase --autostash origin main；Git 自动恢复后不得再 stash pop。
- 再次检查 teammate 是否已提交同类 candidate/patch rerank。
- 首次 `group_artifact_frozen` 是一次性的实验前协调检查点：按第 5、26 节单独汇报并停在 Pass A 前；用户同步后，下一次继续必须重新检查/pull main，不能把未同步的 protocol SHA 直接带入实验。
- 直接在 main 工作，不创建分支。
- 只做本地 commit，不 push；push 由用户执行。
- 只 stage 本实验明确文件，保留所有无关 dirty/untracked 资产。
- 每个实验段结束时汇报 experiment ID、完整命令/config、指标、文件、submission 路径与 checker 结果、本地 commit SHA、origin/main 状态。

## 24. 风险与取舍

- 当前父模型训练/PartToken 选择和多轮人工方向选择都使用过正式验证范围；conditional nested OOF 只能防 PACE scorer 层直接泄漏，不能给出端到端无偏泛化估计。因此本规格 commit 后冻结协议，平台只提交本地通过的单一主候选，平台结果不得回流调规则。
- 第二遍局部推理增加计算，但避免保存约 7.5GB raw patch，并保证候选来自 post-prior 的真实父 Top-2。
- linear classifier 是全局判别器，不保证其权重差一定是局部原型；三方法对照、严格弃权和 promotion gate 用于证伪该假设。
- top/bottom tail 可能关注背景或坏图；跨六视图支持与 leave-one-scale-out 规则限制单 patch 极值。
- 候选覆盖决定理论上限；首版选择 K=2 是为了减少多重比较和验证过拟合。
- clean-core/trusted/proxy 是噪声代理，只作为安全指标。

## 25. 参考依据

- Hao et al., Class-Aware Patch Embedding Adaptation for Few-Shot Image Classification, ICCV 2023:
  https://openaccess.thecvf.com/content/ICCV2023/html/Hao_Class-Aware_Patch_Embedding_Adaptation_for_Few-Shot_Image_Classification_ICCV_2023_paper.html
- Pei et al., Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection, CVPR 2025:
  https://openaccess.thecvf.com/content/CVPR2025/html/Pei_Seeing_What_Matters_Empowering_CLIP_with_Patch_Generation-to-Selection_CVPR_2025_paper.html

上述工作支持“类别相关 patch 证据可能补充全局 CLIP 表示”的研究动机；PACE 的反对称 pairwise residual、nested abstention 和具体工程协议为本仓库独立设计。

## 26. 完成定义

首次正式运行若尚无冻结的 PACE group-map SHA，允许且必须先停在一次性的 `group_artifact_frozen` 检查点，但仅在以下全部完成后：

1. 对 hash-verified 正式 CSV 和官方图片字节独立重建两次，map 字节与 SHA 完全一致；
2. 全覆盖、101,980 个 unique groups、1,238 个 duplicate extras 与 train/val 零 group overlap 检查全部通过；
3. 新 SHA 已写入 machine-readable protocol config，固定路径的 map/config/report 均已确认不受 ignore 并在 main 同一本地 commit 中，且没有 push；
4. 已向用户汇报 map SHA、检查结果、文件、commit SHA 与 `origin/main` 状态，并停在 Pass A 前供团队同步；
5. 后续恢复时重新检查/pull 最新 main 与重叠工作，并 exact-match 已同步 SHA。该前置检查点未运行模型实验，不宣称新 submission；进入 Pass A 后仍必须满足下述正常实验完成定义。

正常实验路径只有在以下全部完成后才可暂停：

1. 当前最佳 checkpoint 已取得并核验 SHA；
2. Pass A/B 缓存通过完整审计；
3. conditional nested-OOF、margin-only、gate-only、oracle、置信区间与 promotion gate 已生成；
4. 两次确定性复跑哈希一致；
5. outer promotion 与 final deployment 两道门都通过时才生成冻结 PACE composite checkpoint；任一道失败都明确关闭并记录原因；
6. 只有 PACE submission 生成、九项 checker、ZIP/CSV 审计与确定性复跑全部成功且 `runtime_closed=false`，才保存唯一 `experiment_outcome=pace_success`；任一步失败都由 runtime 关闭优先覆盖并走 parent fallback/blocker；
7. 最终可提交路径的 pred_results.csv 与 submission.zip 通过九项 checker；
8. 结果报告和审计 manifest 写全；
9. 相关文件已在 main 做本地 commit，但没有 push；
10. 已向用户汇报并停在团队协调检查点。

`preflight_closed` 路径可在不运行 Pass A/B/OOF 的情况下暂停，但必须：

1. 记录规则审计或缺失/错哈希工件的精确关闭原因；
2. 若 hash-verified parent CSV/ZIP 可恢复，则复制、通过 checker 并记录源哈希；
3. 若 parent submission 也不可恢复，则明确标记 external-artifact blocker，而非伪造 submission-ready；
4. 写出 preflight report/manifest，本地 commit 相关实现或报告且不 push；

`runtime_closed` 路径可在不伪造后续 Pass/OOF/gate 的情况下提前暂停，但必须：

1. 记录首个失败阶段、reason、已完成工件哈希，并将尚未产生的 gate 保存为 `null`；
2. 删除或隔离所有未审计的 PACE cache/submission，不把它们用于回退；
3. 若 hash-verified parent CSV/ZIP 可恢复，则复制、通过 checker 并记录源哈希；否则明确标记 external-artifact blocker；
4. 写出 runtime failure report/manifest，本地 commit 相关实现或报告且不 push，并向用户汇报后停在团队协调检查点。
