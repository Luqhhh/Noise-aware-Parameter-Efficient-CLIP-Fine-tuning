# 初赛 75 分冲刺：工程执行方案 v2

日期：2026-09-12。项目：`Luqhhh/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning`。
源码基线：`fece41ba5df33d773de7af575b209fab6e385bf7`，已执行 fetch 和自动 stash 模式的 `git pull --rebase --autostash origin main`，远端无新增提交。

初始状态（83c03dd）：方案及只读资产核对完成，H/V/C未执行。
执行更新：用户已授权“按顺序执行”；新训练/诊断/交付入口已实现并通过初始检查，完整P缓存及H0/H1/V0/V1交付已完成，H0/H1均低于P，H及C关闭，V0平台67.4090%暂保留，V1平台待回填；历史最高70.352866%未刷新。实际执行状态以 [current_execution_plan.md](current_execution_plan.md) 和本地运行目录为准，以下规格不冒称实测提分。
本文件将用户提供的 v2 方案落实到当前仓库，并补齐末轮监督、双 Adapter 残差缓存和几何绑定的实施选择。方案、配方与停止条件由助手选择，不要求用户先判断瓶颈。

## 1. 工作基线与目标

| 对象 | 已登记平台准确率 | 本轮用途 |
|---|---:|---|
| FULLFT_DUAL + 四尺度/Flip + prior 0.90 | 70.352866% | 历史最高，保留原模型和原包 |
| 同一个 FULLFT_DUAL、同视图、无校准 P | 66.7681% | 所有新候选的直接对照 |
| 共享分类头范数对齐、同视图、无校准 | 63.3676% | 已否决，关闭 |
| 恢复 Selftrain R1、单视图、无校准 | 61.4010% | 归档，不作为起点 |

历史最高对应 17,565/24,967；至少 75% 需要 18,726 个正确预测，即**净增加 1,161 个正确预测**。历史最高差 4.647134pp，无校准 P 差 8.2319pp，分开报告。恢复模型与 P 同时改变了模型和视图，不能据此归因 self-training。

我选择先做低成本分类头对照，再做表示学习改动。H 用于判断现有表示能否通过重新学习分类边界提高分数；V 是本轮主要训练投入。不存在已证实足以弥补上述差距的方法，不预填预期成绩。

## 2. 本机资产核对结果

只读核对汇总见 [prelim75_v2_preflight_20260912.json](../results/prelim75_v2_preflight_20260912.json)。以下字段来自实际文件，不只是历史文档路径。

| 资产 | 路径，相对仓库根目录 | 实际状态 |
|---|---|---|
| 完整 P | `reproducibility/aegis_f1/outputs/F1_FLAT_FULL_FT_R3MS/seed42/dual_adapters/best.pt` | 存在；SHA-256 为 `f72b0104257f49d2667fe335553a861dd1dea947753feebdc7301b8890b48765` |
| 原 trust | `reproducibility/aegis_f1/artifacts/trust/selftrain_r2_teacher_v3_multiscale.pt` | 存在；SHA-256 为 `6868041cc7b995a3e8e557ae925d1d25160acf23af09202f46911ce92125b30f` |
| P 的训练清单 | `reproducibility/aegis_f1/artifacts/stages/preliminary/final_full_train.csv` | 103,218 行；原 trust 完整覆盖 |
| 类别映射与旧诊断清单 | `reproducibility/aegis_f1/artifacts/stages/preliminary/seed42/` | 映射、`val.csv` 存在 |
| 原配置引用的 OpenAI 特征张量 | `reproducibility/aegis_f1/artifacts/stages/preliminary/features/clip_vit_b32_openai/features.pt` | 缺失，不能直接运行旧配置 |
| 新恢复的 OpenAI 特征 | `outputs/stage_readiness/20260912_full_rebuild_r1/features/features.pt` | 存在；SHA-256 为 `118f658c1837ee93611ae04f2a4e4630439a02e5ef921de0466d06beedfb6dfe`；不认证与丢失缓存数值等价 |
| 历史最高提交包 | `outputs/delivery/fullft_dual_pa0.9/submission.zip` | 哈希匹配；本次格式、覆盖及包内外 CSV 一致性复检通过 |
| P 无校准提交包 | `outputs/preliminary_no_sweep/20260911_parent_nocal_control_r1/submission/submission.zip` | 哈希匹配；同上复检通过 |

P 实际 checkpoint 含 epoch3 全微调权重、O3 BN32 和 PTA BN64；PTA 的 pooling 是 top8、temperature 0.07。两项 Adapter 都共享原分类头。不得用恢复模型或缺 Adapter 的模型代替。

基于 P epoch3 的目标规则，实际只读统计得到：76,954 行有非零权重，26,264 行权重为零；非零权重且向其他类别软修正的样本为 5,708 行。有效监督总量 74,659.257544；逐类 min/median/max 为 54.858768 / 152.100742 / 214.719667，最大最小比 3.914045，零监督类别为 0。

这些数值支持 H1 能合法计算正先验，并表明筛选后监督存在不均衡；**不能证明长尾是主因，也不是标签真值质量或泛化证据。**历史训练清单与旧验证重叠，不为补齐诊断重跑七节点或完整谱系。

## 3. 执行顺序与预算

| ID | 唯一主要变化 | 训练预算 | 平台候选上限 |
|---|---|---:|---:|
| H0 | 冻结 P，普通加权 soft-target CE 重拟合共享 head | 10 epoch，缓存特征 | 1 |
| H1 | 同 H0，仅训练 loss 加有效监督 log-prior | 10 epoch，同一缓存 | 1 |
| V0 | 从原 P 做 global-only 短微调 | 3 epoch | 1 |
| V1 | 从原 P 做 global/local 联合短微调 | 3 epoch | 1 |
| C | 条件触发，在 V 胜者新特征上重跑 H 胜者配方 | 10 epoch，新缓存 | 1 |

顺序固定为 H0/H1 → V0/V1 → 等待真实平台成绩 → 满足条件才 C。H 的权重不作 V 初始化。H 失败仍完成 V，不要求用户解释失败原因。任何候选提前达到 75%，立即停止剩余新增训练。

总量不超过 6 epoch 在线微调、30 epoch 缓存 head 训练和五个新平台候选。特征生成、定位、锚定和候选推理另计资源；不承诺未经硬件测量的 GPU 小时。已有分支负结果和全部本地/远端 refs 已检查：复用现存方法基础设施；不重开 shared-multiscale frozen-backbone Adapter 实验。

输出候选目录为 `outputs/prelim75_v2_20260912/{H0,H1,V0,V1,C}/`；单个候选目录已存在即停止，不覆盖，也不隐式 resume。每段完成必须生成并验证 CSV/ZIP、记录配置、度量和本地 commit 后才交接；线上回填期间继续不依赖分数的既定工作。

## 4. 固定无校准推理协议

沿用 P 已登记协议：ViT-B/32 原生 224；attention top-k5；crop112/128/144/160，尺度权重 0.2/0.3/0.4/0.1；local weight 0.4；Flip weight 0.5；global/local temperature 均为 1.5；O3/PTA 同时启用，PTA pooling 保留实际 spec。

十个几何视图的概率聚合权重为：global 原图/Flip 各 0.3，local 每尺度原图/Flip 各 `0.2 * scale_weight`。每张图权重合计 1。先得到每视图当前共享 head 的 logits，再按原代码做 softmax 和概率融合，不平均 embedding 后只做一次 softmax。

单个候选只包含一个 checkpoint、一个共享 head、一个确定性推理流程。H/V/P 不互相投票；定位使用候选自身的视觉 attention。源码表明当前定位不依赖分类头，H 只修改 head 时 boxes 可保持，但仍须运行真实新 head 推理；V 改变视觉参数后必须重算 attention/boxes。

仓库 [COMPETITION_RULES_AGENT.md](../COMPETITION_RULES_AGENT.md) 第18节已登记组委会于 2026-08-31 允许同一 checkpoint 的多尺度＋Flip。沿用此确认，不再次将其设为开工审批条件。当前正式 infer 禁止测试批内 prior 拟合；本轮无 prior 校准，不迁移历史 bias。

## 5. H：冻结表示，重新学习共享分类头

### 5.1 冻结 P 的实际监督

从实际训练清单、类别映射和原 trust 导出 `sample_id / original_label / q / w / group_id`，分布与权重固定为 P 末轮规则：

```text
w_i = (0.6 + 0.4 * clean_i) if clean_i >= 0.60 else 0
q_i = (1 - alpha_i) * onehot(original_i) + alpha_i * onehot(safe_pseudo_i)
n_eff[c] = sum_i w_i * q_i[c]
pi[c] = n_eff[c] / sum_k n_eff[k]
```

`safe_pseudo` 与 `alpha` 的合法范围/回退复用 `corrected_targets()` 和 `TrustBundle`。不以 teacher argmax 冒充软目标，不重新清洗或生成教师。关键：旧 trainer 在 `epoch <= correction_start_epoch` 时关闭软修正；本轮明确继承 **P epoch3 已开启的规则**，H 十轮全程固定 q/w，不能因新循环从1开始意外关闭首轮软修正。

先验只来自既有拟合清单，不额外读取旧验证/测试来拟合。既有清单包含旧验证内容的事实保留，不能宣称独立验证。多视图不重复计类别次数。自然样本抽样，关闭 class-balanced sampler、逆频率 loss 权重。任何类别 `n_eff=0` 时关闭 H1，不用 epsilon 捏造监督；本次预检无零类。

### 5.2 两个固定 loss

令共享 head 为 `z_iv = W h_iv + b`，视图权重为 `a_v`，CE 表示目标 q 与预测分布的交叉熵：

```text
H0 = mean_i [ w_i * sum_v a_v * CE(q_i, softmax(z_iv / 1.5)) ]
H1 = mean_i [ w_i * sum_v a_v * CE(q_i, softmax(z_iv / 1.5 + log(pi))) ]
```

训练是 **加** log-prior，位置在除 temperature 之后；推理不加，也不额外再减。H 采用上式按图像数 mean 的固定 reduction；V 保留原 trainer 的 `sum(w * loss)/sum(w)`，不得误把两个 reduction 当成数值等价。

H1 思路来自 [Ren et al., Balanced Meta-Softmax, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/2ba61cc3a8f44143e1f2f13b2b729ab3-Paper.pdf)。加权软目标、多视图以及有效监督计数属于本轮工程扩展，论文不提供本项目收益保证。

固定配方：原 P head 初始化；AdamW；LR `1e-3`、WD `1e-4`；batch1024 个图像组；10 epoch；cosine 到0；seed42；FP32；clip1.0。H0/H1 初始化、样本顺序、更新步数和缓存相同；固定最后一轮，不挑 best epoch。所有冻结模块保持 eval，包括 O3/PTA dropout；只有 `classifier.weight/bias` 允许更新，不能使用包含 `feature_adapter` 的整个现有 head 参数组。

### 5.3 缓存与双 Adapter 的真实数值路径

global 缓存 P 真正 pre-head embedding。local 实际路径见 `adapted_dual_local_view_logits()`：

```text
h_local = O3(h_base) + PTA(h_base, pooled_parts) - h_base
z_local = linear(h_base, W, b) + linear(h_local - h_base, W, None)
```

两份 local residual 在**特征空间相加**，不能改为 O3/PTA 的两个预测投票。不能再归一化 `h_local`，不能用 OpenAI 原始特征代替 P 特征，不能复用旧 base_logits 到训练后的新 head。

缓存每个 global embedding；每个 local 的 `h_base` 和总 residual，使用上式在当前 head 下重构 logits，保存数值运算次序。其数学形式等价于 `W h_local+b`；真实诊断/最终推理保留原 anchored-residual 路径。

缓存分片加载。仅十个最终 embedding 约1.97 GiB；按默认保留两 global＋八 local 的 base/residual，共十八个512维 FP32向量，纯张量约3.54 GiB。另计索引、软目标及临时文件，不能将1.97 GiB当作默认实现总占用。

## 6. V：共享 backbone 与原局部模块联合学习

V0/V1 均从完整 P 独立启动，保留结构、末轮 q/w 和现有许可冻结掩码。两者3 epoch、seed42、FP32、有效 batch32、AdamW、clip1.0；固定末轮。

| 参数组 | LR | WD |
|---|---:|---:|
| 原 full-FT 许可的视觉参数，conv1/位置编码继续冻结 | `1e-6` | 0 |
| 共享 classifier | `1.5e-7` | `1e-4` |
| 已有 O3/PTA，V1 有梯度；V0 保留并冻结 | `3e-6` | 0 |

新建 optimizer，沿用原 full-FT scheduler 形状和 `schedule_epochs=18`，新微调的调度轮次1..3；监督状态明确继承 P 末轮，不随调度重置 correction。OOM 只允许 batch16＋accumulation2，保持每步32张；必须正确跨 microbatch 累积加权 loss 的分子/分母，不能简单平均两个权重总量不同的 loss。记录降档，不另生成候选。

```text
V0 = L_supervised(global) + 2.0 * L_feature_anchor(global)
V1 = 0.6 * L_supervised(global) + 0.4 * L_supervised(local)
     + 2.0 * L_feature_anchor(global)
```

监督 loss 复用原 GCE q0.5、实际软目标与加权 reduction；V 的监督 logits 温度维持原 trainer，推理仍固定1.5。局部分支必须通过当前 backbone、patch pooling、O3/PTA、当前共享 head 的完整残差路径反传，不新增 global/local KL loss，也不复制 infer CLI 的 `no_grad()` 包裹训练。

为落实几何要求，**本方案作一项明确修订：V0/V1 统一使用原生 CLIP center-crop 224＋由 seed/sample_id/epoch 决定的 Flip，替代旧 weak_rrc_flip。**两者使用相同 global 视图和顺序，V1 每图额外抽一个局部尺度，概率0.2/0.3/0.4/0.1。这是成对训练共同的工程选择，不解释为已验证的提分方法。

训练前由冻结 P 对官方训练图的原生224原图/Flip视图生成 top5 boxes，坐标存于各自224张量空间，绑定 sample_id/orientation/尺度。先 native preprocess，再按该张量空间的 boxes 裁切并复用 `extract_attention_crops()` 的 resize 路径；不用原始 PIL 坐标套224 box，不改为另一个 affine-grid crop。翻转视图由该视图实际 attention 生成定位。

global anchor 使用冻结官方 OpenAI ViT-B/32 对**相同 global 像素张量**即时编码，再复用原特征锚定 loss/reduction。它不训练、不参与候选预测。这样 Flip 后也有身份匹配的参考，避免把未翻转/另一RRC的旧整图缓存当参考。原恢复缓存可保留作核对，但本轮默认在线参考；V0/V1 均记录这项额外前向成本。local 不施加整图逐向量 anchor。

冻结 P 只产生训练 boxes，最终推理仅加载候选。V1 比 V0 多 local 前向，不能声称等算力消融。

## 7. 工程检查、平台决策与停止条件

旧10,316行验证和 clean-core 均是 overlap diagnostic，只用于故障/明显退化检查。固定末轮，本地小幅变化不挑 epoch、不追加参数搜索。

实施前的最小验收：严格恢复 P 和两 Adapter，epoch0 完整协议的 logits/argmax 对齐；H 冻结参数和 dropout 状态正确；均匀 pi 时 H1/H0 loss/gradient 等价；非均匀 pi 的正号及 soft q/w 手算一致；prior 不流入 infer；缓存/保存重载数值一致；V1 local 反传至许可视觉参数与两个 Adapter；boxes/anchor 几何身份一致。

缓存/真实路径建议 FP32 初始误差检查 `atol=1e-4, rtol=1e-5`，并检查 argmax 一致；不一致时先定位计算路径，不能用放宽阈值、删除来源验证或 `strict=False` 丢参数使检查通过。数值 finite、500维输出、映射完整、数据路径属于本阶段官方训练域必须通过。

若末轮在完整同协议、无校准的 raw micro 或 clean-core micro 相对 P 下降超过2.0pp，该候选工程止损，不上传、不换LR/epoch重试。2.0pp 是预算限制，不是独立泛化结论。预测类别数仅记录，不强求 argmax 覆盖全部500类。

| 真实平台结果 | 固定后续动作 |
|---|---|
| H0/H1 都不超过66.7681% | 关闭 H，不追加 loss/tau/seed |
| V0/V1 都不超过66.7681% | 关闭 V，不追加 epoch/教师 |
| 候选超过 P | 保存真实胜者及收益；未超过70.352866%不称历史新高 |
| H最佳、V最佳均至少比 P 高0.30pp，即均≥67.0681% | 允许一次 C |
| 已达到75% | 停止剩余新增训练，冻结胜者与提交包 |
| 既定候选全部完成/路线关闭，最多五候选 | 结束本轮，保留各协议下真实最佳，不自动扩展搜索 |

分数并列时 H 选 H0、V 选 V0，节省后续复杂度；0.30pp 是追加一次训练的投入门槛，不是统计显著性。平台分数仅用户实际提交后回填，时间和精确正确数缺失就留空。不能自动操作账号上传。

C 使用 V 胜者真实新 embedding 重新缓存，以 V 的原 head 为初始化重跑 H 胜者固定配方10 epoch；若 H1胜出，仍只使用同一训练 q/w 计算 pi。得到一个单模型 checkpoint，再按固定协议生成一个包。不能将旧 H head 接上新 backbone，也不能平均 H/V 预测。

## 8. 待实现功能与交付

当前可复用 `TrustBundle`、`corrected_targets()`、`class_prior_adjusted_logits()`、local inference/pooling、现有提交器与检查器。`longtail.py` 已有训练先验机制，但目前普通原始标签计数不能直接代替本轮 `n_eff`。

待新增功能：实际 w/q 的监督导出与有效先验；完整 P 的分片 base/residual 缓存；H0/H1共享 head refit；支持完整 composite 的成对 global/local 微调；绑定配置、模型、映射、诊断、CSV/ZIP哈希的候选报告。原 trainer 已有 attention-local loss，但没有本轮完整双 Adapter 联合训练接线，不能仅开现有开关就声称 V1 实现。

方法实现留在 Aegis 现有实验包，CLI 作为薄入口，复用共享数据/映射/提交模块；不复制 `common/`，不为本轮复制整条历史训练链。本文件中的功能名是待实现规格，不给出并不存在的可运行新训练命令。

每个完成的候选交付 checkpoint、resolved config、固定协议、实际命令/源码SHA、训练和诊断摘要、`pred_results.csv`、ZIP、SHA-256 manifest及空白线上成绩。运行末尾使用真实路径执行已有检查器：

```bash
python3 scripts/check_submission.py --test_dir /home/lux1/noise/test --num-classes 500 --csv outputs/prelim75_v2_20260912/H0/submission/pred_results.csv --zip outputs/prelim75_v2_20260912/H0/submission/submission.zip
```

H0只是路径示例，H1/V0/V1/C替换对应候选目录。另核对 ZIP 仅含 CSV 且包内外字节相同。24,967个文件名完整唯一；输出 comma＋space＋4位类别编号，通过动态映射生成，不硬编码类索引。

工作直接在 main；每段验证后只本地 commit，不 push。用户拥有全部 origin/main 推送和平台提交。再次同步混合工作树使用 Git 自动 stash 模式：`git pull --rebase --autostash origin main`，Git 自动恢复，不再执行 `git stash pop`。

## 9. 本轮不启动的工作与规则边界

不重训七节点/完整历史复现，不续训恢复R1，不扫描 prior/温度/crop权重，不继续范数对齐，不重开高分辨率、PACE/SCOPE、已关闭的 multiprototype/LDA、checkpoint averaging、clean-routed LoRA、trusted prototype-contrastive 或 dynamic trust refresh。缺必需资产只报告精确路径/哈希，不换来源、不绕过现有检查。

规则口径：backbone为CLIP ViT-B/32、初始化为官方OpenAI；无外部数据、跨阶段参数、人工必需清洗、测试训练或多模型集成。新增流程按自动执行设计；**完整历史来源闭合与干净环境复现仍未认证**，本次预检不将其改为通过。复现移出当前优化前置任务的用户决定继续有效。

官方规模、单模型及当前阶段数据限制本次已从 [官方赛题](https://www.aicomp.cn/tracks/tracks-1/3714.html) 复核。TTA确认来自仓库登记，不将官网泛化文字冒称为专项答复。

## 10. 证据及只读监督统计复跑

平台证据：[parent_nocal_control_20260911.md](../results/parent_nocal_control_20260911.md)、[fullft_dual_pa0.90_platform_20260806.json](../results/fullft_dual_pa0.90_platform_20260806.json)、[stage_readiness_recovery_20260912.md](../results/stage_readiness_recovery_20260912.md)及平台登记表。
实现证据：P实际 checkpoint config、`configs/f1_flat_full_ft.yaml`、`trainer.py`、`local_inference.py`、`part_token_adapter.py`、`localization.py`，均在 `reproducibility/aegis_f1/`。

下列命令从仓库根目录只读统计 P epoch3 规则，不训练，也不读取测试预测拟合参数；数值精度按 FP32目标/权重、FP64累计计算：

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 - <<'PY'
import pandas as pd
import torch
from aegis_clip.data import TrustBundle
from aegis_clip.features import canonical_sample_path
from aegis_clip.losses import corrected_targets
base = 'reproducibility/aegis_f1/artifacts/'
frame = pd.read_csv(base + 'stages/preliminary/final_full_train.csv')
trust = TrustBundle(base + 'trust/selftrain_r2_teacher_v3_multiscale.pt')
paths = [canonical_sample_path(p) for p in frame.image_path.astype(str)]
assert len(set(paths)) == len(paths) == 103218
trust.verify_coverage(paths)
idx = torch.tensor([trust.path_to_index[p] for p in paths])
y = torch.tensor(frame.label.to_numpy(), dtype=torch.long)
clean = trust.clean_probability[idx]
w = torch.where(clean >= 0.60, 0.6 + 0.4 * clean, torch.zeros_like(clean))
q = corrected_targets(y, trust.pseudo_label[idx], trust.correction_alpha[idx], 500)
assert torch.isfinite(q).all() and torch.isfinite(w).all()
assert torch.allclose(q.sum(1), torch.ones(len(q)))
n = (w.double().unsqueeze(1) * q.double()).sum(0)
corrected = (w > 0) & (trust.correction_alpha[idx] > 0) & (trust.pseudo_label[idx] != y)
print('retained/zero-weight/corrected:', int((w > 0).sum()), int((w == 0).sum()), int(corrected.sum()))
print('sum/min/median/max/zero-classes:', float(n.sum()), float(n.min()), float(n.median()), float(n.max()), int((n == 0).sum()))
PY
```
