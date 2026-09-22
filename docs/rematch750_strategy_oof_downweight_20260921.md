# REMATCH750 策略思路：训练侧 OOF 连续降权

> **状态：实现已落地并推送（`ae54a8c`），训练未跑。** 权重侧车生成中，尚无本地指标、无提交包。
> 登记：clairvoyanttt，2026-09-21。本文件用于防止重复劳动 —— 开工前请先读这里确认边界。
> 2026-09-22：开工。下方「方法概要」的两处事实错误已就地更正，判据已修订，实现细节见文末「实现记录」。

## 一句话

用交叉拟合（OOF）预测给每张训练图打一个**连续质量分**，训练时按分降权：不删样本、不改标签、不动推理。

## 为什么是这条

- **推理侧不是答案。** 推理侧（TTA / 多尺度 / prior 校准）是初赛提分最快的路径 —— 测试集上拟合的 prior 校准把分数推到 70.35%，是当时绝对最高分。但 [lessons_learned](lessons_learned.md) 把它列为**陷阱 #7「不同来源协议」**：分数不可推广。复赛首轮边界也已明确为 224 center crop / 无 TTA / 无 prior。
- **当前无人做噪声建模。** RM-LP / RM-FT / RM-LT 三个候选的唯一变量是**采样**（shuffle vs sqrt_class_balanced），FT 与 LT 的差只有 0.03pp macro。比赛题目是噪声标签，这是当前最大的未覆盖面。
- **执行文档已预告。** [rematch750 执行记录](rematch750_execution_20260921.md) 第 80 行：「首轮后如需噪声处理，另行建立**仅训练侧的 CVT/OOF 连续降权实验**」。
- **初赛经验支持 soft > hard。** P2 的 hard gate 无独立干预，soft gate（+0.042pp）才有增益；方向是软降权而非硬删。

## 与在途工作的边界

| 维度 | 归属 |
|---|---|
| 采样（shuffle / sqrt_class_balanced） | RM-FT / RM-LT（已完成，队友） |
| 推理协议（224 center crop，无 TTA / prior） | 首轮固定边界，本轮不动 |
| **训练侧逐样本噪声加权** | **本策略（本文件占位）** |

## 方法概要

1. **OOF 预测**：在缓存特征上按内容组分层做 K 折，逐折训线性头，取 held-out logits。参照特征（原型 / kNN）**只从训练折取**，防泄漏。
2. **复合质量分** —— 沿用初赛公式，[analysis/oof/quality.py](../analysis/oof/quality.py) L106-112 原文：

   ```
   quality     = 0.35 · p_原标签类内百分位
               + 0.25 · 原型 margin 类内百分位
               + 0.25 · clip(kNN 一致率, 0, 1)
               + 0.15 · clip(flip 一致率, 0, 1)
   soft_weight = clip(0.3 + 0.7 · quality, 0.3, 1.0)
   ```

3. **注入训练**：按规范路径对齐成 `[N_train]` 权重向量，乘进现有的逐样本损失链。~~新框架已有该链，无需改数据加载~~ —— 更正（2026-09-22）：该链存在，但 rematch 首轮闸门禁止启用它（详见下方「需要新写」）。
4. **明确不做**：不硬删样本、不改标签、不启用伪标签 / prior / TTA / 多尺度、不改推理路径。

## 复用清单

**代码可迁移（与初赛产物无耦合）**

- `analysis/oof/quality.py::build_sample_quality` / `add_quality_weights` —— 公式与百分位口径可直接沿用
- `common/diagnostic_metrics.py` —— `chunked_topk_cosine`、`build_trimmed_class_prototypes`
- 折划分与逐折线性头：优先复用新框架内已有的 `aegis_clip/trust.py::build_cross_fitted_trust` 或 `aegis_clip/oof_rebuild.py`，而不是移植初赛的 `analysis/oof/run_oof.py`（后者带若干 `num_classes=500` 与过期路径默认值）

**需要新写 / 有意修订**

- 复赛版折划分（内容组分层，seed 42，与现有 train_dev/val_dev 划分互不干扰）
- ~~权重注入的配置入口：**新增顶层 section**，并**有意修订** `aegis_clip/rematch_assets.py::validate_dataset` 的首轮闸门~~
  **更正（2026-09-22，实测）：两处判断都错了。**
  1. 闸门确实存在，但在 [rematch_assets.py:52](../reproducibility/aegis_f1/aegis_clip/rematch_assets.py#L52)：`require(not trust.enabled and not trust.bundle_path, 'trust disabled in first round')`。它禁的是**启用 trust**，不是「trust 段里出现新键」。
  2. 因此**不需要改闸门，也不需要新增顶层 section**。最终做法是 `trust` 段下新增一个 `trust.sample_weight_path`（默认不设），闸门第 55-56 行只检查 `bundle_path` / `groups_path`，放行。
  3. 也**没有**复用现成的 `trust.enabled` + `bundle_path` 通道：那条路能少改 trainer 一行（`0.3 + 0.7*clean` 恰好等于 `soft_weight`），但该 bundle 被 `a2_gate.py` 等下游按 `cross_fitted_visual_trust_v1` 语义读，塞别的量会污染共享产物。
- ~~flip 一致性需要**第二份特征缓存**（`aegis_clip/cli/cache_features.py` 支持 `--augmentation horizontal_flip`）~~
  **更正（2026-09-22，实测）：这条路不通。** `cache_features.py` 在 rematch 分支直接 `raise ValueError("Rematch uses only its bound canonical feature cache")`，非 `augmentation=none` 一律拒绝。故取退化方案：**去掉 0.15 项，其余三项按 /0.85 重新归一化**（0.411765 / 0.294118 / 0.294118）。

### ⚠️ 不可复用（合规红线）

以下初赛**拟合产物仍在磁盘上，且被 git 跟踪**（2026-09-21 实测）：

- `outputs/phase/phase3/oof/sample_quality.csv`（33 MB / 91,196 行，`train_dedup/...` 键，500 类）
- `outputs/phase/phase3/oof/fold_assignments.csv`、`oof_manifest.json`、各 `*_weight_manifest.csv`
- `outputs/phase4/purification/*/purification_manifest.csv`（22 MB）、`outputs/phase4/global_rejected_paths.txt`（990 条 `/home/lux1/noise/...`）
- `outputs/data/d3_strict/seed42/{train,val}.csv`

按比赛规则，数据 / 特征缓存 / 伪标签 / 原型 / 拟合 prior **不可跨阶段复用**。**危险点在于路径存在**：引用它们的代码不会干脆地报 `FileNotFoundError`，而是静默接上上一阶段的 500 类 / `train_dedup` 数据，或跑到一半才 `KeyError` —— 失败会伪装成「方法无效」。复赛阶段一律不读这些文件，只迁移公式与代码。

## 预注册判据

**修订（2026-09-22，在任何 RM_OOFW 结果产生之前）**：主判据从「整体 macro」改为「尾部 macro」。

依据是队友已推的配对 FT/LT 数据（[rematch750_local_comparison_20260922.md](rematch750_local_comparison_20260922.md)，与本实验无关的第三方轮次）：

| 指标 | RM_FT | RM_LT | 差 |
|---|---|---|---|
| 整体 macro | 71.8362% | 71.8655% | **+0.03pp** |
| bottom-10% macro | 22.5567% | 23.5166% | **+0.96pp** |
| 固定训练尾部 10% macro | 55.5827% | 58.1485% | **+2.57pp** |

整体 macro 在 14,880 张上的分辨力只有 ~0.03pp 量级 —— 拿它当主判据会把真实效应淹没在噪声里。降权的**作用机制本来就是改善尾部**（压低疑似错标样本的权重），应在上表后两行上验证。

- **主判据（晋级）**：**固定训练尾部 10% macro** 相对 RM-FT **≥ +0.20pp**（沿用项目晋级门，只是换到尾部口径）
- **次判据（佐证）**：bottom-10% macro 同向、且整体 macro 不低于 RM-FT −0.30pp（允许整体略降换取尾部收益，但不得倒退到止损线）
- **止损**：本地 **−2pp** 立即关闭，不派生参数扫描
- **提交额度**：平台每日 2 次
- **单变量**：除 `trust.sample_weight_path` 外其余字段与 `configs/rematch750_ft.yaml` **逐字段一致**（同初始化 RM-LP best.pt、同采样 none、同 loss gce q=0.5 / ce_warmup 2 / distill 2.0、同增强 weak_rrc_flip、同 8 epoch），否则无法归因
- 本地 noisy validation 不等于干净测试准确率，本地胜负不等同于平台胜负

## 开工前置条件（2026-09-22 已全部满足）

- ~~`artifacts/stages/repechage/20260921/` **不存在**~~ → 已在本地重建（`prepare` 9m41s → `verify` 通过 → `cache` 148,695 条）。9 份资产与队友登记的 manifest **逐字节一致**；只有 `train_root` / `test_root` 是机器相关字段，这正是 `dataset_manifest.json` 的 SHA 跨机器不同的原因，也是特征缓存不可跨机器复用的原因。
- ~~配置中 `data.train_root: ../train` 本机不存在~~ → 已用符号链接把 `train` / `test` 指向 `复赛数据集/{train,test}`，**未改任何共享配置**（改路径会改 config SHA-256，与提交登记表和 checkpoint 绑定冲突）。符号链接已加入 `.git/info/exclude`（本地生效，不入库）。
- CUDA 必需，禁止静默回落 CPU —— 本机 8G 显存，单卡。

## 实现记录（2026-09-22）

**改动文件**（commit `ae54a8c`，已推 `origin/main`）

| 文件 | 性质 | 说明 |
|---|---|---|
| `aegis_clip/sample_weights.py` | 新增 | sidecar 加载器，按规范路径对齐，重复/缺失/越界一律抛错 |
| `aegis_clip/cli/build_sample_weights.py` | 新增 | 折划分 → OOF logits → 折内几何 → 复合分 → sidecar |
| `aegis_clip/trainer.py` | +15 行 | 守卫分支注入；不设该键的配置走逐位相同路径 |
| `aegis_clip/config.py` | +1 行 | 登记新路径键 |
| `tests/test_sample_weights.py` | 新增 | 8 条回归测试 |
| `configs/rematch750_oofw.yaml` | 新增 | 与 `rematch750_ft.yaml` 只差 `trust.sample_weight_path` |

**复用而非重写**（这是刻意的）

- 折划分：`analysis/oof/build_folds.py::assign_group_stratified_folds`（sklearn `StratifiedGroupKFold`，group 键用 `content_group`）。已经它自己的断言验证「无内容组跨折」。
- 质量分：`analysis/oof/quality.py::build_sample_quality` + `add_quality_weights`，只覆盖复合系数。
- OOF logits：`aegis_clip/oof_rebuild.py`（`--num-classes` 已参数化，无需改）。
- 几何：`common/diagnostic_metrics.py` 的原型 / kNN 原语，k=10，与仓库既有 `compute_knn_top1_agreement.py` 同口径。

**偏离预注册之处（全部有意，逐条记录）**

1. **flip 项去掉**（原因见上「更正」），三项重新归一化。
2. **权重不做均值 1 归一化**。原计划写「归一化到均值 1」，但 trainer 的逐样本损失是 `(per_sample * w).sum() / w.sum()` —— **加权平均**，权重整体缩放对梯度精确无影响（分子分母同比例）。归一化不但无收益，还会把权重推过 1.0 而被加载器拒绝。故保留 `0.3 + 0.7·q ∈ [0.3, 1.0]` 原样。
3. **折数 5 折**，而仓库历史 OOF 用 3 折。理由：每折线性头只看 80% 而非 67% 的样本，OOF 信号更准；成本可忽略。
4. **加权可靠性下限 `--min-class-support 10`**（新增，同样在出结果前声明）。样本数少于 10 的类，其全部样本权重恒为 1.0，不参与降权。

   为什么必须有这一条：实测 `train_dev` 的类支持度 **min=4 / 中位数=176**，只有 **1 个类（class 183，4 个样本，各折分布 `[1,0,1,1,1]`）** 低于 10。而本轮**主判据「固定训练尾部 10%」正是取支持度最小的 75 个类**（该尾部 min=4 / 中位数=137），class 183 就在其中。

   若不设下限：class 183 的 4 个样本在 OOF 里必然信号极差（每折训练划分只有 3 个样本），其中 `knn_agreement` 几乎必为 0（近邻都是别的类），复合分被拉到 ~0.44、权重 ~0.61。**这是「类太小」而非「标签有噪声」导致的降权**，会让 RM_OOFW 在该类上比 RM_FT 更差，而 RM_FT 完全不受影响。该混淆项量级 ≈ 1/75 × 若干 pp，与 +0.20pp 晋级门同量级 —— 足以污染裁决。

   注意复合分里 2/3 的项（`p_original` / 原型 margin）本来就走**类内百分位**，已天然免疫类规模差异；只有 `knn_agreement`（权重 0.294）是绝对值，这也是把下限设在这里的直接原因。

   **附带要求**：裁决时主判据要**同时报「含 class 183」与「不含 class 183」两个版本**，证明结论不依赖这一个 4 样本类。

**未改动（有意保留稀释，需在结果里如实说明）**

- `trainer.py` 的 2.0 倍特征蒸馏项 `loss + distill_weight * drift.mean()` **不受逐样本权重影响**。理由：该损失项不含标签、是纯特征空间正则，标签噪声权重对它没有语义基础。**后果**：降权对总梯度的实际影响被稀释，测出的效应是下界。这是有意接受的，不是疏漏。

**验证状态**

- 新增 8 条测试全通过
- 全量套件 **2 failed / 632 passed**，与改动前基线**完全一致**（两个失败均为 `test_scope_protocol.py` 因上一阶段冻结资产被删而失败，属已知项）
- 尚未跑训练，**无任何指标**。不得在结果产出前声称有效。

## 状态

实现已推送（`ae54a8c`），sidecar 生成与配对训练待跑。下一步：

1. 生成 sidecar：`python -m aegis_clip.cli.build_sample_weights`（参数见 `cli/build_sample_weights.py` 的 `--help`）
2. 训练 `configs/rematch750_oofw.yaml`，与 `RM_FT` 配对比较
3. 按上文修订后的判据（**尾部 macro 为主**）裁决
