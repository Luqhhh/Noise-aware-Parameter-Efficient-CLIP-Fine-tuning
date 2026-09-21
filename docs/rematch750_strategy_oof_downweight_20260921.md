# REMATCH750 策略思路：训练侧 OOF 连续降权

> **状态：占位，未开工。** 已登记方向，未写代码、未跑训练、无提交包。
> 登记：clairvoyanttt，2026-09-21。本文件用于防止重复劳动 —— 开工前请先读这里确认边界。

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

3. **注入训练**：按规范路径对齐成 `[N_train]` 权重向量，乘进现有的逐样本损失链（新框架已有该链，无需改数据加载）。
4. **明确不做**：不硬删样本、不改标签、不启用伪标签 / prior / TTA / 多尺度、不改推理路径。

## 复用清单

**代码可迁移（与初赛产物无耦合）**

- `analysis/oof/quality.py::build_sample_quality` / `add_quality_weights` —— 公式与百分位口径可直接沿用
- `common/diagnostic_metrics.py` —— `chunked_topk_cosine`、`build_trimmed_class_prototypes`
- 折划分与逐折线性头：优先复用新框架内已有的 `aegis_clip/trust.py::build_cross_fitted_trust` 或 `aegis_clip/oof_rebuild.py`，而不是移植初赛的 `analysis/oof/run_oof.py`（后者带若干 `num_classes=500` 与过期路径默认值）

**需要新写 / 有意修订**

- 复赛版折划分（内容组分层，seed 42，与现有 train_dev/val_dev 划分互不干扰）
- 权重注入的配置入口：**新增顶层 section**，并**有意修订** `aegis_clip/rematch_assets.py::validate_dataset` 的首轮闸门（该闸门目前 fail-closed 地禁止 rematch 配置出现 trust / loss dict 机制 / clean_routing）。修正是预期内的，但必须在提交信息与执行记录里留痕。
- flip 一致性需要**第二份特征缓存**（`aegis_clip/cli/cache_features.py` 支持 `--augmentation horizontal_flip`）：148,695 张的额外一遍编码，成本要在预注册里写清。若成本不可接受，退化方案是去掉该 0.15 项并重新归一化权重。

### ⚠️ 不可复用（合规红线）

以下初赛**拟合产物仍在磁盘上，且被 git 跟踪**（2026-09-21 实测）：

- `outputs/phase/phase3/oof/sample_quality.csv`（33 MB / 91,196 行，`train_dedup/...` 键，500 类）
- `outputs/phase/phase3/oof/fold_assignments.csv`、`oof_manifest.json`、各 `*_weight_manifest.csv`
- `outputs/phase4/purification/*/purification_manifest.csv`（22 MB）、`outputs/phase4/global_rejected_paths.txt`（990 条 `/home/lux1/noise/...`）
- `outputs/data/d3_strict/seed42/{train,val}.csv`

按比赛规则，数据 / 特征缓存 / 伪标签 / 原型 / 拟合 prior **不可跨阶段复用**。**危险点在于路径存在**：引用它们的代码不会干脆地报 `FileNotFoundError`，而是静默接上上一阶段的 500 类 / `train_dedup` 数据，或跑到一半才 `KeyError` —— 失败会伪装成「方法无效」。复赛阶段一律不读这些文件，只迁移公式与代码。

## 预注册判据

- **晋级门**：本地 noisy raw macro 相对 RM-FT baseline **≥ +0.20pp** 才值得占一次提交额度
- **分辨力参考**：RM-FT vs RM-LT 仅差 0.03pp macro（14,880 张里差 31 张）—— 低于此量级的差异不可解释，不要据此宣称有效
- **止损**：本地 −2pp 立即关闭，不派生参数扫描
- **提交额度**：平台每日 2 次
- **单变量**：除权重外其余配方与 RM-FT 保持一致（同初始化、同采样、同损失调度、同增强），否则无法归因
- 本地 noisy validation 不等于干净测试准确率，本地胜负不等同于平台胜负

## 开工前置条件（本机，待确认）

- `artifacts/stages/repechage/20260921/` **不存在** → 需先跑 `prepare → verify → cache`
- 配置中 `data.train_root: ../train` 相对 `configs/` 解析为 `<repo>/train`，本机**不存在**；数据实际在 `复赛数据集/{train,test}`（37,444 张测试图已核对一致）。**不要改共享配置的路径** —— 那会改变 config SHA-256，与提交登记表和 checkpoint 绑定冲突；应把数据放到配置期望的位置。
- CUDA 必需，禁止静默回落 CPU

## 状态

占位。下一步待与队友 FT/LT 的平台成绩合并后再细化实现方案。
