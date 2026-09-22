# 复赛推理侧探针：Flip TTA 的量化与输入几何假设的证伪（2026-09-22）

**状态：本轮为「关闭方向」的测量轮，不派生候选、无提交包。** 两个测量都指向同一个结论：推理侧不是答案 —— 与 [策略文档](rematch750_strategy_oof_downweight_20260921.md) 第 13 行既有的定性判断一致，本轮把它量化了。

作者：clairvoyanttt。基线 checkpoint：`outputs/rematch750/RM_FT/seed42/checkpoints/best.pt`（与登记值同源）。

---

## 0. 两个测量与结论

| 测量 | 结果 | 结论 |
|---|---|---|
| Flip TTA 全融合规则扫描 | 本地 micro **+0.155 ~ +0.242pp**，9/9 条规则同号 | 量化了推理侧天花板：**很小**，且按 lessons §一.1 本地不可读 |
| 输入几何模拟（测试集分辨率/长宽比） | 本地 **−0.17 ~ −0.31pp** | **证伪**：11.90pp 的本地/平台差**不是**输入几何造成的 |

---

## 1. Flip TTA：推理侧天花板有多高

工具：`aegis_clip/cli/sweep_flip_tta_online.py`（新增）。一次 GPU 前向缓存「原图 / 翻转」两视图 logits，再离线套全部融合规则与温度。指标函数**从 `aegis_clip.evaluation` 导入**，不重写，保证与 `best_evaluation.json` 同定义。

> **命名说明**：仓库里**已有一个** `aegis_clip/cli/sweep_tta_fusion.py`（来自 `97d0a3f`，初赛 LoRA 期）。它走的是**配对增强特征缓存**（`--augmented-feature-dir` + `model.adapt_features`），**不能**跑全量微调 + 在线图像这条链路，因此与本工具不通用。两者按各自的 regime 保留，本工具不改动前者。

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.sweep_flip_tta_online \
  --checkpoint outputs/rematch750/RM_FT/seed42/checkpoints/best.pt \
  --config configs/rematch750_ft.yaml \
  --output outputs/rematch750/RM_FT/seed42/tta_sweep.json \
  --anchor-macro 0.7180540561676025 --anchor-micro 0.7286962270736694
```

**锚点通过**：`fusion=none` 复现已登记的 `raw_macro` / `raw_micro`（差 <1e-6，即 float32 累加舍入量级）。所有 TTA 数字都建立在这次复现之上。

翻转视图与原始视图的 **top-1 一致率 90.4503%** —— 即翻转在 14,880 张里改变了 1,421 张的判决，这不是一个「几乎等价」的视图。

| 融合规则 | T | raw_micro | Δmicro | raw_macro | Δmacro | 配对 SE | 配对 t |
|---|---|---|---|---|---|---|---|
| none（对照） | 1.0 | 72.8696% | — | 71.8054% | — | — | — |
| entropy_weighted_probabilities | 0.5 | 73.0645% | +0.1949 | 71.9939% | +0.1885 | 0.1248pp | +1.51 |
| entropy_weighted_probabilities | 1.0 | 73.0712% | +0.2016 | 71.9810% | +0.1756 | 0.1272pp | +1.38 |
| **entropy_weighted_probabilities** | **2.0** | **73.1116%** | **+0.2419** | **72.0178%** | **+0.2124** | 0.1266pp | +1.68 |
| max_margin | 1.0 | 73.0242% | +0.1546 | 71.9545% | +0.1491 | 0.1256pp | +1.19 |
| **mean_logits（代码默认）** | **1.0** | **73.0712%** | **+0.2016** | **71.9813%** | **+0.1759** | 0.1282pp | +1.37 |
| mean_probabilities | 0.5 | 73.0780% | +0.2083 | 72.0065% | +0.2011 | 0.1247pp | +1.61 |
| mean_probabilities | 1.0 | 73.0511% | +0.1815 | 71.9623% | +0.1569 | 0.1276pp | +1.23 |
| mean_probabilities | 2.0 | 73.0712% | +0.2016 | 71.9761% | +0.1707 | 0.1265pp | +1.35 |
| standardized_logits | 1.0 | 73.0914% | +0.2218 | 72.0012% | +0.1958 | 0.1281pp | +1.53 |

**读法（三点）：**

1. **效应是真的，但很小。** 9/9 条规则同号，量级集中（micro +0.155~+0.242pp），说明这不是某条规则的偶然，而是「两视图平均」本身的降方差效应 —— 融合规则几乎不影响结果。
2. **配对 t 全部落在 +1.2 ~ +1.7**，配对 SE ≈ 0.127pp。即收益高于 1 SE、低于 2 SE，属于「方向可信、显著性不足」。
3. **收益集中在 head / medium，尾部为零**（按训练样本量等分三段，取 Δmicro 最大的规则）：head **+0.3075pp**、medium **+0.3393pp**、tail **+0.0249pp**（tail 的 macro 甚至是 **−0.0196pp**）。**TTA 不解决长尾，也不触碰噪声拟合机制** —— 它与 RM-LT / RM-OOFW 想解决的问题不重叠。

**本轮没有选「val 最优规则」。** 若按 val 取 argmax 会选 `entropy_weighted_probabilities@T2.0`，但那正是 lessons §一.1 禁止的「用本地指标挑晋级」；且规则间差异（0.155~0.242）远小于其可分辨度。任何后续动作应使用**参数自由的代码默认 `mean_logits`**。

---

## 2. 输入几何假设：证伪

**动机。** 本地 val micro 72.8696%，平台 60.9657%，差 **11.9039pp**。先查一个最可能的结构性原因：输入分布不同。查下来差异惊人：

| 划分 | 图片数 | 不同 (w,h) 形状数 | 长边量级 | 主导长宽比 |
|---|---|---|---|---|
| train_dev | 133,815 | **33,631** | 800–1390+ | 1.33（21,120） |
| val_dev | 14,880 | **6,406** | 800–1390+ | 1.33（2,400） |
| test | 37,444 | **807** | **全部 ≤ 500** | 0.75（14,177） |

测试集 375×500 一种形状就占 37.6%，前四种形状占约 72%；**测试集每一个维度都不超过 500px**，而训练/验证集长边普遍在 800–1390。同时朝向相反：val 主导横向 1.33，test 主导纵向 0.75。

**做法（合规）。** 用**带标签的验证集**模拟测试集几何：每张 val 图把长边缩放到 L 再走原 CLIP 预处理。测试集只贡献尺寸统计量，**不读任何测试图、标签或预测**。

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.simulate_input_geometry \
  --checkpoint outputs/rematch750/RM_FT/seed42/checkpoints/best.pt \
  --config configs/rematch750_ft.yaml \
  --output outputs/rematch750/RM_FT/seed42/geometry_sweep.json \
  --anchor-micro 0.7286962270736694
```

| 长边 | 正确 / 总数 | micro | Δ vs 对照 |
|---|---|---|---|
| 原始（对照） | 10843 / 14880 | 72.8696% | — |
| 800 | 10817 / 14880 | 72.6949% | −0.1747pp |
| 640 | 10808 / 14880 | 72.6344% | −0.2352pp |
| **500（测试集量级）** | **10816 / 14880** | **72.6882%** | **−0.1815pp** |
| 400 | 10797 / 14880 | 72.5605% | −0.3091pp |

**结论：把验证集压到测试集的几何，只掉 0.17–0.31pp，而不是 11.90pp。假设证伪。** 输入分辨率与长宽比**不能**解释本地/平台差，因此这个差距**不可能**靠推理侧改预处理关掉。

> 锚点说明：对照返回 10843/14880 = 0.7286962365591397，登记值 0.7286962270736694。两者**不是同一个浮点数**，但 `登记值 × 14880 = 10842.99986`，四舍五入即 10843 —— 差异来自登记值以 **float32 累加**求得，而脚本用 float64 求 `k/N`。正确张数完全一致。工具的默认锚点容差因此设为 `1e-6`（远小于一张图、远大于该舍入量级），而非 1e-9。

---

## 3. 对既有判断的两处订正

### 3.1 「推理侧不是答案」是本项目**既有**决定，本轮只是把它量化

我在上一轮报告里把 flip TTA 推荐为「期望值最高的未探索杠杆」，并称「首轮已结束、推理协议边界现在可以动」。**这个推荐是在没查既有记录的情况下给出的，是错的。** 既有记录明确：

- [策略文档:13](rematch750_strategy_oof_downweight_20260921.md)：**「推理侧不是答案。」** 推理侧（TTA / 多尺度 / prior 校准）是初赛提分最快路径（测试集拟合 prior 达到 70.35%，当时绝对最高分），但被 lessons 列为**陷阱 #7「不同来源协议」**，分数不可推广；复赛首轮边界已明确为 224 center crop / 无 TTA / 无 prior。
- [策略文档:23](rematch750_strategy_oof_downweight_20260921.md)：`| 推理协议（224 center crop，无 TTA / prior） | 首轮固定边界，本轮不动 |`
- **代码闸门仍在生效**：`aegis_clip/cli/infer.py:231-233` 在 rematch 配置下**拒绝** TTA / local-view / prior，报 `Rematch first-round inference is global only, without prior`。我实测触发了该闸门，**没有绕过它**（没有去清 `dataset_manifest` 或改闸门）。

所以本轮的净贡献不是「提议 TTA」，而是**给这个既有决定一个数字**：推理侧最好的免费动作值 +0.155~+0.242pp 本地，而按 §一.1 本地读数不可用于晋级 —— 即该决定是对的，而且余地比原先想象的更小。

**合规性另行确认（与策略无关）**：Flip TTA **本身合规**。[COMPETITION_RULES_AGENT.md:804](COMPETITION_RULES_AGENT.md) 记录组委会答复「多 crop 或多尺度 TTA … 已获组委会答复：合规」，[repechage_prep_20260831.md:19](repechage_prep_20260831.md) 同。`infer.py` 的 `--acknowledge-tta-risk` 是**留痕**机制（`tta_risk_acknowledged` 写进 manifest），不是合规阻断。它拒绝 TTA 的原因是**首轮阶段策略**，不是合规。

### 3.2 RM-OOFW 违反了一条既有纪律：不得设本地晋级门槛

[lessons_learned §一.1](lessons_learned.md) 的 How to apply 原文：

> 平台实测是候选晋级的**唯一**标准。本地指标只用于安全审计、复现检查和工程止损（−2pp），**不设本地晋级门槛**。

而上一轮 [RM-OOFW 预注册](rematch750_strategy_oof_downweight_20260921.md) 设的主判据是**本地**「固定训练尾部 75 类 macro ≥ +0.20pp」。这属于该项目已明文禁止的「本地晋级门槛」。事后看这加重了该轮的问题：即便本地过了门，按 §一.1 也不构成晋级理由；而它没过门，同样不构成关闭理由 —— **该轮实际上无法用本地数据裁决**。

这不改变 RM-OOFW 的最终处置（无提交包、关闭、不派生扫描），但它说明**判据类型本身选错了**，下一轮登记时应直接以平台实测为准。

---

## 4. 本轮为什么不产出提交包

- flip TTA 提交被 `infer.py` 的**首轮闸门**按设计拒绝，我不绕过它。
- 即便绕过，按 §一.1 本地 +0.2pp 级收益不能作为晋级依据，而本次实测的效应量（+0.155~+0.242pp 本地）远不足以支撑一次平台提交的判断。
- 与 [RM-OOFW 轮](rematch750_strategy_oof_downweight_20260921.md) 的先例一致：**关闭方向的测量轮以文档收尾，不出包。**

---

## 5. 复现与产物

```bash
# 全部在仓库根目录；解释器为 WSL venv（python 3.12.13 / torch 2.13.0+cu130）
PYTHONPATH=reproducibility/aegis_f1 .venv/bin/python -m aegis_clip.cli.sweep_flip_tta_online <见 §1>
PYTHONPATH=reproducibility/aegis_f1 .venv/bin/python -m aegis_clip.cli.simulate_input_geometry <见 §2>
PYTHONPATH=reproducibility/aegis_f1 .venv/bin/python -m pytest reproducibility/aegis_f1/tests/test_input_geometry.py -q
```

| 产物 | 路径 | 入库 |
|---|---|---|
| 工具 | `reproducibility/aegis_f1/aegis_clip/cli/sweep_flip_tta_online.py` | ✅ |
| 工具 | `reproducibility/aegis_f1/aegis_clip/cli/simulate_input_geometry.py` | ✅ |
| 测试 | `reproducibility/aegis_f1/tests/test_input_geometry.py`（6 项） | ✅ |
| 汇总 | `results/rematch750_inference_side_probe_20260922.json` | ✅ |
| 原始扫描 | `outputs/rematch750/RM_FT/seed42/tta_sweep.json` | ❌（`/outputs/rematch750/` 被 gitignore） |
| 原始几何 | `outputs/rematch750/RM_FT/seed42/geometry_sweep.json` | ❌（同上） |

**注意**：解释器是 **WSL venv**，不是 Windows python。本机 Git Bash 下的 `python3` 是 Windows Python 3.14.5 / torch 2.11.0+cu128，与产出 checkpoint 的环境不同，且经 UNC 路径访问 WSL 符号链接会 `Input/output error`；一切训练/推理必须走 `wsl.exe -e bash -lc '... .venv/bin/python ...'`。
