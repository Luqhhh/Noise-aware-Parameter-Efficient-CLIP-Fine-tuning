# 复赛特征漂移分析：FT 到底在哪里提升（2026-09-22）

**状态：测量轮，不改判据、不派生候选、无提交包。** 它给 [LP–FT 迁移对照预注册](rematch750_lp_ft_transfer_prereg_20260922.md) 的判据提供一个**预期读数**，并**收回**我在口头汇报里给过的一个方向性判断。

作者：clairvoyanttt。工具：`aegis_clip/cli/analyze_feature_drift.py`（本轮新增），单测 `tests/test_feature_drift.py`（6 项）。测试集未被触碰（只跑 val）。

---

## 0. 动机

`RM-FT` 相对 `RM-LP` 本地 micro **+9.23pp**，且这个增益在**第 2 个 epoch** 就已基本到位（69.29%），之后饱和 —— 说明它来自**「解冻」这个动作**，不是「学得更久」。

问题：这 +9.23pp 是**更好的表征**（应能跨来源迁移），还是**更彻底地拟合了池内域**（不应迁移）？

本地无法裁决迁移（[预注册 §1](rematch750_lp_ft_transfer_prereg_20260922.md)：val 是训练池的忠实 iid 抽样）。但有一个**与迁移有理论联系**的本地量可测：FT 把每一类的特征推离冻结 OpenAI 特征多远，以及它推得最多的类是不是它提升最多的类。

---

## 1. 做法与锚点

对 val 每张图取 FT 特征与缓存中的冻结 OpenAI 特征（均已 L2 归一化），

```
drift = 1 − cos(FT 特征, OpenAI 特征)
```

按类取均值；与两份 `logs/evaluation_epoch_*.json` 的 `per_class` 明细里的 `Δrecall = FT − LP` 配对（750 类）。

```bash
PYTHONPATH=reproducibility/aegis_f1 .venv/bin/python -m aegis_clip.cli.analyze_feature_drift \
  --checkpoint outputs/rematch750/RM_FT/seed42/checkpoints/best.pt \
  --config configs/rematch750_ft.yaml \
  --ft-evaluation outputs/rematch750/RM_FT/seed42/logs/evaluation_epoch_8.json \
  --lp-evaluation outputs/rematch750/RM_LP/seed42/logs/evaluation_epoch_20.json \
  --output outputs/rematch750/RM_FT/seed42/feature_drift.json \
  --anchor-micro 0.7286962270736694
```

**两处独立锚点，都通过：**

- **val micro = `0.7286962270736694`**，与记录值**逐位一致**。
- **mean drift = `0.037974752`**，与训练日志里的 `mean_feature_drift: 0.037974775` **前 7 位吻合** —— 这独立确认了我算的 drift 与 trainer 内部用的是同一个定义。

> 锚点第一次是**失败**的：我默认 `batch_size=64`，得到 `0.7285618`，差 **2 张**。原因是记录那次评测用 `evaluation.batch_size: 128`，autocast 下不同 batch 形状让 cuDNN 选不同 kernel，边界图翻转。已改为**从 config 取 `evaluation.batch_size`**，而不是放宽容差 —— 放宽容差就等于放弃了这个护栏。

---

## 2. 结果

**主相关**：`drift` vs `Δrecall`，750 类

| | 值 |
|---|---|
| Pearson | **+0.0199** |
| Spearman | **+0.4013** |

单调但不线性：Pearson 被尾部拉平，Spearman 抓住单调性。等分五分位梯度干净：

| drift 五分位 | 类数 | val 数 | mean_drift | mean_Δrecall | Δmicro |
|---|---|---|---|---|---|
| Q1 | 150 | 2,993 | 0.0250 | +0.0509 | **+5.05pp** |
| Q2 | 150 | 2,887 | 0.0330 | +0.0612 | +5.85pp |
| Q3 | 150 | 3,000 | 0.0386 | +0.0871 | +8.90pp |
| Q4 | 150 | 3,006 | 0.0433 | +0.1124 | +11.14pp |
| Q5 | 150 | 2,994 | 0.0498 | +0.1470 | **+15.10pp** |

**增益按 drift 拉开 3 倍。** FT 移动特征最多的类，正是它提升最多的类。

---

## 3. 分辨检验：哪一种读法？

单看 §2 的相关**不足以分辨**「域适配」与「真实学习」—— 两种读法都预测正相关。于是做分辨检验：

| 相关对 | Pearson | Spearman |
|---|---|---|
| drift vs `train_samples` | +0.032 | +0.033 |
| **drift vs `LP_recall`** | **−0.567** | **−0.603** |
| drift vs `FT_recall` | −0.451 | −0.521 |
| Δrecall vs `LP_recall` | −0.448 | −0.485 |
| Δrecall vs `train_samples` | +0.012 | +0.004 |

drift 在 support 五分位上几乎恒定：**0.0380 / 0.0376 / 0.0372 / 0.0382 / 0.0387**。

> **口径**：Pearson 一律按 `val_samples` 加权，与 §2 主统计量（工具内同样是加权的 +0.0199）一致。Spearman 不受加权影响。
> 早先我在临时脚本里算的是**无权** Pearson（`drift vs LP_recall` 得 −0.540），与本表不同 —— 上表是统一口径后的值。两条读法的结论不受影响：加权/无权下 `drift vs LP_recall` 都在 −0.54 ~ −0.57，`drift ⊥ support` 都在 +0.03。

**读法：**

- **域适配**该在**高样本类**上最强（那里有更多可拟合的东西）→ `drift ⊥ support`（+0.033）**否掉**。
- **真实学习**该在**冻结特征最弱**的类上最强 → `drift vs LP_recall = −0.603` **对上**。

这与前一轮独立测到的「FT 的增益在 support 五分位上均匀（+7.97~+9.99pp）」两两印证。

> **结论：FT 的 +9.23pp 更像真实的表征改善，而不是池内域适配。** 它移动最多、提升最多的，是**冻结 CLIP 本来就分不开的那些类**，且与类样本量无关。

---

## 4. 一处收回

我在 2026-09-22 的口头汇报里说过：「drift 与提升正相关 → 预测 RM-LP 平台分**偏高**」。**这个推断下得太快，是错的。**

它默认了「特征被移动 = 域适配」，而 §3 的分辨检验恰恰否掉了这个默认。按当前证据，正确的预期方向**相反**：

> **FT 的优势会迁移 → RM-LP 的平台分更可能落在低档（< 53.30%）**，即 FT 家族仍是对的。

这不改变明天的决定（LP 仍是唯一能买的测量，且下行与 FT 相同，见[预注册 §0.1](rematch750_lp_ft_transfer_prereg_20260922.md)），但**改变了预期读数**：若明天 LP 只有 50 出头，那是**符合预期**的，不代表实验失败。

---

## 5. 诚实边界

- 这仍是**弱证据**。`drift vs Δrecall` 的相关对两种读法都成立；真正有分辨力的只有「`drift ⊥ support`」这一条，而它依赖「域适配会集中在高样本类」这个先验。
- **它不能裁决迁移。** val 是训练池的忠实 iid 抽样（[预注册 §1](rematch750_lp_ft_transfer_prereg_20260922.md)），本地对平台无信息这一结论不受本轮影响。
- **判据不变**：[预注册 §4](rematch750_lp_ft_transfer_prereg_20260922.md) 的三档迁移率判据仍然有效，本轮只更新对读数的预期。

---

## 6. 产物

| 产物 | 路径 | 入库 |
|---|---|---|
| 工具 | `reproducibility/aegis_f1/aegis_clip/cli/analyze_feature_drift.py` | ✅ |
| 测试 | `reproducibility/aegis_f1/tests/test_feature_drift.py`（6 项） | ✅ |
| 汇总 | `results/rematch750_20260922_feature_drift.json` | ✅ |
| 逐类明细 | `results/rematch750_20260922_feature_drift_per_class.csv`（750 行） | ✅ |
| 原始输出 | `outputs/rematch750/RM_FT/seed42/feature_drift.json` | ❌（`/outputs/rematch750/` 被 gitignore） |

**附**：本轮顺带修掉了 `spearman` 的一个真实缺陷 —— 原实现用 `argsort` 给并列值按索引顺序分配不同秩，会让**常数列算出伪相关**。已改为平均秩（标准 Spearman）。修正后重跑，数值**完全一致**（+0.4013），因为 drift 是连续量、并列极少；修正的价值在于常数列不再产生假信号。
