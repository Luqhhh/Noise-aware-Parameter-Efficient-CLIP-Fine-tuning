# RM_FT / RM_LT 平台结果（2026-09-22）

本页记录用户于 2026-09-22 回报的复赛平台成绩，并把两个提交包与其训练策略明确绑定。这里的两次平台提交均为既有 RM_FT / RM_LT 单模型包，**不是** RM_OOFW，也没有使用 TTA、prior、模型融合或测试时适配。

## 策略对应关系

| 平台候选 | 策略 | 唯一变量 | 推理 |
|---|---|---|---|
| RM_FT | full fine-tune baseline | 普通 shuffle；每个 `train_dev` 样本每轮使用一次 | 单 checkpoint，224 center crop，无 TTA、无 prior |
| RM_LT | long-tail sampling control | 每样本采样权重 `1/sqrt(n_class)`，有放回；每轮仍抽取 `train_dev` 样本数 | 与 RM_FT 完全相同 |

两组均从同一个 RM_LP best checkpoint 独立初始化，采用相同的 OpenAI CLIP ViT-B/32、CE 2 epoch → GCE q=0.5、特征蒸馏权重 2.0、弱 RRC+水平翻转、8 epoch 和 seed 42。完整配方与本地成对比较见 [RM_FT / RM_LT 严格本地比较](rematch750_local_comparison_20260922.md)。

## 用户回报的平台结果

测试集共 37,444 张。平台百分比可以精确还原为整数正确数：

| 候选 | 平台得分 | 正确数 | 相对 RM_FT |
|---|---:|---:|---:|
| RM_FT | **60.96570879179575%** | **22,828 / 37,444** | — |
| RM_LT | 60.885589146458706% | 22,798 / 37,444 | −0.080119645pp（−30 张） |

平台选择 RM_FT：RM_FT 比 RM_LT 多正确 30 张。该差异很小，但方向明确；RM_LT 在本地 macro 与尾部指标上的微弱优势没有转化为平台总体准确率优势。不得据此否定长尾采样对尾部类别的局部作用，只能判定本次平台总体指标上 RM_FT 更高。

## 回执完整性

当前记录来源为用户直接回报的两个精确平台分数。尚未提供：

- 平台 submission ID；
- 带时区的实际提交时间；
- 平台 reset-period 标识。

因此 `results/rematch_submission_registry.csv` 暂不伪造 `valid` 回执，仍保留原 `ready` 行；机器可读的 scores-only 记录保存在 `results/rematch750_20260922_platform_scores.json`。收到上述真实字段后，应通过 `aegis_clip.cli.rematch record` 补齐正式回执。两次平台额度已经实际使用，不得重复上传相同预测。

## 结论

- 当前已测平台候选中，RM_FT 胜出；RM_LT 不晋级。
- 用户已取消 RM_FULL，本记录不自动恢复或生成 RM_FULL。
- RM_OOFW 是后续独立训练侧降权实验，尚无平台分数，不能与本页两次提交混写。
