# RM_FT94 + RM_LT06 本地融合研究记录（2026-09-22）

## 定位与结论

这是用户在 RM_FT / RM_LT 严格本地比较之后完成的自有探索候选：对两个既有单模型的 logits 做固定线性融合，

```text
blend_logits = 0.94 * RM_FT_logits + 0.06 * RM_LT_logits
prediction = argmax(blend_logits)
```

该候选在 14,880 张本地验证图上比 RM_FT 多正确 12 张（+0.0806pp micro），方向为正但幅度很小。它没有上传平台；2026-09-22 实际占用第二次平台额度的是原始 RM_LT 单模型包。

本项目的 `COMPETITION_RULES_AGENT.md` RULE-05 明确禁止“多模型 logits/概率加权”。因此本候选只作为已完成的本地研究与止损记录，**不是合规候选，不得进入正式 submission pipeline，也不得补传平台**。其增益同时低于项目内部 +0.30pp 迭代投资回报门，研究段在此关闭。

## 输入血缘

| 输入 | SHA-256 |
|---|---|
| RM_FT best checkpoint | `94a361e8672a0db67f43e1e35e91f22715964124ab59ee5157fd3d86c67be305` |
| RM_LT best checkpoint | `088f6464564e82c15a268e35e2aaef3b82699f2a0ca57c24a9013bb28bbd778e` |
| `val_dev.csv` | `d91106df365b62f9bf56eff51474c222925301546642fa427f6778258cb833ab` |
| RM_FT val logits | `aa0875adf6e3fec36192673eb5e2c1dc209f3657f1fc559ba2789adb56a073cf` |
| RM_LT val logits | `860f6ade95158ff7bef203953283249e477d6c50aa1bd9146f66cd3e28327c51` |
| RM_FT test logits | `12efeb847e1066dcfe05f513769b8a2c390c19a602dea70c2b890d498b7b3896` |
| RM_LT test logits | `d50ffb3dc4ba064d504218e2c3cbff52125339c5dab7f734ffa6276392eaad51` |

两组 logits 来自既有 best checkpoint 的同一推理协议与同一顺序导出；模型训练配方、split 与推理细节见 [RM_FT / RM_LT 严格本地比较](rematch750_local_comparison_20260922.md)。没有重新训练或重新选 checkpoint，也没有用测试标签或测试预测分布选择融合比例。

## 本地验证结果

| 方案 | 正确数 | Micro | Macro | 相对 RM_FT micro |
|---|---:|---:|---:|---:|
| RM_FT | 10,848 / 14,880 | 72.9032% | 71.8362% | — |
| RM_LT | 10,817 / 14,880 | 72.6949% | 71.8655% | −0.2083pp |
| RM_FT94 + RM_LT06 | 10,860 / 14,880 | 72.9839% | 71.9249% | **+0.0806pp** |

相对 RM_FT，融合只改变 75 / 14,880 个验证预测：

- RM_FT 正确、融合错误：9；
- RM_FT 错误、融合正确：21；
- 净增：12；
- 未校正 McNemar exact p：0.0427739453。

该 p 值没有校正验证集上的融合比例筛选，不能宣称严格显著。5 次重复嵌套 5-fold 记录的净正确数变化为 `[1, 6, 6, 7, 8]`，均为正、平均 +5.6 张，支持“弱而稳定的本地信号”，不支持“大幅提升”或平台收益保证。

## 测试预测与本地包

融合相对 RM_FT 改变 269 / 37,444（0.7184%）个测试预测；改动集中在 RM_FT 的低 margin 边界样本。研究时生成过一个格式正确的本地包：

- 本地文件：`D:\\2863949663\\SUBMIT_2_RM_FT94_LT06_BLEND_20260922.zip`；
- ZIP SHA-256：`d876f333083503709406926f24a0616e42251d9057c5bee232b2587f72076c38`；
- ZIP 内只含 `pred_results.csv`；
- 37,444 行、37,444 个唯一文件名、两列、标签 `0000–0749`、CRC 通过。

格式通过不等于规则合规。该 ZIP 仅保留在本地用于审计，未提交 Git、未上传平台，也不得作为未来平台候选。

## 交付复核

提交本记录前，已从上述 val/test logits 独立重算并核对：三组 micro/macro、75 个验证预测变化、9/21 成对转移、McNemar p、269 个测试预测变化、四份 logits SHA、ZIP SHA/成员/行数/唯一性/字段数/标签范围。中心数值均与原研究记录一致。

机器可读摘要见 `results/rematch750_ft94_lt06_research_20260922.json`。

## 决策

- 已完成的自有探索：RM_FT94 + RM_LT06 logits 融合；
- 平台状态：未上传、无平台分数；
- 合规状态：RULE-05 禁止的多模型 logits 加权；
- 投资门槛：+0.0806pp < +0.30pp；
- 最终状态：`closed_rule_ineligible_not_uploaded`；
- 后续只允许研究新的单 checkpoint 候选，不从本融合继续扫比例或衍生提交。
