# A2 LoRA平台测试结果（2026-07-22，历史记录）

本文记录初赛A2 LoRA容量消融的用户报告平台结果；仅修复原文乱码，不新增成绩或重新核验旧包。
当前阶段与现役交付见[当前执行入口](current_execution_plan.md)。
同阶段来源见[独立实验账本](aegis_independent_experiments_2026-07-22.md)。

| 实验 | 推理方式 | 平台分数 | 当时提交包 |
|---|---|---:|---|
| A2_LORA_MIN | 裸推理 | 61.1167% | `outputs/a2_lora_min_knn/A2_LORA_MIN_KNN_DROP/seed42/submissions/submission.zip` |
| A2_LORA_MIN | horizontal_flip TTA | 61.6574% | `outputs/a2_lora_min_knn/A2_LORA_MIN_KNN_DROP/seed42/submissions_tta/submission.zip` |
| A2_LORA_FULL | 裸推理 | 61.5733% | `outputs/a2_lora_full_knn/A2_LORA_FULL_KNN_DROP/seed42/submissions/submission.zip` |
| A2_LORA_FULL | horizontal_flip TTA | 62.1781% | `outputs/a2_lora_full_knn/A2_LORA_FULL_KNN_DROP/seed42/submissions_tta/submission.zip` |

FULL+TTA比MIN+TTA高0.5207pp，为当时这组容量消融的最高分。
MIN裸推理比既有A2 Flip基线61.2128%低0.0961pp，MIN+TTA则高0.4446pp；推理协议不同，分别记录。
这组结果仍低于当时独立实验报告的F1+M1 63.3276%，没有取代其候选。
每份包使用单checkpoint；Flip包使用同checkpoint的固定多视图推理。
旧包路径仅是历史来源，不声明其当前存在或可跨阶段复用；当前规则以[根规则](../COMPETITION_RULES_AGENT.md)为准。
