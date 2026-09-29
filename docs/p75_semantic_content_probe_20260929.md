# P75_SEMANTIC_CONTENT_PROBE：阶段A验收与自动表已交付

用户于2026-09-29明确确认“**不用咨询组委会，这是合规的**”。按此具体用途确认继续，来源记为用户消息，不再等待咨询，也不冒称取得组委会书面答复。[范围记录](p75_semantic_content_authorization_20260929.md)。未引入外部图片、其他视觉模型或替代物种标签；文本不进入最终推理。

**本段已从零命中的像素原型推进到有实际集合的语义规则；阶段B配对训练尚未执行，因此没有识别收益或平台增益结论。** 按CLAUDE.md“每个实验段”收尾提交、合并、校验后暂停。下一段唯一训练点为同RM-LP父权重的control/masked六轮对照，复用已有损失屏蔽，不派生检测器网格。

## 实测与决策

| 项目 | 实测 |
|---|---:|
| 冻结图像缓存 | 148,695 × 512，float32，224px官方中心裁剪 |
| CPU官方图像编码对照 | 4张固定训练图，最小余弦0.99999881，最大L2差0.00160249 |
| 开发内容检查 | 28张：18非外观内容、10保护案例 |
| 独立内容验收 | 48张，与开发内容组重叠0 |
| 验收中的自动命中 | 36张，未观察到应保留的生物外观 |
| 验收中的保护外观 | 7张，均保留分类监督 |
| 自动训练命中 | 1,246 / 133,815（0.9311%），423类、1,233内容组 |
| 自动验证代理命中 | 132 / 14,880（0.8871%） |
| 清空类别 | 0；每类最少剩余4张 |

开发正例包括文字页、地图、表格、统计图和服装商品；保护案例包括标本、显微组织、局部器官、解剖/生物绘图、图文混排。独立验收保护案例主要为足迹、解剖及组织绘图，不能声称每种保护类型都获得了独立验证。36个命中检查混合随机抽样24和边界抽样12，不能把0次观察误伤当作全量精度估计。已逐例看完整视野与实际缓存的224中心输入；未对全部命中逐图核验，中心裁剪遗漏仍是局限。

训练命中按最高非目标描述分为地图782、文字论文页262、统计图88、资料表59、服装商品55。内容名称是代理，不是真值；未命中集合仍混有污染。没有按固定比例凑数量，没有单图人工白名单/黑名单。200张验证核验未重复。

已有L05最终中心快照：命中训练图343张原标签概率≥0.7、229张<0.01，平均sqrt(p_y)=0.4916；其GCE分类logit梯度L1合计占全训练最终快照的1.8922%。这不是训练历史、参数梯度范数、梯度危害或噪声比例；高p_y也不意味着梯度大。该计算仅说明命中图不能整体视为已被鲁棒损失忽略。

固定解码的132张验证代理中原标签错误101张，其余14,748张错误3,373张，合计仍是3,474。这里只冻结未来比较切片，不据此调描述或阈值，也不把代理改称干净集。

## 实现与冻结顺序

1. 验证缓存字节哈希、当前阶段绑定、归一化、官方权重及投影维度。元数据的通用`artifact_scope=unverified`原样保留；本段提供明确绑定与数值对照，不修改共享缓存。CPU/CUDA并非逐元素相等：开发时纠正了比方向误差要求过严的坐标/L2容差，实际差异完整记录。
2. 用同一官方权重的文本编码器编码B生物外观12条、A保护/歧义10条、N非目标6条描述，归一化后分别取组内最高余弦，m=sN−max(sB,sA)。先只计算train_dev评分；描述没有根据验证或平台结果调整。
3. 查看28张真实开发图后冻结[规则](../configs/p75_semantic_content_probe_20260929/frozen_rule.json)：**m≥0.05**。描述仍存于命名为development.json的原文件，由SHA256锁定；名字不表示还允许修改。开发18正例中15命中，3文字页保留；10保护例均保留。这些开发数字不作为独立验收成绩。
4. 在开发组之外按种子20260930抽取48例，先看内容再关联分数。通过有限验收后才计算验证侧分组及生成自动表。人工内容判断只决定整条规则可接受，不参与任何样本权重计算。
5. 真实表成功交给**未修改的**`scripts/p75_text_page_runtime.py::TextPageState`，命中1,246条；旧名称仅是已有通用布尔屏蔽接口，不是“文字页V2”。CE/GCE、局部回退、原分母和feature anchor的6项已有测试通过。76张实际查看图的字节哈希、全部148,695行的自动规则一致性均独立核对。

官方技术来源：[OpenAI CLIP接口及图文归一化示例](https://github.com/openai/CLIP#api)。这只支持技术实现，不是竞赛规则来源，也不保证内容识别准确。

## 可重放命令

工作目录：`/home/lux1/noise-worktrees/p75_semantic_content_preflight_20260929`；分支：`codex/p75_semantic_content_preflight_20260929`。目录/文件已存在时脚本拒绝覆盖；重放时选新的输出位置。人工验收记录是本次开发结果，重算评分不能假装重新完成了人工验收。

```bash
python3 scripts/p75_semantic_cache_preflight.py --stage /home/lux1/noise/artifacts/stages/repechage/20260921 --checkpoint /home/lux1/.cache/clip/ViT-B-32.pt --out results/p75_semantic_content_preflight_20260929/cache_compatibility.json
python3 scripts/p75_semantic_content_probe.py --stage /home/lux1/noise/artifacts/stages/repechage/20260921 --checkpoint /home/lux1/.cache/clip/ViT-B-32.pt --descriptions configs/p75_semantic_content_probe_20260929/development.json --preflight results/p75_semantic_content_preflight_20260929/cache_compatibility.json --out outputs/codex/p75_semantic_content_probe_20260929/development
python3 scripts/p75_semantic_acceptance.py --development outputs/codex/p75_semantic_content_probe_20260929/development --rule configs/p75_semantic_content_probe_20260929/frozen_rule.json --image-root /home/lux1/noise --out outputs/codex/p75_semantic_content_probe_20260929/acceptance
# 人工开发验收记录本次已保存；下面只应用已验收规则，不读取逐图人工标志决定权重。
python3 scripts/p75_semantic_freeze.py --stage /home/lux1/noise/artifacts/stages/repechage/20260921 --development outputs/codex/p75_semantic_content_probe_20260929/development --acceptance outputs/codex/p75_semantic_content_probe_20260929/acceptance --rule configs/p75_semantic_content_probe_20260929/frozen_rule.json --evidence results/p75_supervision_rebuild_20260929/sample_evidence.csv.gz --p0-summary results/p75_supervision_rebuild_20260929/summary.json --out outputs/codex/p75_semantic_content_probe_20260929/frozen
python3 scripts/verify_p75_semantic_content.py --archive results/p75_semantic_content_probe_20260929 --image-root /home/lux1/noise
python3 -m pytest tests/test_p75_text_page_runtime.py -q
```

产物：[冻结摘要](../results/p75_semantic_content_probe_20260929/frozen_summary.json)、[自动逐样本表](../results/p75_semantic_content_probe_20260929/frozen_selection.jsonl.gz)、[类别覆盖](../results/p75_semantic_content_probe_20260929/frozen_class_coverage.csv)、[独立验收](../results/p75_semantic_content_probe_20260929/acceptance_acceptance_report.json)、[缓存对齐](../results/p75_semantic_content_preflight_20260929/cache_compatibility.json)。图像预览仅存本地，不入Git。

## 下一段固定训练与评价边界

从原合法RM-LP父权重开始；不是L05最终续训，父头已接触旧监督的限制保留。两臂相同样本顺序/种子/有效batch/更新次数/16轮LR日程，截在第6轮；第5轮启用局部分支，第4轮仅观察、第6轮主判断。沿用原总计6 GPU小时上限（含验证），超时不加预算，不把未完成判成无效。

干预仅将自动命中的全局/局部分类项置零，feature anchor及其他图的原GCE等均保留，不再实现三通道或替代物种目标。每轮记录命中/其余组的实际训练py、pmax、分类损失。全量原始macro/micro、逐张修正/退化必须报告；固定内容代理仅解释变化。总分下降若主要来自命中组、其余组明确净改善，可保留待平台验证；仅降低命中图置信度不支持完整训练。禁止自动追加阈值/描述/训练配置扫描。

本段CPU完成内容验收和自动表；未启动GPU训练、未操作SAM、未产生新候选包。现役可提交包仍为`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/{pred_results.csv,submission.zip}`，既有37,444行、9/9校验见`results/p75_supervision_rebuild_20260929/submission_check.log`。阶段A检查点按仓库约定提交、合并、重验、推送后暂停，不能把它称为阶段B/C已经完成。
