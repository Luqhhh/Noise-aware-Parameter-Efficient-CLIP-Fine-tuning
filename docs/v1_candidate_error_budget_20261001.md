# V1_CANDIDATE_ERROR_BUDGET_20261001 本机误差诊断

本段完成14,880张独立val_dev的已有候选逐图对齐、误差重叠和预算核对。
四条候选共同判错2,875张，占LR512固定平均候选3,458张错误的83.14%。
LR512前十个混淆类对仅覆盖115张错误；改善少数类对或尾类不足以解释主要误差。
本次没有训练、GPU推理、新候选或平台成绩。现役full v1 SWA仍为用户报告70.98600576861446%。

## 固定输入与可比较范围

使用当前20260921/750类数据、133,815张train_dev和14,880张内容组隔离val_dev。
逐图文件名、顺序和原标签完全一致；验证集14,658个decoded内容组与训练集无交集。
19份输入文件SHA核对通过，全部既有native父/候选指标重新复现。
输入路径与摘要见[完整报告](../results/v1_candidate_error_budget_20261001/report.json)。

不同路线保留原解码，不能把跨路线分差解释成单因素效果：

| 路线 | 实际输入/视图 | micro | macro | 相对其原解码父模型 |
|---|---|---:|---:|---|
| LR512 EMA2–4 | 512输入，512中心+flip，无bias | 76.7608% | 75.7631% | 修正324、退化214、净+110 |
| 普通768头 | 448单中心，无bias | 76.5188% | 75.6027% | 修正308、退化248、净+60 |
| 均衡768头 | 448单中心，无bias | 76.1290% | 75.3420% | 修正319、退化317、净+2 |
| WFT448末轮EMA | 448输入，448/512/576中心及flip，无bias | 76.5457% | 75.5200% | 修正323、退化236、净+87 |

同一DEV父checkpoint仅改变既定解码就有较大的逐图交换：448单中心→512两视图，
修正263/退化277/净−14；→448输入六视图，修正261/退化284/净−23。
因此三份原父分数不是一个逐图相同的基线。本段不改已有候选解码，不恢复TTA搜索。

## 主要误差预算与未知项

下面都是原标签诊断，分组互有重叠，不能相加，也不直接换算平台净修正。
尺寸属性是描述性代理，没有真实来源标签；本段回顾既有候选，不把这些代理补写成原训练的预注册判据。

| 问题/代理 | 组内图数 | LR512错误 | 证据与可恢复收益边界 |
|---|---:|---:|---|
| 小图：短边<224 | 1,848 | 612 | 原尺寸即可确定；LR512相对native父修正71/退化33/净+38，恢复比例及平台联系仍未知 |
| 尾75类 | 1,003 | 333 | 固定train_dev支持量；LR512净+14。均衡头比LR512尾部净+20，但总体净−94，含解码差异 |
| 跨标签相同内容 | 367 | 251 | 177个val内容组含互斥原标签；同内容确定性分类器至少189张原标签错误，不能确定哪张标签干净 |
| A已冻结混淆类集合 | 417 | 161 | 普通768头原错误173；四候选共同错误135。20类仍有探索规模，但不是平台错误清单 |
| 长宽比≥2 | 261 | 106 | 官方尺寸代理；LR512相对native父净+3，因果与可恢复比例未知 |
| native父对、LR512平均后错 | 214 | 214 | 可观察退化，不等于已证实灾难性遗忘；与小图交33、尾类交11、A类集合交12 |

前五项代理的LR512错误去重并集1,205张，补入native父退化后并集1,359张；
另2,099张不在这些代理中，仍未解释。小图与尾类错误交61，与相同内容跨标签错误交56；
完整两两交集见[独立核对](../results/v1_candidate_error_budget_20261001/independent_verification.json)。
相同内容冲突至少189张错误的下界，只证明评价标签存在矛盾，不能据此宣称标签噪声解释了其余错误。
更一般的错标、真实来源漂移、细粒度可恢复比例和平台迁移仍未知。

LR512前10/50/100个无向错误类对分别覆盖115/340/510张，约3.33%/9.83%/14.75%。
覆盖不是可实现收益；重新选类对、单独学习这些类对或按真值路由都没有获得启动依据。

## 候选重叠与采样不确定性

四候选共同错误2,875张，其中1,651张预测相同、1,224张预测不同；一致不证明标签错。
只被一个候选判对的图数分别为LR512138、普通头34、均衡头47、WFT149。
这描述各路线仍有不同错误，不能当成可提交的集成收益，也没有导出融合预测。

以LR512为起点换成普通头：修正376/退化412/净−36；换均衡头381/475/净−94；
换WFT311/343/净−32。跨路线比较同时包含结构、训练与解码差异。
用2,000次seed42整decoded内容组重采样，LR512相对native父micro差的95%百分位区间
为[+0.4430,+1.0469]pp；普通头减LR512为[−0.6062,+0.1146]pp。
它们只是这些含噪原标签的抽样区间，不涵盖标签偏差、平台漂移或多重比较，不能证明平台排序。

决定：保留已出包LR512固定平均候选及A/B既定独立配对优先级。
短边不足224组规模较大，可在A/B完成后作为**回顾性辅助诊断**，不能替代已冻结目标组/投入门。
本段不新增第三条训练路线、不自动执行LR512_FULL或解冻旧配方；主要共同失败的机制和平台恢复收益尚缺证据。

## 精确命令、实现与核验

方案分支`codex/v1_candidate_error_budget_20261001`，从`origin/main@8238dc3`建立，独立目录：
`/home/lux1/noise/worktrees/v1_candidate_error_budget_20261001`。
没有spawn子agent，不操作A/B机器或远端v2，不改变公共Python依赖。
CPU完整诊断实测3.73秒；用户已取消后续墙钟上限，本段没有设置运行超时。

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 scripts/diagnose_v1_candidate_errors.py \
  --config configs/v1_candidate_error_budget_20261001.json \
  --output outputs/codex/v1_candidate_error_budget_20261001/diagnostic_r2

python3 scripts/verify_v1_candidate_error_report.py \
  --report results/v1_candidate_error_budget_20261001/report.json \
  --config configs/v1_candidate_error_budget_20261001.json \
  --output results/v1_candidate_error_budget_20261001/independent_verification.json

python3 -m pytest reproducibility/aegis_f1/tests/test_v1_candidate_error_diagnostic.py -q
git diff --check
```

重放使用新的独立output路径；脚本拒绝覆盖已有输出。
实现：只读已有NPZ/协议和官方split，按原decoder报告native配对、群体计数和整内容组bootstrap。
测试6项通过，覆盖错序/错标签资产拒绝、整组重采样、macro定义和wrong→wrong计数。
另外用stdlib集合/Counter独立重计全14,880行、81份群体指标、135份配对群体和9份重叠群体，全部通过。
逐图CSV保留在独立output内、不入Git；聚合report和独立核对入库。

本段修改为两份脚本、固定诊断JSON、6项测试、聚合结果、本记录、当前执行入口及private输出ignore。
不改模型训练实现；实测结论仅限本页所列预测与原标签。
方案验证后立刻commit/push；main使用自动`pull --rebase --autostash`模式同步，合并、重新核验、push后停在本检查点。

## 现役可提交包

现役仍为`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/submission.zip`，
同目录`pred_results.csv`。ZIP SHA256
`1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`已在本段重新核对。
37,444行/9项通过及内外CSV一致引用[既有正式校验](../results/v1_full_swa_platform_20261001/submission_check.log)
和[包复核](../results/v1_full_swa_platform_20261001/artifact_verification.json)。
本段是已验证诊断交付，不生成新的提交包，不声称已完成新训练候选。
