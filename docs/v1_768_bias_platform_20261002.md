# V1 768 full 无bias / 测试均衡bias 平台配对（2026-10-02）

用户回填「768 bias合规吗 74.41512658903963」。按明确的候选名称和连续对话，
将 **74.41512658903963%** 关联到已交付的 `v1_768_full_test_bias_submission.zip`，
成为当前最高用户报告平台分。随后用户回填「71.53883132144003 无Bias」，
关联同一checkpoint、同一六视角logits的raw包。尚无独立平台回执、提交ID或实际上传文件绑定核验。

## 分数与解释边界

| 项目 | 数值 |
|---|---:|
| 768 full SWA + 固定测试均衡bias | 74.41512658903963% |
| 同一768 full SWA，无bias | 71.53883132144003% |
| 此前最高full v1 SWA | 70.98600576861446% |
| 新包减此前最高 | +3.42912082042517个百分点 |
| 同模型bias减无bias | +2.87629526759960个百分点 / 1,077张 |
| 768无bias减此前最高 | +0.55282555282557个百分点 / 207张 |
| 按37,444张换算的新包正确数 | 27,864 |
| 按37,444张换算的768无bias正确数 | 26,787 |
| 相比此前最高的正确数增量 | 1,284 |
| 到75%的缺口 | 0.58487341096037个百分点 / 219张 |

计数由用户报告的总体分推算，不是逐图真值或独立平台回执。
同一768 checkpoint和同一六视角logits的两个预先交付包，仅bias不同，
因此用户报告的+2.8763pp支持本固定bias在该模型/测试集上的配对收益；
1,077是总体正确数增量，没有逐图真值，不能拆成具体修正/退化数。
768无bias比旧方案高0.5528pp，但两者读出、训练目标和校准路径都有差别，
不能将207张全部归因于768维度。总增量1,284张中1,077张来自本固定bias配对，
不据此自动扫描强度、迭代数或派生训练，也不外推其他模型/阶段同样受益。

## 固定先验的含义与合规来源

采用官方已知的测试类别均衡先验，750类的目标占比固定为1/750，约0.1333%。
37,444张对应每类约49.925张；它不是从预测频次学习出的未知先验。
对单checkpoint六视角求和logits，固定200次迭代拟合750维类别bias，强度1，
使全测试集平均softmax概率接近均匀。最终逐张argmax，不强制每类恰好50张；
本包预测类别计数为43–55张。这个范围是输出统计，不是合规或准确率的证明。

2026-10-01用户转述官方答复：「利用测试集类别均衡做先验合规，来自官方」。
该固定bias流程符合已记录的确认范围，见[确认原文及用途](v1_test_prior_authorization_20261001.md)。
本段重新访问[官方赛题页](https://www.aicomp.cn/tracks/tracks-1/3714.html)：
页面说明测试类别均衡，禁止测试数据参与模型训练，但未单列这项bias许可。
因此许可来源仍明确为用户转述的官方答复，不伪写为网页原文或独立取得的官方回执。

代码和固定配置核对：测试标签未使用，骨干/LoRA/分类头不更新，只有类别bias进行
测试统计拟合；固定200次/强度1，不按平台反馈选择强度。768是同一官方CLIP ViT-B/32
的投影前CLS读出，没有更换更大骨干。单个导出checkpoint和六视角流程沿用原交付。
本次确认范围是测试均衡bias；不把它扩展为EMA/SWA的新增官方裁定，
[规则第18节](../COMPETITION_RULES_AGENT.md#18-尚需向组委会确认的规则灰区)尚未记录该项单独答复。

## 包、配置与核验

原[训练和交付协议](v1_768_full_test_bias_20261001.md)以及
[固定配置](../configs/v1_768_full_test_bias_experimental_20261001.yaml)保持不变。
原交付JSON保留当时快照，最新平台结果单独记录于本段。

- 桌面现位置：`/mnt/c/Users/lqh22/Desktop/noise/v1_768_full_test_bias_submission.zip`。
- 源包：`/home/lux1/noise-worktrees/v1_768_full_test_bias_20261001/outputs/codex/v1_768_full_test_bias_20261001/submission/submission.zip`。
- 两份ZIP SHA256均为`0f83a2458199d524d3c70f19744fc7043497a531ba968d740ab6f15ebfaf4fca`。
- 无bias桌面包：`/mnt/c/Users/lqh22/Desktop/noise/v1_768_full_raw_submission.zip`；源包在同一输出根的`submission_raw/submission.zip`。
- 无bias两份ZIP SHA256均为`c8d7e0a5fb74f78bbb86796c67bc419f66dfd47733ac6cbe9d26d03875bf462e`。
- 单checkpoint SHA256为`013b8e9c038d21e4dbb677f3cd52ac3ed9fb1bf7d2f74ae826056138275dacfe`。

CPU核对两包各自源CSV、源ZIP、桌面ZIP与原交付摘要；ZIP内外CSV逐字节一致，
两包各37,444行、9项正式提交检查通过。核对共同checkpoint和已保存bias报告的配置/血缘。
原完整NumPy预测重放与独立FP64 bias拟合仍引用
[原交付核验](../results/v1_768_full_test_bias_20261001/delivery_verification.json)，本段不重做模型推理。

可复核命令（仓库根目录，CPU）：

```bash
python3 results/v1_768_bias_platform_20261002/verify.py
```

[机器记录](../results/v1_768_bias_platform_20261002/record.json)、
[本段核验](../results/v1_768_bias_platform_20261002/validation.json)与
[bias检查日志](../results/v1_768_bias_platform_20261002/submission_check.log)与
[无bias检查日志](../results/v1_768_bias_platform_20261002/submission_raw_check.log)。

方案分支`codex/v1_768_bias_platform_20261002`，独立目录
`/home/lux1/noise/worktrees/v1_768_bias_platform_20261002`，起点`e11a5335d91f18454c1a01a6235c8ef87b5c6d29`。
本段只登记反馈、核对规则来源及现有产物；无模型实现改动、新训练、推理或平台上传。
原70.9860%包保留；A/B/v2在途工作与冻结边界保持。核验后推送方案，
main采用自动autostash pull、合并、复核及push，在本反馈检查点暂停。
