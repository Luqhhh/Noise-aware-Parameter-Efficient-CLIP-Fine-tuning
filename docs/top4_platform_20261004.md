# TOP4_PLATFORM_20261004

用户回填“1.74.89317380621728 2.73.99049246875335”，按前一轮桌面四包编号绑定。
01 晋级现役；02 不晋级。分数来自用户，未取得平台提交ID或独立平台回执。
本段复核实际桌面文件与既有交付 SHA、CSV 和九项提交检查，没有训练、推理或平台上传。

## 分数与判定

| 编号 / 配方 | 平台分 | 按37,444张推算正确数 | 结论 |
| --- | ---: | ---: | --- |
| 01 v2 full第5轮raw，六视图logits求和＋固定bias | 74.89317380621728% | 28,043 | 晋级现役 |
| 02 v1 512 full SWA，六视图logits求和＋固定bias | 73.99049246875335% | 27,705 | 不晋级 |
| 旧现役：v2同权重六视图概率均值＋固定bias | 74.6688387992735% | 27,959 | 留作旧解码对照 |
| v1 768 full SWA，六视图logits求和＋固定bias | 74.41512658903963% | 27,864 | 留作另一血统对照 |

01比旧现役 **+0.22433500694378pp / 净+84张**，满足预先指定的超过74.6688%晋级条件。
01比02 **+0.90268133746393pp / +338张**；比v1 768同解码bias **+0.47804721717765pp / +179张**。
02比v1 768同解码bias低 **0.42463412028628pp / 159张**。
75%需28,083张正确，当前差40张；原目标80%需29,956张，当前差1,913张。
上述张数为平台聚合分按测试行数推算，不是逐图真值复核。

## 提交包绑定

桌面目录：`C:\Users\lqh22\Desktop\noise_top4_20261003`。

| 编号 | 文件名 | ZIP SHA256 |
| --- | --- | --- |
| 01 | `01_v2_full_sixview_logit_sum_bias.zip` | `51e372c390f65702ef15af82b7de5a6cc7a382cfaf6f54b7c44be418517aec49` |
| 02 | `02_v1_512_full_sixview_logit_sum_bias.zip` | `260fc98bff216236e2d1bb4b4d6423f10a779f14bd6e508e9bcf0c9a8ae11048` |

原件分别位于 Downloads 的 `v2_continuation_20261001/submission_bias_sixview_logit_sum/submission.zip`
和桌面 `noise/v1_512_test_bias_control_20261001_512_test_bias_submission.zip`，与编号副本字节一致。
编号交付回执：`/home/lux1/noise/outputs/deliveries/top4_20261003/receipt.json`。
本段[record.json](../results/top4_platform_20261004/record.json)保留该回执的摘要、具体路径和检查命令。

## 结论与待测

同一v2权重、六视图及固定bias协议下，把归约改为logits求和并在新分数空间重新拟合bias，
平台净增84张。当前观测支持这一次固定解码改动有效；不能把3,656张改判都当成修正，
也不能从聚合分拆出修正/退化各自张数。此前74.4–74.7压缩假说不是性能上限。

先补 **03 `03_v2_full_sixview_logit_sum_raw.zip`**，再补
**04 `04_v1_512_full_sixview_logit_sum_raw.zip`**，可得到各自纯bias的配对收益。
03仍未知，不能用旧概率均值raw的72.96229035359471%与01作纯bias配对；04仍未知，
也无法算出v1 512的bias增益。这里的下一步是平台测既有包，不派生训练或解码搜索。
v2+768完整路线仍只有实现与CPU验证，未因本次反馈启动长训练。

## 校验与交付

两份ZIP都通过37,444行、750类、内容唯一性、格式/覆盖等九项检查，CSV字节与原交付一致。
复核记录：[validation.json](../results/top4_platform_20261004/validation.json)。复现命令：

```bash
python3 results/top4_platform_20261004/verify.py
git diff --check
```

`verify.py`重新核对桌面包及源包SHA、按Decimal复算净增与目标差距，并在临时目录重跑
`scripts/check_submission.py`；只读已有预测。变更为本记录、当前执行入口和原配方文档的回填链接。
方案分支 `codex/top4_platform_20261004`，验证后提交推送；main使用自动autostash同步、合并、
重验及推送，在该检查点暂停。
