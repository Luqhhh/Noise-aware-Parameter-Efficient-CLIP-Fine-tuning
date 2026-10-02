# V2_SIXVIEW_PLATFORM_20261003

状态：`reported_platform_best`。用户先后回填同一checkpoint的两包：六视图固定bias
**74.6688387992735%**、六视图raw **72.96229035359471%**；bias包超过此前最高的
768 full SWA同解码bias 74.41512658903963%。本段只登记反馈、绑定文件、复跑提交校验、
给出配对算术与待测包优先级；**没有训练、没有新推理、没有平台上传**。

## 反馈与文件绑定

两条用户消息分别是：`74.6688387992735 这是bias+six`
`C:\Users\lqh22\Downloads\v2_continuation_20261001\submission_bias_sixview\submission.zip`，
以及`72.96229035359471 v2 六视图 raw（同 checkpoint 对照）`。
bias包ZIP SHA256 `7164b26d9662a8c0e78c9c0636393efe0927739cb84df8dc88ed33e35f47659a`
（806,198 bytes），raw包ZIP SHA256 `4228a78a390e34744e42fb9a8ef861c6807d37b4e28dedc61bc3984c98353c6c`
（805,956 bytes），两者均与[六视图交付记录](../results/v2_sixview_bias_20261002/report.json)、
[配对补交付回执](../results/v2_sixview_pair_handoff_20261002/pair_receipt.json)登记一致，
各37,444行；2026-10-03对**实际Windows文件**重跑`scripts/check_submission.py`九项全部通过。
CSV SHA256分别为`194c0a0a…2e90`（bias）与`446165ca…f8e`（raw）。

这是用户报告分，不是独立平台回执：提交ID、实际上传字节与平台侧文件未独立核验；无逐图真值，
只能给总体净变化，不能拆成修正/退化。评分绑定证据见[record.json](../results/v2_sixview_platform_20261003/record.json)
与[validation.json](../results/v2_sixview_platform_20261003/validation.json)。

## 平台算术与排序变化

| 项 | 数值 |
|---|---:|
| 本次v2六视图bias | **74.6688387992735%**（27,959张） |
| 同checkpoint六视图raw（2026-10-03回填） | 72.96229035359471%（27,320张） |
| 本次固定bias配对收益 | **+1.70654844567879pp / +639张** |
| 此前最高（v1 768 full SWA4–12 + 固定bias） | 74.41512658903963%（27,864张） |
| v1同checkpoint六视图raw | 71.53883132144003%（26,787张） |
| v1固定bias配对收益 | +2.87629526759960pp / +1,077张 |
| 修正后v2−v1 | +0.25371221023387pp / +95张 |
| 原始(raw)v2−v1 | +1.42345903215468pp / +533张 |
| 80%至少需要 | 29,956张 |
| 距80%仍差 | **1,997张 / 5.3311612007265pp** |

两步配对都可确认（各自同checkpoint、同缓存、只差固定bias），因此两个bias收益是可比的内部量。
**v2模型在修正前确实更强（+533张），但固定均衡bias在v2上只修回639张，在v1上修回1,077张**，
修正后两者只差95张。原始收益到修正收益的透过率约0.178（95/533）。
这支持“固定先验校正会吸收一部分模型差异、修正后分数向74.4–74.7收敛”的假说；
它仍是假说而不是定律，且**未解开的混杂**是两条血统的多视图归约不同：
v1是六视图logits求和（等价于逐视图概率的几何平均），v2是`log(六视图概率均值)`，
固定200次、damping1的bias在各自的分数空间里作用，收益差不能单独归给模型或解码。
若这个透过率继续成立，靠“继续提高raw正确率”补上剩余1,997张需要不现实的raw涨幅；
剩余错误更可能是内容/混淆错误，而不是先验偏移。

## 免费诊断：固定bias对硬判决的作用强度（2026-10-03）

只用已交付的六份CSV（无标签、无推理、只读）统计每个包argmax类别计数与均匀先验1/750的偏离，
脚本与结果见[analyze_predictions.py](../results/v2_sixview_platform_20261003/analyze_predictions.py)、
[prediction_distribution_diagnostic.json](../results/v2_sixview_platform_20261003/prediction_distribution_diagnostic.json)
（期望每类49.9253张）：

| 包 | 解码 | raw chi2 | bias后 chi2 | bias后min/max | bias翻转数 |
|---|---|---:|---:|---:|---:|
| v1 768 六视图 | logits求和 | 7,039.1 | **29.9** | 43 / 55 | 6,775 |
| v1 512 六视图 | logits求和 | 7,364.9 | **30.2** | 44 / 57 | 6,838 |
| v2 六视图 | log概率均值 | 4,586.1 | **1,507.5** | 4 / 95 | 3,077 |

两个v1血统模型在固定bias后都落到几乎精确的硬配额（chi2≈30），而v2只从4,586降到1,507，
离均衡很远。**同样的固定bias在v2的分数空间里对硬判决的作用弱得多**，且这个差异在两个v1模型上稳定复现，
说明它主要是解码/分数尺度属性，而不是单个模型的性质；v2 raw本身的偏斜也更小（4,586 对 7,039），
所以模型侧也贡献了一部分。两者共同解释+1.7065pp 对 +2.8763pp的收益差，但符号方向仍需平台判定。

**对“v2六视图logits求和（≈逐视图概率几何平均）”的判断**：
它是一个**无自由参数**的确定性归约，等价于把v1已在平台上验证过（74.4151%）的解码协议套到更强的v2 checkpoint上，
按上面的诊断会把v2推回chi2≈30的强配额状态。这既可能补回一部分收益（v1两条血统在该状态下都拿到约+2.9pp），
也可能**过校正**——历史阶段里“精确配额”变体相对调好的prior曾下降0.3284pp，方向本地无法判定：
full段没有独立留出，且没有逐视图缓存（本机只存`mean_probabilities.npy`），必须重跑一次六视图推理。
因此它值得做，但应作为**一个预先限定的单一候选**，不是解码扫描。

## 已就绪、尚未测过的包（2026-10-03按新证据重排）

以下包都已交付并通过九项校验，不需要新推理；桌面副本SHA与其交付记录逐一相同
（见[validation.json](../results/v2_sixview_platform_20261003/validation.json)）。
第1条已由本次raw回填完成，其余按更新后的信息量排序：

| 顺序 | 包 | 路径（Windows） | 回答什么问题 |
|---|---|---|---|
| 1 | v1 512 full SWA 六视图 + 固定bias | `Desktop\noise\v1_512_test_bias_control_20261001_512_test_bias_submission.zip` | 第三个checkpoint，且与v1 768用同一套“六视图logits求和+固定bias”协议：若它的bias收益也接近+2.9pp，则v2较小的+1.71pp更可能是解码（概率均值 vs logits求和）造成，而不是模型；若也只有约+1.7pp，则收益随模型质量收缩 |
| 2 | v1 512 full SWA 六视图 raw | 同前缀 `_512_raw_submission.zip` | 与上一条配成同checkpoint对，给出第三个(修正前, 修正后)点 |
| 3 | v2 原四视图 raw（服务器解码） | `Downloads\v2_continuation_20261001\submission\submission.zip`（桌面`v2_full_epoch5_raw_submission.zip`同字节） | 在最佳checkpoint上分离“六视图解码”与“模型”；注意任何“四视图+bias”后续包都需要新的GPU推理，本机没有四视图概率缓存 |
| 4 | CRT768 均衡/普通头 + bias（各两包） | `Desktop\noise\v1_crt768_dev_20261001_{balanced,unbalanced}_{test_bias,raw}_submission.zip` | 768头/冻结塔路线；训练人口为133,815张train_dev，低于full，期望值未知 |

不建议占用名额：A（`lp_lora768` LoRA/仅头）、B（`strong_aug448` 强/弱增强）、`lr512_dev`、
v3两臂、`v1_detail_aug_pair`、`v1_cosine_margin_probe` 与 `v2_preprojection768_pair` 四包。
它们的共同问题是**没有bias臂**（放弃已知的bias杠杆）、训练人口更小，或已按原门关闭。

## 权重平均候选：在新证据下降级（需另行授权与可用GPU）

现交付包用的是**full_576末轮raw**；v1两个平台最高包、以及v2自己的s2/s3阶段选模都用了
平均权重（EMA/SWA）。下载包`runs/full_576/last.pt`内同时保存`model`与`ema`
（154个张量、`ema_decay=0.9995`、`ema_updates=9290`），另有两条**零训练但需新实现段**的候选：

1. **v2 full_576 EMA + 同六视图 + 固定bias**：只换单checkpoint权重来源，与现最高包同解码；
2. **V2_FULL_LAST3_SWA**：三份raw快照`epoch_03/04/05_raw.pt`已在Windows下载目录，
   对应binding sidecar在`delivery_metadata.tar.gz`内，冻结的CPU平均导出实现见
   [swa.py](../reproducibility/aegis_f1/v2/swa.py)。

两条都**不是现成命令**：六视图推理入口按SHA绑定`runs/full_576/last.pt`，而`swa.py`的preflight
校验plan绑定的绝对输入路径（原服务器`/root/autodl-tmp/...`），这些路径本机不存在；
需要一个新的有界实现+核验段（导出新单checkpoint、登记新binding、跑同一解码与出包校验）。
**本次raw回填后它们的优先级下降**：在“修正后分数压缩、原始收益透过率约0.178”的观察下，
权重平均预计只能带来零点几个pp的raw变化，对应修正后收益更小；这不是撤回候选，
只是不再是当前名额的最佳用途。full段也没有独立留出，本地无法排序raw/EMA/SWA。

## 未做与资源边界

没有训练、没有测试集推理、没有平台上传；不据本次反馈扫描bias强度、温度或解码变体；
不自动恢复A/B/v3/监督阶梯或任何已关闭配方；不连接已关机远端。
本会话`/dev/dxg`缺失且`torch.cuda.is_available()`为False，新候选推理需要先具备可用GPU，
本段不自动启动该动作。

## 核验命令

```bash
sha256sum /mnt/c/Users/lqh22/Downloads/v2_continuation_20261001/submission_bias_sixview/submission.zip
python3 scripts/check_submission.py --test_dir /home/lux1/noise/test \
  --class-mapping /home/lux1/noise/artifacts/stages/repechage/20260921/class_to_idx.json \
  --csv /mnt/c/Users/lqh22/Downloads/v2_continuation_20261001/submission_bias_sixview/pred_results.csv \
  --zip /mnt/c/Users/lqh22/Downloads/v2_continuation_20261001/submission_bias_sixview/submission.zip
```

同一命令对`submission_raw_sixview`与`submission`（四视图raw）执行，三包均九项通过。
