# V2_SIXVIEW_PLATFORM_20261003

状态：`reported_platform_best`。用户回填六视图固定bias包的平台分 **74.6688387992735%**，
超过此前最高的768 full SWA同解码bias 74.41512658903963%。本段只登记反馈、绑定文件、
复跑提交校验并给出待测包优先级；**没有训练、没有新推理、没有平台上传**。

## 反馈与文件绑定

用户消息给出分数与包路径：`74.6688387992735 这是bias+six`
`C:\Users\lqh22\Downloads\v2_continuation_20261001\submission_bias_sixview\submission.zip`。
该文件在磁盘上的ZIP SHA256为`7164b26d9662a8c0e78c9c0636393efe0927739cb84df8dc88ed33e35f47659a`，
与[六视图交付记录](../results/v2_sixview_bias_20261002/report.json)登记的产物摘要一致，
806,198 bytes、37,444行；2026-10-03对**实际Windows文件**重跑`scripts/check_submission.py`九项全部通过。
CSV SHA256 `194c0a0a4da379e6f340e6b078ae48439d00978f3924feb03df9806ed67d2e90`。

这是用户报告分，不是独立平台回执：提交ID、实际上传字节与平台侧文件未独立核验；无逐图真值，
只能给总体净变化，不能拆成修正/退化。评分绑定证据见[record.json](../results/v2_sixview_platform_20261003/record.json)
与[validation.json](../results/v2_sixview_platform_20261003/validation.json)。

## 平台算术与排序变化

| 项 | 数值 |
|---|---:|
| 本次v2六视图bias | **74.6688387992735%** |
| 此前最高（v1 768 full SWA4–12 + 固定bias） | 74.41512658903963% |
| 差值 | +0.25371221023387pp / 推算+95张 |
| 按37,444张推算本次正确 | 27,959张 |
| 80%至少需要 | 29,956张 |
| 距80%仍差 | **1,997张 / 5.3311612007265pp** |

两者是不同的checkpoint、不同训练血统（v2 full末轮raw 对比 v1 768 full SWA4–12），
因此这0.2537pp同时混合了模型、权重平均方式与解码差异，**不能归因为“v2模型更好”**。
v1同checkpoint的固定bias配对收益是+2.87629526759960pp，v2这条纯bias贡献仍然未知。

## 已就绪、尚未测过的包（按信息量排序）

以下包都已交付并通过九项校验，不需要新推理；桌面副本SHA与其交付记录逐一相同
（见[validation.json](../results/v2_sixview_platform_20261003/validation.json)）。

| 顺序 | 包 | 路径（Windows） | 回答什么问题 |
|---|---|---|---|
| 1 | v2 六视图 raw（同checkpoint对照） | `Downloads\v2_continuation_20261001\submission_raw_sixview\submission.zip` | 固定bias在v2上值多少；v2 raw是否高于v1 raw 71.5388，即v2血统是否真的领先。这是解释74.6688的唯一低成本途径 |
| 2 | v2 原四视图 raw（服务器解码） | `Downloads\v2_continuation_20261001\submission\submission.zip`（桌面`v2_full_epoch5_raw_submission.zip`同字节） | 分离“六视图解码”与“模型”：用户指定的448/512/576×flip是否优于原四视图。若为正，是可迁移到任何checkpoint的免费解码杠杆 |
| 3 | v1 512 full SWA 六视图 + 固定bias | `Desktop\noise\v1_512_test_bias_control_20261001_512_test_bias_submission.zip` | 第三个checkpoint上检验固定bias杠杆是否稳定；若同样约+2.9pp，说明模型间差异小于解码/先验差异 |
| 4 | v1 512 full SWA 六视图 raw | `Desktop\noise\v1_512_test_bias_control_20261001_512_raw_submission.zip` | 与上一条配对；也可看512的六视图相对其四视图70.9860的变化 |
| 5 | CRT768 均衡/普通头 + bias（各两包） | `Desktop\noise\v1_crt768_dev_20261001_{balanced,unbalanced}_{test_bias,raw}_submission.zip` | 768头/冻结塔路线；训练人口为133,815张train_dev，低于full，期望值未知 |

不建议占用名额：A（`lp_lora768` LoRA/仅头）、B（`strong_aug448` 强/弱增强）、`lr512_dev`、
v3两臂、`v1_detail_aug_pair`、`v1_cosine_margin_probe` 与 `v2_preprojection768_pair` 四包。
它们的共同问题是**没有bias臂**（放弃已知约+2.9pp的杠杆）、训练人口更小，或已按原门关闭。

## 可能超过74.67的低成本新候选（需另行授权与可用GPU）

现交付包用的是**full_576末轮raw**；v1两个平台最高包、以及v2自己的s2/s3阶段选模都用了
平均权重（EMA/SWA）。下载包`runs/full_576/last.pt`内同时保存`model`与`ema`
（154个张量、`ema_decay=0.9995`、`ema_updates=9290`），因此有两条**零训练**候选：

1. **v2 full_576 EMA + 同六视图 + 固定bias**：只换单checkpoint权重来源，与现最高包同解码；
2. **V2_FULL_LAST3_SWA**：三份raw快照`epoch_03/04/05_raw.pt`已在Windows下载目录，
   对应binding sidecar在`delivery_metadata.tar.gz`内，冻结的CPU平均导出实现见
   [swa.py](../reproducibility/aegis_f1/v2/swa.py)。

两条都**不是现成命令**：六视图推理入口按SHA绑定`runs/full_576/last.pt`，而`swa.py`的preflight
校验plan绑定的绝对输入路径（原服务器`/root/autodl-tmp/...`），这些路径本机不存在。
因此需要一个新的有界实现+核验段（导出新单checkpoint、登记新binding、跑同一解码与出包校验），
成本约为一次六视图推理（本机GPU实测1,236.34秒，约21分钟）加实现与校验，无训练。
**限制必须写明**：full段没有独立留出（val_dev在full训练内），本地无法排序raw/EMA/SWA，
只能由平台判定；这属于“机制有先验、结果未知”的低成本候选，不是已证实的提分，
也不因为GPU空闲或本次反馈自动启动。

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
