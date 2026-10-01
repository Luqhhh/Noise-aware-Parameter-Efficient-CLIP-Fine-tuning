# WFT448_FULL 平台反馈（2026-10-01）

用户在要求WFT448_FULL ZIP写入桌面后回填`70.43585087063347`。
按连续对话将该值关联到桌面`wft448_full_submission.zip`的固定末轮EMA候选。
这是用户报告的平台分；没有独立平台回执、提交ID或实际上传包绑定核验。

| 项目 | 数值 |
|---|---:|
| WFT448_FULL用户报告 | 70.43585087063347% |
| 现役full v1 SWA用户报告 | 70.98600576861446% |
| 本候选减现役 | −0.55015489798099个百分点 |
| 37,444张下本候选推算正确数 | 26,374 |
| 相对现役推算正确数变化 | −206 |
| 本候选距75% | 1,709张 / 4.56414912936653个百分点 |

计数由总体分换算，不是逐图真值审计，也不是独立平台回执。
本次固定末轮EMA候选未超过现役，保留full v1 SWA70.9860%的模型/推理/包。
WFT DEV原标签验证曾净修正87张、micro+0.5847pp；该提升没有迁移为本次平台收益。
这不能确定下降来自全参数训练、平均策略、校准差异或其他因素，未测试导出的平台能力仍未知。
本段只归档反馈和核对产物，未启动新训练、推理、平均窗口搜索或平台上传。
LR512_FULL保持`evidence_requires_review`、未启动。
下一低训练成本备选是已有LR512_DEV固定EMA2–4权重的推理出包；该平均权重目前尚无CSV/ZIP，不能把现有末轮EMA包标成76.7608%的平均模型。

[机器可读记录](../results/wft448_full_platform_20261001/record.json)、
[核验结果](../results/wft448_full_platform_20261001/validation.json)、
[原固定4轮配置/命令/交付](wft448_full_20261001.md)。
本反馈段分支`codex/wft448_full_platform_20261001`、独立worktree
`/home/lux1/noise/worktrees/wft448_full_platform_20261001`，起点`40722ca7d515b94cbe1cf5423df04a6518b38a27`。

WFT源包：
`/home/lux1/noise/worktrees/wft448_full_20261001/outputs/codex/wft448_full_20261001/submission/submission.zip`。
桌面包：`/mnt/c/Users/lqh22/Desktop/wft448_full_submission.zip`。
两者SHA-256相同：`833dcd651f2d8698a552fa5ef3ff1d53d198ae328c4b56405a2a67de78434af0`。
37,444行、9项校验来源见[原交付核验](../results/wft448_full_20261001/final_validation.json)，
桌面ZIP内部CSV与源CSV逐字节一致。

现役包继续保留：
`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/submission.zip`，
37,444行、9项校验来源为[现役包复核](../results/v1_full_swa_platform_20261001/artifact_verification.json)。

复核命令（标准库/CPU，无GPU）：

```bash
python3 results/wft448_full_platform_20261001/verify.py
```

核验后立即提交推送本反馈分支；main使用自动autostash模式pull、合并、复核、push，停在反馈检查点。
