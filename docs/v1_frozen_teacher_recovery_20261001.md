# V1_FROZEN_TEACHER_RECOVERY_20261001：冻结teacher信号转移预算

## 问题与一次固定检查

既有四条DEV学生路线共同错误2,875张；低细节训练配对相对控制全量净−37、实际小图净−2，已关闭该固定配方。
官方冻结CLIP特征的原512/768 cosine teacher尚未与现役DEV学生做逐图恢复/退化预算。
先检查是否保留足够的独立识别信号，再考虑teacher logit转移；本段不训练或扩图像核验。
官方teacher只使用train_dev拟合，但其224中心解码与LR512中心+flip不同；比较不是遗忘的因果检验。

从`origin/main@c794a4b`建立独立分支`codex/v1_frozen_teacher_recovery_20261001`和
`/home/lux1/noise/worktrees/v1_frozen_teacher_recovery_20261001`。开工fetch及全部分支/main历史核对已完成：
A的`clairvoyanttt/lp_lora768_20261001`已有正式启动记录，B的448强增强配对进行中；本段不重复它们，也不操作各机或v2。

## 结果前冻结的协议

[配置](../configs/v1_frozen_teacher_recovery_20261001.json)：仅一次CPU重放原20轮末步teacher512/768。
原768冻结特征诊断已有净+247和近邻净+116的方向证据，因此本次预指定768为唯一主teacher；512为既有路径参照，不看本次结果改主路径。
复用原特征/原头/原split/原标签，源摘要、特征行顺序与完整DEV teacher top1必须严格重现。
固定规则来自原v1 `pseudo_threshold=0.7`、`pseudo_margin=0.2`：teacher最大类别概率≥0.7且前两类概率差≥0.2。
不按验证标签调阈值，不增加训练侧来源、人工标签、强度扫描或模型融合预测。

全14,880张保留teacher/学生原标签macro/micro、修正/退化；另外保留冻结支持组、四学生共同错误、
实际小图、tail75和其余学生错误的交集预算。支持规则只看teacher输出，不能把teacher高置信当作干净真值。
主teacher支持组需修正≥75、净≥75、修正/退化比≥1.25，且支持组覆盖的四学生共同错误中teacher判对≥75，
才能标`supports_transfer_review`；否则关闭这个固定高置信teacher logit转移入口。
通过仅说明有限训练值得复核，不是可实现收益；不自动训练，不生成组合预测或提交包。
现有缓存CPU计算，固定两路径、一规则、两份原头，不设墙钟截止。

## 命令与交付

```bash
env PYTHONPATH=reproducibility/aegis_f1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 scripts/diagnose_v1_frozen_teacher_recovery.py --config configs/v1_frozen_teacher_recovery_20261001.json --output outputs/codex/v1_frozen_teacher_recovery_20261001/diagnostic_r1
```

private逐图score/supported标记留在独立output，不入Git；聚合报告、独立复算、配置/代码/文档入Git。
CPU源输入与重放问题可修复，但不得改冻结门、原头或为收益追加编码/训练。
当前为实现准备；尚无诊断结果或新候选。本段完成后立即commit/push，main自动pull/合并/复核/push，停在检查点。

现役仍为用户报告full v1 SWA70.98600576861446%，未独立绑定平台提交ID。
可提交包`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/submission.zip`，
同目录CSV；SHA256`1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`。
[9项检查](../results/v1_full_swa_platform_20261001/submission_check.log)和
[现役包记录](../results/v1_full_swa_platform_20261001/artifact_verification.json)作为本纯诊断段交付依据，不为新包启动训练。
