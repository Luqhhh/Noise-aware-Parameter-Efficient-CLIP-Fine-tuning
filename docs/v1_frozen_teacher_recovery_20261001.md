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
协议在结果前以`0857ece`提交并推送；CPU源预检通过，4项支持规则与门禁测试通过。
实现包括本段诊断、独立Torch/Counter复算脚本、配置及测试；主线学生训练代码未变。

## 实测预算与决策

CPU诊断4.31秒，28份输入摘要通过；两套原teacher全部29,760条预测重现。
完整14,880张原标签micro/macro：LR512固定平均76.7608/75.7631%，
原512 teacher59.8320/58.7772%，原768 teacher61.4919/60.4555%。
这些是不同native解码的既有模型，不把跨路径差值归因于单独的预训练遗忘或512/768维度。

| teacher对LR512的原标签比较 | 行数 | 修正 | 退化 | 净值 |
|---|---:|---:|---:|---:|
| 512，全部val | 14,880 | 319 | 2,838 | −2,519 |
| 768，全部val（主路径） | 14,880 | 331 | 2,603 | −2,272 |
| 512，冻结高置信支持组 | 1,141 | **0** | 1 | −1 |
| 768，冻结高置信支持组 | 1,890 | **0** | 1 | −1 |
| 768，四学生共同错误全组 | 2,875 | 195 | 0 | 195 |
| 768，共同错误内高置信支持组 | 30 | **0** | 0 | 0 |

原768 teacher能判对195/2,875=6.78%的共同错误，但这195张全部在冻结支持门之外。
另外583张非共同学生错误中teacher判对136张，也均在门外。
高置信支持组只发生2次预测变更：1次退化、1次wrong→wrong；没有新正确，不能用1,890张支持量冒充1,890张可恢复错误。
512参照重复同一风险结论，没有据结果改选路径。

主路径原小图组修正62/退化364，尾75类25/189；支持门内分别0/0、0/1。
原标签可能错误，四候选共同失败不证明标签噪声；teacher正确也不构成干净真值。
本段比较的是已有分类输出，支持组净−1不是蒸馏训练的实测效果，也不是可提交的组合模型分数。

冻结投入门未通过，决策`close_fixed_confident_teacher_logit_transfer`。
关闭这个固定0.7/0.2高置信teacher logit转移入口，不降阈值找收益、不自动训练或扩为feature-anchor实验。
无条件teacher输出仍有大量退化风险；更广的表示保持、软分布转移或新来源证据尚未证实，不能从331个可修正样本直接推断可实现净收益。
这不改变A的微调768学生配对、B的448强增强或v2任务，不替换现役。

结果：[聚合报告](../results/v1_frozen_teacher_recovery_20261001/report.json)；
[独立核验](../results/v1_frozen_teacher_recovery_20261001/independent_verification.json)用CPU Torch float64重新计算两套head概率，
与NumPy最大confidence/margin误差分别≤7.22e−15/1.38e−14；29,760条预测及全部支持组成员一致。
另用Python Counter/逐行条件独立复算20份群体指标与20份配对群体，通过；源摘要、行序、完整DEV人口均核对。
private逐图文件：`/home/lux1/noise/worktrees/v1_frozen_teacher_recovery_20261001/outputs/codex/v1_frozen_teacher_recovery_20261001/diagnostic_r1/scores.npz`，
SHA256`a60f3851a20046bb241d0a04074acd676ce0072b6f4ddd1e702a6fc6ebc33eb7`。
其中只有两份原teacher预测、原学生预测、置信度/概率差与诊断标记；没有融合预测列或提交包。

```bash
env PYTHONPATH=reproducibility/aegis_f1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 scripts/verify_v1_frozen_teacher_recovery.py --config configs/v1_frozen_teacher_recovery_20261001.json --report outputs/codex/v1_frozen_teacher_recovery_20261001/diagnostic_r1/report.json --output outputs/codex/v1_frozen_teacher_recovery_20261001/diagnostic_r1/independent_verification.json
env PYTHONPATH=reproducibility/aegis_f1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -m pytest reproducibility/aegis_f1/tests/test_v1_frozen_teacher_recovery.py -q
```

重放请使用新的output/独立核验路径，脚本拒绝覆盖既有产物。本段仅为已核对诊断，没有新训练、GPU编码或平台分。
已核对现役CSV/ZIP摘要与ZIP内外字节，引用下方既有9项校验；不为新包启动训练。
收尾立即commit/push，main采用自动`git pull --rebase --autostash origin main`、合并、重新校验及push；在该检查点暂停。

现役仍为用户报告full v1 SWA70.98600576861446%，未独立绑定平台提交ID。
可提交包`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/submission.zip`，
同目录CSV；SHA256`1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`。
[9项检查](../results/v1_full_swa_platform_20261001/submission_check.log)和
[现役包记录](../results/v1_full_swa_platform_20261001/artifact_verification.json)作为本纯诊断段交付依据，不为新包启动训练。
