# V2_TASK_TEACHER_RECOVERY_20261002

固定单次本机CUDA验证前向，检查已训练v2是否提供可选择的监督信号；没有学生训练、
test预测、融合候选或平台分。现役768 full+bias仍为74.41512658903963%，80%未达到。
LR512仅作为有完整独立DEV预测的旧学生参照，不代表现役768 full；普通768头为另一DEV参照。

## 新证据与固定问题

已完成的[V2 DEV诊断](v2_dev_error_budget_20261002.md)显示：任务训练s3 EMA相对LR512
修正810/退化704，能判对旧四候选共同错误中的464张。无条件替代风险很大，
但这批任务训练权重在先前冻结teacher检查时尚未交付。
原冻结CLIP224 teacher微调分类头只有61.4919% DEV，其固定支持组修正0/退化1，已关闭。
本段不复活原teacher或降低阈值；仅检查新交付的任务训练s3 EMA是否在相同规则下具有不同恢复预算。

可证伪假设：teacher最大类别概率≥0.7且前两类概率差≥0.2的固定支持组，
相对LR512能保留足够的新增正确并限制退化，且能覆盖旧四模型共同错误。
阈值和复核门完整沿用[原协议](../configs/v1_frozen_teacher_recovery_20261001.json)，
不观察结果后选择阈值、teacher、温度或学生。通过只支持有限转移实验的复核，
不证明训练能实现该收益，不自动学生训练/完整训练或训练集重编码。

## 结果前冻结的执行边界

[配置](../configs/v2_task_teacher_recovery_20261002.json)：唯一teacher为s3第4轮EMA，
checkpoint SHA `05f4c1ace5647ce9afa166f6af629389262a44da3570918c83bc53a916a6425f`。
保持原576中心、短边round(576×1.14)=657、无flip/bias、BF16；不选择full_576。
初版误将未生效的配置项`val_batch_size=128`当作原生batch，后按真实执行路径修正为16；
首次失败与修正依据见下方工程纠错，原协议提交`a92207a`及实现`5765be9`保留。
只对同一14,880张val_dev前向一次，逐图原文件SHA核对、路径/标签/内容组绑定；
未入训，但此val用于原有checkpoint选择。full模型不能混入独立DEV比较。

本机CUDA无计算任务时才启动，不抢占，不连接旧远端。累计诊断进程运行上限900秒，
前三批即正式前向的一部分；排除首批冷启动后，用其余两批最慢时间外推，超剩余预算则停止。
OOM、明显重放不一致或预算失败只交付失败证据，不改变batch/数值条件自动重启。
唯一已记录工程修正为恢复代码实际使用的原生batch16；不是按结果选择batch。
不同机器BF16重放最多容许14/14,880个top1变化（<0.1%），每个变化的本机前二概率差须≤0.01，
且不能有任何变化进入0.7/0.2支持组；超出任一条件拒绝预算结论。
保存本机logits和原归档top1，所有数字以本机同一次前向为准，并报告变化，不伪称原预测完全复现。

主比较为LR512 EMA2–4；普通768头只作固定次比较，两者均不代表现役768 full。
全部原标签micro/macro、修正/退化及沿用9组都保留；另列旧四模型共同错误、支持/非支持交集。
支持门只看teacher输出，不用标签选样本。主支持组修正≥75、净≥75、修正/退化比≥1.25，
并且共同错误内支持修正≥75，四门全部满足才标`supports_transfer_review`。
不通过则关闭这个固定任务teacher高置信入口，不降门槛补跑。
只统计已有两种预测的分组预算，不生成按teacher/学生切换的组合预测。

先提交协议，再执行。完成后独立CPU float64复算完整概率、支持组、预测及计数；
报告源摘要、耗时和结果，提交/推送并在main自动autostash同步集成复核，在检查点停下。

## 既有可提交产物

现役：`C:\Users\lqh22\Desktop\noise\v1_768_full_test_bias_submission.zip`，
SHA `0f83a2458199d524d3c70f19744fc7043497a531ba968d740ab6f15ebfaf4fca`，
[现役校验](../results/v1_768_bias_platform_20261002/validation.json)。
明天待测v2六视图bias/raw及两包37,444行9项校验见[配对交付](v2_sixview_pair_handoff_20261002.md)。
本诊断不产生新提交包，不改变平台比较基准。

## 执行与结果

独立分支`codex/v2_task_teacher_recovery_20261002`，目录
`/home/lux1/noise/worktrees/v2_task_teacher_recovery_20261002`，基线`origin/main@5913fa1`。
开工已fetch、检查所有分支及main历史；没有重叠任务teacher概率检查，原V1冻结teacher和CRT近邻入口保持关闭。
### 原生batch读取错误与修复

协议`a92207a`和实现`5765be9`先推送，9项测试通过后首次前向。
195.3752秒完成全部14,880张，但39张top1与归档不同，超过14张门，故没有进入预算/推进结论。
39张全部在支持门外，最大前二概率差0.0099351；正确数11,526而归档为11,528。
原目录`outputs/codex/v2_task_teacher_recovery_20261002/diagnostic/`的failure、logits和scores原样保留。

随后核对`v2/runtime.py`的真实调用发现，val loader传入的是
`cfg["local_replay"]["micro_batch_size"]`（本阶段为16），没有使用`data.val_batch_size`（128）。
这是本次分析器将未使用字段误读为原生条件，不是概率门不合适。
修复只恢复实际batch16，原0.7/0.2、14张容差、0.01概率差及全部推进门不变；
增加128误用必须拒绝、16必须接受的回归测试。既有前向花费从900秒预算中扣除，
修复后的单次上限704秒，两次总上限899.3752秒，不再自动追加其他重放。
修正提交推送后使用新的`diagnostic_native16/`目录，不能覆盖首次失败或据失败缓存放宽门。

### 最终状态：重放门未通过，转移收益未知

修正`c268e40`先推送，11项规则/批量/拒绝full模型测试通过，再按真实batch16执行。
176.7396秒完成全部14,880张；两次本机logits逐元素完全一致，均与原归档有39张top1差异
（0.2621%，上限为14张）。因此误读batch是已修复的实现错误，但**不是剩余差异的已证实原因**。
最大变化样本前二概率差0.0099351，支持组内变化0；虽然这两门通过，数量门仍失败。
本机原标签正确11,526，micro/macro77.4597%/76.5800%；原归档11,528，77.4731%/76.5927%。
这些分数只说明重放偏差，不能用来宣称新的模型、收益或平台排序。

CPU Torch float64从保存logits独立重算全部14,880条预测及支持成员，与NumPy一致；
confidence/margin最大误差均1.7764e-15。18份源摘要、原初版Git内容、四份归档源码一致：
V2模型、runtime、training_utils和Aegis位置编码实现。两次原图文件摘要全部通过，
没有从概率门、标签或样本筛选解释/修饰这39张差异。
原远端只保存top1，没有逐张logits；剩余差异可能涉及运行环境，但本段未证明原因，
不连接或重启服务器，不改dtype/后端/容差再扫描。

**决定：`close_audit_native_replay_mismatch_transfer_unknown`。**
未执行支持组修正/退化推进门，不能称teacher有效或无效；不启动转移训练。
首次195.3752秒＋修正176.7396秒＝372.1148秒（6.20分钟），在累计900秒预算内。
本段无新提交包或平台分；既有v2六视图raw/bias平台待测与现役768基准继续按前述交付记录处理。

### 确切命令及产物

以下前向命令是已关闭运行的复现记录，不构成再次执行许可；诊断目录已有产物，禁止覆盖。

```bash
cd /home/lux1/noise/worktrees/v2_task_teacher_recovery_20261002
PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python3 -m pytest tests/test_v2_task_teacher.py reproducibility/aegis_f1/tests/test_v1_frozen_teacher_recovery.py -q
PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python3 -u scripts/probe_v2_task_teacher.py \
  --config configs/v2_task_teacher_recovery_20261002.json \
  --output outputs/codex/v2_task_teacher_recovery_20261002/diagnostic_native16
PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python3 scripts/verify_v2_task_teacher.py \
  --directory outputs/codex/v2_task_teacher_recovery_20261002/diagnostic_native16 \
  --output results/v2_task_teacher_recovery_20261002/independent_verification.json
```

初次batch128使用相同入口、初版配置，输出`diagnostic/`；两个私有目录都保留`failure.json`、
`scores.npz`和`cost.json`。主目录最后`progress.json`为异常前的进度快照，终态以`failure.json`和
已退出进程为准，不能据进度文件自动重启。
实现为[诊断入口](../scripts/probe_v2_task_teacher.py)、[独立失败复算](../scripts/verify_v2_task_teacher.py)、
[测试](../tests/test_v2_task_teacher.py)、固定配置、私有输出ignore和本段文档，正式训练/推理代码未修改。
证据见[首轮及源码定位](../results/v2_task_teacher_recovery_20261002/first_attempt.json)、
[原生批量失败](../results/v2_task_teacher_recovery_20261002/native_failure.json)、
[独立复算](../results/v2_task_teacher_recovery_20261002/independent_verification.json)和
[验证汇总](../results/v2_task_teacher_recovery_20261002/validation.json)。
没有放宽0.7/0.2、14张容差或推进门，没有导出融合预测。
