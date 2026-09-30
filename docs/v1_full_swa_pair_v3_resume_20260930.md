# full v1 SWA提交配对与v3恢复：2026-09-30

> **状态说明（2026-10-01）**：v3已按后续“继续”指令完成并交付；
> 本文暂停、恢复、在途进程和现役包描述保留本段发生时的口径，不作为当前运行状态。
> 最终v3结果见[交付记录](v3_final_delivery_20261001.md)，
> 当前最高分及现役full v1 SWA包见[当前执行入口](current_execution_plan.md)。


用户要求“相对V1 swa提交包改变了多少判断，v3继续”。
只比较已有已校验包，并恢复原v3暂停的内存现场；本段不新建训练候选。

## 提交预测配对

参考`V1_SWA_TRIAL_20260930`（用户报告平台70.4091%）与新
`V1_FULL_SWA_20260930`（平台分未知），按相同图片名对齐37,444条预测：
**6,952张改判（18.5664%），30,492张保持一致（81.4336%）**。
这里是预测不一致数，无测试真值，不能称净修正或据此推算平台收益。

参考ZIP SHA256：`7db9e15c1615e8767280e374a0d2b44b93fe7cae8632f970881ae7cf29f9aa5f`。
新ZIP SHA256：`1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`。
两包均为单一SWA checkpoint的推理，均37,444条、图片名唯一且覆盖完全相同，
CSV/ZIP SHA与各自推理manifest相符，ZIP中的CSV与外部文件字节一致。
两包既有9项校验来源与确切路径，以及改判清单路径和SHA见
[配对结果](../results/v1_full_swa_pair_v3_resume_20260930/submission_pair.json)。
逐图改判CSV只存本地独立输出目录，包含旧/新类别，非真值判断。

可复现计数命令（仓库根目录）：

```bash
python3 - <<'PY'
import csv
from pathlib import Path
old = Path('/home/lux1/noise/worktrees/v1_swa_trial_20260930/outputs/codex/v1_swa_trial_20260930/candidate/submission/pred_results.csv')
new = Path('/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/pred_results.csv')
def predictions(path):
    with path.open() as handle:
        rows = [(r[0], r[1].strip()) for r in csv.reader(handle)]
    assert len(rows) == len(dict(rows)) == 37444
    return dict(rows)
a, b = predictions(old), predictions(new)
assert a.keys() == b.keys()
changed = sum(a[name] != b[name] for name in a)
print(dict(samples=len(a), changed=changed, unchanged=len(a)-changed,
           changed_percent=100*changed/len(a)))
PY
```

## v3恢复

原服务`noise-v3-after-v1-exif-recovery-20260930.service`的runner、训练、
资源追踪及两个数据加载进程此前均为T状态，PID分别
361828、361861、362190、362191、362252。
按本次用户明确授权发送一次CONT，五进程只读核对为S/S/S/S/R，
2026-09-30 14:10:52 CST心跳更新为`running / train`。
恢复原内存模型、优化器、随机数和数据迭代现场，没有重启服务或重跑已完成更新。
随后同一`original`臂第1轮完整落盘更新轨迹从236推进至333次，
确认实际优化器持续更新；完整轨迹片段与核验时间写入恢复记录。

冻结计划SHA保持`615fc646d20cf5d7b02f64e3adcc4d659bd607036302b895d08901140fa9e224`，
沿用原v1 train_dev去噪目标与v2全视觉微调两臂协议：384、每臂2轮、
固定末轮EMA、原标签监督对比v1监督目标。没有切换到full train目标或SWA学生初始化。
原先的用户暂停约束已由“v3继续”撤销；不自动重试、派生搜索或上传平台。
完成后原runner继续推理、正式9项校验及单checkpoint CSV/ZIP交付。
截至恢复检查点，两臂全程及最终交付仍未完成，没有v3新完整指标或平台分。

恢复命令（已执行，不需要重复）：

```bash
kill -CONT 361828 361861 362190 362191 362252
ps -o pid,ppid,state,comm -p 361828,361861,362190,362191,362252,361817,370041
```

[恢复记录](../results/v1_full_swa_pair_v3_resume_20260930/v3_resume.json)、
[运行心跳快照](../results/v1_full_swa_pair_v3_resume_20260930/v3_resumed_status.json)。
实时状态仍在原v3独立目录的`handoff_exif_recovery/status.json`，
日志与更新轨迹仍在原计划目录，详见[原接力协议](v3_after_v1_20260930.md)。

现役可提交包为桌面`v1_full_swa_submission.zip`，已核对的9项校验见
[full v1交付记录](../results/v1_full_swa_20260930/submission_check.log)；
原`v1_swa_submission.zip`保留，平台最高报告仍为70.4091%。

本段在配对统计、恢复事实及文档同步检查点结束；v3持续运行。
方案分支`codex/v1_full_swa_pair_v3_resume_20260930`，独立worktree同名。
