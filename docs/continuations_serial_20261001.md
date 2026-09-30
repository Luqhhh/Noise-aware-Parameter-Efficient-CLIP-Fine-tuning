# V1_CONTINUATIONS_SERIAL_20261001：本机串行执行

用户明确要求“在本机串行进行剩下几个不依赖v2的实验”。执行队列冻结为：
WFT448_DEV → LR512_DEV → 有证据支持的WFT448_FULL → 有证据支持的LR512_FULL。
每项使用`codex/<路线>_<dev/full>_20261001`新分支与独立worktree/输出。
v2仍未启动；V2_FULL_LAST3_SWA依赖其完整full_576，不进入本队列。
v3已完成两臂正式复核，整体/尾部均不支持自动扩大监督阶梯，该路线本段不执行。

四轮、LR、目标、可靠度、增强、EMA .999、固定EMA2–4窗口均沿用
[原固定方案](v1_continuations_20261001.md)。GPU适配固定micro8、逻辑batch32，
每逻辑batch仍只混合一次、更新一次；最后短batch保留。
FP32推理使用batch32，与训练累积拆分无关；同一baseline/候选协议，增加进度侧车。
没有改变头、监督、采样、输入视图、损失归一化、梯度裁剪或学习率日程。

CUDA预检保留原micro2检查及最终micro8的3次实际更新，不保存探针权重。
WFT最终预检峰值约2.13GiB，预热后逻辑更新0.435/0.387秒，FP32 batch32推理约0.154/0.149秒。
固定四轮训练估计1.80小时，含十次DEV评估/基线及一次完整测试推理估计3.29小时。
这些是短测外推，不是保证耗时或识别成绩；训练输出从原v1 SWA重新开始。
成本和血缘见[CUDA预检](../results/continuations_serial_start_20261001/wft448_cuda_probe.json)。

LR512最终CUDA预检也通过：实际512×512、257 tokens，3次更新均有限，
峰值训练0.817GiB、推理1.006GiB，预计四轮训练2.21小时、含评估出包约3.13小时。
两DEV合计短测外推约6.42小时，数据读取和GPU频率会改变实际耗时。
见[LR512 CUDA预检](../results/lr512_dev_start_20261001/cuda_probe.json)与
[148项相关测试记录](../results/lr512_dev_start_20261001/validation.json)。

## FULL投入边界

在新结果出现前，持久串行controller冻结一个保守的自动投入条件：
末轮EMA、EMA2–4均值两份固定导出均需全量净修正>0且macro改善；
尾部净修正和macro不得下降，有验证行的少支持组净修正/macro不得下降；
末轮raw同时提供全量净修正/macro改善，末轮EMA保留第2轮的净收益。
满足才引用DEV报告/哈希及明确支持分析，准备对应FULL独立人口。
FULL仍为4轮，固定末轮EMA提交，另导出原固定EMA2–4；不读取重叠val选模或校准。

这是保守的自动算力投入门，不是候选有效性裁决。小幅原标签下降、raw/EMA方向不同或组间混合
都保留DEV模型、报告和校验包，标为`evidence_requires_review`，继续另一DEV，不自动否定路线，
也不因此追加训练或扫描参数。原标签及分组代理不是干净真值，平台收益仍未知。
用户对串行工作的授权覆盖有上述证据支持的固定FULL；没有自动平台上传。

## 持久执行与段交付

controller脚本为`python3 scripts/run_serial_continuations.py --queue <queue.json> --execute`。
逐项`train → infer → 独立重算/ZIP核验 → 提交并推送方案 → main自动autostash同步/合并/校验/推送`。
每个已完成训练段交付37,444行CSV/ZIP、9项校验、来源SHA与配对报告，然后执行下一独立段。
不并发训练，已有CUDA计算任务时等待本机资源，不抢占、不用远端/NPU。
训练失败不会自动重启；完整状态、PID、命令与日志均落盘。

运行队列来源见[冻结queue](../results/continuations_serial_start_20261001/queue.json)。
实时controller位于
`/home/lux1/noise/worktrees/wft448_dev_20261001/outputs/codex/continuations_serial_20261001/status.json`。
WFT DEV输出位于`worktrees/wft448_dev_20261001/outputs/codex/wft448_dev_20261001/`，
LR DEV输出位于`worktrees/lr512_dev_20261001/outputs/codex/lr512_dev_20261001/`。
此启动记录没有新识别结果或新提交包；源模型仍为已校验v1 SWA，平台起点为用户回填full v1 SWA70.9860%。

命令（每项从各自worktree执行，FULL prepare额外引用已通过的DEV证据）：

```bash
PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 python3 -m v1_continuation.runtime train \
  --plan outputs/codex/wft448_dev_20261001/prepared_r1/plan.json \
  --output outputs/codex/wft448_dev_20261001/run --execute
PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 python3 -m v1_continuation.runtime infer \
  --plan outputs/codex/wft448_dev_20261001/prepared_r1/plan.json \
  --run-root outputs/codex/wft448_dev_20261001/run \
  --checkpoint outputs/codex/wft448_dev_20261001/run/training/selected.pt \
  --output outputs/codex/wft448_dev_20261001/submission --execute
```

WFT/LR每项各自输出固定末轮EMA和EMA2–4报告；所生成包始终使用一份明确checkpoint。
现役包与既有校验继续引用
`worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/`，
37,444行、9项通过的来源为[既有验证](../results/v1_full_swa_20260930/final_validation.json)。
