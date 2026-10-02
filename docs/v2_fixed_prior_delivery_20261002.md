# V2_FIXED_PRIOR_DELIVERY_20261002

2026-10-02用户明确要求“生成对应提交包”，指V2的固定类别均衡先验bias包。
本段执行已合入main的[V2_FIXED_PRIOR_20261002](v2_fixed_prior_20261002.md)，不新增算法、训练或参数搜索。

## 执行归属与当前状态

执行分支：`codex/v2_fixed_prior_delivery_20261002`；独立目录：
`/home/lux1/noise/worktrees/v2_fixed_prior_delivery_20261002`；起点：`714d672`。
准备分支`codex/v2_fixed_prior_20261002`已有验证实现与成本检查，本段复用其代码和固定配置，
不写入准备者的worktree或输出目录。启动前已fetch并检查全部分支、main近期历史与本机进程：
仅准备者64张架构探针存在，未发现正式V2 bias推理进程或候选目录。
后续完整执行由本分支承接，同一checkpoint和固定配方只跑一次，协作者请复用本段结果。

当前仍等待原V2完整本地下载回执；尚未启动正式推理，尚无bias候选或平台分。
原始无bias ZIP已保存在Windows下载目录并通过9项检查；原始权重下载及核验后关机流程继续。
本段只使用本机CUDA，服务器关机不等待本机bias推理，也不修改原始包。

## 固定输入和命令

完整本地交付目录：`C:\Users\lqh22\Downloads\v2_continuation_20261001`。
使用原full第5轮raw单checkpoint，保留四视图概率平均，取log后固定200次均衡bias、强度1、目标1/750。
沿用原配置`configs/v2_fixed_prior_20261002.json`和全部输入/数值迁移门，不重复64张成本探针。
正式启动前再次检查准备分支与本分支是否已有候选或CUDA进程，避免重叠执行。

在本段worktree执行：

```bash
export PYTHONPATH=reproducibility/aegis_f1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python3 scripts/run_v2_fixed_prior.py \
  --config configs/v2_fixed_prior_20261002.json \
  --output outputs/codex/v2_fixed_prior_delivery_20261002/candidate --action check
python3 -u scripts/run_v2_fixed_prior.py \
  --config configs/v2_fixed_prior_20261002.json \
  --output outputs/codex/v2_fixed_prior_delivery_20261002/candidate --action run
```

最终应交付同缓存raw与bias两包、9项检查日志及NumPy/float64独立复算报告；
核验后将bias ZIP/CSV及记录复制到Windows下载目录的独立`submission_bias`子目录。
这些是预期产物，当前不能引用为已经完成。没有平台反馈前不声明bias提高准确率或达到80%。
