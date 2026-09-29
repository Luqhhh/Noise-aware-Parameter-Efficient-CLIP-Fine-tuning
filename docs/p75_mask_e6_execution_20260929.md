# P75_MASK_E6_EXTENSION：已授权的新执行段

用户在工程交付后明确回复 **“授权启动”**。该授权用于 A 的新 21,600 秒上限和 B 的新 3,600 秒上限；不重置旧 E4 预算，不启动 C1/C2，不上传平台。原 `proposal_only` 是工程交付时的状态，已由此次新指令解除启动授权缺项，其他配方/资源/成本边界保持。

方案分支 `codex/p75_mask_e6_execution_20260929`，独立工作目录 `/home/lux1/noise/worktrees/p75_mask_e6_execution_20260929`，输出 `outputs/codex/p75_mask_e6_execution_20260929/`。从 main `464cebf` 开始，检查现有分支和本机实际进程后执行。实测 RTX4070 Laptop 8188 MiB；启动前未见其他 CUDA 训练任务。不连接远端/NPU，不操作 SAM。

## 启动前工程修正

1. 新增一次10更新的 local 吞吐测量：从 control E4完整状态到第534次更新，仍384px、micro4×accum256、原16轮日程、local启用。只测成本，权重丢弃，绝不作为真实E5或正式续接输入。正式两臂仍各自恢复原E4。测量任务墙钟计入同一个新6小时上限，不另给预算。
2. A调度器校验测量记录SHA，并把已用测量秒数带入启动状态及20%余量估算，拒绝重置计费。
3. 验证缓存命令显式传 `--tta-temperature 1.4`，避免缓存元数据默认值与冻结解码不一致。最终仍使用各臂自己的验证侧prior。
4. 用户服务独立设置PATH，含`/usr/lib/wsl/lib`。首次服务在调用nvidia-smi前因PATH缺项退出，未执行模型更新；失败记录保留。随后仅修正服务环境重新测量。

## B 的实际缺项

授权已收到，但当前阶段资产目录及本机相关输出/工作目录未发现可绑定的 OpenAI val flip 图像特征；已有的是center特征、文本特征或L05预测概率，不能冒充所需缓存。因此 B **未启动**，不是负结果；不自动重编码、不以center分替代最终固定解码，也不消耗B训练预算。

## 复现与后台执行

```bash
python3 -m pytest tests/test_p75_pipeline.py tests/test_p75_text_page_runtime.py tests/test_p75_semantic_pair.py -q
python3 scripts/p75_pipeline.py prepare --out outputs/codex/p75_mask_e6_execution_20260929
python3 scripts/p75_pipeline_cost_probe.py --out outputs/codex/p75_mask_e6_execution_20260929 --authorization outputs/codex/p75_mask_e6_execution_20260929/authorization_start.json
python3 scripts/p75_pipeline_service.py --out outputs/codex/p75_mask_e6_execution_20260929 --authorization outputs/codex/p75_mask_e6_execution_20260929/authorization_A.json --expected-head <执行段冻结提交>
```

上述prepare/probe/正式run均拒绝重复启动既有目录/预算。测量中写出的partial checkpoint即使文件名带E5，也只有534更新，不能通过正式E5所需655更新检查。正式流程：control E5、masked E5、control E6、masked E6；仅E6全量固定解码，记录所有冻结分组与逐类变化。

`p75_pipeline_service.py`在有界任务退出后，重算配对表并归档状态/日志/报告；不派生训练或平台上传。验证通过后尝试提交推送方案，并在main用自动`pull --rebase --autostash`模式集成、重新校验与推送。检测到并发改动或冲突则保留结果并记`needs_manual_review`，不强推、不覆盖协作者内容。

运行信息：`A/status.json`、`A/logs/`、`A/cost_probe/status.json`；终态交付状态：`delivery_status.json`。`running`不等于已有E6配对结果或新可提交候选。预算超时标记incomplete，不判机制无效；本次授权不包含自动完整训练或“为名额出包”。旧现役L05包及既有校验继续引用[工程任务文档](p75_supervision_pipeline_20260929.md)，不重建/重验。

## 冻结启动门禁

实测10次local更新，测量任务耗时252.583秒；更新区间实测22.319–22.948秒，采用最慢区间×1.2=27.53719秒/更新。两臂验证预留600秒（旧同配方两臂实测合计275.829秒），启动/保存/报告开销预留900秒。连同测量总估算16182.069秒（4.495小时），低于17,280秒门槛；新预算保持21,600秒。22项针对性测试通过。

## 终态

两臂从各自E4恢复，E5和E6各131次更新，至E6累计786次；两臂样本顺序、实际更新和学习率一致。
E5保存完整状态及统计，E6使用同一固定Flip/T1.4/prior0.60流程，各自拟合验证侧prior。
全量原标签评价如下，分组重叠，不能相加：

| 验证组 | 张数 | control macro / micro | masked macro / micro | 修正 / 退化 / 净 |
|---|---:|---:|---:|---:|
| 全量 | 14,880 | 74.2950% / 75.0672% | 74.2124% / 74.9933% | 24 / 35 / −11 |
| 命中内容代理 | 132 | 17.9365% / 21.2121% | 9.3651% / 12.8788% | 0 / 11 / −11 |
| 未命中内容代理 | 14,748 | 74.6967% / 75.5492% | 74.6862% / 75.5492% | 24 / 24 / 0 |
| 生物主导代理 | 13,600 | 75.3344% / 76.6838% | 75.3279% / 76.6838% | 19 / 19 / 0 |

全量macro差−0.0827pp、micro差−0.0739pp；净−11集中于命中代理组，未命中与生物主导代理均净0。
全部750类已重算：23类净正、34类净负、693类净0。代理不是真值，复用验证集也不构成独立留出证据。
本轮未观察到预先指定目标代理上的净改善，判断`close_recipe_no_full_training`；不晋级完整训练。
B因缺少已有绑定flip特征未启动，无B负结果；没有新平台成绩或新完整候选。

逐样本预测、全部分组和逐类结果见
[predictions.csv](../results/p75_mask_e6_execution_20260929/A/predictions.csv)、
[paired.csv](../results/p75_mask_e6_execution_20260929/A/paired.csv)、
[report.json](../results/p75_mask_e6_execution_20260929/A/report.json)。

任务状态与成本：

```json
{
  "status": "local_result",
  "used_seconds": 11657.457608794968,
  "preflight_seconds": 252.58251099599875,
  "cost_gate": {
    "estimated_seconds": 16182.068930457608,
    "budget_seconds": 21600,
    "reserve_fraction": 0.2,
    "fits": true
  },
  "authorization_sha256": "0ea623b919c6b29197f8c31077fa642ce1ba56f7f104d9b7f7f77508bc6489b3",
  "local_gpu": "NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB"
}
```

## 发布复核与交付

终态原始提交`654ba2b3381b863b108a55894bec709b2adaa583`已推送到
`codex/p75_mask_e6_execution_20260929`。发布使用独立目录
`/tmp/noise-p75-e6-delivery-20260929`、分支`codex/p75_mask_e6_delivery_20260929`，
从最新`origin/main`开始，先以自动`git pull --rebase --autostash origin main`同步，再合并终态提交。
README保持现行最简说明，结果只更新本执行记录、当前执行入口和`results/`。
归档CSV保留原始CRLF及SHA，目录内`.gitattributes`仅允许标准CSV记录终止符，其余空白检查保留。

本次核对49项归档SHA、534项冻结输入/源文件、14,880行身份与顺序、全部分组和750类指标、
两臂权重/缓存/续接状态绑定、E5/E6批次顺序/更新/学习率、预算终态。22项针对性测试通过。
详细复核结果见[发布验证记录](../results/p75_mask_e6_execution_20260929/delivery_verification.json)。
确切复核命令（仅CPU/只读，不启动训练或重编码）：

```bash
python3 -m pytest tests/test_p75_pipeline.py tests/test_p75_text_page_runtime.py tests/test_p75_semantic_pair.py -q
python3 results/p75_mask_e6_execution_20260929/verify_delivery.py \
  --repo /tmp/noise-p75-e6-delivery-20260929 \
  --runtime /home/lux1/noise/worktrees/p75_mask_e6_execution_20260929/outputs/codex/p75_mask_e6_execution_20260929
git diff --check --cached
```

现役可提交包继续引用
`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/pred_results.csv`
和同目录`submission.zip`，37,444行、9/9既有校验见
[提交校验日志](../results/p75_supervision_rebuild_20260929/submission_check.log)。
本段未生成E6完整候选或新的测试提交包，不把验证预测表当作比赛提交；发布后在检查点暂停。
