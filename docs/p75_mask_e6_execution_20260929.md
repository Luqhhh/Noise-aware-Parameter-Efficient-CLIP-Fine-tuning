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
