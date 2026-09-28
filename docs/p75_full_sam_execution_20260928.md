# P75_FULL_SAM：2026-09-28 启动记录

> **2026-09-29 用户最新指令**：保留已启动实验，按本文件原固定配方与门槛收尾，
> 完成后停在检查点；不自动重启，不派生 rho、ASAM、GSAM、续训、多 seed 或联合训练。
> 即使本地过门，也不能据此自动继续搜索或声称平台提升。后续主线先满足
> [主要误差预算门禁](p75_error_budget_policy_20260929.md)，当前 L05 邻域搜索冻结。

目标仍为平台 75 分；现役 L05_T14_P060 为 66.94797564362783%，尚无 SAM 验证或平台成绩。
本段属于在途训练，不能作为已完成实验或提分证据。

## 去重与独立执行

开工已 fetch、检查全部本地/远端分支与近期 main。Patch Readout 已完成但固定解码仅
+0.0074pp macro / +0.0134pp micro，关闭；本分支合并其结果和日志修复。
K05 由另一工作目录继续，OOF hard filter 尚无可确认的最终结果，不重复启动。
完整局部目标 SAM 尚未执行，沿用已准备实现，不派生参数扫描。

分支 `codex/p75_full_sam_20260928`；目录
`/home/lux1/noise/worktrees/p75_full_sam_20260928`；运行输出为该目录下
`outputs/codex/p75_mechanisms_20260927/`。本机实测 GPU 为 RTX 4070 Laptop 8GB，
启动前无其他训练进程。没有使用或占用远端 NPU。

## 固定配方与证据

当前阶段 20260921、133815 train_dev / 14880 val_dev、750 类；OpenAI ViT-B/32。
384px、16轮、microbatch 4 × 256累积 = 有效1024、SAM rho=0.05；全图、局部和
feature anchor 都参与第二遍。正式局部监督从第5轮开始，gate=0.7。
保持原 L05 其余配方，按中心视图 macro 选模。**AMP 改为 FP32**，因此不是 SAM
机制的严格单变量因果对照。固定配置见
[P75_FULL_SAM.yaml](../configs/p75_full_sam_20260928/P75_FULL_SAM.yaml)。

CPU预检通过：阶段摘要、父权重、内容组隔离、配置与私有运行时代码摘要。
SAM数值/异常恢复及运行器日志测试共5项通过。两次真实有效batch的GPU工程验证通过：
每次1024样本，两遍梯度均有限，实际扰动半径约0.05，峰值张量显存2727412224字节，
总计133.04秒。smoke仅提前启用局部路径并将gate置0，以覆盖最重路径；不保存候选，
正式训练重新从RM-LP父权重开始。完整数值见
[启动证据](../results/p75_full_sam_launch_20260928.json)。

## 可重放命令

```bash
python3 scripts/p75_prepare.py --phase prepare
python3 scripts/run_p75_full_sam.py --phase preflight
python3 -m pytest tests/test_p75_sam.py tests/test_p75_runner_logs.py -q
python3 scripts/p75_sam_gpu_smoke.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 python3 scripts/p75_sam_campaign.py
```

最后一条在本次以独立session后台启动，PID与启动命令存于输出目录 `launcher.json`。
只允许运行一次，已有状态/日志时拒绝覆盖；不自动重启失败任务。
流水线依次运行 `train → cache → evaluate`，仅在固定 Flip/T1.4/prior0.60 的
macro比现役至少+0.30pp且micro不退化时执行deliver。未过门则关闭固定配方。
有提交包后仍需独立审计、9/9校验、提交/合并实测记录；平台上传需用户选择并回填分数。

## 监控和后续

按用户新指令，训练启动后**每3600秒一次**观察进程、GPU、日志尾部，写入
`hourly_health.jsonl`；等待子进程使用阻塞wait而非轮询。`campaign_status.json`
记录当前phase、PID、下次检查时间及终止状态。错误只记录并停止，不自动重复训练。
CPU/GPU工程验证不能证明提分；此轮验证结束前不宣称实验完成。

现役可提交包继续保留：
`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/`
下的 `pred_results.csv` / `submission.zip`，本次37,444行及9/9提交校验通过。
本次尚无SAM新提交包；75分目标保持未完成。
