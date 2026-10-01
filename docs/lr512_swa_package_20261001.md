# LR512固定平均候选出包

状态completed_verified_delivery。用户要求三台本地机器分工探索，并确认各自拉代码执行；本机4070 Laptop承担已有LR512_DEV固定EMA2–4候选出包。没有新增训练、平均窗口搜索、测试bias或平台上传；新平台成绩未知，现役full v1 SWA70.98600576861446%保持。

## 产物和结果边界

checkpoint为已完成LR512_DEV的`run/training/ema_swa.pt`，SHA256 `c103dc3e6039c9fe156a56dfcd196500975b5bd21354be6506447bb1ee636e2e`。固定EMA epochs2–4平均已在原训练段生成，本次不重算或补训。原独立验证macro75.7631%、micro76.7608%，修正324/退化214/净110，引用[原核验](../results/lr512_dev_20261001/final_validation.json)；本段没有重新跑val，不把推理出包称为新验证结果。

单checkpoint固定512中心与flip、无bias，37,444张推理591.98秒。CSV/ZIP通过9项提交检查，另核对ZIP仅含pred_results.csv、内外字节相等、CSV37,444行且图片名唯一，复制桌面后SHA一致。详细路径和摘要见[本段核验](../results/lr512_swa_package_20261001/final_validation.json)与[校验日志](../results/lr512_swa_package_20261001/submission_check.log)。

- 新ZIP：`/home/lux1/noise-worktrees/lr512_swa_package_20261001/outputs/codex/lr512_swa_package_20261001/submission/submission.zip`。
- 同目录`pred_results.csv`是外部预测CSV。
- 桌面：`/mnt/c/Users/lqh22/Desktop/lr512_dev_ema_swa_2_4_submission.zip`。

LR512_FULL原evidence_requires_review不变，没有启动完整训练；DEV训练人口与full现役不同，不能据两包平台分差隔离SWA或512的单因素收益。两个今晚名额仍由用户稍后决定。

## 精确命令和来源

分支`codex/lr512_swa_package_20261001`，独立目录`/home/lux1/noise-worktrees/lr512_swa_package_20261001`。原plan先通过verify，SHA `68973498a7594b3ed76707ec24bcea7a0ba877a995a62ec5fb292839b2ee3651`。从原已冻结代码目录`/home/lux1/noise/worktrees/lr512_dev_20261001`执行：

```bash
PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 -m v1_continuation.plan verify \
  --plan outputs/codex/lr512_dev_20261001/prepared_r2/plan.json

timeout 5400 env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 -u -m v1_continuation.runtime infer \
  --plan /home/lux1/noise/worktrees/lr512_dev_20261001/outputs/codex/lr512_dev_20261001/prepared_r2/plan.json \
  --run-root /home/lux1/noise-worktrees/lr512_swa_package_20261001/outputs/codex/lr512_swa_package_20261001/run \
  --checkpoint /home/lux1/noise-worktrees/lr512_swa_package_20261001/outputs/codex/lr512_swa_package_20261001/run/training/ema_swa.pt \
  --output /home/lux1/noise-worktrees/lr512_swa_package_20261001/outputs/codex/lr512_swa_package_20261001/submission \
  --execute
```

上面的timeout是本次已完成推理的历史命令。用户随后明确取消后续时间上限，不将它作为新任务的时长配置。

原runtime会写run-root/inference_progress.json，因此先将原平均checkpoint和sidecar逐字节复制到独立run/training/；原report只在副本中迁移artifact路径，保留SHA值与原plan binding。原report与派生report摘要、checkpoint复制一致性见[复制来源](../results/lr512_swa_package_20261001/copy_provenance.json)。源已完成目录和原artifact均未改写。数值/资产验证使用原冻结实现，没有新增模型代码。

方案提交推送后，main集成采用自动模式pull --rebase --autostash，再合并、核对文档链接与产物、push，在本出包检查点暂停。
