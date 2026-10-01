# 机器C交付LR512固定平均候选

本机4070 Laptop承担C，另外两台执行A/B，租用4090继续v2。本任务已完成，实际交付见[出包记录](../lr512_swa_package_20261001.md)。只复用已有LR512_DEV的固定EMA2–4平均checkpoint推理，不训练、不搜索SWA窗口、不自动启动LR512_FULL。用户已授权三机各自执行；今晚平台名额由用户稍后决定。

## 输入与执行

先读CLAUDE.md、当前入口和[三机分工](../three_machine_exploration_20261001.md)，fetch检查重叠后建立独立分支`codex/lr512_swa_package_20261001`和worktree。协议见`configs/team_exploration_20261001/machine_c.json`。

已完成训练的源目录是`/home/lux1/noise/worktrees/lr512_dev_20261001/outputs/codex/lr512_dev_20261001`，选择`run/training/ema_swa.pt`，不是`selected.pt`或自行取最佳epoch。原checkpoint SHA为`c103dc3e6039c9fe156a56dfcd196500975b5bd21354be6506447bb1ee636e2e`，selected_policy=ema_swa_2_4、average_epochs=[2,3,4]、complete=true。原plan SHA为`68973498a7594b3ed76707ec24bcea7a0ba877a995a62ec5fb292839b2ee3651`。

CLI存在，可从原已冻结代码目录运行v1_continuation.plan verify及runtime infer。由于infer会写run-root/inference_progress.json，先将平均checkpoint/原sidecar逐字节复制到本任务独立run/training/，复制原report并仅迁移其中artifact路径，保存原与派生report摘要；checkpoint binding继续使用原plan。不能写回原已完成run、改写plan或sidecar。跨机迁移若旧绝对路径不可用，须实现保留原字节的新位置映射并重放，不能直接改plan SHA绕过绑定。

原机命令形式如下，执行前确认本段是否已完成，完成后不重复跑：

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -m v1_continuation.runtime infer \
  --plan /home/lux1/noise/worktrees/lr512_dev_20261001/outputs/codex/lr512_dev_20261001/prepared_r2/plan.json \
  --run-root /home/lux1/noise-worktrees/lr512_swa_package_20261001/outputs/codex/lr512_swa_package_20261001/run \
  --checkpoint /home/lux1/noise-worktrees/lr512_swa_package_20261001/outputs/codex/lr512_swa_package_20261001/run/training/ema_swa.pt \
  --output /home/lux1/noise-worktrees/lr512_swa_package_20261001/outputs/codex/lr512_swa_package_20261001/submission \
  --execute
```

单checkpoint、固定512中心及flip、无bias，完全复用原LR512解码。CUDA空闲再执行；不设时间上限，不抢占、不自动重试失败、不拟合测试分布或派生测试bias。

## 验证与结束

保留9项提交校验日志，独立核对37,444行、ZIP只含pred_results.csv、内外CSV字节一致、checkpoint和源副本SHA一致，复制用户桌面时再比摘要。引用已独立核验的DEV指标macro75.7631%/micro76.7608%、修正324/退化214/净110；本次推理没有新增验证集准确率或平台分。

记录新包路径与摘要、确切命令、原计划和复制来源，在`docs/lr512_swa_package_20261001.md`/对应results落盘。立即方案commit/push，main自动autostash pull、合并复核、push，随后暂停。若已完成，只复用交付记录，不再占GPU。不要向队友自动发消息，不替用户上传或分配今晚名额。
