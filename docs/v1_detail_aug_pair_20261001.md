# V1_DETAIL_AUG_PAIR_20261001：固定低细节训练配对

## 问题与有限投入依据

当前独立DEV的实际短边<224组共1,848张，LR512 EMA2–4错612张；该组占全量3,458张错误的17.70%。
[既有误差预算](v1_candidate_error_budget_20261001.md)不是干净真值，也未证明小图造成全部错误。
[一次受控细节降采样](v1_resolution_degradation_probe_20261001.md)在13,022张较大图上使LR512修正214、退化447、净−233；
相同原DEV父净−236。LR512原有512续训仅多抵消3张，支持检验专门的训练干预；可恢复收益与平台迁移仍未知。

本段回应持续的“本机继续进行优化探索”，只作一个配方、两个训练臂的有限证伪。
它与B机器的448弱/强增强4轮配对不同：从LR512平均权重出发，保持512几何与RA一致，唯一配对差异为训练裁剪的细节降采样。
A的768头/LoRA任务仍属另一条路线。开始前已fetch、检查全部本地/远端分支及main近期历史，没有同名/同配方在途实现。

## 结果前冻结

配置：[v1_detail_aug_pair_20261001.json](../configs/v1_detail_aug_pair_20261001.json)。
父为当前阶段DEV LR512原EMA2–4平均权重，SHA256
`c103dc3e6039c9fe156a56dfcd196500975b5bd21354be6506447bb1ee636e2e`；原监督、750类顺序与独立内容组划分均保留。
原归档plan/sidecar绑定到旧实现，运行采用原LR512工作目录的Python代码，不改写源哈希或绕过严格校验。

- 固定seed42，从119,074张正权重train_dev无放回抽32,768张，每臂1,024次真实更新，逻辑batch32/micro8/worker2。
- 两臂共享样本顺序、裁剪/flip/RA的随机过程、Mixup系数和排列；监督Mixup0.2、label smoothing0.1不变。
- 共同增强：512 RandomResizedCrop scale[0.8,1]、flip、RandAugment2/7。
  候选在此后对每批冻结16/32张裁剪执行PIL bicubic224→512，再共同归一化；控制不执行该操作。
- 同一rank32/alpha64原LoRA及512头；LoRA LR5e−5/WD0，head LR2.5e−4/WD0.01，OneCycle pct0.1/cos，grad clip1。
- 512/1,024步各评估完整14,880张，固定512中心+flip、FP32、无bias；主结果固定1,024步raw。
  不用中途val选步数、平均窗口或EMA，初始平均checkpoint仅作为共同父。
- 原短边<224、其余、原预冻结tail75组在GPU结果前写入CSV。全量原标签macro/micro、修正/退化/净值同时保留。
- 复核门同时要求候选相对控制全量净≥75、相对初始父全量净≥75、实际小图组相对控制净≥25，且全量修正/退化比≥1.25。
  未通过则关闭本固定配方；通过仅进入投入复核，不自动full、续训、强度搜索或平台上传。
- 任一冻结评估点全量micro比初始父低超过2pp则记录失败并停止该段，不能把中途模型称作完成候选。
- 正式训练前每臂8次真实更新测显存/吞吐，探测权重丢弃并从原父重新开始。
  总正式成本上限为2,048次真实更新及固定验证/两包推理。用户已取消时长上限，不按预计时长中止。

## 命令、私有产物与交付

工作目录 `/home/lux1/noise/worktrees/v1_detail_aug_pair_20261001`，方案分支 `codex/v1_detail_aug_pair_20261001`。
下面三条命令按顺序执行；`run`要求已通过成本检查，已有正式状态时拒绝隐式重启。

```bash
env PYTHONPATH=/home/lux1/noise/worktrees/lr512_dev_20261001/reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 scripts/train_v1_detail_aug_pair.py prepare --config configs/v1_detail_aug_pair_20261001.json --output outputs/codex/v1_detail_aug_pair_20261001/pair_r2
env PYTHONPATH=/home/lux1/noise/worktrees/lr512_dev_20261001/reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 scripts/train_v1_detail_aug_pair.py cost --config configs/v1_detail_aug_pair_20261001.json --output outputs/codex/v1_detail_aug_pair_20261001/pair_r2 --execute
env PYTHONPATH=/home/lux1/noise/worktrees/lr512_dev_20261001/reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 scripts/train_v1_detail_aug_pair.py run --config configs/v1_detail_aug_pair_20261001.json --output outputs/codex/v1_detail_aug_pair_20261001/pair_r2 --execute
```

`pair_r1`仅做CPU预检，审阅中补回中途评估后的train模式，实施哈希改变，因此改用新目录`pair_r2`冻结；未在r1使用GPU或训练。
随机子集覆盖749/750类，缺原索引183；保持抽样不改。该局限限制全类收益解释，不据此追加数据/轮次。
每臂完成后从单一主checkpoint对当前37,444张test固定解码，交付各自`submission/{pred_results.csv,submission.zip}`、
9项校验、ZIP内外字节核对及冷加载64张val重放；不能只有checkpoint结束训练段。
私有权重、逐图缓存、schedule与包均不入Git，汇总报告/独立验证与协议入Git。

本段平台分未知，现役仍为full v1 SWA用户报告70.98600576861446%。其可提交包为
`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/submission.zip`，
摘要`1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`；
[既有9项校验](../results/v1_full_swa_platform_20261001/submission_check.log)及
[包绑定核验](../results/v1_full_swa_platform_20261001/artifact_verification.json)仍是现役依据。

## 当前状态

实现及共享随机过程的3项测试已通过；CPU严格初始权重加载、内容隔离和冻结输入检查通过。
GPU成本与正式训练尚未完成，没有新实测候选收益。此处在交付时用实际报告更新。
