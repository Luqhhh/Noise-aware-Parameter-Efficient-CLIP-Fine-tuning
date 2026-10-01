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

## 实测结果与交付检查点

实现及共享随机过程的3项测试已通过；CPU严格初始权重加载、内容隔离和冻结输入检查通过。
协议在GPU结果前以`5669788`提交并推送。
[成本检查](../results/v1_detail_aug_pair_20261001/cost.json)每臂8次真实更新均成功，稳定更新控制0.4784秒/步、
候选0.3972秒/步；CUDA峰值分配分别0.8701/1.2227GiB，各有一次已恢复AMP重算，无跳过更新。
正式训练已丢弃探测状态、重新从原父开始，各完成1,024次更新；各2个batch发生已恢复AMP重算，无跳过更新。
两臂训练、4次全量验证、冷加载及两包正式测试推理总计3,104.19秒（51.74分钟），不含之前成本检查。
控制/候选测试推理分别586.95/569.97秒。原标签结果见[完整报告](../results/v1_detail_aug_pair_20261001/report.json)。

| 权重/点 | 全量micro | 全量macro | 相对初始父全量净修正 | 相对初始父小图净修正 |
|---|---:|---:|---:|---:|
| 初始LR512 EMA2–4 | 76.7608% | 75.7631% | 0 | 0 |
| 控制512步raw | 76.1223% | 75.1148% | −95 | −28 |
| 候选512步raw | 76.0820% | 75.0835% | −101 | −26 |
| 控制1,024步raw（固定主结果） | 76.4987% | 75.4819% | −39 | −19 |
| 候选1,024步raw（固定主结果） | 76.2500% | 75.2274% | −76 | −21 |

| 候选末步相对控制末步 | 行数 | 修正 | 退化 | 净修正 |
|---|---:|---:|---:|---:|
| 全量 | 14,880 | 128 | 165 | **−37** |
| 实际短边<224 | 1,848 | 35 | 37 | **−2** |
| 其余 | 13,032 | 93 | 128 | −35 |
| 冻结tail75 | 1,003 | 7 | 16 | −9 |

实际小图micro控制65.8550%、候选65.7468%；macro控制67.5319%、候选67.7148%，略增0.1829pp。
本固定配方没有交付全量或实际小图净修正，冻结复核门未通过，决策`close_fixed_detail_augmentation_recipe`。
不推进此配方full/续训/强度扫描；这不排除其他细节鲁棒性方法，也不能把控制短续训的−39归因于增强。
本段监督、单次seed、有限人口和raw导出均限制外推；没有平台分或干净真值收益声明。

[独立核验](../results/v1_detail_aug_pair_20261001/independent_verification.json)通过：
4×14,880=59,520条val logits→预测、20份群体指标、20份配对群体、2×37,444=74,888条test logits→CSV字节重放，
全部源输入/新checkpoint sidecar/完整步数/冻结训练日程核对；每臂原运行另有64张冷加载预测一致。
实现/共享随机过程及拒绝缓存错序、不一致和非有限logits的15项相关测试通过。
校验器9项约束打印8条成功行（两字段解析无单独成功行）；独立重放另核对字段和精确CSV字节，避免仅按日志条数判断。

两份完整可提交包均为各自单一末步checkpoint，固定512中心+flip、无bias，37,444行、9项约束及ZIP内外字节通过：

- 控制：`/home/lux1/noise/worktrees/v1_detail_aug_pair_20261001/outputs/codex/v1_detail_aug_pair_20261001/pair_r2/control/submission/{pred_results.csv,submission.zip}`。
  ZIP SHA256 `628240b92a7a46efa354196f0f3f7d463fabf4ea21497f2cceca45e6ecd48abc`；
  [校验日志](../results/v1_detail_aug_pair_20261001/control_submission_check.log)、[包manifest](../results/v1_detail_aug_pair_20261001/control_manifest.json)。
- 候选：`/home/lux1/noise/worktrees/v1_detail_aug_pair_20261001/outputs/codex/v1_detail_aug_pair_20261001/pair_r2/detail_aug/submission/{pred_results.csv,submission.zip}`。
  ZIP SHA256 `15a5dc9e1df719211aec3a3111f8ec2588ccee7b27360a3731b61c413025d8dd`；
  [校验日志](../results/v1_detail_aug_pair_20261001/detail_aug_submission_check.log)、[包manifest](../results/v1_detail_aug_pair_20261001/detail_aug_manifest.json)。

独立重放原命令如下；重复核验须为`--result`指定新的路径，不覆盖既有记录。

```bash
env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 scripts/verify_v1_detail_aug_pair.py --config configs/v1_detail_aug_pair_20261001.json --output outputs/codex/v1_detail_aug_pair_20261001/pair_r2 --result outputs/codex/v1_detail_aug_pair_20261001/pair_r2/independent_verification.json
env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -m pytest reproducibility/aegis_f1/tests/test_v1_detail_aug_pair.py reproducibility/aegis_f1/tests/test_v1_detail_aug_verification.py reproducibility/aegis_f1/tests/test_v1_candidate_error_diagnostic.py reproducibility/aegis_f1/tests/test_v1_resolution_degradation_probe.py -q
```

收尾fetch发现B的448配对实现及正式启动已进入`origin/main`（`735b2eb`）；它保持4轮、448几何强增强，未重复本段512低细节干预。
方案通过merge同步新main，保留B启动证据；main集成采用自动`git pull --rebase --autostash origin main`模式，合并后重新校验再推送。
本检查点关闭本固定配方，保留两包以供复用；现役full v1 SWA和A/B/v2既定工作保持，不上传平台。
