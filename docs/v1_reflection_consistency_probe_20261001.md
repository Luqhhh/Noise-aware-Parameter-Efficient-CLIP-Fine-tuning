# V1_REFLECTION_CONSISTENCY_PROBE_20261001：镜像一致性机制检查

## 问题、重叠与边界

上一轮四候选共同原标签错误2,875张，现役LR512有3,458张错误；主要共同失败仍未解释。
v1训练有随机水平翻转，推理固定原图/flip平均，但未显式约束同图镜像的输出一致。
假设：较大的镜像分歧关联一块原标签错误，且正确原类仍在某一视图靠前；这可能支持一次显式一致性训练的有限证伪。
此处仅查当前原DEV父与LR512固定平均模型的一种反射，不扫描TTA或选择新推理配方。

开工fetch、检查全部本地/远端分支和近期main，从`origin/main@9f763da`建立
`codex/v1_reflection_consistency_probe_20261001`，独立worktree同名。
历史`88bbf97`是2026-07-15的500类teacher/student镜像MSE，只有可迁移代码；不读取旧资产或重写那一算法。
本轮当前750类机制尚未测量，不启动旧teacher配置。
B强增强配对报告flip一致性，但不含本固定原父/LR512的逐视图全量误差预算或显式一致性损失；
A为768 head/LoRA，保持各既定任务，不操作A/B/v2或NPU，不spawn子agent。
L05 crop/readout/TTA等冻结项保持，不扩人工核验或继续拆旧P0缓存。

## GPU结果前冻结

[配置](../configs/v1_reflection_consistency_probe_20261001.json)。当前20260921/750类，内容组隔离14,880张val_dev。
只读原v1 DEV EMA4–12父与LR512 DEV EMA2–4单checkpoint，沿原512 Resize/CenterCrop/Normalize，
每个输入只作一次水平反射；原图和镜像各FP32前向一次，原native预测仍为两者logits平均，无bias。
原源配置、plan、checkpoint、sidecar与18份原绑定输入严格核验；使用原LR512代码目录加载，不放宽旧binding。

- primary为LR512；原DEV父只作历史参照，不从两者中选更有利门槛结果。
- 每图报告两分布Jensen–Shannon divergence（自然对数，温度1）。
  高不一致固定为top1不同且JS≥0.1 nats，规则不使用原标签；同时报不同top1/其余、common wrong、tail75、小图及交集。
- 保留native/原图/镜像各全量和分组macro/micro、相对native逐图修正/退化；报告native错误中任一视图原类top1/top5预算（相同logit固定类索引升序）。
  任一视图包含原类是有原标签的诊断上限，不能用真值选视图交付，不能保证训练能恢复。
- 机制门同时要求primary高不一致≥500图、其中native错误≥200、其错误率比全量高≥10pp，
  且该组native错误中任一视图原类top5≥100张。通过仅支持一次有限一致性干预的投入复核，不自动训练。
- 先每模型64张seed42冷加载native预测全部一致，再全14,880 native逐图一致才出完整报告。
  数据/数值/复现失败即封存，不自动重启；只两模型、一次反射、一个阈值，无参数网格或墙钟截止。
- 原标签含噪，镜像对全部数字类是否保标签未知；分歧/原类排名不证明失效原因、真实标签或平台收益。

## 命令与交付

目录`/home/lux1/noise/worktrees/v1_reflection_consistency_probe_20261001`：

```bash
env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -m pytest reproducibility/aegis_f1/tests/test_v1_reflection_consistency.py reproducibility/aegis_f1/tests/test_v1_candidate_error_diagnostic.py -q
env PYTHONPATH=/home/lux1/noise/worktrees/lr512_dev_20261001/reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 scripts/probe_v1_reflection_consistency.py prepare --config configs/v1_reflection_consistency_probe_20261001.json --output outputs/codex/v1_reflection_consistency_probe_20261001/probe_r1
env PYTHONPATH=/home/lux1/noise/worktrees/lr512_dev_20261001/reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u scripts/probe_v1_reflection_consistency.py run --config configs/v1_reflection_consistency_probe_20261001.json --output outputs/codex/v1_reflection_consistency_probe_20261001/probe_r1 --execute
env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 scripts/verify_v1_reflection_consistency.py --config configs/v1_reflection_consistency_probe_20261001.json --run-root outputs/codex/v1_reflection_consistency_probe_20261001/probe_r1 --output outputs/codex/v1_reflection_consistency_probe_20261001/probe_r1/independent_verification.json
```

逐图view logits/组成员留在private输出；聚合报告与独立NumPy64/Counter复算记录入Git。
当前仅协议实现准备，无新GPU机制结果或训练候选。
纯诊断沿用现役full v1 SWA可提交包：
`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/{pred_results.csv,submission.zip}`，
ZIP SHA256`1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`；
[既有9项检查](../results/v1_full_swa_platform_20261001/submission_check.log)、
[资产核验](../results/v1_full_swa_platform_20261001/artifact_verification.json)。本段核对摘要和ZIP字节，不为凑新包训练。
已验证交付立即方案commit/push，main采用自动`pull --rebase --autostash`后合并、复验、push，再停在本段检查点。
