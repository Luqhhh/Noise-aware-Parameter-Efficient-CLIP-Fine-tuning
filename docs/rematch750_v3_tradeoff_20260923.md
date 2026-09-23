# REMATCH750_V3：精度优先的耗时权衡

状态：**2026-09-23 已完成扫描、审计和提交包校验；用户同日冻结当前 winner，V3 后续搜索关闭**。用户要求加入 batch1024 精度补偿、允许使用当时空闲的 NPU0、每十分钟监控。精度目标为同 NPU B32 基准 macro 下降≤0.1pp。实施分支 `codex/rematch750_v3_scan`，独立目录 `/home/lux1/noise/worktrees/rematch750_v3_scan`。

实施结果：物理 NPU0 经空闲检查后执行三次完整训练。B64、B128、B1024 均通过 16/8 轮动态审计、checkpoint/optimizer 有限性及逐类重载检查。条件 B64_LR2 因 B128 达线而跳过。下表为预注册点位，实测值见后文。

参考为V2同后端NPU batch32八轮：macro0.7190471887588501、micro0.7298387289047241、纯训练2670.459578s、完整CLI2870.871812s。精度可接受下界raw_macro=0.7180471887588501（71.80471887588501%）。一次达界只表示本次独立验证满足约束，不代表统计等价。

## 冻结决定（2026-09-23）

用户指令：**冻结当前 winner**。本段据此关闭 V3 搜索，不再追加 LR、epoch、batch、worker/prefetch 或 loss 扫描；后续只有用户明确新指令才允许启动新实验。

- 冻结候选：`RM_V3_B1024_E16_LR4`
- 冻结配置：`configs/rematch750_v3_b1024_e16_lr4.yaml`，config SHA-256 `db802b635a908204bbb1c9e946a0e6e2648a4f24dc9a42a1a61ffd1c4ba3c0a5`
- best checkpoint SHA-256：`a8de990eb0fd347e52dd5387d6cd986a30268ed316111dce6ce8d9e4c42c2f27`；last checkpoint SHA-256：`342058a6027a75ca164d6027a5a83090ca0c49083056703805eb6897be5978cf`
- 提交包：`submission.zip` SHA-256 `5c58f2cda01a7d9eeb7d2e31d5e46504371b2c8780dae52e9e43d6537ffe63bd`；`pred_results.csv` SHA-256 `175c5be75eb9b18a291c34d7a20c035576269b5732695eea321d9bf83ad430bb`；37,444 行，9/9 独立提交检查通过；平台分数仍为 `null`。
- 远端冻结标记：`/workspace/noise-worktrees/rematch750_v3_scan/outputs/codex/rematch750_v3_tradeoff/RM_V3_B1024_E16_LR4/seed42/FROZEN.json`；best/last checkpoint 与提交文件已设为只读（444）。
- 冻结依据：本次受控验证 macro **72.3472%**、micro **73.4140%**，比同后端 NPU B32 基准高 **+0.4425pp / +0.4301pp**，完整 CLI 快 **1.397×**，达到 ≤0.1pp 精度约束；但单 seed、同验证集选模，固定尾部 75 类 macro 低 0.1140pp，平台迁移仍未验证。
- 机器可读记录：`results/rematch750_v3/freeze.json`。

## 有界比较

完整运行 `configs/rematch750_v3_b64.yaml`（8/8轮）、`configs/rematch750_v3_b128_e16.yaml`（16/16轮）和新增 `configs/rematch750_v3_b1024_e16_lr4.yaml`（16/16轮）。B1024相对已完成的八轮点延长到16轮，同时head/visual LR各乘4（4e-4 / 1.2e-5）；这只是有界精度补偿假设，不保证收敛或精度等价。旧B1024八轮结果只作参照，不重跑。只有B64与B128均未达精度约束时，最多补一次 `configs/rematch750_v3_b64_lr2.yaml`：batch64、head LR2e-4、visual LR6e-6，其余不变。最多四次新完整训练；不扫描workers/prefetch，不恢复LoRA或关闭路线。

共同设置：910B2单卡、只选择当时空闲的设备；B64/B128使用workers16、prefetch2，B1024沿用V2已测高吞吐配置workers40、prefetch4；均使用pinned、fused AdamW、foreach norm、OMP4。CPU绑定按所选卡实际拓扑确定；NPU0沿用已核对的144–191编号。同NPU LP SHA `d5cb8f5265754fd900d3efde23e24fefbcf616c747f2cab13e4dc2201fdd689b`，133815/14880同内容组划分；B64为8 epochs/cosine horizon8，B128与B1024补偿点为16/16；原head1e-4/visual3e-6（补偿点例外）、CE2轮后GCE q=.5（B64六轮，16轮点十四轮）、anchor2.0、shuffle、无LR warmup。每20步及轮末记录成功更新，每两轮验证（B64：2/4/6/8；16轮点：2/4/6/8/10/12/14/16），raw_macro主选模、同值raw_micro。

B64每轮2091步、八轮16728步；B128每轮1046步、十六轮16736步；B1024每轮131步、十六轮2096步。各自完整跑满预先确定的轮数，不能只用CE段或重建optimizer续训冒充。B1024仍只有B32成功更新数的1/16；它以较高学习率和更多轮数测试能否弥补，不把更新预算称为等价。各自独立目录，共享训练/验证数据、特征和初始化只读，Python依赖不修改。

## 选择与交付

执行、optimizer有限性、成功更新、checkpoint及逐类重载审计先通过；将已验证B32也纳入候选集合，在macro满足阈值的配置中选**完整CLI耗时最短**者，另列纯训练秒数、macro/micro差、固定尾部75类及其余675类。若新点均不满足精度约束，或完整耗时未优于基线，则保留已验证B32。不用吞吐短窗口估算代替完整耗时。即使16轮run选中更早checkpoint，仍计实际跑完整个run的成本；不能事后按所选epoch倒算节时。

用户的效率筛选与+0.20pp平台精度晋级门分开：满足≤0.1pp精度损失的最快新配置可生成工程交付CSV/ZIP并独立校验，标记为效率候选，不自动声称平台更优、不自动上传。若无新点满足约束，保留并复核原FT交付包。报告包含所有失败点，不只报告胜者。

## 实测比较与交付

| 点位 | 选中轮次 | macro | micro | 完整 CLI 秒数 | 相对 B32 macro | 精度下界 | 结论 |
|---|---:|---:|---:|---:|---:|---|---|
| B32 基准 | 8 | 71.9047% | 72.9839% | 2870.872 | — | 达线 | 已测基准 |
| B64×8 | 8 | 70.8299% | 71.8683% | 1946.714 | −1.0748pp | 未达线 | 淘汰 |
| B128×16 | 16 | 71.9320% | 72.9906% | 3663.711 | +0.0272pp | 达线 | 比 B32 慢 |
| **B1024×16/LR×4** | **16** | **72.3472%** | **73.4140%** | **2055.272** | **+0.4425pp** | **达线** | **效率胜者** |

B1024 纯训练 1816.882 秒、完整 CLI 比 B32 快 815.599 秒（1.3968 倍），2,096 次 optimizer 更新均成功；best checkpoint SHA-256 为 `a8de990eb0fd347e52dd5387d6cd986a30268ed316111dce6ce8d9e4c42c2f27`，epoch16 重载 macro/micro 和逐类结果完全一致。固定尾部 75 类 macro 55.9589%，其余 675 类 74.1682%；B32 对应 56.0729% 与 73.6638%，因此总 macro 改善不代表尾部提升。原始审计、各验证点曲线与计时在 `results/rematch750_v3/RM_V3_B1024_E16_LR4.json`；完整筛选表在 `results/rematch750_v3/comparison.json`。这是一轮单种子独立验证，未声称跨种子稳定性或平台得分。

胜者在远端 `/workspace/noise-worktrees/rematch750_v3_scan/outputs/codex/rematch750_v3_tradeoff/RM_V3_B1024_E16_LR4/seed42/submission/` 生成 `pred_results.csv` 和 `submission.zip`，不上传平台。独立运行 `scripts/check_submission.py --test_dir /workspace/noise/test --class-mapping /workspace/noise/artifacts/stages/repechage/20260921_npu/class_to_idx.json --csv <submission>/pred_results.csv --zip <submission>/submission.zip`，9/9 检查通过：37,444 行、750 类范围、文件名全覆盖且无重复，ZIP 只含 `pred_results.csv`，包内外 CSV 字节相同。CSV SHA-256 `175c5be75eb9b18a291c34d7a20c035576269b5732695eea321d9bf83ad430bb`；ZIP SHA-256 `5c58f2cda01a7d9eeb7d2e31d5e46504371b2c8780dae52e9e43d6537ffe63bd`。推理清单的 checkpoint 哈希与审计一致；机器可读交付记录在 `results/rematch750_v3/delivery.json`。

精确重放命令（先用 `scripts/bind_rematch750_v3.py` 生成远端 `.npu0.local.yaml`，验证空闲 NPU0 与相同数据/LP 哈希）：`ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH=reproducibility/aegis_f1 taskset -c 144-191 /workspace/noise-npu-venv/bin/python -u -m aegis_clip.cli.rematch train --config configs/rematch750_v3_b1024_e16_lr4.npu0.local.yaml`；推理将 `train` 换为 `infer`。单模型 best checkpoint 推理，未融合其他 checkpoint。

## 隔离与重放

本机方案worktree `/home/lux1/noise-worktrees/rematch750_v3_tradeoff` 与实施worktree `/home/lux1/noise/worktrees/rematch750_v3_scan` 分开；远端实施源码目录 `/workspace/noise-worktrees/rematch750_v3_scan`。绑定脚本 `scripts/bind_rematch750_v3.py` 将本阶段原始数据/特征绝对路径指向 `/workspace/noise` 并校验同哈希NPU LP；输出在实施目录的 `outputs/codex/rematch750_v3_tradeoff/<candidate>/seed42/`，不覆盖原运行。

模板仍以 `train.device: npu:UNASSIGNED` 表示未绑定设备，不能直接运行。选定空闲物理卡后，单卡可见环境使用 `ASCEND_RT_VISIBLE_DEVICES=<物理卡号>`、配置内逻辑 `npu:0`；绑定脚本拒绝错误的可见设备环境变量。执行前还须复核CPU拓扑和设备空闲状态。

`scripts/audit_rematch750_v2.py` 已改为按配置epochs和interval生成验证点与成功更新数；保留checkpoint有限性、optimizer步数、重载精确一致等审计标准。

正式B64在物理NPU0完成，PID `989345`，环境 `ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH=reproducibility/aegis_f1`，CPU `taskset -c 144-191`，命令为 `/workspace/noise-npu-venv/bin/python -u -m aegis_clip.cli.rematch train --config configs/rematch750_v3_b64.npu0.local.yaml`。启动前空闲显存65,068,498,944字节，同哈希NPU LP与数据校验通过。8轮16,728次更新全部成功，按macro选epoch8；macro **70.8299%**、micro **71.8683%**，比B32 macro低1.0748pp，未达精度下界71.8047%。纯训练1752.747秒，完整CLI1946.714秒；checkpoint、optimizer有限性和重载审计通过，记录在 `results/rematch750_v3/RM_V3_B64.json`，无候选提交包。顺序运行器 `scripts/run_rematch750_v3_scan.py --wait-for-b64-pid 989345` 接续 B128、B1024，每次启动前检查≥50GiB 空闲显存，并于远端 `results/rematch750_v3/scan_status.json` 记录状态；已以 `all_runs_audited` 结束。

B128×16也已完成且审计通过：16,736次更新、选epoch16，macro **71.9320%**、micro **72.9906%**，比B32 macro高0.0272pp，达到71.8047%精度下界；纯训练3317.859秒、完整CLI3663.711秒，比B32多792.839秒，不是效率胜者。记录在 `results/rematch750_v3/RM_V3_B128_E16.json`，无该点位提交包。由于 B128 达线，条件 B64_LR2 按预设规则跳过。

训练不串接平台上传。方案分支验证后推送；main 集成目录使用自动模式 `git pull --rebase --autostash origin main` 同步、合并、重新校验、推送；不执行手动 stash pop。

修订原则：不同epoch可以胜出；按真实完整耗时评价，不要求相同样本遍历预算。B64×8与B128×16的预期更新数接近（16728 vs16736），但后者图像遍历翻倍、CE/GCE更新分配不同，不能称优化过程等价。

## 方案摘要

| 点位 | batch | epochs / cosine | head / visual LR | CE / GCE轮数 | 预期总更新 | 定位 |
|---|---:|---:|---|---:|---:|---|
| 既有B32 | 32 | 8 / 8 | 1e-4 / 3e-6 | 2 / 6 | 33456 | 已测基准，不重跑 |
| B64 | 64 | 8 / 8 | 1e-4 / 3e-6 | 2 / 6 | 16728 | 中间batch参照 |
| B128_E16 | 128 | 16 / 16 | 1e-4 / 3e-6 | 2 / 14 | 16736 | 用更多epoch补偿较大batch |
| B1024_E16_LR4 | 1024 | 16 / 16 | 4e-4 / 1.2e-5 | 2 / 14 | 2096 | 大batch延长训练并提高LR的补偿点 |
| 条件B64_LR2 | 64 | 8 / 8 | 2e-4 / 6e-6 | 2 / 6 | 16728 | 仅前两点均精度不达标时的一个补偿点 |

后续可研究的“大batch前期、小batch收尾”暂不纳入本轮：它需要新增调度与恢复逻辑，也会扩大比较范围。本轮1024只测一个预先固定的延长训练与LR补偿点。

本轮已完成完整训练、审计、推理和提交校验。平台成绩仍待用户实际提交，不能由本地验证外推。
