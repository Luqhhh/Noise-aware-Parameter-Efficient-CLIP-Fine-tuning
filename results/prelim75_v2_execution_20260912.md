# PRELIM75_V2_20260912 执行记录

用户授权：按顺序执行。源基线83c03dd，main，远端fece41b。H0/H1 → V0/V1顺序执行；C只在真实平台反馈满足方案门槛时启动。不push、不操作平台账号。

## 初始实现与检查

实现完整P十视图global/local base/residual分片缓存；P epoch3实际软目标与权重；H0/H1共享头重拟合；V0/V1完整双Adapter联合短微调；真实候选attention诊断；固定单checkpoint、无校准推理及提交校验；顺序队列与每段冻结源码。

H checkpoint将表示训练scope登记为 `frozen`，严格加载的plain visual参数、O3/PTA及共享linear结构不变；这是实际冻结状态，不将backbone LR guard关闭。V继续登记full_finetune许可掩码。

首批8个手算/几何/有效batch检查通过，随后增加冻结scope严格重载和工程止损检查，专项共11 passed。首次CPU Aegis回归448 passed/1 skipped（12.21秒），根回归424 passed/1 skipped（51.89秒）。CPU回归设置CUDA_VISIBLE_DEVICES为空以隔离运行缓存；真实官方训练图的CUDA初始检查另行完成：局部原生logits与base/residual重构最大差0，argmax相同，global重构通过。V1首步还将单独核验local loss对visual proj和两个Adapter的梯度，不能仅由总loss的视觉梯度代替。

## 实际命令

以下从仓库根目录运行，缓存命令已启动，其余由缓存完成后的队列顺序执行。候选目录已有即停止，不自动覆盖/恢复。

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -u -m aegis_clip.cli.cache_prelim75_features --config configs/prelim75_v2.yaml
python3 scripts/run_prelim75_queue.py --config configs/prelim75_v2.yaml --segment head --execute
python3 scripts/run_prelim75_queue.py --config configs/prelim75_v2.yaml --segment visual --execute
```

队列默认不带 `--execute` 时只审计。执行时保存配置、main/ref/历史状态和冻结源码；每个候选完成固定训练→真实10视图诊断→未触发工程止损则真实测试推理→24967行CSV/ZIP检查和包内外一致性→候选报告。GPU时间、显存峰值和各阶段退出码来自真实运行，线上分数留空。

## 当前截面

运行目录 `outputs/prelim75_v2_20260912/`。完整训练缓存已完成；H队列正在执行，尚无新提交包或平台成绩。初赛75%目标未达到，不把实现或本地重叠指标写成达成。后续在本记录追加各段实际末轮结果、哈希与提交路径。

## 缓存与直接对照（实际执行至2026-09-13）

缓存103218行完成，耗时1356.5038秒；8条原生局部logits与base/residual重构的最大绝对差0，argmax分歧0。CUDA峰值分配652132352字节。缓存manifest SHA-256：`738c58c2d3b2d2dd39e8701cd0d1ecad6597c293aa9a980ebbe98d751956486c`；记录位于 `outputs/prelim75_v2_20260912/cache/manifest.json`。缓存仅接受官方训练清单，完整w/q及身份记录已绑定哈希。

H队列已注册冻结源码并启动，代码提交84a82ec。P真实十视图重叠诊断完成，10316行，raw micro=79.4978678%，clean-core micro=91.9247031%（7331行），耗时136.0690秒。这里是工程止损直接对照，与P已登记的平台66.7681%分开。

最终CPU Aegis回归451 passed、1 skipped（CUDA单元测试主动隔离，10.71秒），日志 `outputs/prelim75_v2_20260912/aegis_final_tests.log`；此前根回归424 passed、1 skipped。真实CUDA前向另有实际数值检查。当前仅实现与缓存/对照完成，H实验完成与提交包将在运行后逐项追加。

CPU回归中跳过的CUDA有效样本数检查单独在GPU补跑：`PYTHONPATH=reproducibility/aegis_f1 python3 -m pytest reproducibility/aegis_f1/tests/test_longtail.py::test_effective_number_preserves_cuda_device_and_matches_cpu -q`，1 passed（1.98秒）。日志 `outputs/prelim75_v2_20260912/cuda_unit_tests.log`。

## H段已完成交付（2026-09-13）

实验PRELIM75_V2_H0/H1，精确配置 `configs/prelim75_v2.yaml`，命令及退出码见 `results/prelim75_v2_head_20260913.json`。两者固定10 epoch，训练代码84a82ec，H0耗时33.5902秒，H1耗时31.5541秒；两者仅共享head训练，其他表示参数保持P不变。

| 候选 | 末轮loss | 重叠raw micro | 相对P raw pp | 重叠clean-core micro | 相对P clean pp | 实际平台 |
|---|---:|---:|---:|---:|---:|---|
| H0 | 0.37411068 | 79.439706% | -0.058162 | 92.333925% | +0.409222 | 待回填 |
| H1 | 0.37419679 | 79.652965% | +0.155097 | 92.415768% | +0.491065 | 待回填 |

两者未触发2pp止损，均使用真实候选attention重新定位，固定无校准协议输出24967行并通过全部提交检查，ZIP包内外CSV字节一致。交付位于 `outputs/prelim75_v2_20260912/{H0,H1}/submission/`，各目录的 `artifact_sha256.json` 绑定checkpoint、配置、源码、诊断、CSV/ZIP等。桌面副本 `submission_prelim75_H0_20260912.zip` 和 `submission_prelim75_H1_20260912.zip` 与仓库包哈希相同。平台分数/上传时间未知，registry新增待测行，不晋级、不预填。

H段实现和实际训练/提交交付已完成，可由队友pull复用；只本地提交，用户负责push和平台上传。H路线是否有效尚未裁决。V段继续从原P独立启动；C未触发。

## V0已完成交付，V1已开跑（2026-09-13）

实验PRELIM75_V2_V0，配置 `configs/prelim75_v2.yaml`，实际命令 `python3 -u -m aegis_clip.cli.train_global_local_joint --config /home/lux1/noise/configs/prelim75_v2.yaml --candidate V0`（队列使用冻结PYTHONPATH源码，登记代码491c7f9）。3 epoch、9678 optimizer steps，耗时1843.3736秒，CUDA峰值分配2733108736字节，未OOM降档；末轮loss=0.13968389。首步视觉/head梯度非零，局部Adapter梯度零；最终冻结参数字节不变检查通过。

真实十视图重叠诊断raw micro=82.2120965%，clean-core micro=94.8165298%，比P分别+2.7142286pp/+2.8918266pp。未触发2pp止损；这些指标不是独立泛化或平台成绩。V0真实测试推理已完成，24967行与全部提交检查通过、包内外CSV字节相同。ZIP `outputs/prelim75_v2_20260912/V0/submission/submission.zip`，SHA-256 `bf866f45fa959e552f28006940f9dbd4be0ea68ed6ce385e0d45032ab4a45b64`；桌面同哈希副本 `submission_prelim75_V0_20260912.zip`。checkpoint SHA-256 `2deaa155069636e5673fc75575efbd45deaed872af101a31182d78edb8bfda5f`。完整命令/结果/交付哈希见 `results/prelim75_v2_v0_20260913.json`。平台未知，不晋级。

V0结果可pull复用；只本地提交，用户负责push。V1已从原P独立启动。首步local-only梯度范数visual proj=6.99440、O3 up=1.37985、PTA up=1.59095，证明局部损失能真实反传至三者；总loss梯度审核通过，冻结参数无泄漏，batch32无需降档。V1尚未完成，不把首步检查称作结果交付。V1接续前再次自动stash模式pull，origin/main仍fece41b，refs无新重叠；自动stash93b2cfd已恢复，没有第二次pop。证据在运行目录 `V1_additional_preflight.json`。

## H0真实平台回填（2026-09-13）

用户回复H0包问题：65.8629%。绑定已交付H0 ZIP SHA-256 `08fb4339f63f0b3ecd78dad7eab5f88dfe8e9d0356b7189076fb939f23702abe`。比匹配无校准P66.7681%低0.9052pp，H0不晋级；原本地重叠clean-core+0.4092pp不能当成平台收益。精确正确数、上传时间未提供。registry及current_platform_summary已回填；不新增扫描，H1尚待真实成绩，V1继续固定训练。C只可能由满足门槛的其他真实H/V胜者触发。

## 四个独立候选交付完成（2026-09-13）

H0/H1固定各10epoch，V0/V1固定各3epoch，均从原P独立启动。四个候选均未触发2pp工程止损，全部完成真实候选attention重算、24967行测试推理和提交校验；包内外CSV字节相同。V1冻结参数终态检查及local-only反传检查均通过。

| 候选 | 末轮loss | 重叠raw micro | 重叠clean-core micro | 实际平台 |
|---|---:|---:|---:|---:|
| H0 | 0.37411068 | 79.439706% | 92.333925% | 65.8629% |
| H1 | 0.37419679 | 79.652965% | 92.415768% | 待真实回填 |
| V0 | 0.13968389 | 82.212096% | 94.816530% | 待真实回填 |
| V1 | 0.18329918 | 81.785578% | 94.093573% | 待真实回填 |

V1耗时2979.7419秒，CUDA峰值分配4012382720字节，batch32未降档。V1 checkpoint SHA-256：`a52548511f5971836dbd6ffc07f62685050e2dc9b4318818573c53e9a7c5a696`；ZIP SHA-256：`7f28c1300e4a50bcb8e99aabff5c9d8df50db813c3ff165dbfbe916c8e37b77e`。四组完整配置/命令/产物绑定见 `results/prelim75_v2_visual_20260913.json`、各候选 `artifact_sha256.json`。桌面副本 `submission_prelim75_{H0,H1,V0,V1}_20260912.zip` 已逐个核对同哈希。

已运行预算：在线6epoch、缓存head20epoch、4个独立平台包；第五个C尚未启动。实际反馈及C判定为 `pending_real_platform_feedback`，仅真实H/V胜者各比P高至少0.30pp（67.0681%）才执行C；本地指标不触发组合。当前75%尚未被实际平台反馈验证。四组工程交付完成，可复用本地checkpoint/配置和受跟踪报告；只本地提交，用户负责push/上传。训练源snapshot及其逐文件哈希不变，历史包不覆盖。

最终复核：历史最佳及匹配无校准P的CSV/ZIP四个哈希均与执行前一致；C入口只读审计退出0、返回pending_real_platform_feedback，未生成C目录。四个桌面副本、候选checkpoint及ZIP实际哈希再次核对通过。

## H1真实平台回填，关闭H及C（2026-09-13）

用户回复H1包问题：66.1914%，绑定H1 ZIP SHA-256 `3640997770dd9bd56f105ab8a17e2bc96da4dbe584e3067b14e7ce955a16cf22`。比匹配H0高0.3285pp，但比无校准P66.7681%低0.5767pp。H0=65.8629%、H1=66.1914%均未超过P，按既定规则关闭H，不追加loss/tau/seed/epoch拟合；本轮C不满足门槛，取消，不需要等待V平台成绩才能作此否定判断。V0/V1工程交付已完成，平台成绩待回填以决定视觉路线。精确正确数及上传时间未提供。registry/current_platform_summary已登记，候选交付报告和其哈希保留为交付时截面；当前平台字段见新增platform结果/路线决策。

C入口修正：某条路线的全部有效候选已评分且最好仍低于门槛时立即关闭，不要求另一条路线提供无法改变结论的分数。均达到门槛时仍必须完整评分后选实际胜者。该变更不启动训练、不改变任何checkpoint、预测或提交包。

H及C关闭验收：组合/几何专项8 passed（包括路线负结果提前关闭、门槛边界、拒绝未评分胜者、已止损成员不参与胜者选择）；`PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.train_prelim75_combination --config configs/prelim75_v2.yaml --execute`真实退出0，返回closed_below_combination_gate。C及C_cache均不存在。四包ZIP哈希未变，registry四个唯一行中仅H0/H1写入真实成绩；V0/V1保持空白。日志 `outputs/prelim75_v2_20260912/gate_closed_tests.log`。

## V0真实平台回填：无校准工作候选改善，历史最佳未刷新（2026-09-13）

用户回复V0包问题：67.4090%，绑定ZIP SHA-256 `bf866f45fa959e552f28006940f9dbd4be0ea68ed6ce385e0d45032ab4a45b64`。比同模型起点/同推理协议的无校准P66.7681%高0.6409pp，暂保留V0；比历史FULLFT_DUAL四尺度/Flip+prior0.90的70.352866%低2.943866pp，不能标为刷新历史最好。两项差值按用户提供的精度计算，精确正确数及实际上传时间未提供。训练起点仍是历史方案同一份完整P；区别是当前新推理未应用旧prior。V1工程包已交付，平台待回填；H/C已关闭，不因V0超过组合的单路线门槛而重新开启H或C，不追加epoch或扫描。registry/current_platform_summary已登记，交付报告/产物哈希保留为交付时截面，平台结果另存；模型与预测文件未变。

## V1真实平台回填及本轮结束（2026-09-13）

用户回复V1包问题：**67.5932%**，绑定ZIP SHA-256 `7f28c1300e4a50bcb8e99aabff5c9d8df50db813c3ff165dbfbe916c8e37b77e`、checkpoint SHA-256 `a52548511f5971836dbd6ffc07f62685050e2dc9b4318818573c53e9a7c5a696`。比同协议无校准P66.7681%高0.8251pp，比V0 67.4090%高0.1842pp，保留V1为本轮最终无校准胜者。比历史最高70.352866%低2.759666pp，距75分仍差7.4068pp；历史最好及原包保留，75目标未达到。精确正确数及实际上传时间未提供，不从四舍五入成绩推算。

| 固定候选 | 真实平台 | 对无校准P差值 | 最终决策 |
|---|---:|---:|---|
| H0 | 65.8629% | -0.9052pp | 不晋级，H关闭 |
| H1 | 66.1914% | -0.5767pp | 不晋级，H关闭 |
| V0 | 67.4090% | +0.6409pp | 保留备选 |
| V1 | 67.5932% | +0.8251pp | 本轮无校准胜者 |

四个预定候选均完成训练、真实推理、提交校验和实际平台回填；在线6epoch、缓存head20epoch。C因H路线未达到既定门槛而取消，不重新开启H或C；本轮结束，不自动追加训练、prior拟合或参数扫描。旧交付报告/阶段记录保持交付时截面和原哈希；当前成绩、最终选择与包绑定见 `results/prelim75_v2_v1_platform_20260913.json`、`results/prelim75_v2_final_20260913.json`。本次仅更新结果及文档，四包字节不变，只本地提交，不push，不操作平台账号。

回填结束复核：四个真实成绩均绑定对应ZIP且registry/current_platform_summary各只有一个对应行；四包CSV/ZIP与交付报告、桌面副本哈希不变，V1全部manifest文件哈希核对通过。历史最好及匹配P的CSV/ZIP四个哈希均保留。最终C入口只读运行退出0，返回closed_below_combination_gate，C/C_cache不存在；git diff --check通过。元数据更新未改训练代码，无需重复整套训练回归。
