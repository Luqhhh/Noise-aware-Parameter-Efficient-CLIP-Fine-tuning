# A机 LP_LORA768_20261001 四轮配对交付

成员clairvoyanttt；方案分支`clairvoyanttt/lp_lora768_20261001`。两臂各4轮、每轮3,722个逻辑更新，共各14,888次。两份单checkpoint提交包与独立审计均已完成，冻结判据结论为 **closed_fixed_recipe**。没有本方案平台实测分数，不自动追加full或参数搜索；在完成方案和main推送的交付检查点暂停。

## 实测结果

主输出固定为同一轨迹EMA第2/3/4轮的权重算术平均，未选最佳轮。独立val共14,880张，448中心、FP32、无bias。

| 模型 | 正确数 | micro | macro |
|---|---:|---:|---:|
| 本机共同父 | 11387 | 76.525538% | 75.611025% |
| 仅训练头 | 11373 | 76.431452% | 75.511373% |
| LoRA＋头 | 11383 | 76.498656% | 75.553560% |

源父报告为micro76.518817%、macro75.602691%；源逐图预测与本机父有3张差异，详见后节。两臂使用同一本机实际父作为共同配对基线。

| 比较 | 群体 | 修正 | 退化 | 净正确 | 修正/退化 |
|---|---|---:|---:|---:|---:|
| 候选 vs 对照 | all | 236 | 226 | +10 | 1.0442477876106195 |
| 候选 vs 对照 | non_target | 229 | 213 | +16 | 1.0751173708920188 |
| 候选 vs 对照 | other | 214 | 191 | +23 | 1.1204188481675392 |
| 候选 vs 对照 | tail75 | 15 | 22 | -7 | 0.6818181818181818 |
| 候选 vs 对照 | target | 7 | 13 | -6 | 0.5384615384615384 |
| 候选 vs 对照 | target_tail_overlap | 0 | 0 | +0 | None |
| 候选 vs 本机父 | all | 282 | 286 | -4 | 0.986013986013986 |
| 候选 vs 本机父 | non_target | 267 | 273 | -6 | 0.978021978021978 |
| 候选 vs 本机父 | other | 250 | 248 | +2 | 1.0080645161290323 |
| 候选 vs 本机父 | tail75 | 17 | 25 | -8 | 0.68 |
| 候选 vs 本机父 | target | 15 | 13 | +2 | 1.1538461538461537 |
| 候选 vs 本机父 | target_tail_overlap | 0 | 0 | +0 | None |

候选相对对照全量仅净+10（micro+0.067204pp），相对本机父净−4；目标组相对对照净−6。全量净+75、目标净+25及比率≥1.25门均未满足，因此关闭本固定配方。目标组在训练前由源父冻结：前10个无序类对、20类、417张、173张父错误；源与本机目标预算均为173，3张重放差异均在目标组外。尾75类共1,003张，与目标组重叠0。10对全部成员的详细指标、完整标签和四模型逐图预测见交付目录的`final_report.json`、`paired_validation.csv`与独立复算报告。分组与原标签是诊断代理，不能把这些本地结果称作平台收益。

末轮仅作诊断：

| 臂 | 导出 | micro | macro |
|---|---|---:|---:|
| head_only | ema_swa_2_4 | 76.431452% | 75.511373% |
| head_only | last_ema | 76.377688% | 75.459067% |
| head_only | last_raw | 76.404570% | 75.481304% |
| lora_and_head | ema_swa_2_4 | 76.498656% | 75.553560% |
| lora_and_head | last_ema | 76.465054% | 75.526639% |
| lora_and_head | last_raw | 76.465054% | 75.526247% |

## 固定实现与确切运行

协议`configs/team_exploration_20261001/machine_a.json`。两臂同一个普通CRT768末轮cosine头＋原48处LoRA父，原119,074条正权重train_dev监督；control仅更新头，candidate更新全12块rank32/alpha64 LoRA和头，其他视觉权重冻结。逻辑32/micro8/worker2/seed42、448训练、AdamW LoRA5e-5/WD0、head2.5e-4/WD0.01、OneCycle pct_start0.1/cos、clip1、EMA0.999、LS0.1、Mixup0.2、RRC下界0.8、flip、RA2/7。监督和采样不平衡化，不额外拟合teacher或第二seed，无时间截止。

独立工作目录：
`/mnt/c/Users/clairvoyant/.codex/worktrees/clairvoyanttt-lp-lora768-20261001/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning`

位置文件`.planning/2026-10-01-clairvoyanttt-lp-lora768/locations.json`；原资产只读，共享Python依赖未更改。训练实际实现提交`05d6a1c`，实现内容绑定SHA为`12b0bec095ebefa1c7d4d4e875f3b9c409a547ca184b3f2fd3d1fef2f535a270`；完整输入、26份资产绑定和原sidecar见`current_inputs_verified.json`。最终交付提交与main合并SHA在本次交付回复报告。

首次正式命令（必须使用全新输出目录，不覆盖已完成运行）：

```bash
PYTHONPATH=reproducibility/aegis_f1 /home/clairvoyant/.venvs/noise-clip/bin/python -u -m lp_lora768 \
  --locations .planning/2026-10-01-clairvoyanttt-lp-lora768/locations.json \
  --output /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/outputs/clairvoyanttt/lp_lora768_20261001/20261001T222332_retry1 \
  --execute --allow-local-parent-baseline
```

本目录发生已记录异常后，按同一命令加`--resume`续跑；完整结果已经存在，禁止再次对本目录启动训练。跨机器仅改locations，保留父资产与原sidecar字节。严格源重放先用同一入口`--action replay --execute`和全新目录；如果逐图严格一致，正式运行无需本机基线例外标志。

## 源重放偏离与故障证据

26份交接资产、原图字节、官方权重、完整train/val有序路径与标签、mapping、原监督绑定一致。仍有3/14,880张父预测与源不同，故严格源逐图门未通过；batch32、头CPU/FP64、禁TF32、math SDPA与CPU FP32检查保留本机选择，根因未知。用户得知差异并被询问源环境版本后回复“这些不影响我本地实际运行啊”，据其本地执行优先指导继续既定任务。此处明确记录本机执行偏离，不声称严格源门通过或队长批准例外。原源预测列保留，所有源文件未改写；证据见结果根目录的parent_replay、replay_diagnostics、backend_diagnostics、local_baseline_ruling，以及交付目录的独立报告。

首次目录`20261001T211600`在head_only第2轮batch1268发生CUDA unknown error，仅完成1轮，失败产物完整保留，不计为四轮结果。当前`20261001T222332_retry1`从原父重新开始。两次完整零更新父logits逐元素相同（最大差0，NPZ SHA2667c2e9d80a22d58322b16d7f720fa8e428a95b5b617c803bdf24b74896a20b）；重复rows字段入口错误已修正，真实完整NPZ用于补写报告，修复标记保留。

当前轨迹对照臂四轮完成后，candidate第1轮batch556再次发生同类设备错误（香港时间2026-10-02 00:05:14）。新进程CUDA矩阵运算正常，根因仍未知。按完整step0状态恢复该臂，原556条尝试日志归档，完成的对照臂直接复用；不拿EMA替代raw，失败尝试不计入正式14,888步。恢复首批损失与原首批一致，最终两臂全部配对JSONL字节相同。异常与诊断在first_run_failure.json、retry1/cuda_failure_before_resume1.json及原输出目录保留。

## 实际验证与成本

5项真实父资产/恢复测试通过：首128图768父、冻结control全状态、两臂实际更新与独立RNG/Mixup质量、冷加载，以及完整raw/AdamW/OneCycle/GradScaler/EMA/平均状态恢复后下一步逐项一致。另验证恢复采样位置和实际增强图像一致。5步成本探针权重丢弃，正式从原父重新开始。

实际完成轨迹内部计时：control 4453.568秒，candidate 6073.243秒；各自包含四轮验证及三份导出的冷加载验证，故障尝试的额外耗时另保留，不把内部计时当全部墙钟。两份六视图提交推理分别1608.068/1627.785秒。成本探针、峰值显存和启动前预计投入保留在cost_probe与probe报告，没有据成本提前停轮。

完整状态独立核对：每臂所有AdamW step计数14,888（control2项/candidate98项），OneCycle last_epoch14,888，EMA count14,888，平均count3，cursor为epoch5/batch0；last_raw、last_ema、EMA2–4与最后完整恢复状态逐项相同。两份成功batch日志SHA：`85d4d588b9a3b58bb3681b965ac3bb942c58acc9f41e144c412b171fcd2ecd1e`。每轮全部119,074条正权重人口恰好覆盖一次，全部Mixup置换有效，两臂增强/索引/Mixup证明逐批一致。

最新main回归800 passed/12 skipped/2 failed；两项失败均为CLAUDE声明的test_scope_protocol.py ScopePreflightError（旧阶段冻结资产缺失），其余全绿，日志main_final_latest_tests.log。真实资产测试之前已显式启用，未把默认skip算作通过。历史CRLF问题仅恢复干净跟踪文本的原LF，没有修改冻结摘要。

独立交付审计实际通过：完整98项状态、control LoRA与原父相等、candidate LoRA确实更新、EMA2–4平均复算、六份cold-val logits→预测→指标、原/本机父/两臂完整逐图和全部群体修正退化、固定门、74,888行测试CSV与实际test_logits argmax、ZIP内外CSV与SHA均一致。命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/clairvoyant/.venvs/noise-clip/bin/python \
  scripts/verify_lp_lora768_delivery.py --output /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/outputs/clairvoyanttt/lp_lora768_20261001/20261001T222332_retry1 \
  --assets /home/clairvoyant/code/team_exploration_assets_20261001
```

## 可提交产物

两臂各单个EMA2–4 checkpoint，固定448/512/576中心＋flip、FP32 sum_logits、无bias；未用测试预测调参、未做平台上传。每包37,444行、逗号＋空格＋4位标签、ZIP仅一个pred_results.csv且内外字节相同，九项检查全部通过。

| 臂 | 行数 | 检查项 | CSV SHA256 | ZIP SHA256 |
|---|---:|---:|---|---|
| head_only | 37444 | 9 | `9133d13766ede812e7ea2bf440148c5641eef65212949bb5aeee47bfb1dc3a15` | `9ecea85fdc99cc2c78211672ddd3b5151ae3c6244ee3b84a630863a15ac44f96` |
| lora_and_head | 37444 | 9 | `eabc0885e1e1971c81eeaece9a8641b9f1dfa196f54818891c477357b0d73f29` | `fb609e40f0d4a8fbb287d527b2515309cd19cfd68f5471fecb29f47dcc97a150` |

可共享的CSV/ZIP及完整报告已复制到`results/lp_lora768_20261001/delivery/head_only/`和`delivery/lora_and_head/`，与原生成字节相同，目录.gitattributes固定保留原字节。每份checkpoint、test_logits的绝对路径和SHA见`delivery/artifact_manifest.json`；权重、恢复状态和大logits保留在唯一运行目录`/home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/outputs/clairvoyanttt/lp_lora768_20261001/20261001T222332_retry1`，不塞入Git。原CSV/ZIP路径为该目录各臂下的`submission/`。

通用校验命令（对各臂分别执行）：

```bash
/home/clairvoyant/.venvs/noise-clip/bin/python scripts/check_submission.py \
  --test_dir /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/复赛数据集/test \
  --class-mapping /home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/artifacts/stages/repechage/20260921/class_to_idx.json \
  --csv <臂目录>/pred_results.csv --zip <臂目录>/submission.zip
```

上方数据根与mapping路径已按原dataset_manifest实物核对。改动文件：lp_lora768六个模块、真实资产测试、独立交付验证脚本、next_continue_time.py、此执行记录/当前入口、结果报告和两份预测包。定时任务a按用户要求从02:39每5小时续跑，完整交付推送后暂停；不重复训练或自动扩展方案。
