# v1训练后SWA导出：V1_SWA_EXPORT_20260929

用户要求补一个训练后导出入口。新增`aegis_clip.cli.v1 export-swa`，仅使用CPU读取
已有的同一轨迹epoch checkpoint，输出一份选定权重及独立的校准/推理配置。
方案分支`codex/v1_swa_export_20260929`，工作目录
`/home/lux1/noise/worktrees/v1_swa_export_20260929/`。

默认按原配方的`swa_start`至`epochs`平均EMA权重；当前v1为第4–12轮。
计算复用现有`WeightAverage`，等价于现有在线SWA的参数平均算法。
`--weight-source raw`可显式改为平均原始训练参数；默认是EMA，未自动比较多个窗口或来源。
只平均可训练LoRA/分类头参数，冻结骨干继续使用官方初始化。

## 当前v1训练结束后的导出命令

从main目录`/home/lux1/noise`执行新增入口，配置指向原训练工作目录：

```bash
env CUDA_VISIBLE_DEVICES= PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 python3 -m aegis_clip.cli.v1 export-swa \
  --config /home/lux1/noise/worktrees/v1_train_20260929/configs/v1_train_20260929.yaml \
  --output-dir outputs/codex/v1_swa_export_20260929/candidate \
  --start-epoch 4 --end-epoch 12 --weight-source ema --device cpu --execute
```

默认源目录由原配置解析为
`/home/lux1/noise/worktrees/v1_train_20260929/outputs/codex/v1_train_20260929/training/`；
也可使用`--training-dir`指定完整epoch目录。输出目录必须独立、为空，并位于原运行目录之外。
源文件只读，每轮必须同时具有`epoch_XX.pt`和对应摘要文件；缺任意轮次即报错，
不会用当前可用的子集代替完整窗口。

导出校验包括：本阶段/recipe/实现绑定及文件摘要、真实epoch编号、同一trajectory/seed、
类别映射/架构、完整epoch恢复状态、有效训练配方、同一目标摘要、EMA更新及衰减、
参数名称/形状/dtype和有限值。验证成功后才保存产物，已有输出拒绝覆盖。

输出目录包含：

- `selected.pt`和摘要文件：单份SWA权重，`selected_policy=swa_ema`，记录4–12轮来源；无续训状态。
- `config.yaml`：保留原始训练绑定，只将`output.root`改为导出目录。
- `export_report.json`：来源、平均方法及当前状态`exported_not_evaluated`，local/platform分为null。

原配方的`swa_enabled=false`保留，表示实际训练时SWA关闭；新checkpoint的选定策略及导出记录
表示事后平均。直接换用`configs/v1_swa.yaml`会改变绑定，重放应使用导出生成的配置。

## 评估接口

导出后可使用原有校准和推理入口；SWA应在训练侧val_dev重新拟合自己的bias。
当前训练已有的bias属于另一份checkpoint，导出目录不携带它。

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.v1 calibrate \
  --config outputs/codex/v1_swa_export_20260929/candidate/config.yaml \
  --checkpoint outputs/codex/v1_swa_export_20260929/candidate/selected.pt --device cuda --execute
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.v1 infer \
  --config outputs/codex/v1_swa_export_20260929/candidate/config.yaml \
  --checkpoint outputs/codex/v1_swa_export_20260929/candidate/selected.pt --device cuda --execute
```

校准阶段产生六视角bias前后完整验证指标及逐样本预测；bias后是开发分，非独立测试分。
推理阶段使用单checkpoint并生成`submission/{pred_results.csv,submission.zip,submission_check.log}`。
本段交付导出实现与CPU验证，没有新的SWA比赛验证分、预测包或平台成绩。
本机CUDA评估仍先只读检查资源，当前v1训练继续沿原12轮流水线执行。

## 已验证内容

CPU测试覆盖：事后EMA平均与在线SWA逐参数一致、raw来源、导出权重重新加载、
独立校准/推理及合成3行CSV/ZIP的正式9项校验；另验证缺轮次、混轨迹/seed/目标、
阶段/摘要漂移、架构/类别不一致、缺EMA、参数不一致、非有限值及输出覆盖时拒绝导出。
测试与当前源绑定核验见[验证记录](../results/v1_swa_export_20260929/verification.json)。

```bash
env CUDA_VISIBLE_DEVICES= PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -m pytest reproducibility/aegis_f1/tests/test_v1_export.py reproducibility/aegis_f1/tests/test_v1.py reproducibility/aegis_f1/tests/test_model.py reproducibility/aegis_f1/tests/test_submission.py -q
```

工程段的现役可提交包继续引用
`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/{pred_results.csv,submission.zip}`，
37,444行、9/9既有校验见[校验记录](../results/p75_supervision_rebuild_20260929/submission_check.log)。
合成测试包属于测试夹具，不能当作本轮比赛候选。
