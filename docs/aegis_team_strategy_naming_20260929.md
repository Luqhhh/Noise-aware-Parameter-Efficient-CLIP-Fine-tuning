# AEGIS_TEAM_STRATEGY_NAMING_20260929

用户确认两条策略均由本队自主设计，当前工程统一采用Aegis项目名称。
用户随后要求不保留这两条策略的历史分数、旧校验哈希和许可证；相应记录及来源快照已移除。
本段范围为名称、归属表述和当前CPU验证，未启动GPU或真实项目训练。

| 旧称 | 正式名称 | 工程标识 | 当前入口 |
|---|---|---|---|
| B448 | Aegis-Aligned448 | AEGIS_ALIGNED448_20260929 | aegis_clip.cli.aligned448_strategy |
| V13 | Aegis-ResolutionLadder | AEGIS_RESOLUTION_LADDER_20260929 | aegis_resolution_ladder.plan / runtime |

实现改动包括模型/流水线/CLI、配置、脚本、测试及文档的同步改名。
ResolutionLadder直接读取 `configs/aegis_resolution_ladder_20260929/stages/` 中的项目配置；
全量段名为 `full_576`，checkpoint血缘使用项目 `strategy_revision`。
两条策略的训练参数、EMA选择、固定推理和执行授权保持当前配方。
旧名字生成的plan和绑定不用于当前入口，需按当前源码重新准备。

## 当前实测

- CPU测试 **83 passed**：两条策略、既有模型及提交检查。
- Aligned448官方CPU前向：448输入、有限 `[1,750]` 输出、48处适配、5,102,593个可训练参数，CUDA未初始化。
- ResolutionLadder官方CPU前向：224与native视觉实现最大绝对差0.0；384/448/576均为有限 `[1,750]`，88,233,966个训练参数均为FP32。
- 两份Aligned448配置通过当前阶段preflight；ResolutionLadder重新prepare并verify通过，133,815/14,880内容组隔离划分、四段38,266次逻辑更新。

当前实测输出见[verification.json](../results/aegis_team_strategy_naming_20260929/verification.json)，
测试原始输出见[tests.log](../results/aegis_team_strategy_naming_20260929/tests.log)。
当前官方权重、代码及数据绑定按执行协议校验；不引用已删除的历史分数或旧校验记录。
这些是工程验证，没有新local macro/micro、平台成绩或候选包。

在方案工作目录执行的命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=reproducibility/aegis_f1 python3 -m pytest tests/test_aegis_resolution_ladder.py reproducibility/aegis_f1/tests/test_aligned448_strategy.py reproducibility/aegis_f1/tests/test_model.py reproducibility/aegis_f1/tests/test_submission.py -q
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.aligned448_strategy preflight --config configs/rematch750_aegis_aligned448.yaml --report outputs/codex/aegis_team_strategy_naming_20260929/aligned448_preflight.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.aligned448_strategy preflight --config configs/rematch750_aegis_aligned448_swa_experimental.yaml --report outputs/codex/aegis_team_strategy_naming_20260929/aligned448_swa_preflight.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=reproducibility/aegis_f1 python3 scripts/check_aegis_aligned448_cpu.py --config configs/rematch750_aegis_aligned448.yaml --report outputs/codex/aegis_team_strategy_naming_20260929/aligned448_cpu_forward.json
CUDA_VISIBLE_DEVICES='' PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_resolution_ladder.plan prepare --output outputs/codex/aegis_team_strategy_naming_20260929/resolution_ladder_prepared
CUDA_VISIBLE_DEVICES='' PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_resolution_ladder.plan verify --plan outputs/codex/aegis_team_strategy_naming_20260929/resolution_ladder_prepared/plan.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python3 scripts/check_aegis_resolution_ladder_cpu.py --output outputs/codex/aegis_team_strategy_naming_20260929/resolution_ladder_cpu_forward.json
```

方案分支为 `codex/aegis_team_strategy_naming_20260929`，工作目录为
`/home/lux1/noise/worktrees/aegis_team_strategy_naming_20260929`。
main集成采用**自动模式** `git pull --rebase --autostash origin main`，方案合并同样使用Git自动stash；
不另行stash pop，合并后重新校验、推送并在检查点暂停。

现役可提交包为
`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/{pred_results.csv,submission.zip}`，
37,444行、9/9校验来源见[既有校验](../results/p75_supervision_rebuild_20260929/submission_check.log)。
本段未重生成该包，未启动训练或上传。
