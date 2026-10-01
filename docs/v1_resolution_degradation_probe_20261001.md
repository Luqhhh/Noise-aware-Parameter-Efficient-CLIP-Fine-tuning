# V1_RESOLUTION_DEGRADATION_PROBE_20261001

## GPU结果产生前冻结的协议

问题来自[上一段独立误差预算](v1_candidate_error_budget_20261001.md)：LR512的3,458张错误中，
短边<224的1,848张图贡献612张错误；这只是尺寸代理，没有证明信息损失或来源漂移。
本段用同一图像的固定降采样检验对细节损失的敏感性，比较原DEV父与既有LR512平均候选。
不是新推理配方搜索，不从诊断分数选择提交图像，也不产生新模型。

全部14,880张val_dev、当前20260921/750类；两模型仅训练过train_dev。
原图短边>224的13,022张：先PIL/torchvision BICUBIC缩短边到224、保持宽高比，
再使用原512 Resize/CenterCrop、Normalize和flip平均。原短边≤224的1,858张保持原预测，
其中1,848张短边<224、10张恰为224。两模型都用512输入/中心及flip，无bias。
各native原预测来自原LR512段已审计NPZ，完整图名、顺序、标签相等；不能拿448单中心父来比较。

明确可证伪假设：高分辨率原图减少细节后，native父出现至少75张新增原标签错误，
且被降采样组micro下降至少0.5pp，则记录supports_resolution_robustness_review；未达关闭这个固定假设。
这只支持分辨率鲁棒性问题存在，不证明实际小图错误有相同比例原因、不证明有可恢复收益，
也不自动授权训练。进一步训练需提出固定干预、对照与目标评估，并与A/B在途工作核对。

只跑两个已冻结模型的一种降采样点，不拟合head/bias或梯度、不做阈值/分辨率/TTA网格。
完整报告各模型原/降采样macro、micro及逐图修正/退化，尾75类与小图不变组；
先各重放64张seed42固定native样本，预测必须全部一致，再执行降采样。
数值、资产或native重放错误则立即封存失败，不自动重启。无墙钟时间上限；记录显存、吞吐与实际耗时。
只用本机空闲CUDA，不抢占、不使用远端/NPU，不修改依赖，不spawn子agent。

## CPU准备核验

3项测试通过：小图与224边界像素转换逐元素不变，大图确实降到224且最终保持3×512×512。
真实官方模型和2张val图CPU前向有限；旧父448→512零更新logits误差0。
checkpoint/plan/sidecar/原预测均校验SHA；分类顺序和原EMA2–4窗口严格匹配。
frozen_groups.csv及配置/实现摘要在GPU输出前保存。

首次调用当前main的模型模块被原binding拒绝：main新增768可选路径，
与原LR512所绑定代码不同。本段改用**原已冻结LR512代码目录**完成严格验证和加载，
不放宽原校验、不改任何旧plan/sidecar/checkpoint。新探针代码仍在本段独立worktree运行。

## 精确命令

方案分支`codex/v1_resolution_degradation_probe_20261001`，独立目录
`/home/lux1/noise/worktrees/v1_resolution_degradation_probe_20261001`，基于`origin/main@a8a028a`。
配置为[固定JSON](../configs/v1_resolution_degradation_probe_20261001.json)。从该worktree执行：

```bash
env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 -m pytest reproducibility/aegis_f1/tests/test_v1_resolution_degradation_probe.py -q

env PYTHONPATH=/home/lux1/noise/worktrees/lr512_dev_20261001/reproducibility/aegis_f1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 scripts/probe_v1_resolution_degradation.py prepare \
  --config configs/v1_resolution_degradation_probe_20261001.json \
  --output outputs/codex/v1_resolution_degradation_probe_20261001/probe_r1

env PYTHONPATH=/home/lux1/noise/worktrees/lr512_dev_20261001/reproducibility/aegis_f1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 -u scripts/probe_v1_resolution_degradation.py run \
  --config configs/v1_resolution_degradation_probe_20261001.json \
  --output outputs/codex/v1_resolution_degradation_probe_20261001/probe_r1 --execute
```

prepare输出必须新建，run拒绝存在status.json的任务以避免自动重跑。
预检记录见[preflight](../results/v1_resolution_degradation_probe_20261001/preflight.json)。
当前此节仅记录准备，不声称新GPU结果或平台收益；终态将在独立复算后补入同一记录。

## 现役包与交付纪律

本段是诊断，无新增训练候选。现役包仍为
`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/submission.zip`，
同目录pred_results.csv。37,444行/9项通过及ZIP内外CSV一致引用
[既有正式校验](../results/v1_full_swa_platform_20261001/submission_check.log)；SHA256
`1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`。
诊断完成后独立重计、提交推送方案，再在main以自动autostash模式pull、合并、复核、push，停在本段交付检查点。
