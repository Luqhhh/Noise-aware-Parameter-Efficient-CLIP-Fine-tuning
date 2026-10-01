# V1_RESOLUTION_DEGRADATION_PROBE_20261001

状态`completed_verified_diagnostic`。本机CUDA两模型固定降采样已完成，耗时432.34秒，
128张native冷加载预测全部一致，26,044条降采样预测由保存logits独立重放一致。
父模型净损失236张正确，LR512净损失233张；原512续训没有显示明显缓解这一敏感性。
无新训练候选或平台成绩，现役full v1 SWA保持用户报告70.98600576861446%。

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
上面是实际prepare/run命令及GPU输出前协议。聚合GPU结果见
[report](../results/v1_resolution_degradation_probe_20261001/report.json)，独立核对命令：

```bash
python3 scripts/verify_v1_resolution_degradation.py \
  --config configs/v1_resolution_degradation_probe_20261001.json \
  --run-root outputs/codex/v1_resolution_degradation_probe_20261001/probe_r1 \
  --output results/v1_resolution_degradation_probe_20261001/independent_verification.json
```

## 实测结果与决定

完整14,880张原标签诊断，短边≤224的1,858张保持native原预测，其余接受固定干预。
表中修正/退化均以同模型的native流程为基线，不代表训练效果。

| 模型 | native micro/macro | 降采样后micro/macro | 修正/退化/净变化 | 被降采样组micro变化 |
|---|---|---|---|---|
| 原DEV父 | 76.0215% / 75.0287% | 74.4355% / 73.4463% | 206 / 442 / −236 | −1.8123pp |
| LR512 EMA2–4 | 76.7608% / 75.7631% | 75.1949% / 74.2183% | 214 / 447 / −233 | −1.7893pp |

父和LR512共同退化234张，父独有退化208张、LR独有退化213张；共同修正96张。
尾75类父净−4，LR净−8；真实小图组按设计没有改变，不能把不变组说成此次修复收益。
LR512相对native父的全量原优势修正324/退化214/净+110；干预后372/259/净+113。
优势仅多3张、约0.0202pp，不能宣称512续训带来明显的细节损失鲁棒性。

父442张退化、被降采样组下降1.8123pp，超过预先冻结的75张/0.5pp机制门，
故`supports_resolution_robustness_review`。这证明本流程对这一次合成信息减少有原标签识别损失；
它不说明真实小图的612张错误有多少由相同原因造成，也不证明降采样训练能恢复它们。
原标签含噪、真实来源未知、图像的有效目标占比未知，平台恢复比例仍未知。

本轮决定：不把原LR512小图净+38自动归因为分辨率鲁棒性，不以继续加分辨率或原配方续训
作为该问题的修复依据。已有LR512包与A/B独立配对保持；若以后设计针对低细节的训练，
需单独的固定训练干预、完整DEV对照及退化控制，并核对B增强路线重叠，不能由本诊断自动启动。
不扫描更多降采样点、不把合成降采样变成提交解码。

## 验证与产物

新增3项转换测试与依赖的6项误差统计测试合计9项通过。
独立stdlib集合/Counter复算20份原/干预群体指标和10份逐图配对群体，全部一致；
13,022×2条降采样预测由保存FP32 logits重放，1,858×2条不变预测逐条相等；
冻结分组14,880行按官方尺寸和tail类定义重新核对，18份原绑定输入SHA再次验证。
原包/权重/旧sidecar均未改写，运行期间无梯度、0次训练更新。
详细核验见[独立JSON](../results/v1_resolution_degradation_probe_20261001/independent_verification.json)。

实测432.34秒，CUDA峰值allocated为1,321,474,560字节（1.2307GiB）；nvidia-smi进程列表在结束后为空。
读取中出现EXIF/palette元信息警告，全部13,022张成功前向，无替代图像或跳样。
逐图分组、NPZ/logits、progress/status位于独立目录
`/home/lux1/noise/worktrees/v1_resolution_degradation_probe_20261001/outputs/codex/v1_resolution_degradation_probe_20261001/probe_r1/`，不入Git。
入库改动为两份探针/复算脚本、固定JSON、3项转换测试、聚合结果、本执行记录、当前入口和private输出ignore。

协议在GPU输出前以`2509cfe`提交并推送；最终结果提交后立即推送方案并在main复核集成。
这是新机制证据，未产生新checkpoint或提交CSV/ZIP。

## 现役包与交付纪律

本段是诊断，无新增训练候选。现役包仍为
`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/submission.zip`，
同目录pred_results.csv。37,444行/9项通过及ZIP内外CSV一致引用
[既有正式校验](../results/v1_full_swa_platform_20261001/submission_check.log)；SHA256
`1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`。
诊断完成后独立重计、提交推送方案，再在main以自动autostash模式pull、合并、复核、push，停在本段交付检查点。
