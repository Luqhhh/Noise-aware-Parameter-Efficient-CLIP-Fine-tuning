# V1_COSINE_MARGIN_PROBE_20261001：一次固定类间间隔训练探针

2026-10-01启动准备，正式运行跨午夜按实际时间记录；实验标识不随日期滚动改名。

## 问题、方法与重叠

LR512的3,458个原标签错误中，1,881个原类仍在原图或镜像top5；这只给排名探索空间，不是可达收益。
两视图较大JS分歧已关闭，本方案不改JS阈值、做一致性训练或恢复TTA。
当前cosine classifier只按混合CE训练。假设：训练中要求目标cosine额外领先，可改善近边界分类，
但含噪原标签也可能被进一步拟合，实际识别净收益必须用完整DEV实测。

公式参考[CosFace原论文](https://arxiv.org/abs/1801.09414)，其归一化特征/权重和训练目标类cosine间隔思想可迁移；
原论文为人脸实验，不提供本750类任务或平台收益证据。本轮是保留v1 LS与加权Mixup的适配，不复现其训练配方。
启动已fetch、检查全部分支及main历史，未找到cosine target-margin损失的已有当前方案。
从`origin/main@e3533fa`创建`codex/v1_cosine_margin_probe_20261001`独立worktree；
A为768 head/LoRA、B为增强配对及恢复，本方案不重复它们、不碰其任务或NPU/远端，不spawn子agent。
L05冻结项、旧teacher支持阈值、内容代理和监督阶梯均保持。

## 结果前冻结的一个配方

[配置](../configs/v1_cosine_margin_probe_20261001.json)。仅训练一个候选，原已交付控制只读复用，不重复训练。
同LR512 EMA2–4父checkpoint，32,768 active train_dev无放回样本、seed42、1,024成功更新，
batch32/micro8/workers2；512 RRC0.8–1/flip/RA2,7、Mixup0.2、LS0.1、原target/可靠度均保持。
优化器、OneCycle、学习率、衰减、clip与原控制完全相同；无EMA。原schedule SHA256
`42cf04b89482d279c8b2a4fd251cecc58d590dfb546b50761aefaae05f231ff4`；
控制主checkpoint SHA256`7ccf90560293dfdb4456e98f6c9a8f543a34a2960dc4bee9f369eaeaf1bdcaa1`。
原固定子集缺第183类，不补采样。模型/目标/sidecar/代码严格使用原绑定，公共依赖不变。

唯一干预：cosine单位`m=0.05`，沿head原可学习scale `s`，对每个Mixup来源独立计算
`CE(q_i, z−s*m*one_hot(target_i))`，再以原`λ*w_i`及`(1−λ)*w_j`组合并除原整batch可靠质量。
LS分布不改，混合图像不改，不把两来源监督改为一个硬标签。m=0在loss及梯度上回到原加权Mixup目标。
推理完全用原cosine logits，不添加margin、不使用原标签或bias。

候选输出前按固定LR512 native logits的top1−top2差升序、图路径升序取最低四分位，
不依据标签挑图；3,720张/2,369个原父错误，覆盖原3,458错的68.51%，原类是否干净未知。
组定义、logit源摘要与原控制逐图预测在CPU prepare固定，首batch原增强tensor摘要亦固定。
完整报告all、low_gap/补集、tail75、小图、四候选共同错误的macro/micro及相对父/控制修正退化。

先8步真实CUDA成本检查，状态丢弃并从原父/原seed重新开始；只固定1,024步，512步诊断、1,024末步raw为唯一主结果，
不根据中间分数换checkpoint或提前停accuracy，不设墙钟截止。数值、资产、资源或复现错误即封存，不自动重启。
完成后必须给末步单checkpoint的37,444行CSV/ZIP、9项检查、64张冷加载一致及独立logits→CSV字节复算。

复核门同时要求候选相对控制全量净≥75、相对初始父净≥75，低间隔组相对控制净≥25，
全量macro不低于控制且修正≥1.25×退化。通过只支持后续投入复核；未达关闭本固定配方，
不扫描m、延长/重复seed、组合或自动full，不平台上传。平台迁移和干净标签均未知。

## 准备验证与精确命令

7项新增测试（m=0 loss/gradient、Mixup两端、正间隔/scale梯度、microbatch总质量、有限梯度单次更新、
非有限梯度拒绝更新）及3项原日程测试通过，共10项。
父零更新FP32误差0，分类顺序/257 tokens保持；目标组和首batch控制增强已冻结。
初次静态/单测发现并修正重复JSON字段和混合dtype scatter，修复在任何GPU输出前完成。

工作目录`/home/lux1/noise/worktrees/v1_cosine_margin_probe_20261001`：

```bash
env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -m pytest reproducibility/aegis_f1/tests/test_v1_cosine_margin.py reproducibility/aegis_f1/tests/test_v1_detail_aug_pair.py -q
env PYTHONPATH=/home/lux1/noise/worktrees/lr512_dev_20261001/reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 scripts/train_v1_cosine_margin_probe.py prepare --config configs/v1_cosine_margin_probe_20261001.json --output outputs/codex/v1_cosine_margin_probe_20261001/probe_r1
env PYTHONPATH=/home/lux1/noise/worktrees/lr512_dev_20261001/reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u scripts/train_v1_cosine_margin_probe.py cost --config configs/v1_cosine_margin_probe_20261001.json --output outputs/codex/v1_cosine_margin_probe_20261001/probe_r1 --execute
env PYTHONPATH=/home/lux1/noise/worktrees/lr512_dev_20261001/reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u scripts/train_v1_cosine_margin_probe.py run --config configs/v1_cosine_margin_probe_20261001.json --output outputs/codex/v1_cosine_margin_probe_20261001/probe_r1 --execute
```

已有输出拒绝覆盖，重放须新路径。
2026-10-02北京时间00:09完成8次真实CUDA成本更新：后4步均值0.461751秒/步、峰值allocated934,223,872字节，
1次数值AMP重试后全部成功；首batch原增强摘要一致，成本状态丢弃。
00:09正式从原父重新初始化；本轮采用固定成功更新数，不设时长中止。
训练代码、损失、配置与源摘要仍为原preflight绑定；运行中merge新的A/B恢复记录，未修改绑定输入。
完整末步实测及新提交包见下文；中段分数不选模。
日志位于同方案私有目录`logs/{cost_r1,run_r1}.log`；模型、图片、逐图NPZ和分组不入Git。
最终聚合报告/独立核验/提交校验日志入Git，立即方案commit/push，再main自动autostash pull/合并/复验/push，检查点暂停。

现役full v1 SWA仍为用户报告70.98600576861446%；沿用
`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/{pred_results.csv,submission.zip}`，
ZIP SHA256`1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`；
[既有9项检查](../results/v1_full_swa_platform_20261001/submission_check.log)、
[包核验](../results/v1_full_swa_platform_20261001/artifact_verification.json)。新候选未获平台分时不替换现役。

## 完整DEV末步实测与交付

唯一主结果为1,024步raw；下表来自14,880张完整原标签验证。

| 群体 | 张数 | 候选micro | 候选macro | 相对控制修正/退化/净 | 相对初始父修正/退化/净 |
|---|---:|---:|---:|---:|---:|
| all | 14880 | 76.6667% | 75.6587% | 101/76/+25 | 181/195/-14 |
| low_gap | 3720 | 35.9677% | 37.4258% | 98/76/+22 | 180/193/-13 |
| other_gap | 11160 | 90.2330% | 86.8416% | 3/0/+3 | 1/2/-1 |
| tail75 | 1003 | 66.6002% | 60.1751% | 12/10/+2 | 11/13/-2 |
| small | 1848 | 66.5043% | 67.9028% | 28/16/+12 | 33/40/-7 |
| common_candidate_wrong | 2875 | 1.8783% | 1.9734% | 13/15/-2 | 54/0/+54 |

初始LR512父micro/macro76.7608/75.7631%，复用控制76.4987/75.4819%。
候选相对控制micro+0.168011pp、macro+0.176809pp；
相对初始父micro-0.094086pp、macro-0.104325pp。
全量净+25<75、相对父−14<75、目标组净+22<25三项未达；macro和101≥1.25×76两项通过。
固定门失败，关闭本0.05配方，不据此否定所有cosine-margin方法或声称平台效果。
512步micro/macro76.5121/75.5006%仅为过程诊断；相对末步控制的诊断比较有更新数差异，不能用作同日程主归因。
正式更新中第1/160步AMP降scale重试后成功，共2批；没有跳过更新或改动1,024步配方。
原控制首batch像素未在历史训练时存档；当前使用原实现摘要、相同确定性日程/worker种子/增强，并在本轮prepare/成本/正式首batch及CPU独立重放保持同一像素摘要。
这不宣称历史32,768张增强像素曾逐张归档。

正式本机CUDA训练、两次完整val与全test出包共1558.083736秒（25.9681分钟）。
10项相关测试通过；独立重放29,760条候选val、37,444条test，
14,880行冻结gap/分组、原控制14,880行预测、30份指标及24份配对一致；
原源资产摘要未变，32张CPU首batch像素亦复现。9项约束、ZIP内外字节、64张冷加载一致通过。

- [聚合实测](../results/v1_cosine_margin_probe_20261001/report.json)
- [CPU冻结与源摘要](../results/v1_cosine_margin_probe_20261001/preflight.json)、[8步成本](../results/v1_cosine_margin_probe_20261001/cost.json)
- [独立核验](../results/v1_cosine_margin_probe_20261001/independent_verification.json)、[提交校验](../results/v1_cosine_margin_probe_20261001/submission_check.log)

单checkpoint末步候选目录：
`/home/lux1/noise/worktrees/v1_cosine_margin_probe_20261001/outputs/codex/v1_cosine_margin_probe_20261001/probe_r1/candidate/submission`。
CSV：`/home/lux1/noise/worktrees/v1_cosine_margin_probe_20261001/outputs/codex/v1_cosine_margin_probe_20261001/probe_r1/candidate/submission/pred_results.csv`；SHA256`70aa47749c7be066d243118109e8e33afc28d5a99963add5987a5fc4e8b86f45`。
ZIP：`/home/lux1/noise/worktrees/v1_cosine_margin_probe_20261001/outputs/codex/v1_cosine_margin_probe_20261001/probe_r1/candidate/submission/submission.zip`；SHA256`ff65d4430b27ae6e6608f192f4e5ead19c9093b194f12a59b5706ba149c3b647`。
Checkpoint：`/home/lux1/noise/worktrees/v1_cosine_margin_probe_20261001/outputs/codex/v1_cosine_margin_probe_20261001/probe_r1/candidate/selected.pt`；SHA256`f9313b247c335102c2ab1f44af2ce93767f5dc106429c88a6eafb70dffc39afa`。
包仅保留为本固定已验证候选，无平台上传/分数，不替换现役full v1 SWA。

独立核验精确命令（同工作目录；main集成复验使用上述绝对输出路径和新result路径）：

```bash
env PYTHONPATH=/home/lux1/noise/worktrees/lr512_dev_20261001/reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 scripts/verify_v1_cosine_margin_probe.py --config configs/v1_cosine_margin_probe_20261001.json --output outputs/codex/v1_cosine_margin_probe_20261001/probe_r1 --result outputs/codex/v1_cosine_margin_probe_20261001/probe_r1/independent_verification.json
```

实现文件：固定JSON配置、`train_v1_cosine_margin_probe.py`、`v1_cosine_margin_loss.py`、
`verify_v1_cosine_margin_probe.py`及margin测试；文档、执行入口和上述小型聚合结果入Git，
原控制只读复用、未重复训练，私有checkpoint/逐图预测/图片不入Git。
末步三个规模门失败，关闭本固定0.05方案、不扩full或扫描；下一步仍等A/B/v2既定任务证据，
本轮不能将+25相对控制、宏分上涨或包校验解释为平台提升或已弥合1,503张平台缺口。
