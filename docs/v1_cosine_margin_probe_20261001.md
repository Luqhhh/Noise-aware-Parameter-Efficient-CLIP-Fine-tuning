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

5项新增测试（m=0 loss/gradient、Mixup两端、正间隔/scale梯度、microbatch总质量）及3项原日程测试通过。
父零更新FP32误差0，分类顺序/257 tokens保持；目标组和首batch控制增强已冻结。
初次静态/单测发现并修正重复JSON字段和混合dtype scatter，修复在任何GPU输出前完成。

工作目录`/home/lux1/noise/worktrees/v1_cosine_margin_probe_20261001`：

```bash
env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -m pytest reproducibility/aegis_f1/tests/test_v1_cosine_margin.py reproducibility/aegis_f1/tests/test_v1_detail_aug_pair.py -q
env PYTHONPATH=/home/lux1/noise/worktrees/lr512_dev_20261001/reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 scripts/train_v1_cosine_margin_probe.py prepare --config configs/v1_cosine_margin_probe_20261001.json --output outputs/codex/v1_cosine_margin_probe_20261001/probe_r1
env PYTHONPATH=/home/lux1/noise/worktrees/lr512_dev_20261001/reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u scripts/train_v1_cosine_margin_probe.py cost --config configs/v1_cosine_margin_probe_20261001.json --output outputs/codex/v1_cosine_margin_probe_20261001/probe_r1 --execute
env PYTHONPATH=/home/lux1/noise/worktrees/lr512_dev_20261001/reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u scripts/train_v1_cosine_margin_probe.py run --config configs/v1_cosine_margin_probe_20261001.json --output outputs/codex/v1_cosine_margin_probe_20261001/probe_r1 --execute
```

已有输出拒绝覆盖，重放须新路径；当前为CPU准备通过，尚无正式候选结果/新提交包。
日志位于同方案私有目录`logs/{cost_r1,run_r1}.log`；模型、图片、逐图NPZ和分组不入Git。
最终聚合报告/独立核验/提交校验日志入Git，立即方案commit/push，再main自动autostash pull/合并/复验/push，检查点暂停。

现役full v1 SWA仍为用户报告70.98600576861446%；沿用
`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/{pred_results.csv,submission.zip}`，
ZIP SHA256`1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`；
[既有9项检查](../results/v1_full_swa_platform_20261001/submission_check.log)、
[包核验](../results/v1_full_swa_platform_20261001/artifact_verification.json)。新候选未获平台分时不替换现役。
