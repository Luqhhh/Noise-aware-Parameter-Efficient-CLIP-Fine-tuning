# L05 训练侧 logit 幅度惩罚（预注册）

实验 `L05_LOGIT_PENALTY_20260926`，分支 `codex/l05_logit_penalty`，基于main `1ea4cf7`。
开始前fetch全部分支，核对V4/V5所有训练机制、focus及现有候选，未发现spectral decoupling/
logit penalty/logit norm配置或结果。共享协方差轮已关闭，本轮改训练目标，不派生该轮参数。

## 假设与固定配方

[Gradient Starvation作者论文](https://arxiv.org/abs/2011.09468)提出对logit加L2惩罚。
本轮采用 `GCE(q=.5) + lambda/2 * mean_class(logit²)`，作用于global及local分类损失，
再由原训练器执行相同的局部置信门控、.25局部权重、feature anchor=2和原weight decay。
**保留GCE/weight decay、用于L05续训是本轮假设，不是原论文理论保证的设置。**
eval不加惩罚；模型结构、参数数目、推理流程不变；无多头/多模型/多checkpoint融合。

lambda只从本阶段train_dev的冻结L05中心特征确定：
`lambda = .2 * mean_train(GCE) / mean_train_class(logit²)`。
因此初始训练中心特征上惩罚均值是GCE均值的10%；此比例预先固定，不扫描。
标定只用133,815张训练图特征与原标签。lambda落盘后冻结；val只用于选模/固定解码，
不用于选择lambda，测试集不用于拟合或调参。

复用已审计的MS00普通续训作对照，不重复训练：同L05父checkpoint、同split/seed42、同
2epochs cosine、384px弱RRC+Flip、microbatch4×accum256、backboneLR3e-6/headLR1e-4、
WD1e-4，无CE warmup，local从epoch1开始。新runtime config直接继承MS00 YAML，
仅改实验/输出身份及新增惩罚元信息；formal的model/data/train/loss/evaluation逐项保持一致。
MS00原hook概率0且不耗RNG；新loss wrapper也不耗RNG。复用的不是新独立seed证据。
控制配置SHA、checkpoint SHA由固定JSON钉死，并验证实际文件及已完成审计。

原生训练器源码固定`f050ecb`，复制至本worktree输出并核对全部Git blobs；只在内存包装
`_per_sample_loss`，不修改快照或共享框架。每次保存将lambda/激活计数/代码SHA与checkpoint
绑定。显式resume必须验证并恢复计数；不因观察超时重启训练。

## 判据

先跑2次更新smoke，完整val和原生重载通过、确认惩罚激活且有非零梯度后才启动正式训练。
正式epoch0中心指标必须复现L05；按center raw macro选epoch0/1/2，micro破平局。
任一epoch中心macro或micro比L05低≥2pp则止损。固定Flip/mean_probabilities/T1.4/prior0.60；
SD01解码macro须同时比L05与MS00高≥.30pp，micro均不下降才晋级。
未过门关闭，不派生lambda/预算/惩罚位置扫描；过门后独立解码、图像重载、条件交叉校准、
单checkpoint生成CSV/ZIP及9/9校验。平台上传仍由用户执行。
失败交付现役L05_T14_P060保底CSV/ZIP（原路径，重新校验），不生成伪新候选。

## 可重放入口

```bash
python3 -m pytest tests/test_l05_logit_penalty.py -q
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 scripts/run_l05_logit_penalty.py \
  --config configs/l05_logit_penalty/fixed.json --phase prepare
python3 scripts/run_l05_logit_penalty.py --config configs/l05_logit_penalty/fixed.json --phase queue
```

所有输出在`outputs/codex/l05_logit_penalty/`；训练前检查当时空闲GPU，不占卡、不改共享依赖。
运行监控每30分钟一次。实现、smoke与正式方法成绩分开登记；当前无新平台成绩，
已知最佳66.94797564362783%，70分以上目标未达。


## 启动前实测（不是方法结果）

3项测试通过；484份源码blob及父checkpoint/阶段资产验证通过。训练特征标定得到
`lambda=0.004359280240465393`，mean GCE=0.2600840193，mean logit²=11.9324294366；
标定惩罚均值0.02600840193，正好是固定10%比例，未使用val选择强度。
formal的model/data/train/loss/evaluation/trust与MS00逐项一致。
完整分片哈希和预检见[启动记录](../results/l05_logit_penalty_launch_20260926.json)。
