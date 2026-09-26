# L05 训练侧 logit 幅度惩罚（预注册与实测，已关闭）

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

队列于2026-09-26 23:53:42 Asia/Shanghai启动，driver PID215110、start_ticks5138776。
先smoke后正式；实现提交5360562。下一次监控不早于2026-09-27 00:23:42。
进程记录不是完成证据，正式结果待验证；当前只推送方案分支，不合并未验证结果到main。


## 2026-09-27 00:24运行检查（历史快照，非方法结果）

smoke两次更新、14,880张中心验证及原生重载完成，penalty calls=4/examples=16，
累计penalty_sum=0.3607485592；checkpoint/sidecar SHA复核通过。
正式epoch0复现L05中心macro/micro=75.2497%/76.2970%，原队列自动启动正式训练，
累计59/262次更新全部成功；driver215110、child215869仍存活，未观察到训练错误。
尚无正式完整验证/解码结果，不将smoke数值当作方法成绩。下一次约00:54检查。


## 最终实测（2026-09-27）

原队列正常完成，262/262次更新全部成功，未触发止损。按center raw macro选择epoch1：
epoch1 macro/micro=75.2842%/76.3105%；epoch2=75.2670%/76.3105%。
最佳checkpoint原生重载中心验证及独立进程的14,880张val双分支缓存生成均完成。

| 固定Flip/T1.4/prior0.60 | macro | micro | 对L05 Δmacro/Δmicro |
|---|---:|---:|---|
| 现役L05 | 75.8265% | 76.6532% | — |
| 复用MS00普通续训 | 75.7948% | 76.6129% | −0.0317pp / −0.0403pp |
| SD01 logit惩罚 | 75.8253% | 76.6465% | −0.0012pp / −0.0067pp |

SD01比MS00高0.0306pp/0.0336pp，但未超过现役，也未过+0.30pp门。
**本固定配方关闭，不派生lambda、预算或惩罚位置扫描，不生成新测试候选。**
训练惩罚calls=133,816，examples=535,260，penalty_sum=14,093.384134；examples含global/local
及多轮重复前向，不是独立图片数。lambda始终为0.004359280240465393。

最佳checkpoint SHA256=`47a9f1dd37de03a8d127e7e013da333d5d3f91ec8de682dc666db2910f7b7ee7`。
best/epoch1/last与各自惩罚sidecar、源代码及系数文件SHA全部一致。
CPU NumPy float64独立解码与Torch的14,880/14,880个预测完全一致，prior bias最大差6.62e-7，
阶段/划分/模型/缓存绑定及484份Git源码blob复核通过。该审计复核缓存解码，
不等同于再跑一次独立图像推理，也不证明平台收益。3项损失与恢复测试通过。

新增实现包括惩罚模块、训练队列、独立解码审计、配置和测试；完整实测、逐轮指标、
控制引用、checkpoint与交付哈希见[最终机器可读记录](../results/l05_logit_penalty_20260927.json)。
此前启动JSON保留为历史运行快照。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/audit_l05_logit_penalty.py \
  --config configs/l05_logit_penalty/fixed.json
python3 scripts/check_submission.py \
  --test_dir /home/lux1/noise/test \
  --class-mapping /home/lux1/noise/artifacts/stages/repechage/20260921/class_to_idx.json \
  --csv /home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/pred_results.csv \
  --zip /home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/submission.zip
```

本轮交付上方现役L05_T14_P060保底CSV/ZIP，37,444行，9/9校验通过；不是SD01新候选。
CSV SHA256=`51e0efe7178528d23993a44351069d2829b0cd3669c195a77d8d879e66776a75`，
ZIP SHA256=`e788f07636b80abb61685335cd8df108803863ea36ffbde8b353f8e8317fcd7f`。
未上传平台，无新平台成绩，最佳仍为66.94797564362783%，70分以上目标未达到。
在本段交付检查点停止扩展；后续长训练仍按用户要求每30分钟监控。
