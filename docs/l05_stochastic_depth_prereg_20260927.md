# L05 随机深度完整训练（预注册与实测）

状态：2026-09-27 已跑满 16 轮并关闭；固定解码低于现役，不晋级。用户明确要求后额外生成了 DP01 测试提交包，未上传平台，不能把它当成通过晋级门的候选。

实验`L05_STOCHASTIC_DEPTH_20260927`，分支`codex/l05_stochastic_depth`，base main `b48a0af`。
开始前fetch，检查全部本地/远端分支、近期main和复赛/focus机制，未找到stochastic depth/
DropPath重复实现。全量数据方案仍按用户决定关闭，不扩大训练数据范围。
原L05中心macro在epoch12/14/16为75.1091%/75.2358%/75.2497%，晚期接近平台期；
本轮从其**同一个LP父checkpoint**重新走固定16轮，在表征形成过程中加入新机制，
不是失败续训方案的epoch/LR扫描，也不从L05末期追加预算。

## 固定随机深度

参考[Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)。
对OpenAI CLIP视觉Transformer的12个完整residual block，训练时定义
`out = x + m/(1-p_l) * (block(x)-x)`，`p_l=0.1*(l+1)/12`（l从0起）。
每个样本、每个block独立Bernoulli，mask共享该样本所有token和通道。
这是整个block残差的屏蔽，非attention/MLP分别屏蔽，不是原论文CNN设置的精确复现。
所有block仍执行前向，不宣称计算量节省；推理逐位使用原完整block，不做多模型集成。

每个可求梯度的图像forward一次性生成12×batch masks，独立NumPy SeedSequence([42,calls])，
不改变训练器Torch RNG。global/local训练前向各自采样；no-grad attention选区路径和验证
不采样。原始特征anchor、GCE、局部置信门控照常执行。保存每块实际调用/丢弃计数，
smoke和正式结尾要求12块调用数均等于图像forward数，且实际丢弃非零，防止hook被绕过。

无新参数、无结构持久改动；checkpoint仍为单一CLIP ViT-B/32。
仅当前阶段train_dev入训，val选模/校准，测试只允许冻结确定性推理。

## 控制与训练预算

直接从原始L05 checkpoint内保存的**实际CUDA运行配置**复制，而非NPU模板。
formal的model/data/train/loss/evaluation/trust/longtail逐项相等，只改变实验身份、输出和
project中的随机深度元信息。包括：

- 原L05共享RM_LP父权重，父SHA、L05 SHA及8个原始偶数epoch结果SHA写入固定JSON。
- seed42，384px弱RRC+Flip，16epoch cosine，backboneLR1.2e-5/headLR4e-4、WD1e-4。
- microbatch4×accum256，有效batch1024；每轮131次更新，计划共2,096次。
- CE warmup2轮→GCEq0.5，feature anchor2；attention-local从epoch5开始，权重0.25、门0.7。
- 每2轮验证，按center macro选模、micro破平局，不将train重叠指标当验证。
- 复用原L05历史对照，非新增独立seed对照。固定框架f050ecb独立导出并核验484个Git blob；
  配置一致和已有原生前向重放不等于原训练全轨迹已重新复现，该证据边界保留。

先4步smoke，积累设为1并从smoke epoch1启用local以覆盖两路hook，验证每块激活、完整
14,880张val及原生重载后才自动正式运行。smoke仅工程验证，训练配方修改不带入formal。
正式每个偶数epoch的macro或micro比**原L05同epoch**低≥2pp即止损。
由于从LP重新训练，不能拿早期结果与L05末期相比作为止损；八个同轮参照已在训练前冻结。
止损若发生在第一次验证、best尚未写出，则用已落盘last作诊断，始终不晋级。

最终固定Flip/mean_probabilities/T1.4/val uniform prior0.60，macro相对现役≥0.30pp且micro
不下降才晋级；再过独立解码、图像重放、条件交叉校准（+0.20pp/micro不下降），生成CSV/ZIP
并9/9提交校验。失败关闭，不派生深度概率、层位置、预算或学习率扫描。
现役平台最佳仍66.94797564362783%，70分目标未达，平台上传由用户执行。

## 复现与交付

```bash
python3 -m pytest tests/test_l05_stochastic_depth.py -q
python3 scripts/run_l05_stochastic_depth.py --config configs/l05_stochastic_depth/fixed.json --phase prepare
python3 scripts/run_l05_stochastic_depth.py --config configs/l05_stochastic_depth/fixed.json --phase queue
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/audit_l05_stochastic_depth.py --config configs/l05_stochastic_depth/fixed.json
```

输出`outputs/codex/l05_stochastic_depth/`；4项测试包含独立mask/残差梯度核对、eval/no-grad
逐位恒等、无模型权重变更、恢复确定性与共享RNG不变、关闭机制恒等。
启动前仅实现并通过测试，当时尚无方法成绩。训练中每30分钟监控，不因观察超时重启；
只在GPU空闲时启动，不占卡、不改共享依赖，不自动派生其他实验。
失败交付现役L05_T14_P060保底CSV/ZIP并重新校验，路径：
`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/{pred_results.csv,submission.zip}`。

## 实测与交付（2026-09-27）

DP01 从同一 RM-LP 父权重跑满 16 轮、2,096/2,096 次更新，选择 epoch16。
第 16 轮中心视图 macro/micro 为 **74.1373% / 75.1815%**，同轮原 L05 为
**75.2497% / 76.2970%**。原生 checkpoint 重载一致；12 个 block 都实际触发随机深度，
累计图像前向 936,712 次，逐层丢弃数均非零。没有触发同轮 −2pp 止损。

固定 Flip/mean_probabilities/T1.4/验证集拟合 uniform prior0.60 解码：
DP01 macro/micro **74.3077% / 75.1277%**，现役 **75.8265% / 76.6532%**，
相差 **−1.5189pp / −1.5255pp**，未过晋级门，固定配方关闭，不做概率/位置/预算扫描。
独立 CPU float64 解码与正式结果的 **14,880 个预测逐张一致**；4 项定向测试与
原生重载审计通过。没有 DP01 平台成绩；现役平台最佳仍为 **66.94797564362783%**。

按用户明确要求，使用同一 DP01 epoch16 单 checkpoint 另生成独立测试包；这不改变本地
未晋级裁决，也不代表建议上传。推理命令如下（在 `focus/f05-four-lines` 工作目录执行，
该目录的框架提交为 `f050ecb59e0a885da0b43cdc6c8c316953fb5e7d`）：

```bash
python3 scripts/build_l05_tta_prior_submission_final.py \
  --checkpoint /home/lux1/noise-worktrees/l05_stochastic_depth/outputs/codex/l05_stochastic_depth/runs/DP01/seed42/checkpoints/best.pt \
  --config /home/lux1/noise-worktrees/l05_stochastic_depth/outputs/codex/l05_stochastic_depth/configs/DP01.yaml \
  --val-branch-cache /home/lux1/noise-worktrees/l05_stochastic_depth/outputs/codex/l05_stochastic_depth/runs/DP01/seed42/val_branch_logits.pt \
  --temperature 1.4 --prior-strength 0.6 --fusion mean_probabilities --tta horizontal_flip \
  --tag DP01_UNPROMOTED_20260927 \
  --output-root /home/lux1/noise/outputs/codex/l05_stochastic_depth_delivery \
  --skip-desktop-copy --device cuda:0 --batch-size 32 --num-workers 2
```

测试包目录：`/home/lux1/noise/outputs/codex/l05_stochastic_depth_delivery/DP01_UNPROMOTED_20260927/`。
37,444 行 CSV 与 ZIP 内部字节一致，9/9 提交校验通过；CSV SHA-256
`4649c3c79ac776734c7dbd877c62de22f0905a09969f35e4fc633aacd55ab2f5`，ZIP SHA-256
`955305a385699e04d624c5f3de4f20563e869d2aa7da39edd9241eb0d4f3fe71`。
现役 L05_T14_P060 保底包也重新通过 9/9 校验。完整机器可读结果见
[结果文件](../results/l05_stochastic_depth_20260927.json)。平台上传由用户执行。
