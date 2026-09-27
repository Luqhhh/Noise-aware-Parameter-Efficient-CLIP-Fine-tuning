# L05 随机深度完整训练（预注册）

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
当前只实现并通过测试，尚无方法成绩。训练中每30分钟监控，不因观察超时重启；
只在GPU空闲时启动，不占卡、不改共享依赖，不自动派生其他实验。
失败交付现役L05_T14_P060保底CSV/ZIP并重新校验，路径：
`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/{pred_results.csv,submission.zip}`。
