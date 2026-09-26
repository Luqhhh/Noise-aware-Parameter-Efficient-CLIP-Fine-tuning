# L05 类别级缩放校准（预注册与实测，已关闭）

实验`L05_VECTOR_CALIBRATION_20260927`，分支`codex/l05_vector_calibration`，base main `b9b52d2`。
已fetch并检查全部分支和近期结果。历史初赛v8只有训练重叠分数上的有监督类别偏置，
已失败关闭；本轮不重跑v8、不读取任何初赛资产。复赛已有uniform prior为无监督边际对齐。
本轮新增类别级乘法缩放，并在当前复赛val内严格隔离校准拟合与计分，未找到重叠实验。

## 固定机制

[Guo等校准论文](https://proceedings.mlr.press/v70/guo17a.html)讨论类别级vector scaling。
本轮为固定L05上的变体：令`s=log(mean(softmax(original/1.4),softmax(flip/1.4)))`，
先按现役流程加`0.6*prior_bias`，再逐行减去类别均值，得`x`。
学习每类`a_c,b_c`，输出`a_c*x_ic+b_c`，初值`a=1,b=0`；
不改变单一父checkpoint，不组合模型/折预测；折只用于评估最终单套全val校准参数。

目标为类均衡CE加`1/(2C) * sum((a-1)^2+b^2)`，正则系数固定1。
每个拟合集内部样本权重为`1/(出现类别数 * 该标签样本数)`。
约束`a∈[0.5,2]`、`b∈[-log(4),log(4)]`；缺失类别固定`a=1,b=0`。
数据口径检查发现当前val每类1～25张、3类不足3张，因此预先声明缺类处理。
CPU float64、SciPy L-BFGS-B、maxiter100/maxfun300/maxls30/maxcor10、ftol1e-12/gtol1e-8。
必须收敛、目标不增、投影梯度≤1e-6；任一失败停止，不改求解预算或强度。

缩放和偏置联合拟合区别于旧v8偏置求解器；保留现役温度、prior强度和视图，不派生扫描。
逐行中心化确保公共logit平移不改变结果；该变体不是原论文设置的精确复现或收益保证。

## 三折条件交叉校准与裁决

按`SHA256('42:'+content_group)`前8字节大端整数mod3固定分折。
对每折：只在另外两折拟合prior，再用另外两折标签拟合a/b；本折只推理和计分。
同条件baseline的prior也只拟合另外两折。所有折预测恢复原14,880张顺序后聚合macro/micro。
每条样本的标签不参与产生自身校准参数；内容组不能跨折。
既有checkpoint和基础解码曾用整个val选择，因此这仍是**条件交叉校准**，不是全链路独立OOF。

候选聚合macro须同时比同条件crossfit baseline、现役全val冻结baseline高≥0.30pp，
micro均不下降。该单门同时严格于既有+0.20pp条件校准门，不用全val重拟合成绩晋级。
未过门关闭，不派生正则、边界、温度、先验或参数化扫描，不生成测试候选。
通过才全val拟合一套最终参数，核验原生图像重放、绑定和独立解码，单checkpoint确定性
测试推理，生成CSV/ZIP并9/9校验。测试图及测试预测分布不参与拟合，平台上传由用户执行。

## 审计和重放

预检当前阶段manifest、train/val路径及内容组隔离、父checkpoint、参考cache SHA与标签顺序。
先独立NumPy重现现役14,880个预测；各折优化目标/解析梯度与独立Torch autograd比对≤1e-10，
各折held-out预测与独立Torch应用逐一比对。缓存、折号、拟合参数、预测和实现SHA独立落盘。
本轮直接复用已审计原图/Flip缓存，不宣称重新完成全量图像前向。

```bash
python3 -m pytest tests/test_l05_vector_calibration.py -q
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 scripts/run_l05_vector_calibration.py \
  --config configs/l05_vector_calibration/fixed.json
```

输出`outputs/codex/l05_vector_calibration/`拒绝覆盖，CPU拟合、不占GPU、不改共享依赖。
长任务每30分钟检查一次。失败交付现役L05_T14_P060保底包并重新校验：
`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/{pred_results.csv,submission.zip}`。
当前只完成实现和定向测试，尚无新方法成绩；平台最佳66.94797564362783%，70分目标未达。


## 最终实测

三折留出样本分别为5,012 / 4,837 / 5,031，合计14,880；各内容组只属一折。
三次拟合均收敛，分别53/53/51次迭代，最大投影梯度2.7543e-7。
三折训练CE+正则目标均降低，但留出预测准确率退化：

| 评估口径 | macro | micro |
| --- | ---: | ---: |
| 现役全val拟合prior（参照） | 75.8265% | 76.6532% |
| 同条件三折baseline | 75.5888% | 76.4180% |
| 类别级缩放三折候选 | 75.3765% | 76.0551% |
| 候选对三折baseline Δ（百分点） | −0.2123 | −0.3629 |
| 候选对现役参照 Δ（百分点） | −0.4500 | −0.5981 |

对同条件baseline纠正116张、退化170张，净少判对54张。未过门，固定配方关闭，
没有全val最终重拟合、没有新测试候选、不派生参数扫描。同条件baseline也低于全val
prior参照，说明两种校准数据量/计分口径不同，不能拿拟合集改善代替留出效果。

4项测试通过。每折解析目标/梯度与独立Torch autograd一致（最大梯度差3.03e-17以内）；
保存参数重新加载后，以独立NumPy float64从原始双路logits重新融合、加prior、中心化、
缩放与加偏置，两套三折baseline/candidate各14,880/14,880预测完全一致。
现役完整解码也独立复现；折预测文件SHA为
`31733bc64ac5cf2e5ba1de6e9f7aed4600125bf44e78373add3678328262b78f`。
本轮只用缓存，不是全量图像重放；父checkpoint此前用全val选模，条件交叉校准不等于
无偏全流程OOF，也不能推断平台成绩。

新增文件是`scripts/l05_vector_calibration_support.py`、`scripts/run_l05_vector_calibration.py`、
`tests/test_l05_vector_calibration.py`及`configs/l05_vector_calibration/fixed.json`。
全部配置、逐折拟合、独立核验、交付路径和确切提交校验命令见
[最终记录](../results/l05_vector_calibration_20260927.json)。
现役保底CSV/ZIP 37,444行及9/9校验通过，未改预测：

- CSV SHA256：`51e0efe7178528d23993a44351069d2829b0cd3669c195a77d8d879e66776a75`
- ZIP SHA256：`e788f07636b80abb61685335cd8df108803863ea36ffbde8b353f8e8317fcd7f`

无新平台成绩；最佳仍为66.94797564362783%，70分目标未达。
本段Git集成后停在检查点，后续长任务每30分钟监控。
