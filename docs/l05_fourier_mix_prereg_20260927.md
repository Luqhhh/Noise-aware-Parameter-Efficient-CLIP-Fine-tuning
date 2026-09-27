# L05 低频幅度混合续训（预注册与实测，已关闭）

实验`L05_FOURIER_MIX_20260927`，方案分支`codex/l05_fourier_mix`，base main `b6f42b4`。
已fetch并检查全部分支、近期main及复赛/focus实现和报告，未发现Fourier/amplitude mixing
重叠方案。MixStyle改patch特征均值/方差，本轮在RGB频域混合低频幅度，独立检验图像外观
增强；不是MixStyle/RSC失败配方的参数扫描。原始数据共享只读，使用独立目录和环境现有依赖。

## 假设与固定机制

参考[Fourier-based DG论文](https://openaccess.thecvf.com/content/CVPR2021/html/Xu_A_Fourier-Based_Framework_for_Domain_Generalization_CVPR_2021_paper.html)
的幅度混合思路。本轮保留当前图相位，混合同一训练microbatch另一张图的低频幅度，
测试减少颜色/低频外观依赖是否有帮助。仅使用当前阶段train_dev，不引入测试或外部图片。

只增强可反向传播的global输入（固定训练器`return_features=True`分支），概率0.5。
将CLIP标准化图按原preprocess mean/std还原RGB，float32 rFFT，低频矩形掩码为
`|ky|≤floor(H*0.05), |kx|≤floor(W*0.05)`（384px时半径19个频率bin）。
每张图的lambda独立Uniform[0,1]，幅度`(1-lambda)*A + lambda*A_donor`，掩码外不变；
同microbatch随机非零循环位移选donor，不允许自配对，标签不混合。
原频谱为零时相位定义为0。逆变换后RGB截断到[0,1]再按原mean/std标准化。
相位与掩码外频谱只在截断前保持；截断会破坏严格频域不变性，必须记录截断比例和像素L1变化。

随机流为独立NumPy SeedSequence([42,call_index])，不改变训练器RNG；仅训练global分支计数。
local crop实际也被resize到384，不能按尺寸区分；因此以`return_features=True`显式限定global。
local图和no-grad attention提取用原始输入，feature anchor仍将增强global表征对齐原缓存教师。
local置信门控照常使用global logits。验证/推理无增强；无新head、无多模型/多checkpoint融合。
这只是固定的低频增强变体，不是含co-teacher的完整FACT复现，亦不宣称颜色混合必然保标签。

## 配对、预算与裁决

复用已审计MS00普通续训对照，其配置/权重SHA固定在JSON并验证原审计。
formal的model/data/train/loss/evaluation逐项与MS00相同：L05父checkpoint、同split/seed42、
384px弱RRC+Flip、GCEq0.5、anchor2、local权重0.25/置信门0.7、2epoch cosine、
microbatch4×accum256、backboneLR3e-6/headLR1e-4、WD1e-4，无CE warmup。
复用控制不构成新独立seed证据。原生框架固定`f050ecb59e0a885da0b43cdc6c8c316953fb5e7d`，
独立复制、核验484个Git blob；只加内存输入hook，不修改框架源码。
checkpoint旁保存增强源码SHA、配方、激活/裁剪/像素变化计数；resume必须绑定并恢复。

先4次更新smoke、完整14,880张中心验证与原生重载，增强确有激活才自动启动正式两轮。
采用4步是为了覆盖固定seed42的概率序列（前两次不激活、第3/4次激活），不是方法调参。
正式epoch0必须复现L05中心75.2497%/76.2970%，按center raw macro选epoch0/1/2、micro破平局。
任一epoch中心macro或micro比L05低≥2pp止损。
最终固定Flip/mean_probabilities/T1.4/val uniform prior0.60；macro同时比L05与MS00高≥0.30pp，
micro均不下降才晋级。通过后独立解码/图像重载和条件交叉校准（+0.20pp/micro不下降），
单checkpoint测试预测CSV/ZIP及9/9校验；未过门关闭，不派生概率/频带/预算/位置扫描。

失败交付现役L05_T14_P060保底CSV/ZIP并重新校验：
`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/{pred_results.csv,submission.zip}`。
平台最佳66.94797564362783%，70分目标未达；平台上传由用户执行。

## 重放入口与数值验证

```bash
python3 -m pytest tests/test_l05_fourier_mix.py -q
python3 scripts/run_l05_fourier_mix.py --config configs/l05_fourier_mix/fixed.json --phase prepare
python3 scripts/run_l05_fourier_mix.py --config configs/l05_fourier_mix/fixed.json --phase queue
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/audit_l05_fourier_mix.py --config configs/l05_fourier_mix/fixed.json
```

4项测试覆盖独立NumPy full FFT对照（float32变换对float64参照，绝对误差≤1e-6）、
恢复确定性/不改变Torch RNG、hook训练与分支边界、零频谱及单样本边界。
初稿4e-7阈值出现4.69e-7的FFT舍入差；在任何正式图像评估前按float32精度明确为1e-6，
方法配方和晋级阈值未改。最终核验还需实测smoke/正式结果，不把单测当作方法成绩。

输出`outputs/codex/l05_fourier_mix/`，framework从固定Git commit导出。
只在当时GPU空闲时启动，不占卡、不改共享依赖；长训练每30分钟监控，不因观察超时重启。
当前无新训练/平台结果。


## 最终实测与关闭（2026-09-27）

原队列正常完成：smoke 4次更新并完成14,880张验证/重载，formal两轮262/262更新成功，
未触发−2pp止损。正式训练共5,214.1558秒（训练器计时，不含预检和验证）。
两轮center raw均未超过epoch0，因此最终保留**epoch0原始L05权重**：

| 中心视图选模 | macro | micro |
| --- | ---: | ---: |
| epoch0 L05 | 75.2497% | 76.2970% |
| epoch1 | 75.1834% | 76.1828% |
| epoch2 | 75.2374% | 76.2366% |

最终固定Flip/T1.4/prior0.60解码为**75.8265% / 76.6532%**，与现役L05完全相同，
增益为0。它不是FM01新训练权重的提分结果；相对MS00的差也是已有L05与MS00的差。
本固定配方未过门，关闭；不派生频带、概率、增强位置或预算扫描，不生成新测试候选。

训练侧共66,908次global调用，33,401次激活（49.9208%），覆盖133,603个样本前向，
含重复epoch；增强图RGB通道值中2.5966%被截断，平均绝对RGB变化0.107696。
这些计数证明增强实际运行；不代表独立样本数或语义保持已经得到证明。

选中best checkpoint文件SHA256为
`d01620a4b48d149daef38e4bc3ea28e0ff93ac551080d8e60ed88eb1905fd866`。
文件元信息不同导致整体SHA不同，但154个模型状态张量与父checkpoint逐位相同。
独立进程重新生成的14,880张原图和Flip两路logits与冻结L05参照各逐位相同，最大误差0。
NumPy float64独立解码14,880/14,880预测一致，bias最大差7.617448897212498e-7；
best/epoch1/last sidecar、配置/源码/阶段与缓存绑定全部复核，4项测试通过。

精度勘误：旧binding输出`autocast=false`，实际固定脚本依照`train.amp=true`执行CUDA AMP，
缓存float32。原始binding不改写；本轮双分支缓存与已确认AMP协议的参照逐位一致。

新增实现包括RGB Fourier增强模块、固定续训队列、解码审计、JSON配置和4项测试。
确切重放命令见上文；完整逐轮结果、实现绑定、增强计数、独立审计、交付路径及提交校验
命令/输出见[最终记录](../results/l05_fourier_mix_20260927.json)。
本轮交付既有L05_T14_P060保底CSV/ZIP，37,444行及9/9校验通过：

- CSV SHA256：`51e0efe7178528d23993a44351069d2829b0cd3669c195a77d8d879e66776a75`
- ZIP SHA256：`e788f07636b80abb61685335cd8df108803863ea36ffbde8b353f8e8317fcd7f`

无新平台成绩，最佳仍为66.94797564362783%，70分目标未达。
本段Git集成后停在检查点，长任务继续每30分钟检查一次。
