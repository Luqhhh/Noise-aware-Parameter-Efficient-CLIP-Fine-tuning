# V2_SIXVIEW_LOGIT_SUM_20261003

用户于2026-10-03明确要求执行现役v2第五轮raw、448/512/576各原图与水平翻转、
六份logits直接求和、固定200次/强度1均衡bias，并同时交付raw/bias两包。
晋级依据为用户回填平台分严格超过现役74.6688387992735%；预测变化与类别均衡程度不是准确率证据。

## 固定实现与核验

从最新origin/main `a33af94`创建独立分支`codex/v2_sixview_logit_sum_20261003`及同名worktree。
已fetch、检查所有分支与近期main；旧六视图概率平均实现和现役产物保持原样。
单checkpoint SHA256为`3b9fcce3a4a2a113f2eb2ad3a2028c7f224fd1b11f1bdaadd6641ee1ccc85138`，
仍用本机已核验完整下载包，不连接已关机远端。

固定顺序`scale448/scale448_flip/scale512/scale512_flip/scale576/scale576_flip`；
沿用原bicubic短边round(size×1.14)、中心裁剪及裁剪后水平翻转，CLIP归一化、BF16前向、batch16。
每视图logits转float32后直接相加，不除以6，不经过softmax再合并。
使用既有`fit_test_uniform_bias`，200次、damping1、strength1、目标1/750；
最终加bias的argmax采用float64加法，保留既有v2的高精度决策口径。
无训练、EMA/SWA、参数或温度搜索，无平台上传。

每视图完整logits落盘，同次前向另存原概率平均结果作为数值控制，不为该控制生成新平台候选。
对已绑定现役概率缓存要求最大概率误差≤0.01、argmax变化≤0.1%；缓存SHA在执行前绑定。
冷加载同一raw权重，前64张全部六视图logits须逐元素相等。
独立NumPy/float64复算六视图求和、概率控制、200次bias、全部37,444张预测及CSV/ZIP字节；
求和逐次复现float32舍入，并核对相对精确float64总和的标准浮点误差界，不误将正常舍入判成错误。
bias最大绝对差≤2e-5且校正决策全量一致。raw/bias两包分别执行9项提交检查。
上述失败保留原始输出，不自动改阈值、改协议或隐式重启。

仅一轮完整六视图推理加64张冷加载检查；同机前段实测约20.6分钟为成本参考，
不因卡忙抢占任务。运行前已在沙箱外只读核对RTX4070 Laptop CUDA可用、无计算进程。

在独立worktree执行：

```bash
PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python3 -u scripts/run_v2_sixview_logit_sum.py \
  --config configs/v2_sixview_logit_sum_20261003.json \
  --output outputs/codex/v2_sixview_logit_sum_20261003/candidate
```

## v2＋768的队列判断

用户已明确指投影前768维特征/分类头。该方向已有独立分支
`origin/codex/v2_preprojection768_pair_20261002`，已推送实测commit `61ecddb`。
原s3父上固定512次更新的配对：768对512控制修正37/退化33/净+4，
对共同父修正181/退化198/净−17，共同错误组对控制修正8/退化4/净+4。
原推进门未过，因此**本队列不重复加入768短续训或扩full**。
这不等于否定从官方初始化全程使用768；现有证据尚不足以支持该完整训练投入。
原分支交付状态由原工作目录维护，本段仅引用原始诊断，不宣称其余交付完成。
机器可读证据与原始commit定位见[队列决定](../results/v2_sixview_logit_sum_20261003/v2_768_queue_decision.json)。

## 实际交付与核验

首次运行在448视图阶段主动中断：补充合成数值检查发现独立校验错误要求float32累加与
不舍入的float64总和逐元素相等。6×100,000个BF16合成数出现4个正常舍入差，最大1.1921e-7。
修正仅涉及独立验证的IEEE加法重放，不改变模型、解码、bias或预测晋级门；补充该边界测试。
首次部分缓存、日志和实现快照保留在`engineering_attempt_01`，没有完整候选，正式运行显式重开。

正式同配方推理、冷加载、拟合及双包生成1,202.720959秒。首次独立复算在概率控制项停止：
NumPy float64与GPU float32 softmax平均有10/28,083,000项超过代码中的2e-7绝对比较值，
最大差2.604900100067198e-7。已修正为明确计入C+V+2次float32归约的相对误差界
`gamma_n=n*eps/(1-n*eps)`（本例n=758），该控制并非候选打分。
原始源码全部按运行前binding的SHA保存，原失败report和日志保留；
新版验证器仅重放原缓存，未重推理、未改变候选字节、bias容差、原生重放门或平台晋级门。
修订证据见[verification_revision.json](../results/v2_sixview_logit_sum_20261003/verification_revision.json)。

最终33项CPU回归2.76秒通过；最终CPU独立复算48.019551秒通过：

- 现役概率平均缓存全部28,083,000项逐元素相等，原生raw决策变化0；
- 同权重冷加载前64张全部六视图logits逐元素相等；
- 六份完整logits相加逐元素复算通过，852项有正常float32舍入、最大2.9802e-7；
- bias最大绝对差5.4885695e-6，低于冻结2e-5；37,444张float64校正决策全量一致；
- raw/bias两包源文件与实际Windows文件各9项通过，CSV/ZIP字节一致。

| 观察量（没有测试真值） | 实测 |
|---|---:|
| sum raw相对概率平均raw改判 | 514张 |
| 固定bias相对sum raw改判 | 5,804张 |
| sum bias相对74.6688%现役bias改判 | 3,656张 |
| sum bias硬类别数量min/max | 44 / 56 |
| sum bias对均匀先验chi2 | 33.205961 |
| 现役概率平均bias同口径min/max、chi2 | 4 / 95、1,507.527615 |

上述只确认归约改变了固定bias的作用，不证明准确率提升。平台分未知，
现役保留74.6688387992735%，不自动扫描温度、强度或添加768训练。
完整结果见[report](../results/v2_sixview_logit_sum_20261003/report.json)、
[独立复算](../results/v2_sixview_logit_sum_20261003/independent_verification.json)、
[预测比较](../results/v2_sixview_logit_sum_20261003/prediction_summary.json)。

Windows根目录`C:\Users\lqh22\Downloads\v2_continuation_20261001`：

| 包 | 子路径 | ZIP SHA256 |
|---|---|---|
| 新bias候选 | `submission_bias_sixview_logit_sum\submission.zip` | `51e372c390f65702ef15af82b7de5a6cc7a382cfaf6f54b7c44be418517aec49` |
| 同缓存raw对照 | `submission_raw_sixview_logit_sum\submission.zip` | `290ce80216c2b19006b581d5671aabe834924749883b36f75c439ee34cdd2a33` |

每包另有CSV、manifest、独立复算和Windows检查日志，
`sixview_logit_sum_delivery/`保留总报告、配置、binding与校验修订记录。
[Windows复制回执](../results/v2_sixview_logit_sum_20261003/windows_copy_receipt.json)记录每个文件的路径、字节数与SHA。

本次既有缓存的独立验证复现命令（在本方案worktree中）：

```bash
PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python3 scripts/verify_v2_sixview_logit_sum.py \
  --config configs/v2_sixview_logit_sum_20261003.json \
  --output outputs/codex/v2_sixview_logit_sum_20261003/candidate \
  --source-snapshot outputs/codex/v2_sixview_logit_sum_20261003/candidate/verification_source_snapshot \
  --report /tmp/v2_sixview_logit_sum_recheck.json
```

新包已完成交付，不自动上传平台；在该检查点暂停。现役仍为
`C:\Users\lqh22\Downloads\v2_continuation_20261001\submission_bias_sixview\submission.zip`，
平台74.6688387992735%，ZIP SHA `7164b26d9662a8c0e78c9c0636393efe0927739cb84df8dc88ed33e35f47659a`；
既有九项及字节核验见[平台反馈](v2_sixview_platform_20261003.md)。
