# V1_MIXUP_MASS_DIAGNOSTIC_20261001：现役训练Mixup质量占比检查

## 可观察问题与一次固定检查

当前v1图像按`λ*x_i+(1−λ)*x_j`混合，监督按`λ*w_i*q_i+(1−λ)*w_j*q_j`混合。
归一化后来源i的监督占比为`λ*w_i/(λ*w_i+(1−λ)*w_j)`，可靠度不同时可偏离图像占比λ。
现有实现正确执行加权目标；这是目标的行为，不是代码bug，也未证明它损害识别。
假设：对齐图像占比与可靠质量占比，可能减小细粒度类别的训练偏移。先量化当前实际日程，规模小或未关联成块误差就关闭，不先开训。

开工已fetch、检查本地/远端全部分支及main近期历史，从`origin/main@3dac3c3`建立
`codex/v1_mixup_mass_diagnostic_20261001`及独立worktree。
旧`clairvoyant/search-mixup-anchor`为20260923的α/feature-anchor强度扫描，没有本占比对齐算法；不重跑旧点。
A的768 LoRA和B的448强增强仍保留Mixup0.2，本段不重复它们，不使用NPU/远端或子agent。

## 结果前冻结

[配置](../configs/v1_mixup_mass_diagnostic_20261001.json)；只读V1_DETAIL_AUG_PAIR已完成控制臂的1,024步×32=32,768行日程，
原active train_dev、样本顺序、Beta(0.2,0.2)系数和配对全部保持，原目标/可靠度/LS0.1不改。
源schedule SHA256`42cf04b89482d279c8b2a4fd251cecc58d590dfb546b50761aefaae05f231ff4`；
控制主checkpoint SHA256`7ccf90560293dfdb4456e98f6c9a8f543a34a2960dc4bee9f369eaeaf1bdcaa1`。

- 只计不同原v1目标类的混合：监督/图像来源占比绝对差≥0.1为displacement event；两占比相对0.5的符号相反为dominance flip。
- 每混合行归属图像中占比更大的来源类；λ=0.5固定归来源i。只在有实际日程暴露的类内按event频率降序、类索引升序取前1/4；缺失类不补入。
- 在候选产生前冻结该训练侧类群体，报告LR512原父和既有控制末步全量、该组/其余、tail75/小图macro/micro及配对。
  训练暴露不是验证错误，类群体是代理，可靠度不等于真实噪声概率。
- 机制门同时要求event占全部混合≥1%、不同目标dominance flip≥75行、类群体原父错误≥75，
  且该组原父错误率比全量高≥5pp。通过仅支持一次固定占比对齐有限探针的投入复核，不自动训练；未通过关闭这条机制入口。
- 不按结果改阈值、选类或α，不生成新模型/提交包；固定CPU一次日程、一个事件规则、一个类群体，无时长截止。

若后续独立有限探针获支持，应固定同父/同日程/同增强和原监督质量，仅将图像混合占比改为上述可靠质量比例。
已有完整控制可只读复用，不重复其训练；目标识别改善和全量净修正仍须实测，不能把占比差本身叫提分。

## 命令与交付

工作目录`/home/lux1/noise/worktrees/v1_mixup_mass_diagnostic_20261001`。

```bash
env PYTHONPATH=reproducibility/aegis_f1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 scripts/diagnose_v1_mixup_mass.py --config configs/v1_mixup_mass_diagnostic_20261001.json --output outputs/codex/v1_mixup_mass_diagnostic_20261001/diagnostic_r1
env PYTHONPATH=reproducibility/aegis_f1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 scripts/verify_v1_mixup_mass.py --config configs/v1_mixup_mass_diagnostic_20261001.json --report outputs/codex/v1_mixup_mass_diagnostic_20261001/diagnostic_r1/report.json --output outputs/codex/v1_mixup_mass_diagnostic_20261001/diagnostic_r1/independent_verification.json
env PYTHONPATH=reproducibility/aegis_f1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -m pytest reproducibility/aegis_f1/tests/test_v1_mixup_mass.py reproducibility/aegis_f1/tests/test_v1_candidate_error_diagnostic.py -q
```

独立核验用Torch64倒数形式重算比例、Python Counter逐行计数、Fraction有理数排序选类，
核对32,768行、750类聚合和全14,880张val的10份指标/5份配对；不复用诊断的比例/选类函数。
逐行日程效应与val标记留在private输出，聚合指标/逐类暴露/独立核验入Git。
纯诊断交付引用现役full v1 SWA可提交包：
`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/submission.zip`及同目录CSV，
ZIP SHA256`1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`，
[既有9项检查](../results/v1_full_swa_platform_20261001/submission_check.log)、
[包核验](../results/v1_full_swa_platform_20261001/artifact_verification.json)。
本段会核对包摘要/ZIP字节，不为凑新包训练；平台仍仅用户报告70.98600576861446%，绑定未知项保留。
完成后立即方案commit/push，main自动pull/合并/重新校验/push，再停在检查点。

## 固定一次CPU实测与决策

预注册实现/配置/协议在`37513ad`提交并推送后才读暴露结果；阈值和群体未改。
纯CPU诊断0.94秒；[完整报告](../results/v1_mixup_mass_diagnostic_20261001/report.json)、
[750类暴露](../results/v1_mixup_mass_diagnostic_20261001/class_exposure.csv)、
[独立核验](../results/v1_mixup_mass_diagnostic_20261001/independent_verification.json)。

32,768次混合中31,679次目标类不同，占比差≥0.1有3,654次（11.1511%）；主来源翻转1,174次。
全部行平均占比绝对差0.0317163、最大0.3733144，LS0.1后相对不加权混合的平均目标L1差0.0570118。
可靠度范围0.200000003–0.990978003；图像主来源覆盖749类，固定子集缺少第183类，不补采样。
受影响最高四分位是187类，对应3,682张val。群体只按训练暴露构成，未依据验证正确与否挑类。

| 固定val群体 | 张数 | 原父micro / macro (%) | 既有控制micro / macro (%) | 控制相对父修正 / 退化 / 净 |
| --- | ---: | ---: | ---: | ---: |
| 全量 | 14,880 | 76.7608 / 75.7631 | 76.4987 / 75.4819 | 172 / 211 / −39 |
| 高占比差187类 | 3,682 | 79.3862 / 77.1849 | 78.8973 / 76.6001 | 33 / 51 / −18 |
| 其余563类 | 11,198 | 75.8975 / 75.2908 | 75.7099 / 75.1105 | 139 / 160 / −21 |
| tail75 | 1,003 | 66.7996 / 60.3838 | 66.4008 / 59.9027 | 13 / 17 / −4 |
| 短边<224 | 1,848 | 66.8831 / 68.4987 | 65.8550 / 67.5319 | 26 / 45 / −19 |

| 结果前冻结的机制门 | 实测 | 判定 |
| --- | ---: | --- |
| event占比≥1% | 11.1511% | 通过 |
| 不同目标dominance flip≥75次 | 1,174次 | 通过 |
| 目标类群体原父错误≥75张 | 759张 | 通过 |
| 群体原父错误率比全量高≥5pp | −2.62545pp | 未通过 |

群体原父错误率20.6138%，全量23.2392%；占比差规模成立，但此冻结类群体没有集中较高原标签错误率。
决策`close_small_or_unlinked_mixup_mass_effect`：关闭本固定误差关联入口，不启动占比对齐训练、调阈值、另选类或扫描α。
已有控制在该组净−18只是既有续训现象；它同时有增强/优化更新，不能归因于Mixup占比。
此结论不证明所有Mixup对齐干预无效，也不证明可靠度、标签或类别群体具有干净语义。
没有新模型、候选、test预测或平台效果；现役full v1 SWA及A/B/v2既定任务保持。

4项新增数学测试+6项依赖诊断测试共10项通过。独立Torch64重算32,768行最大比例差`2.22e−16`，
事件/翻转/类群体及全14,880张val组成员完全一致；750行类表、10份指标、5份配对、45份源摘要通过。
现役包摘要/ZIP字节与既有9项校验记录已核对，未重新训练或推理。
私有逐行产物目录：
`/home/lux1/noise/worktrees/v1_mixup_mass_diagnostic_20261001/outputs/codex/v1_mixup_mass_diagnostic_20261001/diagnostic_r1`。
