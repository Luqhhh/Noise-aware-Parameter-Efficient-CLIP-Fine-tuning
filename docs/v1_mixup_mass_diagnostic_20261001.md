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
逐行日程效应与val标记留在private输出，聚合指标/逐类暴露/独立核验入Git。当前是实现准备，尚无新机制实测。
纯诊断交付引用现役full v1 SWA可提交包：
`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/submission.zip`及同目录CSV，
ZIP SHA256`1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`，
[既有9项检查](../results/v1_full_swa_platform_20261001/submission_check.log)、
[包核验](../results/v1_full_swa_platform_20261001/artifact_verification.json)。
本段会核对包摘要/ZIP字节，不为凑新包训练；平台仍仅用户报告70.98600576861446%，绑定未知项保留。
完成后立即方案commit/push，main自动pull/合并/重新校验/push，再停在检查点。
