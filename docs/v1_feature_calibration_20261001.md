# V1_FEATURE_CALIBRATION_20261001：特征路径与校准迁移诊断

用户要求验证768/512特征路径及当前阶段校准迁移，并核对历史团队仓库复赛记录。
基线main为`7b3bd46`，独立分支`codex/v1_feature_calibration_20261001`。
状态：`completed_diagnostic`。冻结特征及校准诊断均完成并独立复算；没有新LoRA
候选、预测提交包或平台分。现役full v1 SWA保持70.98600576861446%。
已fetch并检查全部分支、近期main及重叠方向：旧L05类别级vector scaling已关闭，
本段只诊断v1已有uniform bias，不恢复旧缩放/强度搜索或完整训练。

## 结果产生前固定协议

- 数据仅用20260921官方训练侧：133,815 train_dev、14,880 val_dev、750类。
- 特征问题：同一OpenAI官方ViT-B/32的768维投影前CLS与512维投影后特征，
  在同一224 center图像前向提取。缓存512与新前向逐行核对；不下载timm权重或其他模型。
- 固定原v1组外kNN初筛的训练样本，原标签及拟合条件不变；两条特征路径各拟合
  一次20轮cosine teacher（seed42、batch8192、AdamW LR .005/WD .01、LS .1、
  dropout .1、cosine eta_min .0005）。固定末轮，不按验证集挑轮或调超参。
  这是冻结特征诊断，不是768 LoRA候选；不从中宣称完整学生/平台收益。
- 在完整val_dev保留macro/micro、逐图修正/退化、按train_dev频次排序最低75类的macro；
  同时报告两条特征路径的k16近邻准确率和邻居标签一致性。
- 新图像编码/近邻/拟合只用空闲本机CUDA，不抢已有进程；合计上限30分钟，
  前128图计时若外推编码超过15分钟则停止并交付成本/未知项，不扩大预算。
  原始数据只读，输出不覆盖已有产物，失败不自动重启。
- 校准复用WFT448_DEV零更新父模型`baseline.npz`（train_dev v1 SWA，六视角），
  必须重现原v1 SWA全部14,880条raw和已冻结bias预测，并验证父checkpoint/binding。
  CPU固定200次uniform bias迭代，无温度/强度扫描。
- 两折按内容组分配，类内用固定hash顺序平衡；无法分到两折的类别作为仅拟合anchor，
  不计入交叉校准评价。每条评价样本及同内容组均不参与它自己的bias拟合。
  报告原始、全val拟合参照及交叉校准结果；分别报告每折拟合集与留出集增益。
  条件交叉校准仍使用含噪原标签，不是独立来源/平台评估；不拟合测试预测分布。
- 正式full v1模型纳入val_dev，不用其87%训练内分评价泛化。平台校准贡献没有配对
  raw提交回执，保持未知，不用crossfit结果冒充平台分。
- 本段是诊断交付：保留head参数/特征/预测和来源摘要用于复算，不新增可提交候选。
  完成诊断和独立核验后提交、推送、main合并复核，并停在检查点。

## 现役交付引用

现役full v1 SWA平台用户报告70.98600576861446%。包：
`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/submission.zip`。
SHA256 `1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`；
37,444行、9项通过及ZIP内外CSV一致见
`results/v1_full_swa_platform_20261001/submission_check.log`和`artifact_verification.json`。

## 执行入口

工作目录`/home/lux1/noise-worktrees/v1_feature_calibration_20261001`，不修改v1正式流水线。

```bash
PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 scripts/probe_v1_feature_calibration.py calibration --output outputs/codex/v1_feature_calibration_20261001/calibration_r1
PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 -u scripts/probe_v1_feature_calibration.py features --output outputs/codex/v1_feature_calibration_20261001/features --device cuda
```

结果摘要见`results/v1_feature_calibration_20261001/summary.json`，其中记录确切命令、
来源、代码与本机产物摘要。本段新增两个诊断/复算脚本及5项测试，没有修改v1正式流水线。

## 768/512实测及独立核验

本机CUDA耗时807.18秒（13分27秒），完成148,695张官方训练侧图像同一前向的
两种特征提取。投影后512与当前阶段缓存逐元素最大误差0、平均cosine=1；
native forward的768@projection与512输出最大误差0。没有换权重或预处理。

两臂共享由原v1 agreement按相同规则重建的120,128张teacher初筛人口；数量与
原目标报告initial_kept一致，原initial mask未保存，CPU topk在同分时不承诺与
历史GPU选择逐张一致。两臂的实际mask和独立seed42 minibatch顺序相同，
dropout/初始化的维度及随机数消费不同；这是单次诊断，不提供跨seed稳定性保证。
固定20轮末轮head，完整14,880张val_dev、不做校准：

| 路径/评价 | macro | micro | 正确数 | 尾部75类macro |
|---|---:|---:|---:|---:|
| 512 cosine teacher | 58.7772% | 59.8320% | 8,903 | 42.7279% |
| 768 cosine teacher | 60.4555% | 61.4919% | 9,150 | 44.3656% |
| 512 k16近邻 | 57.5648% | 58.6223% | 8,723 | 41.8146% |
| 768 k16近邻 | 58.3487% | 59.4019% | 8,839 | 42.7619% |

768 teacher对512：macro +1.6783pp、micro +1.6599pp、尾部macro +1.6377pp；
修正388、退化141、净+247，改判1,316张。768近邻净+116（修正568、退化452），
micro +0.7796pp；平均原标签邻域agreement从0.368262升到0.376281。
近邻没有head初始化/训练随机性，提供了独立于teacher拟合的方向证据。

独立CPU/NumPy应用保存的两套head权重，29,760条val预测全部一致；CPU对全部
148,695张重算768→512并与原缓存核对，最大误差2.98e-7。
见`feature_report.json`、`independent_verification.json`及本机`features/`产物。

**裁决**：768路径具有可观察的冻结特征正信号，值得一个有成本上限的LoRA配对验证。
本段没有训练768 LoRA，也没有在768路径重新做完整训练侧去噪，不能把+1.66pp
加到现役平台70.9860%，不能声称解释了初赛与复赛全部差距。后续应固定共享监督、
初始化/轨迹和评价，先区分student特征路径收益，再决定是否重建768去噪或完整训练。
本检查点不自动执行上述后续任务。

## 校准实测（已核验）

复用的是已交付WFT448_DEV恢复后的`wft448_dev_20261001_r2/run/baseline.npz`，
其摘要与入库交付审计一致。14,880条原始及冻结bias预测均重现原v1 SWA记录，
raw正确11,303、冻结bias正确11,361；未新增图像推理。

两折内容组不交叉；类0046、0183各仅一张val，作为仅拟合anchor，不计入留出评价。
下表三行使用相同14,878张、748个可评价类别，包含1,001张尾部75类图像
（尾部macro实际覆盖73类，另外2类为anchor）。

| 同一留出人口口径 | macro | micro | 正确数 | 尾部macro |
|---|---:|---:|---:|---:|
| 无bias | 75.1563% | 75.9712% | 11,303 | 61.5019% |
| 原全val拟合bias参照（非留出校准） | 75.8581% | 76.3611% | 11,361 | 66.0909% |
| 两折留出bias | 75.2996% | 75.7696% | 11,273 | 65.2119% |

留出bias对无bias：macro +0.1433pp、micro −0.2016pp、尾部macro +3.7100pp；
修正179、退化209、净−30，改判865张。相对原全val拟合参照净−88。
两折各自拟合集micro增益+0.7147/+0.6442pp，留出micro增益−0.1973/−0.2062pp。
结果显示尾部与总体命中数的取舍，不能把拟合集的+58张视为可迁移的平台收益，
也不能仅凭micro下降断言均衡测试先验下的校准必然无效；平台配对贡献仍未知。

独立NumPy float64重拟合两套200步bias，与原FP32 bias最大差1.42e-6/1.43e-6；
14,878条留出预测全部一致，内容组隔离通过。见
`results/v1_feature_calibration_20261001/calibration_report.json`与`calibration_verification.json`。

首次预检因历史artifact实现摘要与当前源码不同而拒绝，未做拟合，原输出目录保留。
修复后按每个历史asset自身的sidecar及payload验证原摘要，同时要求当前数据、
映射、官方权重、特征manifest和有效recipe全部一致，仅显式保留历史实现版本差异；
再以全量预测重放验证已有logits来源。没有放宽跨阶段或checksum检查。

## 历史团队仓库的复赛记录核对

核对`WRw5w/aic_new`公开commit
`fdb42ee9add3c06650bcdf4c256df32a63b1f5d0`：仅main分支、5个公开commit、
81份文本记录及2个Release元信息。公开SCORECARD所有条目均标`initial`，
未找到绑定当前750类/20260921数据的复赛实测成绩。9月26日的两个commit仅更新
共享插件和provenance换行；8月Release为历史交接。

记忆`reach-82-divmix-campaign.md`第70行一度猜测78.5477来自后续阶段，但第77行
明确记录2026-06-30用户确认仍在初赛，否定了阶段/数据切换解释。不能取前一句
猜测作为复赛78分证据。交接文档的复赛段是未来策略建议；其中旧1500类说明
不是我们当前750类数据的权威版本。

边界：未检查私人后续工作；未下载约503MB的8月历史档案。因此结论是
“所检查公开材料没有可核验的复赛分数”，不声称他们从未做过复赛。
逐文件摘要和commit/Release日期见`results/v1_feature_calibration_20261001/external_record_audit.json`。
另检查文档链接的旧公开仓库`WRw5w/lihao`：main最新commit为
`69fa10444812dd581219bb21657c83251a9568d3`（2026-06-20），仓库最后push为6月29日；
完整364条tree未出现当前复赛日期/方案记录。README亦明确为初赛。

独立复算命令（`--features`在特征结果全部落盘后执行）：

```bash
PYTHONPATH=reproducibility/aegis_f1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 scripts/verify_v1_feature_calibration.py --root outputs/codex/v1_feature_calibration_20261001 --features
PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 -m pytest tests/test_v1_feature_calibration_probe.py -q
```
