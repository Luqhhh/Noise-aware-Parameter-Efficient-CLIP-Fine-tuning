# PRELIM75 v9 执行记录

计划：`PRELIM75_V9_20260920`

状态：**R0 平台 69.1633%、R1 平台 69.2033%，两者均低于 L1 无 prior 69.2794%；按预注册停止规则保留 L1、关闭 v9 原图回采样配方。**

## 身份与命令

- 实现提交：`ae75f5ddd5a3db663618dab301e36945578c939b`
- L1 CUDA 数值协议修复提交：`38acbdb89829724936b0b035af1bb07131feebbf`
- 配置：`configs/prelim75_v9.yaml`
- 正式队列：`PYTHONPATH=reproducibility/aegis_f1 python3 -u scripts/run_prelim75_v9_queue.py --config configs/prelim75_v9.yaml --execute`
- 修复后完整推理：`PYTHONPATH=outputs/prelim75_v9_20260920/execution_source python3 -u -m aegis_clip.cli.infer_prelim75_v9 --config configs/prelim75_v9.yaml --action infer`

L1 checkpoint、回退 CSV/ZIP/manifest、500 类映射、103,218 行训练清单、101,980 个内容组、10,316 行重叠诊断清单和 24,967 个测试 basename 均通过只读身份检查。checkpoint SHA-256 为 `cd485c7b…fcee`，包含完整 visual、shared head、O3 与 PTA；未创建 optimizer，参数更新数为 0。

## D0、smoke 与诊断

D0 固定抽取 2,048 个训练内容组，仅解码尺寸；1,532 组满足 `source_gain>=1.5`，占 74.8047%，超过 205 门槛。未运行模型、未读取测试图片。

修复后的 smoke 通过：native global tensor 保持逐元素一致，global logits/features/attention 在固定容差内重放，R0/R1 共用 native 框，低 gain 回退保持 native local tensor，模型全部权重和缓冲区前后哈希一致。

首轮 10,316 行重叠工程诊断中，R0 相对 baseline raw micro 为 −0.0291pp，R1 为 −0.1260pp，均未触发 −2pp 止损。该诊断使用了首轮错误的 `cudnn.allow_tf32=False` 数值上下文，只保留为明显退化筛查，不是独立泛化或平台收益证据，也未用于调参。修复后的重复诊断在 4,224/10,316 处因累计 GPU 预算风险主动中止并保留现场；随后只运行规格必需的完整测试重放。

## L1 重放修复

首轮实现误用了训练队列 `gpu_setup()`，关闭了归档通用推理入口默认启用的 cuDNN TF32，导致归档 L1 重放出现 32/24,967 行差异，守卫正确停止且未生成候选。固定 batch A/B 中，首个不一致样本在 `cudnn.allow_tf32=False` 时预测 0254，恢复 `True` 后回到归档标签 0462。失败现场保存在 `outputs/prelim75_v9_20260920_failed_cudnn_tf32/`，记录见 `results/prelim75_v9_tf32_replay_failure_20260920.json`。

修复后显式使用 matmul TF32=False、cuDNN TF32=True、float32 matmul precision=highest。完整 L1 baseline 按 basename 与归档 CSV 比较，**24,967 行全部一致，差异为 0**；守卫未删除或放宽。

## 提交候选

固定阈值使测试集中 18,545/24,967 张图启用新路径，占 74.2780%；该统计在规则冻结后记录，没有用于修改门槛。

| 项目 | R0 / native 224 canvas | R1 / source RGB |
|---|---:|---:|
| 相对归档 L1 改变预测 | 423 | 1,783 |
| 相对 R0 改变预测 | — | 1,609 |
| 预测类别数 | 500 | 500 |
| CSV SHA-256 | `566ce158…1340` | `bc9557c4…f72b` |
| ZIP SHA-256 | `b99de5fb…8f2f` | `d57d0e14…88e5` |
| 平台分数 | **69.1633%** | **69.2033%** |

两份 CSV 均覆盖全部 24,967 个 basename 恰好一次，标签为合法四位编号；ZIP 只含 `pred_results.csv` 且包内外字节一致。内置和独立 `scripts/check_submission.py` 校验均通过。

仓库路径：

- R0：`outputs/prelim75_v9_20260920/R0/submission/submission.zip`
- R1：`outputs/prelim75_v9_20260920/R1/submission/submission.zip`

C 盘桌面副本：

- `PRELIM75_V9_R0_NATIVE224_20260920.zip`
- `PRELIM75_V9_R1_SOURCE_RGB_20260920.zip`

桌面副本与仓库包 SHA-256 分别一致。

## 预算与结论

首轮失败 GPU 动作为 3,286.396 秒；修复 smoke 为 14.616 秒；中止的重复诊断按 600 秒保守计账；成功完整推理及交付按 2,775 秒保守计账。累计保守值 **6,676.012 秒**，低于 7,200 秒上限。所有失败尝试均计入，没有扩容、缩减测试清单或新增候选。

用户按 R0、R1 顺序回填平台百分比 69.1633% 与 69.2033%。R0−L1 为 **−0.1161pp**，R1−L1 为 **−0.0761pp**，R1−R0 为 **+0.0400pp**。两者均未超过 L1 无 prior 69.2794%，因此按预注册规则保留 L1 并关闭 v9；不降低 `source_gain` 门槛，不扫描插值核、尺度、融合比例或其他回采样变体。相对不同来源协议的 72.4677%，R0/R1 分别低 3.3044pp/3.2644pp；距 75% 分别为 5.8367pp/5.7967pp。

用户只提供百分比，精确正确数与实际上传时间保持未知，不作反推。平台上传不是代理自动执行。72.4677% 仍是不同来源协议的 G0+legacy test-batch prior0.9 单列绝对最高。
