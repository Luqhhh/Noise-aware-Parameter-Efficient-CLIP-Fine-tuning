# PRELIM75 v8 执行记录（2026-09-19）

计划：`PRELIM75_V8_20260919`

状态：**B0、B1 均已生成提交就绪候选并通过独立提交检查；等待真实平台回填，不依据离线分布选择胜者。**

## 执行身份

- 固定审阅基线：`3f83d69f9a4496976e7d50beecd7e0baa0400d4d`
- 正式队列执行快照提交：`18111e51b9d2e856b57d2e39bff47c1911cf5f00`
- 协议描述符稳定化修复：`ad547bddf77779c368050df80329969f42d608fa`
- 配置：`configs/prelim75_v8.yaml`
- 原始队列：`PYTHONPATH=reproducibility/aegis_f1 python3 -u scripts/run_prelim75_v8_queue.py --config configs/prelim75_v8.yaml --execute`
- 修复交付：`PYTHONPATH=outputs/prelim75_v8_20260919/execution_source python3 -u scripts/deliver_prelim75_v8_protocol_repair.py --config configs/prelim75_v8.yaml --candidate B0`（B1 同命令替换候选名）

来源审计确认 103,218 个唯一训练路径、101,980 个内容 SHA-256 组；fit scope 为 `training_overlap_calibration`，`test_data_used=false`，`upstream_provenance_complete=false`。没有创建主干 optimizer，主干更新数为 0。

## 冻结缓存与拟合

| 项目 | B0 / G0 | B1 / L1 |
|---|---:|---:|
| 训练源缓存 SHA-256 | `88b0046f…50ee` | `0e17b25c…40ef` |
| 最大 `abs(logsumexp(s))` | 5.409e-7 | 5.886e-7 |
| 首批包装最大差 | 1.907e-6 | 1.907e-6 |
| 首批 argmax 分歧 | 0 | 0 |
| L-BFGS-B 迭代 / 函数评估 | 41 / 45 | 38 / 43 |
| 最终目标差 | -0.01162834 | -0.01003329 |
| 投影梯度无穷范数 | 2.693e-9 | 8.268e-9 |
| bias 范围 | [-1.3862944, 1.3862944] | [-1.2339126, 1.3862944] |
| bias SHA-256 | `67baf527…b85` | `1b6f0283…d47` |

两者均满足求解成功、目标不增加、投影梯度不超过 `1e-6`、边界与来源绑定检查。目标值是相对 `b=0` 的目标差，不是绝对 CE 或准确率证据。

## 协议守卫事件

原始 B0/B1 交付在读取测试样本前均被 `inference_protocol_sha256` 守卫拒绝。根因是生产者把 `repr(preprocess)` 写入协议，其中 `_convert_image_to_rgb` 携带进程本地内存地址；相同变换在新进程产生不同字符串。

修复没有修改原执行快照、缓存、bias 或来源资产。兼容入口要求删除函数地址后生产/消费描述符完全相等，且原始差异字段必须恰为 `preprocess`；两候选均通过，规范化协议 SHA-256 为 `7977d9d8567f0b5dcc81999c111d06b678e1173475a9fc66d94ab70fedd6ee9c`。失败尝试分别耗时 13.639 秒和 12.082 秒，已保留在原队列账本。

## 提交候选

| 项目 | B0 / CAL_G0_TRAINONLY | B1 / CAL_L1_TRAINONLY |
|---|---:|---:|
| 匹配无 prior 平台对照 | 69.2274% | 69.2794% |
| 相对无 prior 改变预测 | 2,424 | 2,311 |
| 预测类别数 | 500 | 500 |
| CSV SHA-256 | `15474044…6ca2` | `16309ff0…3d7c` |
| ZIP SHA-256 | `77fcff46…fa7a` | `623d18c5…764c` |
| 平台分数 | 待回填 | 待回填 |

B0 与 B1 之间有 1,648 个预测不同。两份 CSV 均覆盖 24,967 个测试文件恰好一次，标签为合法四位编号；ZIP 只含 `pred_results.csv`，包内外字节一致。独立重跑 `scripts/check_submission.py` 均全部通过。

候选路径：

- B0：`outputs/prelim75_v8_20260919/B0/submission/submission.zip`
- B1：`outputs/prelim75_v8_20260919/B1/submission/submission.zip`

## 预算与结论

原队列记录 GPU 动作 2,641.445 秒、CPU 拟合 208.095 秒；两次成功测试推理循环分别为 293 秒与 298 秒。包含修复交付启动和来源复验后的 GPU 动作保守上界为 3,600 秒，低于 7,200 秒预算；CPU 拟合低于 3,600 秒预算。

当前没有平台证据证明训练源偏置提升。等待 B0、B1 两次真实平台回填后，分别计算 B0−69.2274、B1−69.2794、B1−B0，并仅按预注册止损规则保留结果；72.4677% 仍是不同来源协议的已报告绝对最高。
