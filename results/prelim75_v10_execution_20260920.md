# PRELIM75 v10 S0/S1 执行记录（2026-09-20）

计划 ID：`PRELIM75_V10_SAM_20260920`

状态：S0 普通 AdamW 与 S1 标准非自适应 SAM 均已完成固定训练、重叠诊断、测试推理和提交校验。S0 平台为 `69.3195%`；S1 平台结果待用户回填，因此 v10 尚未执行最终胜负与关闭规则。

## 执行来源和命令

- 固定配置：`configs/prelim75_v10_sam.yaml`。
- 实现提交：`cddeaac827773f093fd188a33ca5bc14b9b04191`；S1 运行时干净源码清单登记提交为 `0838880b1fd406f32d29f680326669b4a8108840`，两者之间只有执行记录文档变化，训练实现文件哈希不变。
- S0 原正式队列：`PYTHONPATH=reproducibility/aegis_f1 python3 -u scripts/run_prelim75_v10_sam_queue.py --config configs/prelim75_v10_sam.yaml --execute`。
- 用户要求在 S0 结束后暂停，故在完整 S0 checkpoint 边界结束原自动队列；用户随后明确要求继续 S1。
- S1 训练：`PYTHONPATH=reproducibility/aegis_f1 python3 -u -m aegis_clip.cli.train_prelim75_v10_sam --config configs/prelim75_v10_sam.yaml --action train --candidate S1`。
- S1 诊断与交付使用同一入口，分别令 `--action evaluate` / `--action deliver --candidate S1`。
- 未自动上传平台，未使用 prior、v9 source recrop、测试拟合或多 checkpoint 融合。

## 门禁和共同控制

资产 preflight、A0、真实 AdamW smoke 均通过。A0 的实际 FP32 扰动半径为 `0.04999999329447746`，原损失 `0.047027505934238434`，扰动损失 `0.23755624890327454`；参数逐位恢复、冻结状态、buffer 与 RNG 回放均通过。

S0/S1 common-control 通过：首批索引、路径、图像、Flip/尺度、目标、权重、anchor reference、固定框、global/local logits/features 的 12 项哈希全部一致；首遍完整 loss 差为 0，初始三组 LR 完全相同。两候选都从 L1 独立加载并各完成 9,678 次更新。

## 训练结果

| 候选 | 优化方法 | 训练秒数 | checkpoint SHA-256 |
|---|---|---:|---|
| S0 | fresh AdamW | 4730.7367 | `923e358c68d29aa60f7d92daef3cd97f117ec8732cbb441f97094368d469b143` |
| S1 | 标准 SAM + 同一 AdamW，rho=0.05 | 7031.5225 | `9dd2747b36b0b91958ac07f10d17c1252fb7d9b345c2f33c339abb228710ff6a` |

S1 的 9,678 步全部执行两遍完整有效 batch；`rng_replay_exact_steps=9678`、`sam_nonincreasing_steps=0`。实际扰动半径最小/最大为 `0.04999999329447746` / `0.05000000819563866`，训练结束 scheduler multiplier 为 `0.01`。S1 三轮 base loss 为 `0.1354894744`、`0.1356617210`、`0.1326693694`；扰动 loss 为 `0.5098190769`、`0.4454785575`、`0.4113552295`。这些训练损失不能直接与 S0 的单遍目标均值作平台收益判断。

按实际动作计时及交付文件时间戳保守合计约 `12817.13` GPU 动作秒，包含首次 fail-closed A0、修复后 A0、smoke、两候选训练/诊断/推理，低于 `28800` 秒上限。

## 重叠诊断

| 候选 | raw micro | clean-core micro | 相对 L1 raw | 相对 L1 clean-core | 工程止损 |
|---|---:|---:|---:|---:|---|
| S0 | 84.344709% | 97.121811% | +0.126022pp | +0.136405pp | false |
| S1 | 83.472276% | 96.180606% | −0.746411pp | −0.804800pp | false |

S1 相对 S0 的 raw/clean-core 分别低 `0.872433pp` / `0.941205pp`，但没有触发固定的 `−2pp` 止损，故按预注册继续交付。该诊断与训练数据重叠，不是独立泛化证据，不能替代平台结果。

## 提交产物

两候选均为单 checkpoint、原十视图、温度 1.5、无 prior；均覆盖全部 24,967 个 basename，使用四位合法标签，ZIP 只含 `pred_results.csv`，包内外 CSV 字节一致，并通过 `scripts/check_submission.py` 全部检查。S0/S1 有 2,382 行预测不同。

| 候选 | CSV SHA-256 | ZIP SHA-256 | 桌面包 | 平台 |
|---|---|---|---|---:|
| S0 | `424258a941c2c10b36c9626a85db6b6d7b64d2ec3876193d73c05c97b64d15c8` | `98c2a51bb720ec127d7f0d52cbba200a250460917a3dcd80ba2aca65c7fb17a6` | `C:\Users\lqh22\Desktop\PRELIM75_V10_SAM_S0_20260920.zip` | 69.3195% |
| S1 | `7169336700da80c54123477b6b38504abbd8c83d8bdb345e7030b836149aa50a` | `b92af8bfdbead9762b89f81e0bd54701e66885337777c0c0ab8c9b32823b6d01` | `C:\Users\lqh22\Desktop\PRELIM75_V10_SAM_S1_20260920.zip` | 待回填 |

两个桌面副本均已与仓库源 ZIP 重新计算 SHA-256 并确认一致。S0 相对 L1 69.2794% 为 `+0.0401pp`，是当前无 prior 新高，但未达到 `+0.30pp` 工程投入门槛。

## 待平台反馈的固定决策

- S1 不高于 S0：保留 S0，关闭本轮 SAM，并禁止自动派生 SANER/NCSAM。
- S1 高于 S0：S1 同时超过 L1；保留 S1。若相对 `max(L1,S0)` 的提升低于 `0.30pp`，只登记小收益并关闭；达到门槛也只允许另行预注册后续研究。
- 平台精确正确数与上传时间没有从百分比反推，保持空白。

另一来源协议的 G0 + legacy test-batch prior0.9 `72.4677%` 继续单列，不进入本轮无 prior 对照。
