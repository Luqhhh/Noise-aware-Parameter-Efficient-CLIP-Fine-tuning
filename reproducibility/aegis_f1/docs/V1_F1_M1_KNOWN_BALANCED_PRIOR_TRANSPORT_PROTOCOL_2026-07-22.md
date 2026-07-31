# V1：F1+M1 已知均衡目标先验传输

日期：2026-07-22

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-22
- Verification Status: VERIFIED
- Version Label: V1_closed_v1

## 状态

**CLOSED / NO TEST INFERENCE。** 预注册提交为 `ff64008`；float32 均匀边际总质量容差的纯数值修复提交为 `6869825`。最终自动 gate 未通过，不生成测试包、不扫描参数。

## 研究问题与高收益假设

比赛说明书明确写明测试集“类别分布均衡、经人工精确标注”。当前平台最佳 F1+M1 为 `63.3276%`，但训练监督仍受到类别相关净化率差异影响；N11 也观察到训练保留数与类别 clean-core accuracy 的明显关联。因此检验一个不同于局部视图或损失扫描的宏观假设：单检查点 M1 的后验包含可用排序信息，但输出边际仍残留训练期类别偏差；把同一批次的后验通过固定的已知均衡目标边际做熵正则最优传输，可能恢复困难类而不更新模型。

该假设建立在公开目标先验上，而不是测试预测分布上。Sinkhorn 只用固定 logits 和 `N/500` 的均匀软边际，不估计标签、不反推目标先验、不做参数更新。理论背景仅用于方法来源：Sinkhorn 距离见 [Cuturi, NeurIPS 2013](https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)；已知类别比例约束的标签分配见 [Tai et al., ICML 2021](https://proceedings.mlr.press/v139/tai21a.html)。这些论文不构成比赛合规结论。

## 与既有负结果的区别

L1 使用训练集 cross-fitted OOF 后验估计模型源先验，再做逐样本 Bayes 校正；它明确没有运行 Sinkhorn，clean-core micro 为 `-0.0277pp`，因此已关闭。V1 不估计源先验，而是对固定 M1 后验施加公开、预先已知的目标均衡软边际。两者因果干预不同，但 L1 证明“输出更均衡”本身不是收益证据，所以 V1 必须在 F1/A2 两个检查点上同时获得准确性提升。

## 冻结输入

| 输入 | 形状 | SHA-256 | 说明 |
|---|---:|---|---|
| F1+M1 validation logits | `10,316×500` | `5f927bc9740ec5ce1725a7cfab07fbdc40f3e3dda5213ce59a419092edbf614c` | 当前平台最佳检查点的固定 M1 缓存 |
| A2+M1 validation logits | `10,322×500` | `cea5f7651bf0d24828b7da8b252cb47b52288fa2cb7638a1c6d32359c0c92698` | 跨模型复现检查点的固定 M1 缓存 |

F1 validation 每类 18–23 张，标签计数 CV `0.03006`；A2 validation 每类 20–21 张，标签计数 CV `0.02319`。二者均覆盖 500 类。验证标签只在公式、输入哈希、参数与门槛提交后用于评分。

## 唯一固定方法

1. 输入使用 M1 已融合的 log-probability logits，不改变 global/local 1:1 概率均值；
2. 目标列边际固定为每类 `N/500`，允许小数，避免人为指定哪些类别多一张；
3. `temperature=1.0`，log-space Sinkhorn 固定 100 次；
4. 候选预测为最终软分配的逐行 argmax；不做精确硬配额、类别排序或二次匹配；
5. F1 与 A2 完全相同，不扫描 temperature、迭代数、传输强度、目标先验、融合权重或阈值；
6. 全流程 CPU-only，不读取 test，不训练，不产生 submission。

## 预注册双检查点门槛

只有全部条件同时通过，V1 才能进入“合规复核”，仍不能自动生成测试包。

| 门槛 | F1+M1 | A2+M1 |
|---|---:|---:|
| clean-core micro 变化 | `>= +0.20pp` | `>= +0.10pp` |
| trusted macro 变化 | `>= -0.05pp` | `>= -0.05pp` |
| raw micro 变化 | `>= -0.10pp` | `>= -0.10pp` |
| hard prediction count CV 相对下降 | `>=20%` | `>=20%` |
| 空预测类 | `0` | `0` |
| Sinkhorn 最大行误差 | `<=1e-4` | `<=1e-4` |
| Sinkhorn 最大列误差 | `<=0.05` | `<=0.05` |

同时记录 raw/clean-core 的 wrong→correct、correct→wrong 和净正确数，但不把它们作为事后替代门槛。任一失败即关闭 V1：不扫描参数、不运行 test、不生成提交包。

## 合规边界与风险

- 仅使用官方训练划分产生的两个固定 validation 缓存和比赛说明书公开的均衡目标先验；
- 一个冻结 OpenAI CLIP ViT-B/32 checkpoint、一个固定 M1 推理过程、无外部数据、无第二模型；
- 不使用测试标签、测试伪标签或测试分布估计，也不更新参数；
- 但 Sinkhorn 对整批测试预测进行联合归一化，属于 batch-transductive 固定后处理。赛事网页允许“单模型或单次推理过程”，是否覆盖这种联合后处理存在解释风险。即使本地门控通过，也必须先向队长披露并建议向组委会确认，未经明确批准不得生成或上传测试包。

## 固定执行命令

工作目录：`/home/x28639/projects/AegisCLIP-F6-A2LoRA`

```bash
env PYTHONPATH=/home/x28639/projects/AegisCLIP-F6-A2LoRA \
  /home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
  -m aegis_clip.cli.evaluate_balanced_transport_gate \
  --f1-validation-logits outputs/M1_F1_TRANSFER/seed42/f1_attention_local_global.pt \
  --a2-validation-logits outputs/M1_A2_TRANSFER/seed42/a2_attention_local_global.pt \
  --output-dir outputs/V1_F1_M1_KNOWN_BALANCED_PRIOR_TRANSPORT/seed42
```

## 预期产物

- `f1_m1/evaluation.json` 与 `predictions.pt`；
- `a2_m1/evaluation.json` 与 `predictions.pt`；
- `gate.json`：包含所有门槛、实际值、失败项与唯一决策；
- 成功标准：进程退出 0、输入哈希一致、数值误差门槛通过、`gate.passed=true`。

## 执行异常与修复边界

首次执行完成 F1 后，在读取 A2 结果前因目标边际总质量检查中止：float32 中 `20.644×500` 的双精度观测和为 `10,321.99955`，与 `10,322` 相差约 `4.5e-4`，超过旧固定 `1e-4` 容差。该异常发生在输入合法性检查，不是模型结果。

修复只把总质量检查容差改为 `max(1e-4, 4*eps_float32*N)`，仍拒绝真正不一致的边际；temperature、迭代数、目标先验、预测规则和所有准确性门槛均未改变。新增大尺寸均匀边际回归测试后完整测试为 `206 passed`。首次 F1 中间 JSON 未读取并删除，残留二进制在正式重跑时由临时文件原子覆盖。

## 严格结果

| 指标 | F1+M1 变化 | F1 门槛 | A2+M1 变化 | A2 门槛 |
|---|---:|---:|---:|---:|
| clean-core micro | **+0.2299pp** | `>=+0.20pp` ✅ | **+0.0693pp** | `>=+0.10pp` ❌ |
| trusted macro | **+0.3688pp** | `>=-0.05pp` ✅ | **+0.1406pp** | `>=-0.05pp` ✅ |
| raw micro | **+0.3005pp** | `>=-0.10pp` ✅ | **-0.0775pp** | `>=-0.10pp` ✅ |
| prediction count CV 相对下降 | **50.03%** | `>=20%` ✅ | **51.07%** | `>=20%` ✅ |
| 空预测类 | 0 | 0 ✅ | 0 | 0 ✅ |
| Sinkhorn 最大行误差 | `1.7483e-3` | `<=1e-4` ❌ | `6.6352e-4` | `<=1e-4` ❌ |
| Sinkhorn 最大列误差 | `1.1444e-5` | `<=0.05` ✅ | `1.1444e-5` | `<=0.05` ✅ |

F1 raw 在 10,316 张中改变 866 个预测，wrong→correct/correct→wrong 为 `207/176`，净增 31；clean-core 为 `131/114`，净增 17。A2 raw 为 `199/207`，净减 8；clean-core 为 `121/116`，净增 5。

双侧 exact McNemar：F1 raw `p=0.1252`、F1 clean-core `p=0.3067`、A2 raw `p=0.7283`、A2 clean-core `p=0.7951`。四项均不能排除小样本翻转波动；统计结果只用于解释效应强度，不替代预注册 gate。

## 自动判定与复现

`gate.passed=false`，失败项固定为：

1. F1 最大行边际误差未达数值门槛；
2. A2 clean-core micro 增益未达 `+0.10pp`；
3. A2 最大行边际误差未达数值门槛。

同一固定命令在修复后连续执行两次，五个产物哈希完全一致：

| 产物 | SHA-256 |
|---|---|
| `f1_m1/evaluation.json` | `3530bb6d53ece8676bd3f32ea9b469a6407c988201e33741bd2f3103384edeae` |
| `f1_m1/predictions.pt` | `ab8b0b2125537a00b209cb1d3ca0ad5cb6a1fee6f668321d2dbb7fad8d257ac6` |
| `a2_m1/evaluation.json` | `99a36e399fcc7ed8f060c7a35f9f20d6f677e97c4ef371d20406e381257caf19` |
| `a2_m1/predictions.pt` | `65f94e6eeba34d6a639b3de7d5d64edda3d6aeeedaad7573d0bb7a9fc0ece676` |
| `gate.json` | `7f15b5ed5a556a296a2cd42363f7d29a2a2fa9adaa7931e8326e60598eb4b610` |

## Validation Report

- Overall Confidence: **SOLID for deterministic measurement; RED_FLAG for promotion**
- Reproducibility: **REPRODUCIBLE（exact hashes）**
- Fallacy Scan Coverage: **11/11**

| 谬误 | 判定 | 本实验处理 |
|---|---|---|
| Simpson's paradox | CAUTION | F1 正向、A2 混合，禁止只报告 F1 聚合结论 |
| Ecological fallacy | NOTE | 类别边际只约束批次，不外推为每个样本正确 |
| Berkson's paradox | CAUTION | clean-core 是筛选子集，因此同时报告 raw/trusted |
| Collider bias | CAUTION | clean probability 与难度/预测相关，不能把 clean-core 当独立真值 |
| Base-rate neglect | PASS | 显式使用公开均衡目标并报告本地标签边际 |
| Regression to the mean | N/A | 未按极端表现选择样本或检查点 |
| Survivorship bias | CAUTION | 净化会改变训练保留率，跨 F1/A2 对照但不能完全消除 |
| Look-elsewhere effect | PASS | 预注册两个检查点与合取门槛，无选择性报告 |
| Garden of forking paths | PASS | 无参数扫描；唯一修复为结果前触发的数值合法性检查 |
| Correlation ≠ causation | CAUTION | paired logits 支持本地干预效应，不支持“平台必提升” |
| Reverse causality | N/A | 固定后处理，没有方向性观测关联主张 |

## 最终结论

V1 确认“公开均衡先验约束”能显著降低两模型的输出数量偏差，并在 F1 上得到一致正向指标；但 A2 的 clean-core 增益不足且 raw 为负，四个 paired 检验均不显著。按照预注册规则，**关闭 V1，不读取 test、不生成 submission、不把 F1 子结果单独晋级**。

