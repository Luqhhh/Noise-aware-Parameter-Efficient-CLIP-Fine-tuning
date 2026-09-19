# PRELIM75 v7 执行与平台闭环

计划 ID：`PRELIM75_V7_20260918`

状态：**已闭环，保留 L1 为无 prior 胜者；未达到投入回报门槛，不追加 G2。**

## 平台结果

| 候选 | 平台准确率 | 说明 |
|---|---:|---|
| F1（v5） | 69.1393% | 上游无 prior 参考 |
| G0（v6） | 69.2274% | 本轮父模型 |
| L0 | 69.0632% | 18-epoch cosine horizon |
| L1 | **69.2794%** | 3-epoch cosine horizon；本轮与无 prior 最高 |
| G0+prior0.9 侧包 | **72.4677%** | legacy test-batch balanced-prior；不同协议的绝对最高 |
| 75% 目标 | 75.0% | 未达到 |

## 判定

- L1 − G0 = **+0.0520pp**
- L1 − L0 = **+0.2162pp**
- L1 − F1 = **+0.1401pp**
- L1 − max(G0,L0) = **+0.0520pp**，未达到预注册的 0.30pp 投入回报门槛
- 按预注册规则保留 **L1** 为正式无 prior 胜者，但不据此追加 G2、续训或 horizon/floor/LR/epoch 扫描
- G0+prior0.9 仍为绝对最高，领先 L1 **3.1883pp**，距 75% **2.5323pp**；它不属于 L0/L1 无 prior 对照
- L0/L1 的重叠 raw 诊断相对 G0 分别为 +0.533152pp/+0.339276pp，而平台差值分别为 −0.1642pp/+0.0520pp；重叠诊断不替代平台结果
- 精确正确数与实际上传时间未提供，保持 `null`

## 执行与验证

- 精确队列命令：`PYTHONPATH=reproducibility/aegis_f1 python3 -u scripts/run_prelim75_v7_queue.py --config configs/prelim75_v7.yaml --execute`
- L0/L1 各训练 3 epoch，sample epoch 为 16/17/18；总计 9,678 次更新
- L0 三轮末 scheduler multiplier：0.9924798377 / 0.9701478473 / 0.9336825749
- L1 三轮末 scheduler multiplier：0.7525 / 0.2575 / 0.01
- common-control 通过：首 batch、Flip/尺度、首步 LR、global logits/features 与首更新 probes 对齐；只有 horizon 字段不同
- 队列总耗时 9,148.50s，低于 28,800s GPU 预算；两组均未触发工程止损
- L0 重叠诊断 raw/clean-core micro：84.412563% / 97.108173%
- L1 重叠诊断 raw/clean-core micro：84.218687% / 96.985406%
- 两个提交均为单 checkpoint、无 prior、24,967 行、覆盖 500 个预测类；提交检查退出码均为 0，ZIP 内外 CSV 字节一致

## 资产

- L0 checkpoint SHA-256：`8a55855adb33d77ae7b551dd44223f7d2e8b39c8ac4377af6d75ca13b3234e90`
- L0 CSV SHA-256：`36c5628dc42f545ec82dfc9cd6823051b0fc45dd11c2ec407f4d69f5f8b0373b`
- L0 ZIP SHA-256：`d79ca3759a29319b0abce7d9a56169e77958318fab68bf3dd240e9d0d4d7b150`
- L1 checkpoint SHA-256：`cd485c7beed2781ac13d08fbd33d63b8f1c46971be008966f7280c67cec1fcee`
- L1 CSV SHA-256：`5277b2c33f63b951ea2236ae2e50b8c574ef9f8d2f989e8239fcce7b065c0171`
- L1 ZIP SHA-256：`55d25f1fea9c57ba3431164d53d21f34d20baa2a49c805ad87cca188b5634c5a`
- 正式无 prior 保留包：`outputs/prelim75_v7_20260918/L1/submission/submission.zip`
- 绝对最高侧包：`outputs/prelim75_v7_monitor/g0_prior09/prior09_submission/submission.zip`

平台反馈记录：`results/prelim75_v7_platform_feedback_20260918.json`

