# PRELIM75 v6 执行与平台闭环

计划 ID：`PRELIM75_V6_20260918`

状态：**已闭环，保留 G0，关闭在线几何配方，不追加 G2。**

## 平台结果

| 候选 | 平台准确率 | 说明 |
|---|---:|---|
| F1（旧起点） | 69.1393% | v5 无校准胜者 |
| G0 | **69.2274%** | 冻结 V1 框；本轮回合最高 |
| G1 | 69.1993% | 当前学生在线框 |
| 历史 prior0.90 | 70.352866% | 不同协议最高纪录，未被刷新 |
| 75% 目标 | 75.0% | 未达到 |

## 判定

- G0 − F1 = **+0.0881pp**
- G1 − F1 = **+0.0600pp**
- G0 − G1 = **+0.0281pp**
- G1 − max(F1, G0) = **−0.0281pp**，未达到 0.30pp 投入回报门槛
- 保留 **G0**，不支持“当前学生在线定位优于冻结 V1 框”的平台收益结论
- G1 的重叠 raw 诊断曾比 G0 高 0.048470pp，但平台结果为低 0.0281pp；诊断差值不等于平台收益
- 本轮没有定位真值，不能声称当前学生框更准或 attention 更可解释
- 历史绝对最高仍为 prior0.90 的 **70.352866%**；距 75% 仍差 **5.7726pp**
- 不自动追加 G2、参数扫描、续训或平台上传

## 关键运行指标

- D0：`material_change_groups=1080`，门槛 205，通过
- 共同 smoke：microbatch=32，无 OOM 降档
- G0 末轮重叠诊断：raw 83.879411%，clean-core 96.698952%
- G1 末轮重叠诊断：raw 83.927882%，clean-core 96.685308%
- common-control：首 batch、Flip/尺度、soft targets、weights、global logits/features/anchor 全部对齐
- 队列总耗时：9333.45s，预算上限 28800s
- 平台包：G0、G1 各 1 个；工程止损均未触发

## 资产

- G0 checkpoint SHA-256：`fb164dcac4ce3aae9f66129cf9ad5ba5160fabd26960742b5e00d2a8bdb43bea`
- G0 CSV SHA-256：`bd94d8886e9bae97a486077c89929b98ee60a3d14d135cc618726e5af03967a6`
- G0 ZIP SHA-256：`efccb9f2b601a8b24830ef601739b166c214b8316b441bb9bc7d7aaf6bd4b634`
- G1 checkpoint SHA-256：`dcc4cbe7e3df4725de314168c02f9dcd4b3573800cdf8e68306a37867c68b74e`
- G1 CSV SHA-256：`c63417afe277e40be381509120cd3583fb41ac98c7b9266f9ea773bf217e2759`
- G1 ZIP SHA-256：`14617a1dce399e24fec02ded98091ac92d60f329fd3465249526a48aa8bf835f`
- G0 桌面包：`C:\Users\lqh22\Desktop\submission_prelim75_v6_G0_20260918.zip`
- G1 桌面包：`C:\Users\lqh22\Desktop\submission_prelim75_v6_G1_20260918.zip`

平台反馈记录：`results/prelim75_v6_platform_feedback_20260918.json`
