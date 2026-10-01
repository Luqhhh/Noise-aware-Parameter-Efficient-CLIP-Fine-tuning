# V1_768_FULL_TEST_BIAS_20261001

用户明确授权768路径重跑full train，并转述官方允许利用测试集均衡先验做bias校正。此前冻结特征诊断中768 head相对512净+247/14,880，kNN净+116；这支持本次尝试，但尚不证明LoRA或平台收益。

## 固定配方

- 数据：repechage / 20260921，148,695张full_train，750类；37,444张测试图仅在最终推理/已确认的均衡先验校正中使用。
- 官方CLIP ViT-B/32：原生视觉forward在ln_post后的768维CLS处读出，visual.proj=None；仍从SHA核对的完整官方权重构建，全部48个LoRA位置保持原路径。
- 训练侧缓存：复用刚核验的当前阶段224px FP32原生前向768缓存，不复用上一阶段或现役512去噪目标。缓存SHA 00eefe153a7141f4016c112623dabe796d4fc2468e5a2f6797788085e3058a21；核对源protocol、数据/映射/官方权重和逐行路径。
- 重建k16内容组排除近邻、keep90%初筛、20轮768 teacher及全部去噪/伪标签。参数与原v1固定一致。
- 学生：448px、全12块rank32/alpha64 LoRA、12轮batch32、seed42、LoRA/head LR 0.0002/0.001、LS0.1、mixup0.2、EMA0.999。
- 单个导出checkpoint：EMA epochs4–12的同轨迹SWA；full_train不设独立val，不选最佳epoch，不在重叠val上拟合bias。
- 解码：448/512/576×原图/flip，共6视角logits求和，均衡先验bias固定200次迭代、强度1.0。复用现有prior_alignment算法，固定迭代，无强度扫描。
- 两个包：同一checkpoint/logits的无bias对照，以及测试均衡bias候选。保存test_logits.pt、test_calibration.pt及校正前后分布报告；无测试真值，不把分布变化写成准确率提升。

## 可复现入口

分支codex/v1_768_full_test_bias_20261001；独立工作目录/home/lux1/noise-worktrees/v1_768_full_test_bias_20261001。

```bash
cd /home/lux1/noise-worktrees/v1_768_full_test_bias_20261001
python3 -u scripts/run_v1_training.py \
  --config configs/v1_768_full_test_bias_experimental_20261001.yaml \
  --max-hours 12 \
  --desktop-zip /mnt/c/Users/lqh22/Desktop/v1_768_full_test_bias_submission.zip
```

输出outputs/codex/v1_768_full_test_bias_20261001；校正包submission/，对照submission_raw/，均逐包执行check_submission.py。实现检查、实际运行与平台结果分别记录；尚未启动/完成时不得将本协议当作完成报告。

当前现役70.98600576861446%包保持：/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/submission.zip；其既有37,444行校验与ZIP SHA 1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e见上一段结果。新候选是否提升由后续平台配对反馈判断。

## 启动前实测

CPU官方448px前向通过，750类输出、48处LoRA、5,294,593可训练参数。CUDA RTX4070 Laptop真实5个batch检查通过，2次成功优化更新（初始AMP溢出跳过按现有逻辑处理），不保留拟合权重；训练峰值3,175.21MiB，测得平均0.2563秒/batch，保守全流程估计4.32小时（另加teacher和启动；实际耗时以status为准）。CPU/CUDA检查见results/v1_768_full_test_bias_20261001/。
