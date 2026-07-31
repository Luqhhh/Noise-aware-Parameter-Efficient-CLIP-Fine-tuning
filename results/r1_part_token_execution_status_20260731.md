# R1 Part-Token Residual 执行状态（2026-07-31）

状态：**EXECUTION_CLAIMED_NOT_STARTED**

## 1. 分支与查重

- 专用分支：`agent/r1-part-token-execution`
- 起点：`origin/main@f1fe860`
- 实验 ID：`R1_F1_M1_PART_TOKEN_RESIDUAL`
- 执行人：x28639 / Codex
- 权威协议：
  `reproducibility/aegis_f1/docs/R1_F1_M1_PART_TOKEN_RESIDUAL_PROTOCOL_2026-07-22.md`

接管前已同步最新 `main`，检查远端分支、开放 PR、实验文档、本地输出和 GPU 进程。R1 的
代码、CPU 测试与科学协议已合并，但没有执行分支、真实 cache、训练结果或平台结果；团队
当前新增的 R256 配置属于高分辨率线，与 R1 的冻结 F1 局部 patch-token 残差不共享变量或
输出。本分支只执行 R1，不修改协议参数。

## 2. 固定边界

- 父模型固定为原始 F1 checkpoint，SHA-256
  `7da95e427b959e85cbbf37c99d47d9909b941032e836fc219aaea8e690d72cc4`。
- 仅使用官方 train 与固定 validation；test 在全部门禁通过前禁止读取。
- 全局 F1、M1 定位和唯一线性头冻结；只训练局部 `512→32→512` part-token 残差器。
- attention crop 固定 top-5/crop160；part pool 固定 top-8/temperature0.07；cache batch128。
- 不使用 prior、输出拉平、外部数据、多模型融合、测试时训练或平台分数反推分布。
- 所有产物只写入
  `/home/x28639/projects/AegisCLIP-F6-A2LoRA/outputs/R1_F1_M1_PART_TOKEN_RESIDUAL/`。

## 3. 固定执行顺序与停止规则

1. 每阶段前重新 fetch `main`，检查重复项与团队 GPU 进程；
2. 只运行一次 batch128 train cache；内容门禁通过后立即记录并推送；
3. 只运行一次 batch128 validation cache；center/M1 数值复现门通过后立即记录并推送；
4. 只运行一次固定 CPU Adapter 训练，以 `gate.json` 为权威；
5. 任一命令失败不自动重试，任一门禁失败即关闭 R1；
6. 只有 `gate.json passed=true` 才允许另行评估测试推理；平台上传不自动执行。

用户已给出项目内缓存与训练的持续授权，但团队资源占用、重复实验、规则风险和外部平台上传
仍是自动停止条件。结果无论成功或失败均按阶段单独提交，禁止与其他实验积攒推送。
