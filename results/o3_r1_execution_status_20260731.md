# O3-R1 执行状态与团队边界（2026-07-31）

状态：**PREPARED_NOT_STARTED（已认领准备，未启动缓存或训练）**

## 1. 分支与血缘

- 专用分支：`agent/o3-r1-local-adapter`
- 起点：`origin/main@04c4111`
- 实验：`O3-R1_F1_LOCAL_ONLY_FEATURE_ADAPTER`
- 准备方：x28639 / Codex
- 实际运行人：待团队群内确认，当前未分配给队长或 JJT

开始准备前已执行 `git fetch origin main --prune`，并检查远端分支、开放 PR、实验总表与
O3/O3-R1 协议。未发现正在执行或已产出结果的 O3-R1；最新 `main` 正在推进的 288px
高分辨率 F1 与本实验不共享方法变量或输出目录。

## 2. 实验边界

O3-R1 只验证一个已经预注册的问题：在保持 F1 全局路径和唯一线性分类头冻结的条件下，
能否用一个仅作用于 attention-local 特征的 `512→32→512` 零初始化残差 Adapter，保留
O2 已观察到的局部判别增益，同时避免共享 LoRA/分类头导致的全局退化。

本分支不改变原 O3-R1 的模型、数据、损失、优化器、局部裁剪、融合权重或门槛。唯一允许
的修订仍是原协议规定的 validation cache `batch_size=128` 数值复现修复。

权威协议：

- `reproducibility/aegis_f1/docs/O3_F1_LOCAL_ONLY_FEATURE_ADAPTER_PROTOCOL_2026-07-22.md`
- `reproducibility/aegis_f1/docs/O3_R1_BATCH_REPRODUCIBILITY_AMENDMENT_2026-07-22.md`

## 3. 合规与团队隔离

- 只使用官方 train、OpenAI CLIP ViT-B/32 和单一最终模型；不使用外部数据。
- 完全不使用 prior、输出拉平、prior 强度扫描或平台分数反推测试分布。
- 不读取测试标签，不进行测试时训练；本地门控通过前不生成测试预测或提交包。
- 输出只能写入 O3-R1 专属目录，禁止覆盖团队正在运行的输出。
- 本分支只承载 O3-R1；后续结果无论成功或失败均立即追加并单独推送，不与其他实验合并提交。

## 4. 未授权事项

截至本文件提交时，下列操作均**未执行**：

1. GPU validation cache 重建；
2. GPU train cache 生成；
3. CPU Adapter 训练；
4. 测试集推理、提交包生成或平台评测。

任何 GPU 操作必须同时满足：用户明确授权、团队群内确认实验编号和执行人、启动前再次拉取
最新 `main`、确认没有重复实验，并确认团队 GPU 空闲。若任一条件不满足，保持停止状态。

## 5. 后续固定顺序

1. 再次同步 `main` 并复核重复项；
2. 获得明确 GPU 授权后，仅重建 batch-128 validation cache；
3. 先执行原协议的逐位复现门控，失败即记录并关闭；
4. 门控通过后再生成 train cache，并执行一次固定 CPU Adapter 训练；
5. 结果产生后立即补充本文件并推送；失败结果同样保留；
6. 只有全部预注册门槛通过，才另行请求是否允许生成平台候选。
