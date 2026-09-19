# PRELIM75 v8：训练集来源的冻结类别偏置

计划 ID：`PRELIM75_V8_20260919`

固定审阅提交：`3f83d69f9a4496976e7d50beecd7e0baa0400d4d`

状态：**正式队列已执行；B0 平台 67.8816%，比匹配 G0 无 prior 低 1.3458pp，已淘汰；B1 提交包已通过独立检查并等待平台回填。协议描述符中的进程地址问题已失败关闭并以严格兼容收据修复；完整记录见 `results/prelim75_v8_execution_20260919.md`。**

## 决策与边界

本轮停止自动短续训，不扫描 LR、horizon、floor、epoch、CE、伪标签、在线定位、融合比例、正则、边界或应用强度。固定两个完整模型：

| 候选 | 固定模型 | 匹配无 prior 平台对照 | 新拟合内容 |
|---|---|---:|---|
| B0 / CAL_G0_TRAINONLY | v6 G0 | 69.2274% | 自己的 500 维训练源偏置 |
| B1 / CAL_L1_TRAINONLY | v7 L1 | 69.2794% | 自己的 500 维训练源偏置 |

主干、共享 head、O3 与 PTA 均冻结，不创建主干 optimizer，不做参数更新。偏置是从训练监督拟合的参数，因此准确描述是“零主干训练、训练数据来源的轻量后校准”，不是“完全无学习”。

已记录的 G0+legacy test-batch prior0.9 平台 72.4677% 只作为不同来源协议的分数参照。本轮不读取其 500 维偏置、不拟合测试边际、不复制 G0 偏置给 L1，也不关闭 `infer.py` 与 `calibration_binding.py` 的 test-fitted prior 拒绝逻辑。

## 固定来源与推理协议

- 只允许 `final_full_train.csv` 中 103,218 个官方训练根目录样本；路径、标签、顺序、内容 SHA-256 集合、类别映射、trust 与原监督语义哈希全部绑定。
- 原 `w_i` 与 `Q_ic` 逐元素沿用；不筛图、不改标签、不加样本、不恢复旧大特征缓存。
- 每个模型只缓存最终十视图融合后的 `s=log(p)`，形状 `[103218,500]`、FP32，元素数据恰为 206,436,000 bytes。
- 固定推理为 224 center crop、原图/Flip、112/128/144/160 attention crops、scale 0.2/0.3/0.4/0.1、top-k5、global/local 0.6/0.4、温度 1.5、完整 O3/PTA。
- 缓存逐行检查有限值与 `logsumexp(s)≈0`，首 batch 与仓库既有无 prior 十视图融合实现核对数值和 argmax；不对融合后的 `s` 再除温度。

## 固定数值目标

有效类别监督量和派生拟合权重为：

`n_c = Σ_i w_i Q_ic`，`a_ic = w_i Q_ic / (500 n_c)`，`u_i = Σ_c a_ic`。

每类 `n_c` 必须严格为正；`Σ_i a_ic=1/500`、`Σ_i u_i=1`。只拟合：

`J(b) = -Σ_i,c a_ic log softmax(s_i+b)_c + 0.01/(2×500)||b||²`

并约束 `−log(4) ≤ b_c ≤ log(4)`。实现以等价目标差和解析梯度分块计算；CPU FP64、SciPy L-BFGS-B、chunk 4096、maxiter 100、maxfun 300、maxls 30、maxcor 10、ftol 1e-12、gtol 1e-8，不做任何扫描。求解成功、目标不增加、投影梯度无穷范数不超过 `1e-6`、偏置有限且不越界才允许交付。

测试应用固定为 `corrected_scores = s + 0.90*b`。偏置保存原始拟合值，strength 单独记录；逐行加固定偏置，不调用 `align_logits_to_prior` 的同批拟合入口。

## 来源绑定

每个候选使用独立 version-2 记录，并强制绑定：stage、dataset ID、fit/target checkpoint SHA-256、class mapping SHA-256、完整推理协议 SHA-256、原监督语义 SHA-256、实际训练源清单、源图片内容 SHA-256 集合、原 train CSV、trust bundle 与冻结训练分数资产。`fit_checkpoint_sha256` 必须等于 `target_checkpoint_sha256`。

兼容资产名仍为 `validation_logits.pt`，但 payload 明确写入 `fit_scope=training_overlap_calibration` 与 `score_semantics=log_final_fused_probabilities`，不得伪称独立验证。`source_authenticity_verified` 和 `authorizes_calibration` 只由实际路径、内容、顺序与哈希检查产生。`upstream_provenance_complete=false` 保留；本次来源核对不冒充完整祖先训练链认证。

## 停止条件与预算

任一候选出现父模型/哈希/映射/协议错误、测试路径进入源清单、源文件或顺序不一致、`n_c≤0`、非有限值、求解失败、目标增加、投影梯度超限、偏置越界或来源绑定失败，即停止该候选，不换参数或数据。

预算固定为：零主干 optimizer 更新；两次训练集冻结前向；两次 CPU 拟合；两个测试推理候选；GPU 动作累计最多 7,200 秒、CPU 拟合累计最多 3,600 秒。平台不自动上传，Git 不自动 push。

两个候选都不超过 L1 的 69.2794% 时保留 L1；任一成为非测试拟合路径新高时只保留真实胜者及其对应偏置；提升不足 0.30pp 不派生扫描；达到 75% 后冻结产物并停止新增实验。同分优先已有 L1；B0/B1 同分且均胜过 L1 时保留 B1。

## 实现入口

- `configs/prelim75_v8.yaml`
- `reproducibility/aegis_f1/aegis_clip/source_bias.py`
- `reproducibility/aegis_f1/aegis_clip/prelim75_source_bias.py`
- `reproducibility/aegis_f1/aegis_clip/cli/calibrate_prelim75_v8.py`
- `scripts/run_prelim75_v8_queue.py`
- `reproducibility/aegis_f1/tests/test_prelim75_v8.py`

只读预检命令：

`PYTHONPATH=reproducibility/aegis_f1 python3 scripts/run_prelim75_v8_queue.py --config configs/prelim75_v8.yaml`

正式固定队列命令在实现提交并保持 `main` 干净后才允许使用：

`PYTHONPATH=reproducibility/aegis_f1 python3 -u scripts/run_prelim75_v8_queue.py --config configs/prelim75_v8.yaml --execute`
