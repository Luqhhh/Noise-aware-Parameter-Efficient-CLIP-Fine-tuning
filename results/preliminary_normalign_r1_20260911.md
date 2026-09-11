# PRELIM_NORMALIGN_R1：共享分类头范数对齐负结果

日期：2026-09-11。源基线 `c0d25cddfce8be9f9339508294ed57b139c2e65a`，main；用户本轮明确授权 Agent 代选初赛机制。开始前登记文件 `.planning/2026-09-11-preliminary-norm/decision.json`，SHA-256 `7e2ffa17de6f9ba23249a6426f370bce6692a1bbf8522fc7a8c70648fb1c7e0e`。

**结论：本变体 rejected。** raw 少对 146 张，clean-core 少对 77 张，未通过登记的诊断门槛，未生成候选 checkpoint、未进行新测试集推理或平台上传。没有追加变体或强度扫描。它只排除本次固定范数对齐变体，不推导其他分类器学习方法的效果。

## 固定问题与处理

假设：已有共享线性头的类别权重幅度可能造成不必要的边界偏置。实际 500 行权重 L2 范数 min/median/max 为 47.3752/67.0814/78.7152。这不是已证明的瓶颈；更强的替代解释是幅度反映有效的类别可靠性。

变换 `W'_c = mean_j(norm(W_j)) * W_c / norm(W_c)`，保留偏置、视觉塔、O3 与 Part-Token 参数，使用原单模型的共享分类头。研究依据为[分类器解耦原作者代码](https://github.com/facebookresearch/classifier-balancing)；本次保留平均尺度和 bias 的变体不冒充论文配方复现。零梯度训练运行，单一固定变换。

对照和处理都使用四尺度 112/128/144/160、权重 .20/.30/.40/.10、top-k=5、local=.40、Flip=.50、global/local temperature=1.5、双 Adapter，**均无校准**。与历史 pa0.90 不是同一完整流程，不能继承其平台分数。

原模型与变换后分类头共用一次视觉/局部前向；在每个原始分支 logits 上使用 `(logits-bias)*scale+bias`，再调用既有多尺度/Flip 概率融合函数。它来自共享线性头的代数等价，覆盖两个 Adapter 的加性特征残差。真实首批的 10 次分类头调用与直接变换权重计算比较，最大绝对误差 5.7220459e-6，argmax 差异 0，满足预登记 rtol=atol=1e-5。关闭变换的对照保留原分支数值。

## 实际指标与口径

| 指标 | 原模型，无校准 | 范数对齐，无校准 | 差值 |
|---|---:|---:|---:|
| raw correct / 10,316 | 8,203 | 8,057 | -146 |
| raw accuracy | 79.517257% | 78.101975% | -1.415283pp |
| clean-core correct / 7,331 | 6,739 | 6,662 | -77 |
| clean-core accuracy | 91.924703% | 90.874368% | -1.050335pp |
| 预测缺类数 | 0 | 0 | 0 |

839 张预测变化。门槛预先固定为 raw 正确数严格增加且 clean-core 正确数不下降，仅用于是否允许生成一个候选，不是独立泛化证据。所有 10,316 条验证样本都已参与父模型训练；clean-core 是可信度算法选择的子集，不能视为真实干净测试集。

单次诊断耗时 273.71 秒，PyTorch 峰值 CUDA allocated 645,428,224 bytes（不含所有驱动/显示/其他进程占用）。无解码失败替代进入评价；日志有可见 EXIF 警告。没有训练或正式平台指标。

## 执行与验证

```bash
# cwd: reproducibility/aegis_f1
PYTHONPATH=. python3 -m aegis_clip.cli.evaluate_classifier_norm   --checkpoint outputs/F1_FLAT_FULL_FT_R3MS/seed42/dual_adapters/best.pt   --decision ../../.planning/2026-09-11-preliminary-norm/decision.json   --output-dir ../../outputs/preliminary_no_sweep/20260911_normalign_r1/evaluation
```

退出码 0，研究状态 rejected。两个独立测试入口均完成：仓库根 `python3 -m pytest -q tests` **414 passed**；AEGIS 根以 `PYTHONPATH` 指向该目录运行同命令 **382 passed**；退出码均 0。定向范数/effective-number 测试 30 passed（含实际 CUDA 检查），资产清单测试 4 passed。`git diff --check` 通过。

本地完整结果、配对样本预测、命令与日志在 `outputs/preliminary_no_sweep/20260911_normalign_r1/`，已忽略，不提交样本级结果。结果绑定实际验证 CSV、类别映射、父 checkpoint 与决策哈希。

历史包仍为 `outputs/delivery/fullft_dual_pa0.9/{pred_results.csv,submission.zip}`，重新执行根提交检查退出码 0，24,967 张测试图完整覆盖，ZIP 只含 CSV 且内外字节一致。父 checkpoint、CSV、ZIP、manifest SHA-256 与前次 lock 全部相同；checkpoint 仍为 `f72b0104257f49d2667fe335553a861dd1dea947753feebdc7301b8890b48765`。没有桌面替换，也没有生成另一个正式候选。

## 同段工程修复与未完成范围

方案二 R0 新增只读资产盘点脚本，R1 修复 effective-number 幂运算并保持原数学含义，未启用长尾方法。细节及本轮发现的官网/PDF阶段规模差异见 `docs/stage_readiness_status_20260911.md`。R2–R8 不标完成，方案二已收到，不能再把缺该消息当阻塞理由；其实际实现和未来阶段正式配方仍是后续工作。

变更文件：`.planning/2026-09-11-preliminary-norm/decision.json`、`.gitignore`、`aegis_clip/classifier_norm.py`、`aegis_clip/cli/evaluate_classifier_norm.py`、`aegis_clip/longtail.py`、对应 AEGIS 测试、`scripts/audit_stage_assets.py`、`tests/test_audit_stage_assets.py`、本报告及阶段状态文档。AEGIS 文件均位于 `reproducibility/aegis_f1/`。

本段只本地 commit，不自动 push；代码、决策及汇总报告可共享，受限数据/环境快照不纳入 Git。按登记负结果停止本变体并保留历史包，等待本段复核；不将它描述为初赛提分成功。

## 后续交付与平台回填

上述未生成候选描述本次诊断结束时的状态。随后用户明确要求生成桌面包，见 `results/normalign_desktop_and_reproduction_20260911.md`。该无校准包的平台成绩由用户回填为 **63.3676%**；候选仍不晋级，不再追加范数变体。

原父模型同协议无校准对照随后由用户回填 **66.7681%**，高于本范数变体 **3.4005 个百分点**；平台配对结论见 `results/parent_nocal_control_20260911.md`。固定范数变体维持 rejected。
