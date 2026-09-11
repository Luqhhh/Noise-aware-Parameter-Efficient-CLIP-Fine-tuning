# PRELIM_PARENT_NOCAL_CONTROL_R1：平台同协议对照

2026-09-11，源提交 `89ce92221deb2b7c57daa2a6129d95b9d0e045d3`，main。用户授权 Agent 代选并继续；决策在运行前登记于 `.planning/2026-09-11-parent-nocal-control/decision.json`。

范数对齐包由用户回填 **63.3676%**，按所报精度保存，不推断精确正确数。本轮补齐原实验已定义的原父模型对照：恢复原分类头，四尺度、双 Adapter、Flip、温度与范数包相同，均无 prior。历史 70.352866% 使用不同校准协议，不能直接归因比较。没有重开参数扫描。

## 实际产物

- 本地 `outputs/preliminary_no_sweep/20260911_parent_nocal_control_r1/submission/submission.zip`。
- 桌面 `/mnt/c/Users/lqh22/Desktop/submission_parent_nocal_control.zip`（新文件名，原范数包保留）。
- 24,967 条预测，解码替代数 0；提交检查退出码 0，ZIP 仅含规定 CSV 且内外字节相同。
- 相对范数包 3732 条预测改变；没有测试真值，不将变化数解释为改对数。
- checkpoint SHA-256 `f72b0104257f49d2667fe335553a861dd1dea947753feebdc7301b8890b48765`。
- CSV SHA-256 `3a61d3913c802c36cb0ac21bfdd47ea54e351223a89be258f15f23cad09e6682`。
- ZIP SHA-256 `7cd13a8913feb3eb15b5382182917b836cd0cd5905cccd3118cce0abd61a779b`。
- manifest SHA-256 `ac4cd5d4786e0c73e2b07296f9c59314b21d0a576177aaabe79b9d53354808dc`。

推理退出码 0，耗时 603.31 秒。实际 manifest 除 checkpoint 路径/哈希和预测产物哈希外，与范数包逐字段相同。`prior_alignment=null`。交付时平台分数未知；后续用户回填 **66.7681%**，尚未晋级；无自动上传。

## 命令

```bash
# cwd: reproducibility/aegis_f1
PYTHONPATH=. python3 -m aegis_clip.cli.infer --checkpoint /home/lux1/noise/reproducibility/aegis_f1/outputs/F1_FLAT_FULL_FT_R3MS/seed42/dual_adapters/best.pt --output-dir /home/lux1/noise/outputs/preliminary_no_sweep/20260911_parent_nocal_control_r1/submission --local-view attention_multiscale --local-crop-sizes 112,128,144,160 --local-scale-weights 0.20,0.30,0.40,0.10 --local-top-k 5 --local-weight 0.40 --local-temperature 1.5 --adapt-local-features --adapt-part-token-features --tta horizontal_flip --tta-fusion mean_probabilities --tta-temperature 1.5 --tta-view-weight 0.50 --acknowledge-local-view-risk --acknowledge-tta-risk --batch-size 64
```

完整命令、检查输出、源状态、两套测试日志、协议比较和交付收据位于 `outputs/preliminary_no_sweep/20260911_parent_nocal_control_r1/`，不提交受限产物。

## 工程增量与验证

新增传递作用域声明审计，区分 learned/selected/encoded，带内容组文件绑定和反例测试。实际父模型开发独立性审计 blocked：验证 10,198 个内容组全部被训练接触，上游教师/trust 生产链不完整。它只验证声明一致性，不证明来源真实性，也不解锁正式训练入口。

本轮根测试 **414 passed**，AEGIS **410 passed**，退出码均 0。新入口说明见 `reproducibility/release/README_engineering.md`。R2/R3/R5 的完整正式谱系和来源验证仍未完成，不将软件测试通过写成研究成立。

改动集中于 lineage.py、audit_scope_graph CLI 和测试、决策记录、提交/平台登记表、实验与工程状态文档。本段只本地提交代码和汇总；可供代码同步，不自动 push，不传播模型或样本级数据。

## 平台同协议比较回填

| 固定协议：四尺度 + Flip，无校准 | 用户报告平台准确率 |
|---|---:|
| 原父模型共享分类头 | 66.7681% |
| 共享分类头均值范数对齐 | 63.3676% |

范数对齐相对原父模型降低 **3.4005 个百分点**。该固定变体在本次平台对照中表现更差，与既有重叠诊断的负方向一致；保持 rejected，不追加范数扫描。该结论不推广到所有分类器解耦方法。历史 70.352866% 使用不同校准协议，保留分开记录。按用户报告精度登记，不反推精确正确样本数。
