# 教师生产谱系核对（2026-09-12）

源提交 44f896d，main；运行 ID `20260912_teacher_provenance_r1`。继续方案二 R2/R3 的只读资产与生产记录核对，未启动新机制、正式训练或参数扫描。

新增 `scripts/audit_teacher_producer.py` 从实际历史审计核对文件哈希，并提取明确登记的参数和视图到非执行模板。缺失参数不补默认值；输出/审计/缓存位置保留占位符，禁止直接覆盖历史资产。

实际命令（仓库根）：

```bash
python3 scripts/audit_teacher_producer.py --audit reproducibility/aegis_f1/outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R3_MULTISCALE_FP32/teacher_trust_audit.json --base-dir reproducibility/aegis_f1 --output-dir outputs/stage_readiness/20260912_teacher_provenance_r1
```

返回 2，表示检测到缺失教师 logits 缓存；其他 4 个现存输入/输出哈希匹配。不是工具异常，也不是已完成生产复现。

沿 trust 包 `metadata.teacher_augmentation.base_trust` 追溯到 cvt_v1，共找到 4 个现存 trust 包。9 条 checkpoint/train_csv/base_trust 引用中 7 条哈希匹配，2 个教师 checkpoint 缺失：SELFTRAIN_R1_FP32 和 FULLFIT_R1_FP32 的 epoch_3.pt。此处统计引用边，重复 train CSV 不算新独立资产。

可定位的缺口包括：原 CLIP features.pt、R2 多尺度教师 logits 缓存、上述两个教师 checkpoint。没有借另一个 epoch、seed 或合成模型填补。根 cvt 生产过程、父模型训练和源作用域仍未闭合。生成记录与现存字节相符，不能替代从允许初始化重新运行的证据。

测试：定向新增测试 4 passed；根套件 417 passed、1 skipped；AEGIS 414 passed、1 skipped，退出码均 0。当前环境 CUDA unavailable，两套各跳过一个 CUDA 项，不称作已验证 GPU 行为。代码变化为 CPU 只读审计逻辑。

现有 PRELIM_PARENT_NOCAL_CONTROL_R1 预测包复检退出码 0，24,967 张覆盖、标签及 ZIP 检查通过。本轮未生成新预测、不替换桌面包、不上传平台。平台既有同协议结果仍为原父模型 66.7681%、范数变体 63.3676%。

私有输出：producer_audit.json、trust_metadata_chain.json、transitive_edge_checks.json、测试和提交检查日志。它们只在 outputs/stage_readiness/20260912_teacher_provenance_r1 保存，不发布样本清单、模型或信任权重。

变更：新增审计脚本与测试，更新工程说明及状态报告。本地提交，不推送。下一步仍需核对 cvt 根生成入口及缺失资产的完整生成依赖，不能标记方案一/二全部完成。
