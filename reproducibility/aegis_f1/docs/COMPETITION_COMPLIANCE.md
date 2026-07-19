# 赛规合规矩阵

依据文件：`面向噪声标签数据的细粒度图像识别鲁棒微调-1.pdf`  
文件 SHA-256：`19d6f0fb587bd58530558026e31ba905bf6512c628dd88fa46c9fab99460e148`

本矩阵把赛规转成可执行约束。若后续官方通知与该 PDF 冲突，以最新官方通知为准，并同步更新配置和审计。

| 赛规约束 | 工程落实 | 自动证据 |
|---|---|---|
| 仅可使用 OpenAI 官方 CLIP ViT-B/32 预训练权重 | 配置只接受 `backbone: ViT-B/32` 与 `pretrained: openai`；依赖锁定官方 OpenAI CLIP 提交 `d05afc4` | `config.validate_config`、特征缓存 `manifest.json` |
| 仅使用当前赛段官方训练数据 | `prepare_stage` 直接扫描官方训练目录；配置必须为 `external_data: false` | `split_manifest.json`、`audit` |
| 测试集只用于推理 | 划分、特征缓存、可信度估计只接受训练根目录；配置必须为 `test_usage: inference_only` | 划分与特征清单中的 `test_data_used: false`、`audit` |
| 噪声处理须由提交代码直接复现，不得依赖人工清洗 | 内容哈希分组、OOF 原型、OOF 探针、2-GMM、软修正均由命令行入口自动生成 | `aegis-prepare-stage`、`aegis-cache-features`、`aegis-build-trust` |
| 单模型或单一推理流程，禁止模型集成、融合或投票 | 每次提交绑定一个严格加载的检查点；不实现跨检查点集成 | 检查点 `artifact_manifest.json`、提交 `manifest.json` |
| 以 Top-1 Accuracy 为主指标 | 推理只输出单个 Top-1 类别；内部同时报告 micro/macro 代理指标用于抗长尾选择 | `evaluation.py`、`submission.py` |
| 初赛为 500 类、103,218 张训练图、24,967 张测试图 | 初赛配置写死期望数量并在审计/推理时 fail closed | `a0_fulldata_anchor.yaml`、`audit.py`、`infer.py` |
| CSV 每行格式为 `filename,0001`，标签四位补零且无前导空格 | 使用标准 CSV writer；标签格式和图像覆盖均严格检查 | `submission.py`、`test_submission.py` |
| 压缩包内必须是 `pred_results.csv` | 只在 CSV、ZIP 内字节和覆盖检查全部通过后发布 | `submission.py` |
| 后续赛段需适应 1,500/1,000 类长尾数据 | 类别数和样本数均配置化；提供训练期类先验校正，但默认关闭，必须单独消融 | `class_prior_adjustment_tau`、macro 选择指标 |

## 当前初赛数据证据

- 官方训练图像：103,218；
- 内容 SHA-256 唯一组：101,980；
- 重复内容样本：1,238，所有重复组只能落在同一侧；
- 固定 seed 42 划分：训练 92,902，验证 10,316；
- 新工程重建的两份 CSV 与既有官方全量划分 SHA-256 完全一致；
- 冻结特征：103,218 × 512，路径索引 103,218/103,218 完整覆盖；
- 测试图像不进入划分、特征缓存、OOF 可信度估计或模型选择。

## 上传前硬门禁

1. `bash scripts/run_stage.sh test` 全通过；
2. `bash scripts/run_stage.sh audit` 全通过；
3. 最佳检查点严格加载并完成官方测试集裸推理；
4. `pred_results.csv` 行数等于官方测试图像数，文件名集合完全一致；
5. CSV 第二列均为四位数字标签且范围合法；
6. ZIP 中只有根目录下的 `pred_results.csv`，其字节与外部 CSV 完全一致；
7. 保存检查点、配置、环境、数据划分、可信度资产和提交文件的 SHA-256；
8. 最终只上传一个候选，不上传任何多模型融合结果。
