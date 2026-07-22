# U0：数字类别 Prompt Tuning 可行性审计

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-22
- Verification Status: VERIFIED（相同输入、独立输出路径重跑逐字节一致）
- Version Label: validation_report_v1

## 结论先行

**不能把标准 CoOp/Prompt Tuning 直接套到本赛题的 `0000–0499` 类别名上。** 数字类别提示在 OpenAI CLIP ViT-B/32 的文本空间几乎塌缩为同一方向，且与当前 F1 分类器的类别方向没有对应关系。直接训练共享 context 不具备论文中最关键的“有语义固定类名”正则，优先投入完整 GPU 实验缺乏依据。

该结论只否定“数字类名 + 标准共享 CoOp context”的直接迁移；不否定另立实验研究 class-specific soft token 或 visual-prototype prompt inversion。但后两者已不是论文所证明的语义 Prompt Tuning，必须从 F1 epoch-0 等价性和独立门禁重新论证。

## 研究问题

ICCV 2023 的一手研究指出，Prompt Tuning 在噪声标签下鲁棒的两个关键因素是：

1. 固定且有语义的 class-name token 对噪声梯度形成强正则；
2. CLIP 预训练图文空间提供类间结构先验。

本赛题只公开随机数字类别 ID，没有物种名称。U0 检验：canonical prompt `a photo of a 0000` 到 `a photo of a 0499` 是否仍携带足够的类间结构，可作为标准 CoOp 的固定类名锚点。

一手来源：

- ICCV 2023 正文：<https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Why_Is_Prompt_Tuning_for_Vision-Language_Models_Robust_to_Noisy_ICCV_2023_paper.pdf>
- 作者代码：<https://github.com/CEWu/PTNL>
- 官方赛题对 Prompt Tuning/JoAPR/TrustCLIP 的要求与参考：<https://www.aicomp.cn/tracks/tracks-1/3714.html>

## 输入与方法

- 冻结 OpenAI CLIP ViT-B/32；
- 文本模板固定为 `a photo of a {class_id}`，不扫描模板；
- 只使用官方 train 内的固定 validation `10,316` 张冻结 CLIP 特征；
- clean-core 为 cross-fitted trust `>=0.70` 的 `7,396` 张；
- 对比锚点为平台最佳 F1 checkpoint 的 500×512 线性分类权重；
- 未读取 test，未训练参数，未生成提交。

## 结果

| 指标 | 结果 | 含义 |
|---|---:|---|
| raw validation accuracy | `0.232648%` | 接近 500 类随机水平 `0.2%` |
| clean-core accuracy | `0.229854%` | 高可信子集没有改善 |
| 有预测的类别数 | `210 / 500` | 数字提示导致大量类别从不成为 top-1 |
| 文本两两非对角余弦均值 | `0.978551` | 500 个数字提示高度塌缩 |
| 非对角余弦范围 | `0.844030–0.999638` | 最相似类别几乎完全相同 |
| 90% / 99% 能量秩 | `1 / 5` | 500×512 文本分类器有效维度极低 |
| 与 F1 同 ID 权重余弦均值 | `-0.010036` | 数字 ID 与任务类别方向无对应关系 |
| 同 ID 对齐范围 | `-0.076823–0.047562` | 没有少数强语义锚点可挽救共享 prompt |

权威产物：

- `artifacts/numeric_prompt/seed42/audit.json`
- SHA-256：`694a114eb3dced2b7b298d50948e5738988bc3965d58b0779b7c19efb261cff1`
- 独立重跑 `audit_rerun.json` SHA-256 完全相同，`cmp` 逐字节一致
- 可复现入口：`python -m aegis_clip.cli.audit_numeric_prompts`

团队仓库只整合审计代码、测试、结果指标与产物哈希，不提交机器本地冻结特征、F1 checkpoint 或 JSON 副本。复现时从 `reproducibility/aegis_f1/` 运行入口，并用相同输入哈希重建 JSON；缺失本地大文件不代表审计结果可省略或改写。

## 判定

U0 判定为 **direct numeric CoOp infeasible**：

- 不启动标准 shared-context CoOp/JoAPR prompt 训练；
- 不把数字 token 当作自然物种语义；
- 不使用外部类别名词表或人工识别物种来补齐语义，因为这会引入赛外信息并破坏自动复现边界；
- 若后续研究文本侧，只允许 visual-prototype anchored soft token，并必须证明 epoch 0 与既有 F1 分类器逐位一致；它将作为新的方法而不是“复现 CoOp”。

## Fallacy scan

- 未用一次低 accuracy 推断“所有 Prompt 方法都无效”；结论仅覆盖 canonical numeric class tokens 与标准共享 context 前提。
- 未把带噪 raw validation 当作唯一证据；几何塌缩、clean-core、有效秩和 F1 权重对齐给出互补证据。
- 未用 test 选择模板或方向。
- 未把近随机准确率解释为 CLIP 整体无效；这里只测文本数字类名，F1 视觉分类器仍是独立强基线。
- 未扫描多个模板后挑最差结果；模板在运行前固定。

覆盖的其余统计风险：无 p 值/多重比较/因果效应量声称；无选择性缺失、单位错误、平均数掩盖分组、相关即因果、训练验证泄漏或平台反馈回灌。U0 的确定性诊断已完成独立重跑，故状态为 `VERIFIED`；这不等同于任何新模型已经训练或平台有效。
