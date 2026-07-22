# Q1：交叉拟合训练轨迹与 Wrong-Event 噪声诊断预注册

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-22
- Verification Status: IMPLEMENTED_NOT_RUN
- Version Label: q1_plan_v1

## 1. 实验定位

- **目标**：判断“每个样本在未见过它的折模型中的完整学习轨迹”是否提供了超出单个最终 OOF 置信度的噪声信息，并据此决定是否值得进入 Q1B 鲁棒训练阶段。
- **核心假设**：真正的错标样本更可能在多个 epoch 持续与原标签冲突；困难但干净的样本则可能早期不稳定、随后逐渐收敛。完整轨迹应当比一次最终预测更有区分力。
- **类型**：train-only、三折交叉拟合、冻结 CLIP 特征上的线性头轨迹重建。
- **当前状态**：代码与微型端到端测试已完成；真实 Q1A 尚未运行，不产生平台提交包。
- **明确边界**：Q1A 只重建诊断信号，不训练最终参赛模型，不访问测试集，不使用外部数据。

## 2. 为什么改变方向

现有 91,195 个 OOF-train 样本的只读诊断显示：

| 观察 | 数值 | 含义与限制 |
|---|---:|---|
| 严格 OOF top-1 与训练标签一致率 | 69.4424% | 与旧审计 69.4490% 的差异来自 float16 并列分数的 `argmax/topk` tie-breaking，不是模型变化 |
| 每个折模型在其见过的训练成员上准确率 | 81.555%–81.696% | 比严格 OOF 高约 12.2 个百分点，显示明显的样本记忆效应 |
| OOF 错误、但两个见过该样本的模型都投原标签 | 9,485（10.40%） | 对“见过样本会掩盖噪声”提供直接证据 |
| 三个头一致投同一个替代类别 | 7,878（8.64%） | 是高冲突候选，但不能直接视为真标签 |
| 最终 OOF `p(original)` 与静态跨头标签投票相关 | 0.849 | 再堆叠静态置信度的增量空间有限；需要时间维信号 |
| 各类 OOF 标签保留率 | 5.56%–97.27%，标准差 18.46pp | 说明统一全局阈值风险很高；也可能混合了类别难度与类条件噪声，不能解释为纯噪声率 |

这些结果只支持“轨迹值得验证”，不证明 Wrong-Event 一定提高平台分数。平台测试标签干净，而本地验证标签含噪，最终结论必须以严格门控后的一次平台提交为准。

## 3. 方法来源与取舍

1. [IDO（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/429e7b31625a8b7839f9e4d6e2aa9bb9-Abstract-Conference.html) 使用预测与噪声标签不一致的 epoch 数、标签波动和类内 Beta mixture 描述样本状态。Q1 只借鉴其训练动态思想，不复制实现。
2. [IDO 官方代码](https://github.com/iTheresaApocalypse/IDO) 存在必须规避的实现风险：未深拷贝 `state_dict` 会让“早期检查点”随之后参数更新；置信度张量维度会把逐样本熵广播成 `[B,B]`；常数非零序列的归一化缺少零分母保护。Q1 用独立 CPU 克隆的逐 epoch 权重快照和显式形状检查避免这些问题。
3. [DSS（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_Debiased_Sample_Selection_for_Learning_with_Noisy_Labels_CVPR_2026_paper.html) 提出 marginal distribution alignment 与 class confidence selection，支持“类条件选择优于单一全局阈值”的方向。
4. [DSS 官方代码](https://github.com/Aliinton/DSS) 的核心概率历史为 `[N,E,C]`。按本赛题 `N=103,218, C=500` 估算，仅选择器张量在 20/30 epoch 时就约需至少 4.23/6.15 GiB，尚未计入 CLIP、优化器和激活，无法在 8 GiB GPU 上原样采用。Q1 将历史放在 CPU，只保存原标签概率、top-5 与最终 logits。

## 4. 固定输入与血缘

| 输入 | 路径 | SHA-256 |
|---|---|---|
| 三折分配 | `/home/x28639/projects/Noise-aware-F1-sync-20260719/outputs/phase/phase3/oof/fold_assignments.csv` | `eb2409fdd252450cc8750b8e9ceb5ac6f2d6dd1e9c829b81bc3a609f6824c138` |
| OpenAI CLIP ViT-B/32 冻结特征 | `/home/x28639/projects/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/cache/preliminary/clip_vit_b32_openai/features.pt` | `707b45bedba030a3851737427f664291ec83ad2d2b7d20dfd5964b960efbd143` |
| 特征路径索引 | `/home/x28639/projects/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/cache/preliminary/clip_vit_b32_openai/image_paths.json` | `c77194e18065a5a133e7484bff74248194d015fda6b1669a3d7ce0dcad9388df` |
| 特征标签索引 | `/home/x28639/projects/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/cache/preliminary/clip_vit_b32_openai/labels.json` | `634ebeeca5b3e3182a93959a8bbbb890a6a6027a4faecd705042d0d279cebe90` |
| I0 最终 OOF 参考 logits | `/home/x28639/projects/AegisCLIP-F6-A2LoRA/outputs/I0_OOF_LOGIT_RECONSTRUCTION/seed42/oof_logits.pt` | `b2bb6f72c0f44961cd1665d81443dc5afe58b264702c1c5e9672e55e399440b3` |

输入哈希任何一个不一致时必须在训练前失败，不能静默换数据。

## 5. 固定配置

| 项目 | 值 |
|---|---|
| Backbone/特征 | 官方允许的 OpenAI CLIP ViT-B/32，512 维冻结图像特征 |
| 折数 | 3；每个样本每个 epoch 只由未见过它的折模型预测一次 |
| 线性头损失 | GCE，`q=0.5` |
| 优化器 | AdamW，`lr=0.005`，`weight_decay=1e-4` |
| 调度 | 2 epoch warmup + cosine，末端系数 0.01 |
| 训练 | 50 epoch，batch size 128，折种子 42/43/44 |
| 推理 | 每 epoch 严格 held-out inference，batch size 1024 |
| 轨迹 | Wrong-Event 计数、预测变化计数、原标签概率、top-5 类及概率、最终 logits |
| 检查点 | 每折保存全部 50 个 epoch 的独立权重快照；全局 label-wave 反转点再物化对应折头 |
| 设备 | 单 GPU；轨迹常驻 CPU，AMP 只用于线性头训练 |

“选中 epoch”只使用全体 OOF 预测变化次数的第一处反转，不使用标签正确率、原始验证集或平台反馈。

## 6. 精确执行命令（尚未授权、尚未运行）

```bash
cd /home/x28639/projects/AegisCLIP-F6-A2LoRA
PYTHONPATH=. /home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
  -m aegis_clip.cli.rebuild_oof_trajectory \
  --assignments /home/x28639/projects/Noise-aware-F1-sync-20260719/outputs/phase/phase3/oof/fold_assignments.csv \
  --feature-tensor /home/x28639/projects/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/cache/preliminary/clip_vit_b32_openai/features.pt \
  --feature-paths /home/x28639/projects/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/cache/preliminary/clip_vit_b32_openai/image_paths.json \
  --feature-labels /home/x28639/projects/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/cache/preliminary/clip_vit_b32_openai/labels.json \
  --reference-oof outputs/I0_OOF_LOGIT_RECONSTRUCTION/seed42/oof_logits.pt \
  --output-dir outputs/Q1_CROSSFITTED_WRONG_EVENT/seed42 \
  --expected-assignments-sha256 eb2409fdd252450cc8750b8e9ceb5ac6f2d6dd1e9c829b81bc3a609f6824c138 \
  --expected-feature-tensor-sha256 707b45bedba030a3851737427f664291ec83ad2d2b7d20dfd5964b960efbd143 \
  --expected-feature-paths-sha256 c77194e18065a5a133e7484bff74248194d015fda6b1669a3d7ce0dcad9388df \
  --expected-feature-labels-sha256 634ebeeca5b3e3182a93959a8bbbb890a6a6027a4faecd705042d0d279cebe90 \
  --expected-reference-oof-sha256 b2bb6f72c0f44961cd1665d81443dc5afe58b264702c1c5e9672e55e399440b3 \
  --epochs 50 --batch-size 128 --infer-batch-size 1024 \
  --lr 0.005 --weight-decay 0.0001 --warmup-epochs 2 \
  --q 0.5 --seed 42 --num-classes 500 --top-k 5 --device cuda
```

真实执行前必须同时满足：用户明确授权该命令；再次检查团队 GPU 进程；若团队正在占用 GPU，则不启动。

## 7. 预期产物

| 产物 | 路径 | 成功条件 |
|---|---|---|
| 完整轨迹 | `outputs/Q1_CROSSFITTED_WRONG_EVENT/seed42/trajectory.pt` | 91,195 个样本 × 50 epoch 完整覆盖，无重复/缺失/非有限值 |
| 每折历史 | `fold_{0,1,2}/train_history.json` | 50 条记录，含 train/held-out 指标与 prediction changes |
| 每折全 epoch 头 | `fold_{0,1,2}/epoch_heads.pt` | 50 个独立权重快照，哈希写入审计记录 |
| 选中基础头 | `fold_{0,1,2}/selected_base_head.pt` | 权重逐元素等于 `epoch_heads.pt` 中选中 epoch，不是最终头别名 |
| 参考审计 | `audit.json` | 所有 Q1A 参考门控通过 |
| 血缘清单 | `artifact_manifest.json` | 输入/输出哈希、环境、参数与最终 gate 状态齐全 |

## 8. Q1A 强制门控

以下任一失败都终止 Q1，不进入 Q1B：

1. 所有样本在每个 epoch 恰好记录一次；
2. 当前与参考 logits 均为有限值；
3. 第 50 epoch 与 I0 参考 OOF 的 top-1 一致率 `>= 0.9999`；
4. 原标签概率平均绝对误差 `<= 0.002`；
5. 原标签概率 P99 绝对误差 `<= 0.01`；
6. 三个选中基础头均能从逐 epoch 快照逐元素复核；
7. 输入哈希与第 4 节完全一致。

这些阈值只验证“是否忠实重建旧训练过程”，不代表模型质量提升。

## 9. 轨迹信号决策门控

Q1A 通过后，只做 train-only CPU 分析。进入 Q1B 前至少满足：

1. **稳定性**：epoch 40 与 epoch 50 的 Wrong-Event 排名 Spearman `>= 0.95`；
2. **非冗余性**：最终 Wrong-Event 与最终静态 `p(original)` 的 Spearman 绝对值 `<= 0.90`；若更高，说明轨迹大概率只是昂贵地复述最终置信度；
3. **非塌缩性**：Wrong-Event 与预测变化计数不能退化为近常数，且至少 95% 样本具有可区分的有限轨迹；
4. **类别审计**：报告每类选择率、最小/最大/标准差；任何后续过滤必须使用类内归一化或每类下限，禁止用全局阈值清空困难类别；
5. **反证审查**：分别检查“高 Wrong-Event 但后期稳定回到原标签”和“低 Wrong-Event 但长期低置信”群体，避免把困难干净样本等同于噪声。

`0.95/0.90` 是预先固定的工程决策阈值，不宣称为统计学普适界限。即便门控通过，也只能说明信号值得进入下一阶段，不能证明会提高干净测试集准确率。

## 10. Q1B 及平台验证边界

- Q1B 的样本权重、类内混合模型、LoRA/Adapter 位置、训练轮数和损失必须在看见 Q1A 结果后另建预注册，不能事后挑参数解释结果。
- 不直接照搬 IDO/DSS 代码；只实现经过单元测试、显存审计和比赛合规审查的最小机制。
- 本地含噪验证只用于安全门控和相对诊断，不作为“真实准确率”估计。
- 只有单一模型/单一推理流水线候选才能进入平台；不做多模型融合、投票或隐式集成。
- 最终平台成功标准是相对当前最佳 `F1+M1 = 63.3276` 的预注册比较，而不是本地 OOF 指标好看。

## 11. 主要反例与停止条件

1. 线性头容量有限，Wrong-Event 可能把细粒度难例误判为噪声；因此轨迹只能作软权重候选，不能未经验证直接删样本。
2. 类间可分性不同会制造类条件偏差；全局排序即使整体相关性漂亮，也可能伤害困难类别。
3. OOF 仍使用带噪训练标签计算 Wrong-Event；它是冲突信号，不是真实 clean/noisy 标注。
4. 平台分布与本地分布不同；任何本地提升都不能外推成平台提升。
5. 若 Q1A 参考门控失败、轨迹高度冗余、类别选择塌缩或需要占用团队 GPU，则立即停止，不自动重试。

## 12. 监控

- 监控范围仅限 `outputs/Q1_CROSSFITTED_WRONG_EVENT/seed42`。
- 每折每 epoch 打印训练损失、训练准确率、held-out 准确率、预测变化数与耗时。
- 硬超时在用户授权运行时另行固定；异常不自动重跑。
- 运行期间若发现团队 GPU 任务或显存竞争，报告并停止后续启动，不杀团队进程。
