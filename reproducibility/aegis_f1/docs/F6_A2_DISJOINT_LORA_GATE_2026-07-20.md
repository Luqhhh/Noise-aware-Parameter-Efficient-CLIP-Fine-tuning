# F6：A2 → Visual LoRA 无泄漏因果门控

日期：2026-07-20

## 决策

团队文档已经规划 `NR_COMBINED_UPGRADE`：从平台 TTA 61.2128% 的 A2
检查点出发，叠加 F1 的四层 visual LoRA、clean-core 筛选和特征蒸馏。
独立工程不直接重复这次全量运行，而先回答一个更窄、可证伪的问题：

> 在保持 A2 分类头与官方 CLIP 基座的前提下，只用与选择集不相交的
> 高可信图像学习零初始化视觉 LoRA，能否稳定改善 A2 未见图像的表示？

这与 DeFT 的核心顺序一致：先筛选可信监督，再对视觉编码器做参数高效
适配；同时保留 F1 已经平台验证为正收益的低漂移 LoRA 实现。

## 数据隔离

使用两个已冻结、来源独立的开发划分构造三块数据：

- `adapt_train = A2 train ∩ Aegis validation`：只用于学习新增 LoRA；
- `evaluation = A2 validation ∩ Aegis train`：只用于 epoch 选择；
- `cross_audit = A2 validation ∩ Aegis validation`：不参与选择，仅检查方向。

三块数据必须在规范路径和内容 SHA-256 组两个层面完全不相交。A2 原始
分类头没有看过 `evaluation`；新增 LoRA 也没有看过它。Aegis 的 trust
信号对每张图均来自 OOF 预测，不使用测试集。

真实资产审计结果：

- `adapt_train`：9,135 张、9,122 个内容组；
- `evaluation`：9,315 张、9,306 个内容组；
- `cross_audit`：1,005 张、1,005 个内容组；另有 2 张因与适配集内容组
  冲突而被自动移除；
- 有效 clean-core 分类监督：6,492 张，覆盖 500/500 类；
- A2 黑名单总数 991，其中 104 项位于 `adapt_train`；
- 联合 trust bundle SHA-256：
  `93b282ce91dfbd8dcb662cfaf621bc026d1863f9c94f63f572a5964c545c64bb`。

真实权重迁移审计确认：A2 的 152 个视觉骨干张量与官方基座逐项完全一致；
所有 LoRA 输出矩阵为零；64 个真实缓存特征上的迁移模型 logits 与 A2 直接
线性头计算逐元素完全一致，最大绝对误差为 0，Top-1 完全一致。

## 固定配方

- 父检查点：`NR_CL_KNN_DROP` epoch 48，平台 flip TTA 61.2128%；
- 单一官方 OpenAI CLIP ViT-B/32；
- block 8–11 的 attention Q/V/out，rank 8，alpha 8，零残差初始化；
- Aegis `clean_probability >= 0.70` 且不在 A2 的 991 项黑名单中才承担
  GCE 分类监督；其余样本只有原始 CLIP 特征锚定；
- GCE q=0.5，feature distillation=2.0，无 MixUp；
- head LR 5e-5，LoRA LR 2e-5，4 epochs，seed 42；
- 运行前记录 epoch 0 指标，确认 LoRA 初始化没有改变 A2。

阈值和超参数在平台评测前冻结，不做阈值网格。

## 晋级门槛

F6 只有同时满足以下条件，才允许进入全量 A2+LoRA replay：

1. epoch 0 的模型状态严格加载，视觉基座张量与 A2 完全一致；
2. `evaluation` clean-core micro 相对 epoch 0 至少提高 0.20pp；
3. trusted macro 不下降超过 0.10pp，raw micro 不下降超过 0.20pp；
4. validation feature drift 不超过 0.50%；
5. flip prediction agreement 不下降超过 0.20pp；
6. `cross_audit` 上改进方向不为负。

任一主门槛失败即停止 A2+LoRA 独立分支，不通过延长 epoch 或扫描学习率
挽救；下一分支改为 A2 固定轮数 full-fit，而不是继续堆叠模块。

## 反方审查

最强反对意见是：A2 平台收益未必只来自 991 个样本的筛除，它还包含
训练轮数、MixUp、检查点选择等共同变化；因此“两个平台正收益组件相加”
不保证继续正收益。F6 的作用正是隔离新增视觉适配：同一 A2 父模型、同一
评估集合、epoch 0 配对对照，只改变 LoRA 训练后的参数。小型门控通过也
不能证明平台必涨，只能证明值得进行一次预注册的全量复现。

## 合规性

- 团队仓库和 A2 资产全程只读；实现、输出和提交均位于独立 worktree；
- 不使用外部数据、测试标签、测试分布自适应或人工清洗；
- 最终候选仍是一个 CLIP ViT-B/32、一个分类头、一个检查点；
- 若生成 TTA，只使用同一检查点的原图与水平翻转概率平均。

研究依据：

- DeFT（NeurIPS 2024）：https://proceedings.neurips.cc/paper_files/paper/2024/hash/6af08ba9468f0daca4b8dd388cb95824-Abstract-Conference.html
- PTNL（ICCV 2023）：https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Why_Is_Prompt_Tuning_for_Vision-Language_Models_Robust_to_Noisy_ICCV_2023_paper.html
- NLPrompt（CVPR 2025）：https://openaccess.thecvf.com/content/CVPR2025/html/Pan_NLPrompt_Noise-Label_Prompt_Learning_for_Vision-Language_Models_CVPR_2025_paper.html

## 执行结果

77 项自动测试通过后，F6 于 2026-07-20 在空闲 GPU 上完成固定 4 epochs；
训练过程中没有团队计算进程、非有限值、加载停滞或特征崩塌。best 为 epoch 4：

| 指标 | A2 epoch 0 | F6 epoch 4 | 差值 |
|---|---:|---:|---:|
| clean-core micro | 82.9167% | 83.1019% | **+0.1852pp** |
| clean-core macro | 82.8525% | 83.0191% | +0.1666pp |
| raw micro | 69.4149% | 69.6189% | +0.2040pp |
| trusted macro | 80.5921% | 80.7390% | +0.1469pp |
| proxy macro | 79.2032% | 79.4306% | +0.2274pp |
| flip agreement | 87.3215% | 87.5792% | +0.2577pp |
| feature drift | 0.0001% | 0.1792% | +0.1791pp |

主选择集包含 6,480 个 clean-core 样本；`+0.1852pp` 对应净增 12 个正确
样本，而预注册 `+0.20pp` 至少需要净增 13 个，因而严格判定为未通过，
不能事后下调门槛。

未参与 checkpoint 选择的 `cross_audit`（1,005 张；733 个 clean-core）给出
同方向证据：clean-core micro `+0.6821pp`、trusted macro `+0.4394pp`、
raw micro `+0.1990pp`；但 flip agreement `-0.2985pp`。该旁证说明视觉适配
并非明显错误方向，却不能推翻主门槛失败。

- best checkpoint SHA-256：
  `c791e012319a39519c5d3154e0cb1bb94a00cd4a42a0b0f52605c8743802b341`；
- 结论：**F6 不晋级、不生成平台提交、不在独立工程中重复团队已规划的
  全量 A2+LoRA；下一条预注册分支改为 A2 低学习率 fixed full-fit。**
