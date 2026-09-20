# PRELIM75 v10 研究与预注册：L1 上的严格 SAM 配对

计划 ID：`PRELIM75_V10_SAM_20260920`

## Material Passport

- Origin Skill: `academic-research-suite`
- Origin Mode: deep research + experiment planning
- Evidence cutoff: 2026-09-20
- Repository anchor: `origin/main@a4132de`
- Verification status: repository history and primary-method sources reviewed; **not implemented or trained**
- Version label: `research_preregistration_v1`

## 结论

下一项有信息增益且不与队长 prior 线重叠的正式训练实验，应是从 v7 L1 独立加载的普通 AdamW 与非自适应 SAM 严格配对。L1 是当前无 prior 平台最高 `69.2794%`；`G0 + legacy test-batch prior0.9 = 72.4677%` 只作为不同协议的绝对分数参照，不进入本实验训练、推理或参数选择。

本轮不回退到 V5X/O2L（平台 `64.0806%`），不继续 v8 训练源偏置（`67.8816% / 62.0659%`），也不扫描 v9 原图回采样参数。v9 R0/R1 已回填 `69.1633% / 69.2033%`，均低于 L1 `69.2794%` 并已关闭；其唯一变量是冻结推理像素来源，与本轮优化器机制正交，因此该结果不改动本预注册协议。

## 为什么优先 SAM

1. v5–v7 表明普通续训已经进入低收益区：无 prior 从 C0 `68.4544%` 提高到 L1 `69.2794%`，最近一轮 L1 相对 G0 仅 `+0.0520pp`。
2. 当前训练仍有噪声标签和约 25.45% 零分类权重行，单纯增加训练轮数或恢复同源伪标签已有直接负证据。
3. SAM 直接优化参数邻域内的最坏损失。原始 ICLR 2021 工作报告了图像分类、迁移微调和标签噪声下的稳健性；2026 年的噪声标签研究进一步从梯度分量角度分析了其抑制噪声记忆的机制。
4. SAM 不改变数据、标签、模型架构或最终推理；最终仍是一个 CLIP ViT-B/32 checkpoint，符合单模型和官方数据边界。

主要依据：

- Foret et al., *Sharpness-Aware Minimization for Efficiently Improving Generalization*, ICLR 2021: <https://openreview.net/forum?id=6Tm1mposlrM>
- Luong et al., *Understanding SAM's Robustness to Noisy Labels / SANER*, 2026: <https://openreview.net/pdf/3ee835939c572faaf31b17061d95c0f70661a2ef.pdf>
- Xu and Pang, *NCSAM*, 2026 preprint: <https://arxiv.org/abs/2601.19947>

SANER/NCSAM 只作为后续文献背景，不进入第一轮。只有标准 SAM 在真实平台明确胜出后，才允许另行预注册一个噪声补偿升级；禁止把三者同时扫描。

## 固定父模型与资产

- 父模型：v7 L1
- 平台基线：`69.2794%`
- checkpoint：`outputs/prelim75_v7_20260918/L1/candidate.pt`
- checkpoint SHA-256：`cd485c7beed2781ac13d08fbd33d63b8f1c46971be008966f7280c67cec1fcee`
- fallback CSV SHA-256：`5277b2c33f63b951ea2236ae2e50b8c574ef9f8d2f989e8239fcce7b065c0171`
- fallback ZIP SHA-256：`55d25f1fea9c57ba3431164d53d21f34d20baa2a49c805ad87cca188b5634c5a`
- 固定几何、监督、映射和 split 必须沿用 L1 已绑定资产；正式配置只能在执行机重新计算并写入其精确 SHA 后提交。

当前 `x28639` 执行环境没有 L1 checkpoint、v3 冻结几何及其监督资产，GitHub Releases/Actions 也没有可下载副本。因此本文件仅完成研究注册；在资产精确同步前禁止使用旧 F1/O2L checkpoint 代跑，禁止生成伪 v10 包。

## A0：无更新 sharpness 与资源门禁

A0 从 L1 读取固定 sample epoch 19 的首个有效 batch，不创建 optimizer、不读 test、不更新参数：

1. 用 L1 原损失计算第一次梯度；
2. 对 L1 原许可的全部可训练参数施加非自适应扰动
   `e_w = 0.05 * g / (||g||_2 + 1e-12)`；
3. 用完全相同的 images、Flip、scale、冻结框、targets、weights 和 anchor reference 做第二次前向；
4. 逐位恢复全部参数，记录原损失、扰动损失、绝对/相对 sharpness、梯度范数、实际扰动范数、显存峰值和单 batch 时间。

机械门禁：

- 任何 NaN/Inf、冻结参数梯度泄漏、恢复后参数不逐位相同：停止；
- 实际扰动范数不在 `0.05 ± 5e-6`：停止；
- 扰动损失没有严格增加：停止，不改变 `rho`；
- microbatch 16 仍 OOM，或根据 smoke 推算 S1 超过 8 小时：停止；
- A0 只证明机制和资源可执行，不按准确率选参数。

## S0/S1 唯一正式对照

| 项目 | S0 control | S1 treatment |
|---|---|---|
| 父模型 | L1 独立加载 | 同一 L1 独立加载 |
| sample epochs | 19/20/21 | 19/20/21 |
| 正式轮数 | 3 | 3 |
| optimizer | fresh AdamW | 同一 AdamW，外层加标准 SAM |
| scheduler | L1 的 3-epoch cosine horizon，floor 0.01 | 完全相同 |
| LR/WD | visual `1e-6/0`；head `1.5e-7/1e-4`；O3/PTA `3e-6/0` | 完全相同 |
| loss | L1 的 probability-fusion GCE + anchor | 完全相同 |
| SAM rho | 无 | 固定 `0.05`，不扫描 |
| 选择 | 固定末轮 | 固定末轮 |
| 最终推理 | L1 原十视图、T=1.5、无 prior | 完全相同 |

S1 每个有效 batch 必须执行：第一遍累计完整 batch 梯度 → 保存 FP32 原参数 → 统一归一化扰动 → 清梯度 → 完全相同 batch 第二遍 backward → 逐位恢复原参数 → `clip_grad_norm_=1.0` → AdamW step。第一次梯度只用于扰动，第二次梯度才用于更新。不得改变 loss、权重、视图或随机顺序来节省计算。

## 审计与停止规则

- common-control 必须证明 S0/S1 首 batch 的索引、图像张量、Flip、scale、框、targets、weights、global/local logits、anchor reference 和首遍 loss 对齐。
- S1 必须记录每步 base gradient norm、perturbation norm、perturbed loss、second gradient norm、恢复检查和耗时。
- 重叠诊断只拒绝 raw 或 clean-core 相对 L1 下降超过 `2.0pp` 的明显失败，不用于选 epoch 或调 `rho`。
- 两组通过后各生成一个 24,967 行、单 checkpoint、无 prior ZIP，并运行内置与独立提交检查。
- 平台以 S0、S1 顺序评测：S1 不高于 S0，关闭 SAM/SANER/NCSAM；S1 高于 S0 但未超过 `max(L1,S0)` 至少 `0.30pp`，只记录小收益并关闭；达到 `+0.30pp` 才允许另行研究升级。
- 预算包含 A0、smoke、S0/S1 训练、诊断和推理，GPU 累计上限 `28,800` 秒；不得通过缩小训练清单规避预算。

## 合规边界

- 仅使用当前赛段官方 train、原数字标签和 train 内已冻结监督；
- backbone 固定官方 OpenAI CLIP ViT-B/32；
- 不使用外部数据、类别文本、外部教师、第二模型、模型平均、EMA/SWA、测试时训练或模型投票；
- 不读取、复制或重估 prior0.9，不做输出拉平或平台分布反推；
- test 只在 S0/S1 方案和 checkpoint 冻结后用于确定性推理及格式检查；
- 平台反馈不用于修改 `rho`、LR、epoch、loss、视图或门槛。

## 执行前必须满足

1. v9 R0/R1 已回填 `69.1633% / 69.2033%` 并同步记录，未据此改动本协议；
2. 执行机必须提供 L1 checkpoint、fallback CSV/ZIP、v3 几何、完整监督与官方 train/test；
3. 在最新 `origin/main` 上检查所有 refs，无其他 SAM 实验在运行；
4. 完成实现、单元测试、A0 和 smoke 后才允许正式 S0/S1；
5. 每个阶段按 `AGENTS.md` 立即提交并推送，不积攒结果。
