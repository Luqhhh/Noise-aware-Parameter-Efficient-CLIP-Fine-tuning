# 交给 GPT-6 的项目现状与方法空间说明书（2026-09-23）

**目的**：请求一次**大范围、本地优先、高收益**的方法搜索设计。
**写作口径**：所有数字均标注为「**实测**」或「**推算**」。凡是推算，都写明了它依赖的未验证前提。凡是无量化结论的，直接写「无量化结论」，不用推测填补。

---

## 0. 一句话现状

复赛（750 类 / 148,695 训练 / 37,444 测试）上，**单一 full-FT 模型（LP 初始化 → 全参数微调）本地 macro 72.35% / micro 73.41%，平台 60.9657%**；历史上有过几十轮实验，结论是**这一族已贴自身天花板**，而**本地验证对平台分数结构性无信息**（11.90pp 的本地/平台差在 val 上不可见）。因此需要**换盆地**，不是在现有配方上继续调参。

---

## 1. 不可突破的硬约束（违反即不能作为正式方案）

- **骨干**：必须 CLIP ViT-B/32；**权重**必须 OpenAI 官方公开版本（代码内硬校验）。
- **数据**：只能用当前阶段官方数据；**禁止**外部数据、其他阶段数据、以及**由外部数据训练得到的**教师/分类器/原型/缓存特征。跨阶段不可复用数据、checkpoint、特征缓存、伪标签、拟合的 prior（代码与公式可迁移）。
- **测试集只读**：禁止测试图入训、自监督、无监督域适配、测试时训练/适配/梯度更新、用测试预测分布调参、按测试图像聚类调整、用测试分布造原型。
- **禁止集成**：多模型、多 checkpoint 投票、多 seed 平均、不同 backbone 融合、多头融合、logits/概率加权组合**全部不允许**。最终 = **单 checkpoint + 单确定性推理脚本 + 一份 pred_results.csv**。
- **多尺度 + Flip TTA 已获组委会书面裁定合规**（2026-08-31，前提仍是单 checkpoint + 单确定性流程）。「同一模型多视角预测平均」同样已裁定合规。
- **训练必须全自动可复现**：噪声筛选、样本打分、重加权、伪标签、课程学习、checkpoint 选择都必须由代码完成；**禁止**手工删图、人工维护黑名单、手工改标签作为正式训练输入、依赖未纳入脚本的中间文件或无法重建的缓存。
- **提交格式**：`文件名, 0001`（逗号+空格+4位补零），ZIP 内只含 `pred_results.csv` 且与外部字节一致。**平台每日提交额度 2 次**。

**规则灰区（官方未明确，未获答复前按最保守解释，不得进正式路径）**：EMA / SWA / 权重平均 / model soup 是否算融合；多 prompt 模板平均；是否允许用官方类别名构造文本 prompt；是否允许用 OpenAI CLIP 文本编码器生成类别文本原型；是否允许使用当前阶段数据训练 teacher 再训单一 student；是否允许伪标签替换原始标签；是否允许训练集内部自监督预训练。

---

## 2. 数据与阶段事实（全部实测）

| 项 | 数值 |
|---|---|
| 类别数 | 750 |
| 训练池 | 148,695（133,815 train_dev + 14,880 val_dev，seed 42，每类至少保护 4 个训练组） |
| 测试集 | 37,444 |
| **类别标识形式** | **纯 4 位数字 `0000`–`0749`**；训练目录名即 `0000/0001/...`；测试图名为 md5 哈希。**全仓库不存在任何 synset / 物种名 / 中英文类名映射文件** |
| 类支持度 | `<20` 样本：**仅 2 类 / 16 张**；`20–99`：12 类 / 675 张；`≥100`：736 类 / 133,124 张。最小 4 样本（class 183），中位数 176 |
| 内容去重 | 146,381 独立内容组，其中 **1,882 个跨标签冲突组**（**复赛侧未处理**） |
| 测试集几何 | 只有 **807 种 (w,h)**，**每一维都 ≤500px**；375×500 一种占 **37.6%**，前四种约 72%；主导长宽比 **0.75（纵向）**。train/val 长边 800–1390+，主导 **1.33（横向）** |
| val 的性质 | **训练池的忠实 iid 抽样**。逐切片比对：「长边≤500 且 portrait」这档测试集长相在 train 占 **8.50%**、val 占 **8.58%** |

**⚠️ 由上一行直接推出的硬边界**：**不存在任何本地切法能构造「跨来源代理」**。11.90pp 的来源漂移对 val 上**一切**指标结构性不可见。「留一来源验证」这条路在复赛数据上不成立（已实测否掉）。

---

## 3. 当前最佳与全部平台成绩（实测）

| 候选 | 本地 micro | 本地 macro | 平台分 | 状态 |
|---|---|---|---|---|
| RM-FT（full-FT，LP 初始化） | **72.9032%**（登记值）/ 72.8696%（独立复现） | 71.8362% / 71.8054% | **60.96570879179575%**（22,828/37,444） | **现役最佳** |
| RM-LT（sqrt 类平衡采样） | 72.6949% | 71.8655% | 60.885589146458706%（22,798/37,444） | 比 FT 低 0.0801pp / **30 张** |
| RM-LP（冻结骨干+线性头） | 63.6358% | 62.6319% | **从未上传（null）** | 保留为共享初始化 |
| **V3 winner** `RM_V3_B1024_E16_LR4`（batch1024 / 16 轮 / head_lr 4e-4 / backbone_lr 1.2e-5 / 锚点 2.0） | **73.4140%** | **72.3472%** | 未上传 | 本地最佳；已被用户冻结，`search_closed: true` |

**平台额度**：每日 2 次，**整个复赛阶段总共只用了 2 次**（RM-FT、RM-LT）。登记表 9 行全部 `status=ready`，`platform_score` / `submission_id` / `submitted_at` / `platform_period` 字段全空。

---

## 4. 六条会决定搜索形态的结构性事实

1. **本地/平台差 11.9039pp**（本地 micro 72.8696% vs 平台 60.9657%），且**对 val 上任何指标结构性不可见**。
2. **迁移率（唯一跨域指标）**：FT **0.8366**、LT **0.8375**（= 平台分 / 本地分）。
3. **本地↔平台相关系数：官方文档里不存在。** 手上只有 2 个平台点，且**两次都是排序反向**：① 初赛同批实验中本地 val 最高（69.47%）的候选平台最差（59.89%）；② 复赛本地 macro 上 LT 更高（71.8655 vs 71.8054）而平台 FT 更高（低 30 张）。
4. **单种子噪声 0.90pp**（初赛实测：同配方两 seed 平台 61.21% vs 60.31%）。**复赛阶段所有候选（FT/LT/NPU/LoRA/V3 三点/搜索四点）全是单 seed 42，从未做过第二 seed 复验。**
5. **full-FT 这一族已贴自身天花板**：平台 FT/LT 只差 0.0801pp（30 张），本地全部差异落在噪声内；本地 train_accuracy 86.83% vs val micro 72.87%，泛化裂口 13.96pp，且 epoch 6→8 只 +0.16pp、学习率已退火到 1e-6 → 「单纯延长同一配方期望收益很低」。
6. **唯一一次真正的「换盆地」是 LP→FT**：池内 macro **+9.23pp**、micro +9.267pp，按 support 五分位均匀（**+7.97 ~ +9.99pp**），750 类中 560 好 / 162 平 / 28 差。**且该增益在第 2 个 epoch 就基本到位（69.29%）** —— 来自「解冻」这个动作本身，不是「学得更久」。

### 4.1 由上一条引出、必须点明的逻辑缺口（**这是本说明书最重要的单点**）

若假设两臂迁移率相同（0.8366），则 LP 平台分应约 **53.2%**，即 LP→FT 的 +9.23pp 本地增益换来了约 **+7.7pp 平台分**（60.97 vs ~53.2）。
**但这个推算默认了「两臂迁移率相同」—— 而那恰恰是从未验证过的前提**（RM-LP 从未上传，平台分为 null）。
⇒ 我们**并不知道那次大跃升究竟迁移了多少**。「大跃升能迁移」目前是**假设，不是事实**。
⇒ 而它一旦被测出，就给出**唯一一条把本地读数换算成平台分的函数**，此后所有候选的排序才有依据。

---

## 5. 已明确关闭的方向（按类别，含判据）

### 5.1 数据侧
| 方向 | 实测结果 |
|---|---|
| 数据增强 RRC+Flip | 无增强 69.86% vs 有增强 69.77%，**−0.09pp** |
| ColorJitter / RandomErasing | **67.36%**，显著破坏细粒度判别 |
| MixUp（复赛，α=0.2 / 0.4，p=1.0，8 轮） | micro 基线 72.8696 → α0.2 **72.5000**（−0.37pp）→ α0.4 **71.9288**（−0.94pp）；**8/8 检查点全负**，按 α 单调 |
| 重标 100 个样本（0.1%） | **−0.42pp** |
| 删样本（降噪） | 删 991 高精度 **+0.90pp** > 删 6,354 中精度 > 删 8,680 低精度 **−0.76pp**（即「覆盖面优先」是错的） |
| 输入几何重采样（把 val 压到 test 几何） | 500px **−0.1815pp** → **证伪**「11.90pp 来自输入几何」 |
| 训练侧 OOF 连续降权（RM-OOFW） | 主判据 尾部 75 类 macro **+0.0113pp**（门 +0.20pp，**t=+0.04**）FAIL；整体 macro **−0.1311pp** |
| 在本地解释 11.90pp / 构造跨来源代理 | **已证明不可能**（见 §2 的 ⚠️） |
| RM-FULL（全量 148,695） | **用户已取消**，文档多轮写「不得自行启动」 |

### 5.2 损失侧
Label Smoothing、GCE q=0.9、ELR、EMA Loss、Head EMA、Prototype Weighting、结构化 head、Trusted Prototype-Contrastive、Dynamic Trust Refresh、Clean-Routed LoRA、Classwise CL-only drop、半监督回收、OOF relabel/pseudo-label、OOF 3-tier discrete weight —— **全部已关闭**（其中 ELR / prototype-contrastive / clean-routing / dynamic-trust 另被代码闸门硬禁）。
特征蒸馏权重下调（锚点 2.0→0.5 / 2.0→0.0）：**两次训练崩溃（`cudaErrorUnknown`），无有效结果**，机制至今未定。

### 5.3 优化器与调度侧
| 方向 | 实测 |
|---|---|
| batch 1024 × 8 轮（NPU） | macro **67.3704%**，相对 GPU FT **−4.4658pp**（低于 −2pp 止损）→ 关闭 |
| batch 32 × 8 轮（NPU） | macro **71.9047%**（+0.0685pp），未达 +0.20pp 晋级门 → 仅作效率配置 |
| V3 B64×8 | macro **70.8299%**（−1.0748pp，低于精度下界）→ 淘汰 |
| V3 B128×16 | macro **71.9320%**（达线 +0.0272pp）但 CLI 3663.7s，比 B32 慢 792.8s |
| 同轨迹 checkpoint averaging（linear soup） | 已关闭（代码存在，从未用于复赛） |
| 延长同一配方 | 8 轮 cosine 已退火到 head_lr 1e-6 / visual_lr 3e-8，epoch 8 Δ 已收敛到 **+0.16pp/2 轮** |
| 全部 LR / epoch / batch / worker / loss 扫描 | 用户已冻结 winner，`further_search_authorized: false` |

### 5.4 模型结构侧
| 方向 | 实测 |
|---|---|
| Cosine head | 初赛 **63.61%** vs Linear head **69.86%**，差约 **6pp** → 关闭 |
| CE 下部分解冻 / 4-view TTA / vertical flip / 更多视图 | 已关闭（三视图融合 62.03% < 纯 attention 62.67%） |
| **数字类别语义 prompt** | 固定数字 prompt 的 raw / clean-core 仅 **0.23%**；500 个文本方向 90% 能量秩为 1 → **按语义 prompt 经验处理数字类别文本侧无效** |
| 288px + LoRA r8 | 相对 224 基线 raw +0.30pp / clean-core −0.38pp / drift 21.3%（上限 15%）→ 门禁全失败 |
| 共享分类头范数对齐 / H0·H1 共享 head 重拟合 / 恢复蒸馏 / RECOVERED_SELFTRAIN | 分别 **63.3676%**（低于匹配父模型 66.7681%）、65.8629% / 66.1914%（均低于 66.7681%）、**67.2167%**（低于同轮 68.2982% 共 1.08pp）、**61.4010%**（低于父模型 5.37pp）→ 全部不晋级 |

### 5.5 推理侧
| 方向 | 判据 |
|---|---|
| Flip TTA（9 条融合规则，见 §6.1） | 首轮闸门按设计**拒绝**，未绕过；且低于 +0.30pp 投资门 |
| 多尺度 / prior / 测试批拟合 prior | 属「不同来源协议」，分数不可推广（初赛曾拿到 72.4677% 但被单独登记，不得充当基线） |
| 推理侧改预处理 | 只值 0.17–0.31pp，**11.90pp 关不掉** |
| FT94+LT06 logits 融合 | micro 72.9839%（+0.0806pp）；**RULE-05 禁止多模型 logits 加权** + 低于 +0.30pp 门 → 关闭、未上传 |

### 5.6 长尾侧
| 方向 | 实测 |
|---|---|
| sqrt 类平衡采样（RM-LT） | 本地 macro **+0.03pp** / micro **−0.2083pp**；固定尾部 75 类 macro **+2.57pp**；**平台 −30 张** → 平台选择 FT |
| 尾部 macro 当主判据 | 配对 SE **0.2833pp**，+0.20pp 门只值 **0.71 个 SE**（门设在分辨力边缘）；整体 macro 配对 SE 0.0925pp |
| 「用验证图片占比换算尾部宏观权重」 | **算术错误，已撤销**（尾 75 类占 750 类的 10%，验证图片 6.74% 不是该权重） |

### 5.7 PEFT 侧
| 方向 | 实测 |
|---|---|
| LoRA 后 4 层 / r8 / alpha16（NPU 八轮） | macro **64.1064%** / micro 65.1478%，相对 FT **−7.7298pp** → 关闭，不派生扫描 |
| LoRA（本机 GPU 同配方） | macro **64.1239%** / micro 65.1747%（跨后端差仅 0.0175pp）但**从未入库** |
| 把 LoRA 当「低秩约束」的对照 | **不成立**：它与 `visual` 共用 backbone_lr（2e-5 而非 FT 的 3e-6），**不是单变量对照** |
| PEFT LN-tune | 已关闭 |
| 冻结特征+线性头（RM-LP） | macro **62.6319%** / micro **63.6358%** → 保留为参考基线与共享初始化 |

**由 5.7 推出的判断**：LP 62.63 → LoRA 64.11 → full-FT 72.87。**容量是这里的杠杆**（+9.23pp 来自解冻）。所以**低容量 PEFT 方向（prompt / BitFit / LN-tune）预期收益低**，不建议优先。

### 5.8 噪声鲁棒侧
OOF 连续降权、OOF relabel/pseudo-label、OOF 3-tier discrete weight、Classwise CL-only drop —— 已关闭。初赛对照：**hard gate（硬删）无独立增益，soft gate（软降权）才有 +0.042pp** → 方向应是**软降权而非硬删**。

---

## 6. **已实现、有校验分支、有单测，但 0 个 config 启用过**的机制（最大未开采区）

### 6.1 被 `trust` 闸门挡住的整套机制族

复赛主线的合规闸门 `rematch_assets.py` 里有一条硬禁止：**`trust.enabled` 必须为 false（"trust disabled in first round"）**，另有 `loss_reweighting == 'none'`、`balanced_softmax_tau == 0`、**任何 `loss.<x>` 子字典里 `enabled: true` 一律拒绝**。被这条挡住、**但代码/单测/CLI 全部齐全**的机制：

| 机制 | 实现位置 | 被挡原因 |
|---|---|---|
| `trust.subspace_projection` | 独立模块 `trust_subspace.py`（**全文约 90 处引用**）+ 专用 gate CLI + `config.py` 里**全仓库最长的一段校验**，要求 `trust.enabled` + 在线 `visual_lora` + `loss.name=gce` + 强制 `feature_distillation_weight>0` + 与 11 个其他机制互斥 | `trust` 首轮禁用 |
| `prototype_contrastive` | `trainer.py:419-433`（momentum / temperature / threshold / loss_weight / start_epoch） | 同上 |
| `dynamic_trust` | `trainer.py:435-437` | 同上 |
| `clean_routing` | `trainer.py:413-417` | 同上 |
| `loss.dual_gce` | `losses.py:273` + `trainer.py:288`（suspicious_fraction / clean_q / suspicious_q，要求 `mixup_probability=0` + `trust.enabled`） | 同上 |
| `loss.snscl` | 独立模块 `snscl.py`（**全文 84 处引用**）+ 单测，8 个正数超参 | 同上 |
| `loss.contrastive` | `losses.py:213` + `trainer.py:987`（要求 `peft_mode=feature_adapter` + `use_cached_training`） | 同上 |
| `loss.cyclic_filter` | `losses.py:294/327` + `trainer.py:819`（要求 `peft_mode=frozen` + `cycle_epochs>=2`） | 同上 |
| `loss.active_forgetting` | `losses.py:70` + `config.py:192-216` 完整校验分支 | 同上 |
| `loss.adaptive_cap` | `losses.py:474 class AdaptiveLossCap` + `trainer.py:397` | 同上 |
| `loss.class_prior_adjustment_tau` | `losses.py:14`，`trainer.py` 三处调用点，默认 0.0 | 需要 `balanced_softmax_tau != 0`，而后者被禁 |

**说明**：这是**项目自设的首轮策略**，不是比赛规则。但启用前需注意其中若干机制与规则灰区重叠（伪标签替换原始标签、teacher-student），**应先取得组委会答复**。

### 6.2 枚举里从未被取过的值

| 枚举 | 全部取值 | 复赛用过 | **从未用过** |
|---|---|---|---|
| `loss.name` | `cross_entropy` / `double_softmax_cross_entropy` / `gce` | CE、GCE | **`double_softmax_cross_entropy`**（已实现、已接线、0 config） |
| `classifier_mode` | `linear` / `anchored_residual` | 仅 `linear` | **`anchored_residual`**（0 config） |
| `peft_mode`（11 种） | frozen / feature_adapter / visual_ln / ln_post_proj / visual_lora / visual_lora_last_mlp / visual_lora_mlp_lora / visual_lora_mlp_adapter / visual_mlp_adapter / visual_prompt / full_finetune | **仅 3 种**（frozen、full_finetune、visual_lora） | **8 种从未在复赛用过**（含 `visual_prompt` = Prompt Tuning、`visual_mlp_adapter`、`visual_lora_mlp_adapter` 等，超参全在 schema 里） |
| `selector_metric`（8 种） | raw_micro / raw_macro / trusted_* / proxy_* / clean_core_* | **只用 raw_macro（tiebreak raw_micro）** | 其余 6 种从未作选模指标 |
| `model.unfreeze_last_n_blocks` | 0–12 | 从不设（=0） | 部分解冻从未在复赛做过（CE 下部分解冻在初赛已关闭） |

### 6.3 诊断分支从未启用
`promotion.*`（含 `minimum_selector_gain` / `maximum_mean_feature_drift` / `required_predicted_class_count`）整个 section 在全部复赛 config 里**不存在**；`evaluation.drift_budget` / `drift_penalty`（全写 0.0）/ `clean_core_threshold` / `measure_flip_consistency` / `evaluate_initial_checkpoint` / `train.early_stop_patience`（全 0）/ `save_epoch_checkpoints`（全 false）/ `lr_warmup_epochs`（全 0）—— **全部读到但恒为默认**。

### 6.4 整模块存在、有 CLI、有单测，但从未进入复赛主线
`kta_curriculum.py`、`scale_reweighting.py`、`view_reliability.py`、`fine_trust.py`、`classifier_norm.py`、`source_bias.py`、`trajectory.py`、`soup.py`、`structural.py`、`multiprototype.py`、`local_prototype.py`、`local_residual.py`、`local_inference.py`、`part_token_adapter.py`、`local_adapter.py`、`local_feature_adapter.py`、`cvrg_crossfit.py`、`pace_protocol.py`、`prior_alignment.py`、`calibration_binding.py`、`structured_allocation.py`、`balanced_transport.py`、`balanced_inference.py`。
对应的 CLI 入口（`sweep_*` / `train_local_*` / `evaluate_*_gate`）同样「有入口无运行」。

### 6.5 已测出正读数、但没做成候选的
| 方向 | 正读数 | 为什么没继续 |
|---|---|---|
| **Flip TTA 全融合规则扫描** | 本地 micro **+0.155 ~ +0.242pp**，**9/9 条规则同号**，配对 t 1.19~1.68。最佳 `entropy_weighted_probabilities@T2.0`：micro 73.1116%（**+0.2419**）、macro 72.0178%。分层：head **+0.3075pp**、medium **+0.3393pp**、tail **+0.0249pp**（tail macro 甚至 −0.0196pp） | ① `infer.py` 首轮闸门拒绝，未绕过；② 低于 +0.30pp 投资门；③ **本地指标不设晋级门槛**；④ 若要用，应用**参数自由的代码默认 `mean_logits`**（按 val 取 argmax 是被禁的做法） |
| bottom-10% macro | RM-OOFW 上 **+0.4397pp**；FT/LT 配对 **+0.96pp** 同向 | 事前声明为「事后观察，非判据」——拿看到结果后才挑的口径宣布成功正是预注册要防的事 |
| RM_LORA 本机 GPU 结果 | macro 64.1239% / micro 65.1747% | 产物被 gitignore，纯记录待办，至今未做 |

---

## 7. 从未被触碰的结构性空白（不在任何 config 的维度里）

1. **全量 148,695 训练（+11% 数据）** —— 复赛现存数据里**唯一还没喂给模型的 11%**；同一阶段官方数据，规则上完全合法。RM-FULL 曾被取消，文档多轮写「不得自行启动」。**但当初取消的现实约束是算力，现在 NPU 单点约 34 分钟，算力约束已不成立。**
2. **超过 16 轮的训练** —— 复赛最长 16 轮。而**初赛团队最佳实践是 epoch 44**。本地「延长同配方收益低」的证据（+0.16pp/2 轮）是在 8 轮 schedule 下测的，**16→44 轮是一个从未测过的区间**。
3. **第二 seed 复验** —— 方法论明确要求「任何候选的收益声明必须在第二个 seed 上复验」，而**复赛所有候选全是单 seed 42**。当前 0.90pp 的噪声量级是从初赛借来的。
4. **740/750 与 741/750 类别覆盖的根因** —— LP 只预测 740 类、FT 741 类，**9–10 个类从未被预测**，**上限约 1.3pp**；未查清是样本太少还是类别映射问题。**零 GPU 成本**，此前被 GPT-6 明确建议「立刻做」，**至今未做**。
5. **多尺度 + Flip TTA 从未走过正式提交链路** —— 已获组委会**书面裁定合规**，本地已实测正收益（见 §6.5），但被 `infer.py` 的首轮闸门挡住。这是一条**与训练正交、可叠加在一切成果之上、几乎免费**的增益。
6. **`class_to_idx.json` 与测试集类别的对应关系** —— 若测试集存在训练集没有的类，该类永不可预测；未核。

---

## 8. 我认为最关键的三个判断（供你反驳）

### 判断 1：现有搜索协议的目标函数与「能否换盆地」不匹配
队友 V4 预注册了 **82 个点**（A=GCE/mixup/randaug/cutmix、B=损失族、C=软修复与伪标签、D=锚点/解冻层/层衰减、E=长尾、F=输入分辨率、G=SAM、H=线性探针头），全部围绕**同一个 base**（batch1024 / 16 轮 / head_lr 4e-4 / backbone_lr 1.2e-5 / 锚点 2.0 / LP 初始化）**各改一个旋钮**。
其晋级门槛是 **`local_significant` = macro Δ≥+2.0pp 且 micro Δ≥+1.0pp**。
**问题**：单变量微调**按设计就到不了 +2.0pp**（§5 里几十轮实测的增益几乎全在 0.02–0.09pp 量级），而**该门槛本身又建立在「本地 +2pp 能换来平台收益」这一未被验证的假设上**（见 §4.1）。**门槛与手段、门槛与目标之间都不自洽。**
✅ 需你判断：是**改门槛**（承认本地 0.0x pp 不可判定、改用别的判据），还是**改手段**（放弃单变量网格、只做换盆地），还是两者都改。

### 判断 2：在这种结构下，**信息**比**分数**更值钱
平台额度 2/天、整阶段只用了 2 次。在「只有 2 个平台点、且两次都排序反向」的条件下，**任何候选的本地读数都无法排序**。而**一次 LP 探针提交**能把 §4.1 那个缺口补上：给出**唯一一条本地→平台的换算函数**，此后所有决策才有依据。相比之下，把一个名额用在「本地 +0.2pp」的候选上，**买到的信息量接近于零**。
⚠️ 这与用户当前规则「只有出现显著本地跃升的 candidate 才占用平台提交」**存在直接张力**。我倾向认为该规则对「**方法候选**」是对的，但应给「**标定测量**」留一个例外。请你表态。

### 判断 3：真正的未开采区在「被自家策略闸门挡住的已实现机制」，而不是「再想新方法」
§6 那张表的共同特征是：**代码写完了、单测过了、校验分支写了、然后 0 个 config**。其中 `trust.subspace_projection`（≈90 处引用 + 专用 gate CLI + 最长校验段）与 `snscl`（84 处引用）明显是**当初按主力方案建的**，被首轮策略一刀禁掉后就再没动过。
这类东西**实现成本已经沉没**，试跑成本 = 一次训练；而「再发明一个新方法」要付全额实现费。**在算力便宜、实现贵的现状下（NPU 单点 ≈34 分钟），优先清点这些沉没资产是理性的。**
✅ 需你判断：其中哪些（尤其是否要先就「伪标签 / teacher-student」取得组委会答复）值得插队。

---

## 9. 请你回答的问题（请逐条作答，可以反驳上述任何前提）

1. **搜索空间**：在上面这些约束下，你认为**还有哪些盆地**具备「本地 ≥1–2pp 量级跃升」的潜力？请给出**排序**，并说明每个盆地的**预期量级**与你判断它**能迁移到平台**的理由。
2. **判据**：在「本地对平台结构性无信息 + 单种子噪声 0.90pp + 平台额度 2/天」的条件下，**用什么判据决定一个候选是否占用平台名额**才是对的？我上面的「信息 vs 分数」框架是否成立？有没有更好的判据？
3. **V4 协议**：你同意 §8 判断 1 吗？82 点单变量网格该**关闭 / 缩减为哪几个 / 改造**？如果你认为它仍有价值，请说明我漏算什么。
4. **沉没资产**：§6.1 的机制族里，**哪几个最值得优先启用**？请具体到机制名，并说明预期收益与最坏情况。
5. **规则灰区**：为了打开被 `trust` 闸门挡住的机制，我们应该**优先向组委会确认哪几个问题**（按信息量排序）？如果组委会只答一个，答哪个最值钱？
6. **验证设计**：请给出一个**本地优先**的验证协议 —— 在只有一张 8GB 消费级 GPU（本机 8 轮 full-FT ≈1h49m）+ 一张 NPU（8 轮 ≈34min）的条件下，**如何用最少的训练次数可靠区分「真跃升」与「0.90pp 单种子噪声」**？需要考虑第二 seed 的成本。
7. **零成本项**：§7.4（类别覆盖根因，上限 1.3pp）与 §7.5（已合规的 TTA）这两件零 GPU 成本的事，你建议怎么做、什么顺序？
8. **我可能错的假设**：请指出本文档里**最可能被推翻**的一条结构性事实或判断，并说明怎样用最少的成本去证伪或证实它。

---

## 10. 附：供你调用的原始资料位置

| 内容 | 位置 |
|---|---|
| 比赛规则全文 | `COMPETITION_RULES_AGENT.md` |
| 可迁移经验与方法论陷阱 | `docs/lessons_learned.md` |
| 当前执行计划（权威入口） | `docs/current_execution_plan.md` |
| 平台成绩与回执 | `docs/rematch750_platform_results_20260922.md` |
| LP–FT 迁移对照预注册（含 §4.1 的缺口） | `docs/rematch750_lp_ft_transfer_prereg_20260922.md` |
| 特征漂移分析（drift–Δrecall +0.4013，可复用） | `docs/rematch750_feature_drift_20260922.md` |
| 推理侧探针（TTA 与输入几何，**请勿重复这两项测量**） | `docs/rematch750_inference_side_probe_20260922.md` |
| 本机近日搜索批（4 点，全部为负） | `docs/rematch750_search_batch1_20260923.md` |
| V4 的 82 点预注册与门槛 | 分支 `codex/rematch750_search_v4` 的 `search_manifest.json` + `configs/rematch750_search_v4/*.yaml` |
