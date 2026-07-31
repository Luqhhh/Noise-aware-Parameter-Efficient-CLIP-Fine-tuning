# M1 向平台已知 F1/A2 的冻结迁移协议

日期：2026-07-20

## 目的

M1 已在 J0 上通过预注册门控。迁移阶段不再改变方法，只验证该信号能否复现到具有平台因果反馈的 F1 与 A2 检查点。团队仓库始终只读；缓存、转换检查点、日志和提交包均写入独立工作树。

## 冻结方法

- 单个 OpenAI CLIP ViT-B/32 检查点；
- 原生 224px 全局前向；
- 最后一个视觉 Transformer block 的 12-head 平均 CLS→patch attention；
- attention top-5 patch 的加权中心；
- 固定 160×160 局部裁剪并放大回 224×224；
- 全局/局部按 1:1 概率平均；
- 无参数扫描、无类别先验、无测试集适配、无第二模型。

## 固定顺序

1. F1：原生格式且有平台 bare/flip 对照，先缓存 center，再缓存冻结 M1；
2. A2：先完成只读格式与权重等价审计；只有可无损转换为 Aegis 单模型格式，才执行同样 center/M1 验证；
3. 每个检查点独立判断，不允许用一个检查点的结果调另一个检查点的超参数；
4. 测试集只在对应检查点通过迁移门槛、推理 CLI 与产物审计后运行一次。

## 每检查点迁移门槛

1. candidate 内的原生 global logits 与独立 center cache 最大绝对差为 `0`，预测一致率 `100%`；
2. clean-core micro 至少 `+0.10pp`；
3. trusted macro 不得下降超过 `0.05pp`；
4. raw micro 不得下降超过 `0.10pp`；
5. local/global prediction agreement `<99%`；
6. 失败即停止该检查点，不扫描 crop、top-k、attention block/head 或融合权重。

## 冻结输入

### F1

- checkpoint：`/home/x28639/projects/AegisCLIP-Noise-Robust/outputs/F1_VISUAL_LORA_CLEAN_CORE/seed42/checkpoints/best.pt`
- checkpoint SHA-256：`7da95e427b959e85cbbf37c99d47d9909b941032e836fc219aaea8e690d72cc4`
- validation CSV SHA-256：`54a790b35f836cfba4c19cbb5fe38c4b1b37aab62cc9d477f9285496b2d5568e`
- 已知平台因果对照：bare `60.5159`，horizontal-flip TTA `61.1007`。

### A2

- source checkpoint（团队仓库只读）：`/home/x28639/projects/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/outputs/oof/nr_cl_knn_drop/seed42/checkpoints/best.pt`
- checkpoint SHA-256：`74ad2856e4449a42397edbda599ae79e8a4c6a6fa923624ef4e91a35e20a2a4c`
- validation CSV SHA-256：`607e019165912bb0639efb456b7e8dea122b3e8579a2344dedb8109798921eae`
- 已知平台对照：horizontal-flip TTA `61.2128`。

## F1 迁移结果

F1 的 center 与 M1 缓存均由上述冻结 checkpoint 和其原生验证拆分完整生成。实现审计的 global 最大绝对 logit 差为 `0.0`，预测一致率为 `100%`。

| 指标 | F1 center | F1 + M1 | 变化 |
|---|---:|---:|---:|
| clean-core micro | 81.5306% | 82.2472% | **+0.7166pp** |
| clean-core macro | 82.0538% | 82.6806% | **+0.6267pp** |
| trusted micro | 80.4197% | 81.1555% | **+0.7358pp** |
| trusted macro | 80.6078% | 81.2735% | **+0.6657pp** |
| raw micro | 70.6766% | 71.6363% | **+0.9597pp** |
| raw macro | 70.6445% | 71.6100% | **+0.9655pp** |

补充诊断：local/global agreement 为 `75.8918%`；融合改变 `1,058 / 10,316` 个预测，raw 纠正 `289` 个、破坏 `190` 个，净增 `99` 个；clean-core 净增 `53` 个。被改变样本的全局 top1-top2 平均间隔为 `0.4977`，未改变样本为 `3.8759`。

产物哈希：

- F1 center cache：`dc36bfec1a1b7328035ee65d7729bf9d836c74ed58ad58f1a8234cfe7c0b860c`；
- F1 M1 cache：`5f927bc9740ec5ce1725a7cfab07fbdc40f3e3dda5213ce59a419092edbf614c`；
- F1 evaluation JSON：`555a1751f9022575c332e56678308b3b1569a2834062a3f667b96b54db337795`。

F1 通过全部迁移门槛，结论为 **PROMOTE TO TEST INFERENCE**。该结论只批准冻结的两视图 M1，不批准与水平翻转叠加。

## A2 无损转换审计

团队 A2 源检查点始终只读。转换器首先核对源 SHA-256 和类映射，再把同一 154 个状态张量装入结构等价的 Aegis frozen-CLIP + linear-head 模型；要求键、dtype、shape 和每个张量值逐位相同，保存后还必须严格重建通过。

- source checkpoint SHA-256：`74ad2856e4449a42397edbda599ae79e8a4c6a6fa923624ef4e91a35e20a2a4c`；
- converted checkpoint SHA-256：`1e2c1a4a274c5e466b716ded41ccf58bebf167a2fadbeba75693d08bdb4f039c`；
- conversion report SHA-256：`769b42c5a990e52a874e1c3478279dd1936a88b13d49053ca5bd69fea54831ef`；
- state keys exact：true；state tensors bit exact：true；class mapping exact：true；strict rebuild：true。

## A2 迁移结果

A2 center 与 M1 均强制使用在线图像路径，避免把缓存特征与在线 attention 路径混合比较。实现审计的 global 最大绝对 logit 差为 `0.0`，预测一致率为 `100%`。

| 指标 | A2 center | A2 + M1 | 变化 |
|---|---:|---:|---:|
| clean-core micro | 82.5755% | 83.1716% | **+0.5961pp** |
| clean-core macro | 82.4547% | 82.9844% | **+0.5297pp** |
| trusted micro | 80.6653% | 81.1766% | **+0.5113pp** |
| trusted macro | 80.3732% | 80.8197% | **+0.4466pp** |
| raw micro | 69.4536% | 70.1124% | **+0.6588pp** |
| raw macro | 69.4548% | 70.1100% | **+0.6552pp** |

补充诊断：local/global agreement 为 `74.0845%`；融合改变 `1,216 / 10,322` 个预测，raw 纠正 `322` 个、破坏 `254` 个，净增 `68` 个；clean-core 净增 `43` 个。被改变样本的全局 top1-top2 平均间隔为 `0.5175`，未改变样本为 `3.6789`。

产物哈希：

- A2 online center cache：`1ab0a42f64d49315aa7202bc099eb76c796f599355dd9388c58c537eb35bacd1`；
- A2 M1 cache：`cea5f7651bf0d24828b7da8b252cb47b52288fa2cb7638a1c6d32359c0c92698`；
- A2 evaluation JSON：`e3b191c817fa21e46a7f09772132a3776bf416b02f40008407ca5acfd7a50d3f`。

A2 通过全部迁移门槛，结论为 **PROMOTE TO TEST INFERENCE**。M1 至此在 J0、F1、A2 三个检查点上连续复现，raw micro 增益为 `+0.8525 / +0.9597 / +0.6588pp`。

## 正式测试提交产物

两个提交均使用同一个检查点的确定性双视图推理，坏图为 0，预测数为 24,967，类别数为 500；均未启用测试先验校准、测试集训练或第二模型。审计器验证了样本顺序、文件名、标签集合、CSV 行数、ZIP 根目录结构、ZIP/CSV 字节一致性以及 checkpoint/CSV/ZIP 哈希。

### 优先候选：A2 + M1

- submission ZIP：`outputs/M1_A2_ATTENTION_LOCAL_GLOBAL/seed42/submission/submission.zip`
- ZIP SHA-256：`b73eed1f826b37433962cce547cbfa6f15e57afd7d83b3c56557ce2ab399ecbd`
- prediction CSV SHA-256：`47443ab8ce4f939025ec334e9d903caed769e765ded83b75850e762364f32ef8`
- manifest SHA-256：`45cc791051fe7af166bb0cdbafc5b0506290046db1d072b88086841e19c05734`
- audit：passed。

### 复现候选：F1 + M1

- submission ZIP：`outputs/M1_F1_ATTENTION_LOCAL_GLOBAL/seed42/submission/submission.zip`
- ZIP SHA-256：`eca9e7c6269c6a4a1cdb213228fa11e881a7ed9795df14da721d6799a1dab63c`
- prediction CSV SHA-256：`530de64ccdf7e2d65c64fb4e46885289362cc7eb54b715d723aeefc93a16dd8b`
- manifest SHA-256：`c333f3ad400767b31456bd47e088ce6895566c4c6cf2086a18905fcfed7cc3b8`
- audit：passed。

平台验证顺序固定为 A2 + M1 优先、F1 + M1 次之。该顺序依据是 A2 horizontal-flip 已有当前最高平台基线 `61.2128`，而非测试预测分布或任何测试标签信息。

## 平台结果（2026-07-22）

| 提交 | 平台分数 | 相对各自 Flip TTA | 相对旧平台最佳 A2 Flip |
|---|---:|---:|---:|
| A2 + M1 | **62.6747** | **+1.4619pp**（vs 61.2128） | **+1.4619pp** |
| F1 + M1 | **63.3276** | **+2.2269pp**（vs 61.1007） | **+2.1148pp** |

两个检查点上的固定 M1 都获得了远大于本地门控增益的平台提升，确认“CLIP 自注意力定位局部细节 + 全局/局部概率均值”是当前最强且可复现的方向。F1 + M1 比 A2 + M1 再高 `0.6529pp`，因此真实平台优先级改为 **F1 + M1 第一、A2 + M1 第二**。此前的上传顺序只是平台结果产生前的预注册顺序，不再作为当前推荐。
