# M1 attention-local 推理优化（2026-07-30）

## 结论

在 A2 STRICT 双 seed 上复现了 M1 的最后一层 CLS→patch attention 局部裁剪，并完成固定范围消融。原 M1 定位参数 `crop=160, top-k=5` 仍是最稳组合；将局部分支概率权重从 `0.50` 降到 `0.35` 后，两个 seed 的 raw、trusted、proxy、clean-core 指标全部同向改善，且均保持 500 类覆盖。

该候选只改变同一 checkpoint 的确定性推理，不引入第二模型、外部数据、测试时训练或测试标签。平台最终分数为 **62.6870%**（用户先报 62.8670%，随后更正为 62.6870%，以最新值为准），状态为 `platform_valid_not_promoted`。

## 协议

- 模型：单个 A2 STRICT visual-LoRA checkpoint。
- 全局分支：OpenAI CLIP ViT-B/32 原生 224×224 center/global forward。
- 定位：最后视觉 Transformer block，12 个 head 的 CLS→patch attention 取均值。
- 局部视图：attention top-5 加权中心，裁剪 160×160，双线性放大到 224×224。
- 融合：`0.65 * global_probability + 0.35 * local_probability`。
- 禁止叠加 flip TTA；推理 CLI 会 fail closed。

## 双 seed 结果

| Seed | 协议 | Raw | Δ Raw | Trusted macro | Proxy macro | Clean-core | Δ Clean-core |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | Global | 69.6280% | — | 80.8984% | 79.4704% | 83.2073% | — |
| 42 | M1 0.50 | 70.4805% | +0.8525pp | 81.2939% | 79.8790% | 83.5574% | +0.3501pp |
| 42 | **M1 0.35** | **70.6937%** | **+1.0657pp** | **81.7193%** | **80.2080%** | **84.0196%** | **+0.8123pp** |
| 3407 | Global | 69.7249% | — | 80.9728% | 79.4939% | 83.2493% | — |
| 3407 | M1 0.50 | 70.6452% | +0.9204pp | 81.4276% | 79.9508% | 83.6555% | +0.4062pp |
| 3407 | **M1 0.35** | **70.7324%** | **+1.0076pp** | **81.7491%** | **80.1639%** | **84.0336%** | **+0.7843pp** |

定位网格覆盖 `crop ∈ {144,160,176}`、`top-k ∈ {3,5,9}`。seed 42 上 `crop160/top5/weight0.35` 同时取得最高 trusted macro、proxy macro 和 clean-core；`crop160/top9/weight0.45` 的 raw 仅高 0.0387pp，但其 trusted/proxy/clean-core 均更低，因此不晋级。

另验证了把 144/160/176 三种裁剪概率平均后再与 global 融合的多尺度 M2。seed42 最佳点 `top9/weight0.45` 的 raw 为 70.8777%，但相对单尺度晋级候选，proxy macro 低 0.0156pp、clean-core 低 0.0140pp，没有满足“raw/trusted/proxy/clean-core 全部同向”的预设门槛，因此停止在 seed42，不补第二 seed、不生成平台包。

## 平台候选

- ZIP：`reproducibility/aegis_f1/outputs/F1_VISUAL_LORA_CLEAN_CORE_A2_PARENT_STRICT/seed42/submissions/m1_w035/submission.zip`
- checkpoint SHA-256：`096f3294bebf262c87bc9f8ffa72d08a31c76eb6cba433af90ba36399b543c9e`
- prediction CSV SHA-256：`bbcc0b0551ee4b2ae84c72abbf33cf89240566248d2bc8a3c8f7972ce3f4a0b5`
- submission ZIP SHA-256：`ddbbf0b9e408e9fbcd4fc7d00c8c16e647a872634c61625d9c9e9c935d549e66`
- 预测数：24,967；类别数：500；损坏图：0。
- `aegis-audit-submission --allow-tta`：passed。

## 平台结果与判定

- 平台：**62.6870%**。
- 相对 A2 STRICT Bare 60.6521%：**+2.0349pp**，再次确认 attention-local 是真实有效机制。
- 相对 A2 STRICT Flip TTA 61.1487%：**+1.5383pp**。
- 相对已报告 A2 + M1 62.6747%：+0.0123pp；两者 checkpoint 不同，不能把该差值归因于融合权重。
- 相对已报告 F1 + M1 63.3276%：**−0.6406pp**。
- 本地 raw 70.6937% 与平台 62.6870% 的差距为 8.0067pp；本地 weight 0.35 优势没有形成足以改变平台排序的收益。
- 距离 70% 目标仍差 **7.3130pp**，因此该候选有效但不晋级为当前最佳；停止继续搜索 M1 融合权重，下一轮应回到更强的 F1 表征或新机制。

机器可读双 seed 摘要见 `results/m1_localization_dual_seed.csv`；完整本地 sweep JSON 保存在被忽略的 `reproducibility/aegis_f1/outputs/localization/`。

## 回归验证

- localization/TTA/submission/checkpoint/model 相关测试：25 passed。
- 新推理入口关闭 local view 时，24,967 条 Bare 预测与既有提交逐字节一致；两者 CSV SHA-256 均为 `c390d388a0b7261bc0fab75e1204df2c821a454a4c924c267397d2afaa6bba75`。
- Aegis 子工程全量：71 passed、1 failed。唯一失败是既有 Phase 4 配置使用 `stage: p4_ablation`，而 validator 只接受 preliminary/repechage/semifinal；与本轮推理改动无关。
