# 平台差距的类别级诊断：为什么下一步必须改变上游表征

日期：2026-07-20

## 技术结论

固定 `clean_probability >= 0.70` 的 7,214 个 validation 样本显示，A2 的 1,257 个错误并不集中在少数类别或少数固定混淆对。错误最多的前 10/50/100 类分别仅覆盖 10.66%/38.66%/58.63% 的错误，前 10/50 个有向混淆对仅覆盖 4.77%/15.12%。这排除了“主要靠修几个坏类或几个混淆对即可跨越平台差距”的假设。

M1 与 M3 的改进真实但能力边界很低：三种推理视图的逐样本 oracle 仅为 85.53%，相对 A2 center 的理论上限为 +2.95pp，且仍有 1,044 个样本三者全错。因此继续扫描线性头、局部原型、TTA 权重或类别规则无法解释平台 `61→90` 的巨大差距；下一项主要实验必须改变视觉表示本身。

## 固定口径

- validation：D3 固定 10,322 张，未使用测试集；
- clean-core：`clean_probability >= 0.70`，7,214 张、500 类；
- 单类 clean-core 支持数：1–21，因此单类排名只用于总体结构诊断，不作为类别级调参依据；
- A2/M1/M3：同一 A2 epoch 48 检查点的 center、attention local/global 和 complementary flip/local/global 推理缓存；
- 缓存的路径顺序与标签逐项一致；
- high-clean train：66,929 张、500 类，单类 31–184 张。

## 核心证据

| 方案 | clean-core micro | clean-core macro | 错误数 | 低于 70% 的类别数 | 相对 A2 |
|---|---:|---:|---:|---:|---:|
| A2 center | 82.5755% | 82.4547% | 1,257 | 108 | — |
| M1 attention | 83.1716% | 82.9844% | 1,214 | 101 | +0.5961pp |
| M3 complementary | 83.4350% | 83.2619% | 1,195 | 95 | +0.8594pp |

M1 相对 A2 纠正 194、伤害 151 个 clean-core 样本，净增 43；配对 bootstrap 变化为 +0.596pp，95% 区间 `[+0.111, +1.109]pp`。M3 纠正 127、伤害 65 个，净增 62；变化为 +0.859pp，95% 区间 `[+0.485, +1.248]pp`。区间来自固定 seed 42 的 5,000 次逐样本配对 bootstrap。

训练高可信样本数与 A2 类别准确率的 Spearman 相关为 `0.359`，说明类别可用数据量/质量有一定影响；但训练样本数与 M3 类别增益的相关仅 `-0.002`，说明 M3 的收益不是简单补偿少样本类别。错误分散和三视图共同失败仍是更强的主证据。

## 对实验路线的约束

1. 保留 M3 作为通过训练门控后才启用的同模型推理增强，不再扫描融合权重。
2. 启动 N3：在 A2 上加入零初始化的 last-6 AdaptFormer MLP 并行 adapter，以 J0 attention-LoRA 为严格配对控制。
3. N3 必须首先证明 epoch 0 与 A2 完全一致，再按预注册阈值判定；失败即关闭，不能用追加 epoch 或调参挽救。
4. 若 N3 仍无法显著纠正三视图共同失败样本，下一主方向应是细粒度邻域一致性与显式噪声转移建模，而不是更多后处理。

## 局限

clean-core 是从带噪 validation 标签得到的信任代理，不是真实干净验证集；bootstrap 只描述这个固定代理队列的抽样不确定性，不能外推为平台得分置信区间。平台测试集分布不同且标签干净，因此本诊断只用于选择实验方向，最终结论仍由预注册本地门控和独立平台反馈共同决定。

## 可复核产物

- `artifacts/n3_diagnostics/seed42/diagnostic.json`
- `artifacts/n3_diagnostics/seed42/class_metrics.csv`
- `artifacts/n3_diagnostics/seed42/top_confusions.csv`
- 复现入口：`python -m aegis_clip.cli.diagnose_validation_errors`

源缓存 SHA-256：

- A2 center：`1ab0a42f64d49315aa7202bc099eb76c796f599355dd9388c58c537eb35bacd1`
- M1：`cea5f7651bf0d24828b7da8b252cb47b52288fa2cb7638a1c6d32359c0c92698`
- M3：`51bb51a550fc7e67cfdcc32d9ff1c205836233c47bdc995144ab9eddef1dcb3a`
- high-clean train cache：`9375559010f827dafb3ee48955695bd19a429af37706a08d6f21426265f204c4`
