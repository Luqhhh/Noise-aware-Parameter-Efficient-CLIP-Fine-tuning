# A2 STRICT 局部特征残差 Adapter（2026-07-30）

## 结论

在已审计的 A2 STRICT + M1 `crop160/top5/local_weight0.35` 后方训练局部专用
`512→32→512` 残差 Adapter。全局路径、视觉 LoRA 和唯一线性分类头全部冻结，
训练只使用 trust `>=0.70` 的官方训练样本。

首轮和六个有界消融点均未达到预注册晋级门槛，因此不生成测试提交。最佳点为
`local_loss_weight=0.50`：相对同协议无 Adapter 基线，raw micro `+0.0872pp`、
trusted macro `+0.1219pp`、clean-core micro `+0.1821pp`，500 类覆盖不变，
局部特征平均漂移 `0.6538%`。clean-core 增益低于要求的 `+0.20pp`，状态为
`best_not_promoted`。

## 固定协议

- 父检查点：A2 STRICT seed42，SHA-256
  `096f3294bebf262c87bc9f8ffa72d08a31c76eb6cba433af90ba36399b543c9e`。
- 局部视图：最后视觉 block、12 head 均值、attention top-5、crop160。
- 推理融合：`0.65*global_probability + 0.35*adapted_local_probability`。
- 训练样本：91,195；trust `>=0.70` 的有效样本 64,071，覆盖 500 类。
- Adapter：bottleneck 32、scale 0.25、34,336 参数；上投影零初始化。
- 优化：AdamW、LR `5e-4`、weight decay `1e-4`、GCE `q=0.5`、最多 20 epoch、
  patience 5。
- 晋级门槛：clean-core `>=+0.20pp`、trusted macro 不降、raw
  `>=-0.10pp`、漂移 `<=1%`、500 类覆盖。

缓存先固定全局和局部特征，随后所有消融只在特征层训练，避免重复改变定位数值路径。
epoch-0 指标精确复现 M1 审计值：raw `70.6937%`、trusted macro
`81.7193%`、proxy macro `80.2080%`、clean-core `84.0196%`。

## 有界消融结果

| 配置变化 | Raw Δ | Trusted macro Δ | Clean-core Δ | 漂移 | 判定 |
|---|---:|---:|---:|---:|---|
| 首轮：local loss 0.25 | +0.0775pp | +0.1125pp | +0.1541pp | 0.4701% | 未过门槛 |
| anchor 2.0→1.0 | −0.0484pp | −0.0566pp | −0.0140pp | 0.7316% | 退化 |
| **local loss 0.25→0.50** | **+0.0872pp** | **+0.1219pp** | **+0.1821pp** | **0.6538%** | 最佳但未晋级 |
| train trust 0.70→0.80 | +0.0775pp | +0.0380pp | +0.0980pp | 0.3355% | 未过门槛 |
| bottleneck 32→64 | −0.0484pp | −0.0183pp | +0.0140pp | 0.4442% | 退化 |
| local loss 0.75 | −0.0291pp | −0.0255pp | +0.0140pp | 0.6748% | 退化 |
| local loss 1.00 | −0.0291pp | −0.0200pp | +0.0280pp | 0.8278% | 退化 |

`0.50` 以后局部监督继续增强会同时损害 raw/trusted，说明最优点不是单调强度边界。
降低锚定、扩大容量或只保留更高可信样本也没有形成新收益，因此停止此分支，避免在
同一带噪 validation 上继续无界搜索。

## 产物与审计

- train cache SHA-256：
  `05f89d8fe491cf2fc9644037253808871393dcd322b487384f9c51b9f16d4f1f`
- val cache SHA-256：
  `73d0589bf8c2aa3e597804c2c1331cef2974acccfc6f1b1f9679fabf32762fe6`
- 最佳未晋级 candidate SHA-256：
  `6329ac05a1c221e067a47c0a4ccf6f47fadff5cda809d3b638f5bf17a88b8ac8`
- 机器可读结果：`results/local_adapter_a2_strict_20260730.csv`
- 新增 localization/local-adapter 相关测试：12 passed。

下一步回到 checkpoint 表征：严格重建本机缺失、但已知 F1+M1 平台为 63.3276%
的 F1 轨迹；不把未过 gate 的 Adapter 包装成平台候选。
