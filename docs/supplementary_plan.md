# Supplementary Experiments

> 不在 Phase 3 主线内，但值得在算力允许时补充的实验。  
> 所有实验默认使用与 Phase 3 相同的统一契约（split_seed=42, train_seed=42, A0, linear head, frozen CLIP）。

---

## S-GCE-Q：GCE q 值补充搜索（P1）

当前已知：

| q | 本地 Val | 平台 Bare |
|:--|:--:|:--:|
| 0.5 | 69.49% | 59.62% |
| 0.7 | 69.59% | 58.96% |
| 0.9 | 58.91% | — |

缺失 q=0.3, 0.4, 0.6 的数据点。补全后可以画 q→平台分数曲线，确认最优区间。

### S-GCE-Q-1：GCE q=0.3

```yaml
experiment_id: S_GCE03
loss:
  name: gce
  q: 0.3
```

假设：更接近 CE，可能更适合干净样本，但噪声鲁棒性下降。

### S-GCE-Q-2：GCE q=0.4

```yaml
experiment_id: S_GCE04
loss:
  name: gce
  q: 0.4
```

### S-GCE-Q-3：GCE q=0.6

```yaml
experiment_id: S_GCE06
loss:
  name: gce
  q: 0.6
```

q=0.5 和 0.7 之间插值，判断最优是否在 0.5-0.7 之间。

### S-GCE-Q-4：CE5→GCE q=0.3

如果纯 GCE q=0.3 表现好，补充 CE warmup 版本。

---

## S-AUG：增强消融（P2）

Phase 2 已关闭多数增强方向，但以下单点值得确认：

### S-AUG-1：RandAugment 轻度（magnitude=3）

已被 Phase 3 关闭的是强 RandAugment。轻度版本（magnitude=3, num_ops=2）可能对冻结 CLIP 有帮助。

### S-AUG-2：RandomErasing 小面积（scale=0.02-0.05）

当前关闭的是大面积 Erasing。小面积版本对遮挡鲁棒性可能有微收益。

---

---

# Phase 3 平台测试待办

> 本地完成但尚未提交平台的实验。本地排名 ≠ 平台排名（CE5 本地 +3.65pp，平台仅 +0.09pp），每个实验都需要独立验证。

## 待测清单

| # | experiment_id | 本地 Val | 平台 Bare | 平台 TTA | 状态 |
|:--|------|:--:|:--:|:--:|------|
| 1 | w1_ce5_gce05 | 73.14% | 59.61% | 60.25% | bare=纯 q=0.5，warmup 本地 +3.65pp→平台 0 |
| 2 | w1_ce5_gce07 | 69.78% | 未测 | 未测 | CE warmup + q=0.7 |
| 3 | w1_gce09 | 58.91% | 未测 | 未测 | GCE q=0.9，本地差但平台未知 |
| 4 | w1_gce07_mixup | 69.61% | 未测 | 未测 | MixUp + q=0.7 |
| 5 | w1_gce05_mixup | 71.16% | 待测 | 60.36% | MixUp q=0.5，平台 TTA 当前最佳，bare 已提交待平台 |
| 6 | ft_frozen | 70.64% | 未测 | 未测 | F0-strict frozen control |

## 已测（对照）

| experiment_id | 本地 Val | 平台 Bare | 平台 TTA |
|------|:--:|:--:|:--:|
| gce_q07 | 69.59% | 58.96% | 59.41% |
| b2_gce05 | 69.49% | 59.62% | 60.16% |
| w1_ce5_gce05 | 73.14% | 59.61% | 60.25% |
| w1_gce05_mixup | 71.16% | 待测 | **60.36%** |
| ft_lnpost | 70.78% | — | 56.92% |
| w2_ema_loss | 69.42% | — | 59.39% |
| w2_proto_min04 | 68.76% | — | 58.82% |

---

# S-MIXUP：MixUp q=0.5 扩展实验（P1）

当前 MixUp q=0.5 (alpha=0.2, p=0.2)：bare 待测，TTA=60.36%（最佳）。MixUp 本地 71.16% < CE5 73.14%，但平台反超——MixUp 的输入正则化比 warmup 更利于泛化。以下探索组合：

## S-MIXUP-1：CE5 warmup + MixUp q=0.5

```yaml
experiment_id: S_MIXUP_CE5
loss:
  schedule:
    - start_epoch: 1
      end_epoch: 5
      name: cross_entropy
    - start_epoch: 6
      end_epoch: 50
      name: gce
      q: 0.5
mixup:
  enabled: true
  alpha: 0.2
  probability: 0.2
```

假设：CE warmup 本地 +3.65pp + MixUp 平台泛化 = 组合可能同时提升本地和平台。

## S-MIXUP-2：MixUp alpha sweep

当前 alpha=0.2 沿用 q=0.7 设置，q=0.5 下可能有更优值。

| 实验 | alpha | probability |
|:--|:--:|:--:|
| S_MIXUP_A04 | 0.4 | 0.2 |
| S_MIXUP_A01 | 0.1 | 0.2 |

## S-MIXUP-3：MixUp probability sweep

| 实验 | alpha | probability |
|:--|:--:|:--:|
| S_MIXUP_P04 | 0.2 | 0.4 |
| S_MIXUP_P05 | 0.2 | 0.5 |

## S-MIXUP-4：多 seed 验证

如果组合实验有正收益，补 seed=3407 和 seed=2026。

---

## 实验登记

| experiment_id | status | val_acc | platform | notes |
|------|------|:--:|:--:|------|
| — | — | — | — | — |

---

## 优先级

```
P0: 主线 Phase 3 Wave 1-4
P1: S-GCE-Q（q 值补全，每次 ~4.5h GPU）
P2: S-AUG（在 PEFT 效果确认后可选）
```

P1 实验应在 B 组完成、C 组启动后，利用空闲 GPU 时间跑。
