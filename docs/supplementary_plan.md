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
