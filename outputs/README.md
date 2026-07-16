# Outputs Directory

每个子目录对应一个实验，包含 checkpoint、训练日志、评估结果和提交文件。

## 基线

| 目录 | 实验 ID | 说明 | 本地 Acc | 平台 |
|------|---------|------|----------|------|
| `ref/` | ref | 参考基线：去重训练集 + CE + frozen CLIP + linear head | 70.66% | 57.34% |
| `base_ce/` | base_ce | 基础 CE（无去重），E0 严格重跑 | 70.54% | — |
| `base_b0/` | base_b0 | B0 回归基线，20 epoch，lr=1e-3 | 61.39% | — |

## 噪声鲁棒方法

| 目录 | 实验 ID | 说明 | 本地 Acc | 平台 |
|------|---------|------|----------|------|
| `gce_q07/` | gce_q07 | **训练基线**：GCE q=0.7，与 ref 共享 split | 69.59% | **58.96%** |
| `pw_v1/` | pw_v1 | 原型置信度静态加权，基于 ref split | 70.19% | 58.05% |

## 部分解冻

| 目录 | 实验 ID | 说明 | 本地 Acc | 平台 |
|------|---------|------|----------|------|
| `ft_frozen/` | ft_frozen | 对照组：从 ref init，继续冻结全部 CLIP，lr=3e-4 | 70.64% | — |
| `ft_lnpost/` | ft_lnpost | 实验组：从 ref init，解冻 ln_post + visual.proj | 70.78% | 56.92% |

ft_frozen 与 ref 几乎持平（−0.02pp），证明 ref 的 50 epoch 已收敛。
ft_lnpost 本地 +0.13pp 但平台 −0.42pp vs ref（56.92% vs 57.34%）——又一起 local-platform 反转。CE loss 下的部分解冻已关闭，等待 GCE 重测。

## Dropout 正则化

| 目录 | 实验 ID | 说明 | 本地 Acc |
|------|---------|------|----------|
| `drop_p03/` | drop_p03 | Dropout p=0.3，基于 ref split | 70.09% |
| `drop_p05/` | drop_p05 | Dropout p=0.5 | 69.39% |
| `drop_p07/` | drop_p07 | Dropout p=0.7（未完成） | — |

所有 dropout 均无正收益，已关闭方向。

## 数据增强

| 目录 | 实验 ID | 说明 | 本地 Acc |
|------|---------|------|----------|
| `aug_a1/` | aug_a1 | RandomResizedCrop + HorizontalFlip，lr=3e-3 | 69.15% |
| `aug_a1_lr5e3/` | aug_a1_lr5e3 | A1 对齐 lr=5e-3 | 69.77% |
| `aug_a2/` | aug_a2 | A1 + ColorJitter | 67.36% |
| `aug_a3/` | aug_a3 | A2 + RandomErasing（未完成） | — |

A1 对齐学习率后与 A0 几乎持平（−0.09pp），A2 的 ColorJitter 显著破坏细粒度信息。增强方向已关闭。

## Cosine Head

| 目录 | 实验 ID | 说明 |
|------|---------|------|
| `cos_hyper/` | cos_hyper | Cosine head 超参搜索（5 lr × 3 wd），含 splits |
| `e4_lr_5e-03_wsl/` | — | Cosine head 遗留实验 |

Cosine head 在所有学习率下均显著弱于 linear head（~6pp 差距），已关闭。

## 分析与辅助

| 目录 | 说明 |
|------|------|
| `analysis/` | D3 vs B2 对比分析（feature bank、trusted manifest、findings） |
| `archive/` | 旧 E0 strict seed42 遗留数据 |
| `master_splits/` | 主 split 文件（seed 42/2026/3407），被 ref、gce_q07 等共享 |
| `metadata/` | 类别映射等元数据 |
| `phase2/` | Phase 2 产物：D3 logits、prototype weights、TTA 提交、logit adjustment |
| `dedup/` | 去重扫描结果（duplicate_scan.json） |

## 平台提交记录

| 提交 | 平台分数 | vs ref |
|------|---------|--------|
| **S_MIXUP_CE5 + Flip TTA** | **60.48%** | +3.14pp |
| w1_gce05_mixup + Flip TTA | 60.36% | +3.02pp |
| w1_ce5_gce05 + Flip TTA | 60.25% | +2.91pp |
| b2_gce05 + Flip TTA | 60.16% | +2.82pp |
| gce_q07 + Flip TTA | 59.41% | +2.07pp |
| gce_q07 裸模型 | 58.96% | +1.62pp |
| pw_v1 | 58.05% | +0.71pp |
| ref | 57.34% | — |

详细记录见 `results/submission_registry.csv`。
