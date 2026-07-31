## A2 LoRA 平台测试结果（2026-07-22）

- A2_LORA_MIN：裸推理 **61.1167%**；horizontal_flip TTA **61.6574%**。
- A2_LORA_FULL：裸推理 **61.5733%**；horizontal_flip TTA **62.1781%**。
- 当前 A2 LoRA 消融最高为 **A2_LORA_FULL + TTA 62.1781%**；详见 [A2 LoRA 平台结果](../a2_lora_platform_results_2026-07-22.md)。

# NR_COMBINED_UPGRADE 实施计划

> 在 A2 kNN consensus drop 的基础上，叠加 AEGIS F1 的 visual LoRA + clean filter + feature distillation

**Goal:** 将 AEGIS F1 的微调策略移植到 A2 baseline，实现噪声分层处理 + visual LoRA 的叠加收益。

**Parent:** A2 `NR_CL_KNN_DROP` checkpoint（TTA 61.21%，当前平台最佳冻结 baseline）

**Tech Stack:** 复用 `common/peft.py`、`common/feature_distillation.py`，在 `experiments/baseline/train.py` 主训练入口内实现。

---

## 策略概述

```
A2 baseline (冻结 backbone + GCE + MixUp, 全局黑名单已清洗 991 样本)
    ↓
第 1 层: visual LoRA (block 8-11, rank=8, Q/V/output)
    ↓  零初始化，防止初始扰动
第 2 层: clean probability filter (threshold ≥ 0.70)
    ↓  rejected 样本 feature distillation only，不参与分类 loss
第 3 层: feature distillation (weight=2.0, 锚定冻结的原始 CLIP)
    ↓  防止表征漂移
仅 6 epochs，batch_size=64，无 MixUp
```

**预期：** A2 TTA 61.21% → 叠加 LoRA + distill → 目标 TTA 61.5-61.8%

---

## 实施步骤

### Step 1: 扩展 PEFT 支持 multi-block LoRA

**文件:** `common/peft.py`

当前 `last_block_lora` 仅对最后一个 block 的输出投影加 LoRA。需新增 `visual_lora` 模式：

- 参数: `lora_last_n_blocks` (默认 4), `lora_rank` (默认 8), `lora_alpha` (默认 8), `lora_adapt_qv` (bool), `lora_adapt_out` (bool)
- 对 visual.transformer.resblocks 的最后 N 个 block 的 Q、V、output 投影注入 LoRA
- 零初始化 (A=0, B~N(0,σ²)) 保证训练开始时等价于 parent

### Step 2: 集成 clean probability filter

**文件:** `experiments/baseline/train.py`

- 在 `main()` 中，构建 weight provider 时读取 clean probability 字段
- clean prob ≥ threshold → weight=1, 正常监督
- clean prob < threshold → weight=0, 仅 feature distillation
- 从 OOF `sample_quality.csv` 读取 `p_original_label` 作为 clean probability

### Step 3: 集成 feature distillation

**文件:** `experiments/baseline/train.py` / `common/feature_distillation.py`

- 使用 `FeatureDistillation(parent_model)` 包装原始冻结 CLIP
- 在训练循环中计算 `feat_loss = distill.compute_loss(features, parent_features)`
- `total_loss = task_loss + lambda_feat * feat_loss`（lambda_feat 默认 2.0）
- 仅对 rejected 样本计算 distillation loss（clean 样本已有监督）

### Step 4: 创建配置

**文件:** `configs/nr_combined_upgrade.yaml`

```yaml
experiment:
  id: NR_COMBINED_UPGRADE
  mode: dev
  head_type: linear
  augmentation_preset: a0

data:
  split_dir: outputs/data/d3_strict/seed42
  train_dir: train
  seed: 42
  split_seed: 42
  train_seed: 42

model:
  clip_model_name: ViT-B/32
  freeze_clip: false  # LoRA 需要可训练 backbone
  num_classes: 500

peft:
  type: visual_lora
  lora_last_n_blocks: 4
  lora_rank: 8
  lora_alpha: 8
  lora_adapt_qv: true
  lora_adapt_out: true

loss:
  name: gce
  q: 0.5

mixup:
  enabled: false  # LoRA 训练不使用 MixUp

sample_weighting:
  type: oof_manifest
  manifest_path: outputs/phase/phase3/oof/oof_zero_weight_manifest_thresh0.001.csv
  min_weight: 0.0
  max_weight: 1.0
  missing_weight_policy: error
  clean_prob_threshold: 0.70
  feature_distillation_weight: 2.0

train:
  amp: true
  batch_size: 64
  device: cuda
  epochs: 6
  lr: 5.0e-5       # head LR (LoRA backbone LR 1/2)
  backbone_lr: 2.0e-5
  warmup_epochs: 1
  max_grad_norm: 1.0
  num_workers: 4
  save_dir: outputs/oof/nr_combined_upgrade/seed42/checkpoints
  weight_decay: 0.0001
  backbone_weight_decay: 0.0
```

### Step 5: 训练

```bash
python3 -m experiments.baseline.train \
  --config configs/nr_combined_upgrade.yaml \
  --experiment-id NR_COMBINED_UPGRADE \
  --mode dev --augmentation-preset a0 \
  --init-checkpoint <A2 checkpoint path>
```

全局黑名单自动生效（`outputs/phase4/global_rejected_paths.txt`）。

---

## 与 AEGIS F1 的差异

| 维度 | AEGIS F1 | NR_COMBINED_UPGRADE |
|------|----------|---------------------|
| Parent | E2 epoch 44 (60.48%) | A2 (61.21%) |
| 数据预处理 | clean prob ≥ 0.7 filter | 全局黑名单 (991) + clean prob ≥ 0.7 |
| LoRA | block 8-11 Q/V/out | 相同 |
| 特征蒸馏 | 原始 CLIP anchor | 相同 |
| MixUp | 无 | 无 |
| 训练入口 | 独立 runner | `experiments/baseline/train.py` |

主要升级点：A2 baseline 比 E2 高 0.73pp，且全局黑名单已清洗 991 个确认错标。

---

## Gate 条件

- local clean-core micro ≥ parent 的 clean-core micro
- feature drift ≤ 1%（与原始 CLIP 特征的 cosine distance）
- flip consistency ≥ parent 的 flip consistency
- 满足以上条件后提交 Bare + TTA

## 预期时间线

| Step | 内容 | 预计时间 |
|------|------|---------|
| Step 1 | PEFT multi-block LoRA | 30 min |
| Step 2-3 | clean filter + feature distill 集成 | 30 min |
| Step 4 | 配置创建 | 10 min |
| Step 5 | 6 epochs 训练 | ~15 min |
| Gate check | 本地评估 | 5 min |
| **Total** | | **~1.5 h** |
