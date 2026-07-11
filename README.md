# Noise-Aware Parameter-Efficient CLIP Fine-Tuning

面向噪声标签数据的细粒度图像识别。基于 CLIP ViT-B/32（冻结），对比 Linear Head 与 Cosine Head，系统消融数据增强策略，支持特征缓存加速和多划分验证。

## 项目结构

```
├── common/                              # 所有实验共享的公共代码
│   ├── clip_utils.py                    # CLIP 加载与冻结特征编码（统一入口）
│   ├── class_mapping.py                 # 规范类别映射（生成/复用/一致性检查）
│   ├── transforms.py                    # 训练增强预设 A0-A3
│   ├── cache.py                         # FeatureCacheBuilder + CachedFeatureDataset
│   ├── runtime_config.py               # CLI/YAML 配置优先级解析
│   ├── evaluation.py                    # 多划分 paired delta 报告
│   ├── dataset.py                       # TrainImageDataset / TestImageDataset
│   ├── utils.py                         # load_config / set_seed / setup_logging
│   └── submission.py                    # 提交生成 + 覆盖率验证
├── experiments/
│   ├── baseline/                        # Linear Head 实验（B0, E0, E2-E4）
│   │   ├── model.py                     # CLIPLinearClassifier + build_model
│   │   ├── train.py                     # 训练（dev/confirm/final_fit 模式，head-type 通用）
│   │   ├── evaluate.py                  # 验证集评估（linear/cosine 通用）
│   │   ├── infer.py                     # 测试集推理
│   │   └── b0_regression.py            # B0 回归 fixture
│   ├── cosine/                          # Cosine Head 实验（E1, E5, C0-C2）
│   │   ├── model.py                     # CosineClassifier
│   │   ├── train.py                     # → 薄包装，委托 baseline.train
│   │   ├── evaluate.py                  # → 薄包装，委托 baseline.evaluate
│   │   └── infer.py                     # → 薄包装，委托 baseline.infer
│   └── augmentation/                    # 增强消融实验（E2-E4）
│       ├── train.py                     # → 薄包装，委托 baseline.train
│       ├── evaluate.py                  # → 薄包装
│       └── infer.py                     # → 薄包装
├── configs/
│   ├── b0_regression.yaml               # B0: 原始 baseline 回归
│   ├── e0_hyper_search.yaml             # E0: Linear + A0 超参搜索
│   ├── e1_hyper_search.yaml             # E1: Cosine + A0 超参搜索
│   ├── e2_augmentation.yaml             # E2: Linear + A1 (Crop+Flip)
│   ├── e3_augmentation.yaml             # E3: Linear + A2 (+ColorJitter)
│   ├── e4_augmentation.yaml             # E4: Linear + A3 (+RandomErasing)
│   ├── e5_combined.yaml                 # E5: Cosine + 最佳增强
│   ├── c0_cosine_scale.yaml             # C0: Cosine fixed scale
│   ├── c1_cosine_scale.yaml             # C1: Cosine learnable scale
│   └── c2_cosine_scale.yaml             # C2: Cosine alt init scale
├── scripts/
│   ├── check_data.py                    # 数据完整性检查
│   ├── split_data.py                    # 训练/验证集划分（支持多 seed）
│   ├── cache_features.py                # 构建 CLIP 特征缓存
│   ├── run_hyper_search.py              # 自动超参搜索（lr×wd）
│   ├── verify_cache_consistency.py      # 缓存与在线编码一致性验证
│   ├── check_resolved_configs.py        # 配置解析正确性验收
│   ├── run_acceptance.py                # 50 项验收标准检查
│   ├── check_submission.py              # 提交文件格式验证
│   └── make_tiny_dataset.py             # 生成小型测试数据集
├── tests/
│   ├── test_clip_utils.py
│   ├── test_class_mapping.py
│   ├── test_transforms.py
│   ├── test_cache.py
│   ├── test_cosine.py
│   ├── test_cached_forward.py
│   ├── test_runtime_config.py
│   ├── test_evaluation.py
│   ├── test_integration.py
│   ├── test_label_mapping.py
│   ├── test_split_data.py
│   └── test_submission.py
├── outputs/                             # 实验结果（tracked in git，仅忽略 *.pt）
│   ├── b0/                              # B0 回归结果
│   └── metadata/                        # 规范类别映射
├── requirements.txt
└── README.md
```

## 环境安装

```bash
git clone <repo-url>
cd Noise-aware-Parameter-Efficient-CLIP-Fine-tuning
pip install -r requirements.txt
```

核心依赖：`torch>=2.0`, `torchvision>=0.15`, OpenAI CLIP, `pandas`, `numpy`, `Pillow`, `tqdm`, `pyyaml`, `scikit-learn`

## 数据准备

将比赛数据集放入项目根目录：

```
train/                  # 103,218 张训练图片，500 个类别文件夹 (0000-0499)
test/                   # 24,967 张测试图片，平铺
```

## 实验管线

### 完整执行顺序

```bash
# 0. 验证环境
pytest tests/ -q

# 1. B0 回归 — 验证基础设施未改变 baseline
python scripts/check_data.py --config configs/b0_regression.yaml
python scripts/split_data.py --config configs/b0_regression.yaml
python -m experiments.baseline.train --config configs/b0_regression.yaml

# 2. 构建特征缓存 — 编码全部训练集（一次性，~30min）
python scripts/cache_features.py --config configs/e0_hyper_search.yaml

# 3. E0 超参搜索 — Linear + A0，5 lr × 3 wd = 15 trials
python scripts/run_hyper_search.py --config configs/e0_hyper_search.yaml

# 4. E1 超参搜索 — Cosine + A0，5 lr × 1 wd = 5 trials
python scripts/run_hyper_search.py --config configs/e1_hyper_search.yaml

# 5. E2-E4 增强消融 — 已使用 E0 最优 lr=3e-3，50 epochs + 早停
python -m experiments.baseline.train --config configs/e2_augmentation.yaml
python -m experiments.baseline.train --config configs/e3_augmentation.yaml
python -m experiments.baseline.train --config configs/e4_augmentation.yaml

# 6. 选出最佳增强，写入 E5 config：
python -m experiments.baseline.train --config configs/e5_combined.yaml

# 7. 可选：Cosine scale 消融
python -m experiments.baseline.train --config configs/c0_cosine_scale.yaml
python -m experiments.baseline.train --config configs/c1_cosine_scale.yaml
python -m experiments.baseline.train --config configs/c2_cosine_scale.yaml

# 8. 多划分确认（seeds 3407, 2026）
# 修改 configs 中的 split_seed，重新运行 E0, E1, E5
# 使用 common/evaluation.py 计算 paired delta
```

### 训练模式

训练脚本支持三种模式，通过 config 的 `experiment.mode` 或 `--mode` CLI 参数控制：

| 模式 | 说明 | epoch 来源 |
|------|------|-----------|
| `dev` | 训练+验证，记录 best epoch | config `train.epochs` |
| `confirm` | 用冻结的 epoch 数训练，验证 | `--frozen-epochs` |
| `final_fit` | 全量训练集，无验证，无早停 | `--frozen-epochs` |

### 缓存 vs 在线训练

| | Cached | Online |
|---|---|---|
| 数据 | 预计算 `[B,512]` 特征 | 原始图片 `[B,3,224,224]` |
| CLIP 编码 | 一次性构建缓存 | 每 epoch 每张图 |
| 速度 | ~3s/epoch | ~140s/epoch |
| 增强 | 仅 A0（特征固定） | 支持 A0-A3 |
| 适用实验 | E0, E1, C0-C2 | B0, E2-E5 |

缓存文件位于 `cache/preliminary/clip_vit_b32_openai/`（`.gitignore` 排除）。其他人 clone 后需自行构建或从合作者处拷贝。

训练默认 50 epochs，支持早停（`train.early_stop_patience: 10`），连续 N 个 epoch 无提升自动终止，并在 `eval_results.json` 记录 `early_stopped` 和 `stopped_at_epoch`。

### 配置优先级

`common/runtime_config.py` 统一解析 CLI 与 YAML：

```
CLI 显式指定 > YAML 配置 > 硬编码默认值
```

关键参数：`--mode`, `--head-type`, `--augmentation-preset`, `--use-cached-features`, `--experiment-id`

## 增强预设

| 预设 | 内容 |
|------|------|
| A0 | 仅 CLIP 标准预处理（Resize→CenterCrop→Normalize） |
| A1 | A0 基础上 + RandomResizedCrop + RandomHorizontalFlip |
| A2 | A1 基础上 + ColorJitter |
| A3 | A2 基础上 + RandomErasing（在 Normalize **之后**） |

## B0 回归结果

| 指标 | 原始 Baseline | B0 新实现 | 差异 |
|------|-------------|----------|------|
| Val Acc | 61.38% | **61.39%** | +0.01pp ✅ |
| Epochs | 20 | 20 | — |
| 数据划分 | seed=42 | seed=42 | — |

验证通过，差异远在 0.5pp 阈值内。完整结果见 `outputs/b0/`。

## 常用命令

```bash
# 运行所有测试
pytest tests/ -q

# 验证配置解析
python scripts/check_resolved_configs.py

# 超参搜索（断点续跑）
python scripts/run_hyper_search.py --config configs/e0_hyper_search.yaml --skip-existing

# 超参搜索（预览不执行）
python scripts/run_hyper_search.py --config configs/e0_hyper_search.yaml --dry-run

# 缓存一致性检查
python scripts/verify_cache_consistency.py --config configs/e0_hyper_search.yaml --num-samples 128

# 接受标准检查
python scripts/run_acceptance.py
```

## 实验矩阵

| ID | Head | 增强 | 缓存 | 调参 | 目的 |
|---|---|---|---|---|---|
| B0 | Linear | A0 | No | No | 回归原始 61.38% |
| E0 | Linear | A0 | Yes | lr×wd | 强化 Linear baseline |
| E1 | Cosine | A0 | Yes | lr×wd | 隔离 Cosine Head 收益 |
| E2 | Linear | A1 | No | No | 测试随机裁剪+翻转 |
| E3 | Linear | A2 | No | No | 测试 ColorJitter |
| E4 | Linear | A3 | No | No | 测试 RandomErasing |
| E5 | Cosine | best(A1-A3) | No | No | 测试 Head×增强组合 |
| C0 | Cosine | A0 | Yes | No | scale 消融（fixed） |
| C1 | Cosine | A0 | Yes | No | scale 消融（learnable, init=10） |
| C2 | Cosine | A0 | Yes | No | scale 消融（learnable, init=20） |

## 实验结果 (preliminary, seed=42, val_ratio=0.1)

| ID | Head | 增强 | Best Val Acc | Best Epoch | 备注 |
|----|------|------|-------------|------------|------|
| E0 | Linear | A0 | **69.86%** | 36 | lr=5e-3 (早停 epoch 46) |
| E1 | Cosine | A0 | **63.61%** | 46 | lr=5e-3 (50 epochs 完整) |
| E2 | Linear | A1 | **69.15%** | 40 | lr=3e-3 (epoch 42 终止，未收敛) |
| E3 | Linear | A2 | 待定 | — | 训练中 |
| E4 | Linear | A3 | 待定 | — | 待运行 |

**关键发现：**
- Cosine Head (E1) 在所有 lr 下均不如 Linear Head (E0)，差距 ~6pp
- 扩展 lr 搜索范围后 E0 涨幅显著：61.10% (lr=1e-3) → 69.86% (lr=5e-3)
- Weight decay 对冻结 CLIP + 线性头无影响，后续搜索可固定为 1e-4

## Git 策略

- ✅ 跟踪：所有 `.json/.csv/.log/.yaml` 结果文件
- ❌ 忽略：`.pt` 检查点（二进制大文件）、`cache/`、`train/`、`test/`
