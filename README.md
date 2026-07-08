# Noise-Aware Parameter-Efficient CLIP Fine-Tuning

面向噪声标签数据的细粒度图像识别鲁棒微调。

## 项目结构

```
├── common/                             # 所有实验共享的公共代码
│   ├── dataset.py                      # TrainImageDataset / TestImageDataset
│   ├── utils.py                        # load_config / set_seed / setup_logging
│   └── submission.py                   # 生成 pred_results.csv + submission.zip
├── experiments/                        # 每个实验方法一个子目录
│   └── baseline/
│       ├── model.py                    # CLIP + Linear 分类器
│       ├── train.py                    # 训练脚本
│       ├── evaluate.py                 # 验证集评估
│       └── infer.py                    # 测试集推理
├── configs/                            # 配置文件
│   └── baseline.yaml
├── scripts/                            # 共享脚本
│   ├── split_data.py                   # 训练/验证集划分
│   ├── check_data.py                   # 数据检查
│   └── check_submission.py             # 提交文件检查
├── outputs/                            # 输出（按实验分目录，gitignore）
│   └── baseline/
│       ├── checkpoints/
│       ├── logs/
│       ├── submissions/
│       └── splits/
├── requirements.txt
└── README.md
```

## 环境安装

```bash
pip install -r requirements.txt
```

核心依赖：`torch >= 2.0`, `torchvision >= 0.15`, `clip` (OpenAI CLIP), `pandas`, `numpy`, `Pillow`, `tqdm`, `pyyaml`, `scikit-learn`

## 数据目录格式

```
train/                  # 训练数据（按类别文件夹组织）
├── 0000/
│   ├── xxx.jpg
│   └── ...
├── 0001/
├── ...
└── 0499/

test/                   # 测试数据（所有图片平铺在一个目录）
├── xxx.jpg
└── ...
```

## 快速开始（Baseline）

### 1. 修改配置

编辑 `configs/baseline.yaml`，修改数据路径：

```yaml
data:
  train_dir: "/path/to/your/train"
  test_dir: "/path/to/your/test"
```

### 2. 检查数据

```bash
python3 scripts/check_data.py --config configs/baseline.yaml
```

### 3. 划分训练/验证集

```bash
python3 scripts/split_data.py --config configs/baseline.yaml
```

### 4. 训练

```bash
python3 -m experiments.baseline.train --config configs/baseline.yaml
```

断点恢复：
```bash
python3 -m experiments.baseline.train --config configs/baseline.yaml --resume outputs/baseline/checkpoints/last.pt
```

### 5. 验证集评估

```bash
python3 -m experiments.baseline.evaluate --config configs/baseline.yaml --ckpt outputs/baseline/checkpoints/best.pt
```

### 6. 测试集推理

```bash
python3 -m experiments.baseline.infer --config configs/baseline.yaml --ckpt outputs/baseline/checkpoints/best.pt
```

### 7. 生成提交文件

```bash
python3 -m common.submission --raw outputs/baseline/submissions/pred_raw.csv --out_dir outputs/baseline/submissions
```

### 8. 检查提交文件

```bash
python3 scripts/check_submission.py \
    --test_dir /path/to/test \
    --csv outputs/baseline/submissions/pred_results.csv \
    --zip outputs/baseline/submissions/submission.zip
```

## 添加新实验

以 `lora` 为例，只需创建：

```
experiments/lora/
├── model.py
├── train.py
├── evaluate.py
└── infer.py
configs/lora.yaml
```

复用 `common/` 中的 dataset、utils、submission，以及 `scripts/` 中的数据准备脚本。

## Baseline 设计说明

- **模型**: CLIP ViT-B/32（88M 冻结）+ L2 Normalize + Linear(512→500)（257K 可训练）
- **训练**: AdamW (lr=1e-3, wd=1e-4), CosineAnnealingLR, 1-epoch warmup, AMP fp16
- **结果**: 20 epochs, 61.38% val Top-1 Accuracy, 50 分钟训练时间
- **Sample weight 接口**: `TrainImageDataset` 预留 `get_sample_weights()` 和 `return_path=True`，方便后续接入噪声筛选

## 后续可扩展方向

1. **LoRA / Adapter 微调**: 在 CLIP backbone 中插入低秩适配器
2. **噪声标签筛选**: 基于 small-loss criterion 筛选干净样本
3. **Prototype-based 方法**: 使用类别原型进行特征空间去噪
4. **Co-teaching / DivideMix**: 双模型协同训练筛选噪声
