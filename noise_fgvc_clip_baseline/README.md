# CLIP ViT-B/32 Linear Classifier Baseline

面向噪声标签数据的细粒度图像识别鲁棒微调 — 初赛 Baseline。

## 技术路线

```
CLIP ViT-B/32 (frozen) → L2 Normalize → Linear(512, 500) → Softmax
```

- 冻结 CLIP ViT-B/32 图像编码器，仅训练线性分类头
- 提取 L2-normalized image features
- 500 类线性分类
- Top-1 Accuracy 评估

## 环境安装

```bash
pip install -r requirements.txt
```

核心依赖：
- `torch >= 2.0`
- `torchvision >= 0.15`
- `clip` (OpenAI CLIP)
- `pandas`, `numpy`, `Pillow`, `tqdm`, `pyyaml`, `scikit-learn`

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

类别文件夹名可以是任意字符串，脚本会按排序建立从类别名到 0-499 索引的稳定映射。

## 快速开始

### 1. 修改配置

编辑 `configs/baseline.yaml`，修改数据路径：

```yaml
data:
  train_dir: "/path/to/your/train"
  test_dir: "/path/to/your/test"
```

### 2. 检查数据

```bash
python scripts/check_data.py --config configs/baseline.yaml
```

### 3. 划分训练/验证集

```bash
python scripts/split_train_val.py --config configs/baseline.yaml
```

### 4. 训练

```bash
python -m src.train --config configs/baseline.yaml
```

可选：断点恢复训练
```bash
python -m src.train --config configs/baseline.yaml --resume outputs/checkpoints/last.pt
```

### 5. 验证集评估

```bash
python -m src.evaluate --config configs/baseline.yaml --ckpt outputs/checkpoints/best.pt
```

### 6. 测试集推理

```bash
python -m src.infer --config configs/baseline.yaml --ckpt outputs/checkpoints/best.pt
```

### 7. 生成提交文件

```bash
python -m src.submission --raw outputs/submissions/pred_raw.csv --out_dir outputs/submissions
```

### 8. 检查提交文件

```bash
python scripts/check_submission.py \
    --test_dir /path/to/test \
    --csv outputs/submissions/pred_results.csv \
    --zip outputs/submissions/submission.zip
```

## 项目结构

```
noise_fgvc_clip_baseline/
├── configs/
│   └── baseline.yaml              # 配置文件
├── src/
│   ├── dataset.py                 # 数据集类
│   ├── model.py                   # CLIP + Linear 模型
│   ├── train.py                   # 训练脚本
│   ├── evaluate.py                # 验证集评估
│   ├── infer.py                   # 测试集推理
│   ├── submission.py              # 提交文件生成
│   └── utils.py                   # 工具函数
├── scripts/
│   ├── split_train_val.py         # 训练/验证集划分
│   ├── check_data.py              # 数据检查脚本
│   └── check_submission.py        # 提交文件检查
├── outputs/
│   ├── checkpoints/               # 模型权重
│   ├── logs/                      # 训练日志
│   ├── submissions/               # 预测结果和提交文件
│   └── splits/                    # 数据划分文件
├── requirements.txt
└── README.md
```

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `outputs/splits/train.csv` | 训练集文件列表（image_path, label, class_name） |
| `outputs/splits/val.csv` | 验证集文件列表 |
| `outputs/splits/class_to_idx.json` | 类别名到索引的映射 |
| `outputs/splits/idx_to_class.json` | 索引到类别名的映射 |
| `outputs/checkpoints/best.pt` | 最佳模型（按验证集准确率） |
| `outputs/checkpoints/last.pt` | 最后一轮模型 |
| `outputs/logs/train_log.csv` | 训练日志（每 epoch 的 loss、acc、lr、时间） |
| `outputs/logs/data_stats.json` | 数据统计信息 |
| `outputs/submissions/pred_raw.csv` | 原始预测结果（image_name, pred_idx, pred_label） |
| `outputs/submissions/pred_results.csv` | 最终提交文件（格式：`image_name.jpg, 0001`） |
| `outputs/submissions/submission.zip` | 提交压缩包 |

## 关键设计说明

1. **Class mapping**: 类别按文件夹名字典序排序，映射到 0-499。`class_to_idx.json` 和 `idx_to_class.json` 在 split 阶段生成，训练和推理阶段复用。

2. **Image preprocessing**: 使用 CLIP 自带的 `preprocess` 函数，保证与预训练时的预处理一致。

3. **混合精度训练**: 默认开启 AMP（`torch.cuda.amp`），使用 `GradScaler` 防止梯度下溢。

4. **学习率调度**: CosineAnnealingLR，带线性 warmup（默认 1 epoch）。

5. **Sample weight 接口**: `TrainImageDataset` 提供 `get_sample_weights()` 方法和 `return_path=True` 选项，方便后续接入噪声筛选或样本加权。

6. **损坏图片处理**: 遇到损坏图片打印 warning，返回空白 RGB 图，确保训练不中断。

## 后续可扩展方向

以下方向可在当前代码基础上扩展，但不包含在当前 baseline 中：

1. **LoRA / Adapter 微调**: 在 CLIP backbone 中插入低秩适配器，部分微调视觉编码器
2. **噪声标签筛选**: 基于 small-loss criterion 筛选干净样本
3. **样本加权**: 利用 `get_sample_weights()` 接口对疑似噪声样本降权
4. **Prototype-based 方法**: 使用类别原型进行特征空间去噪
5. **数据增强**: 在训练集加入 RandAugment / MixUp / CutMix
6. **Co-teaching / DivideMix**: 双模型协同训练筛选噪声
7. **模型集成**: 多模型投票（初赛阶段不适用）
