# Project Restructure Design

## Motivation

当前 `noise_fgvc_clip_baseline/` 把所有代码（configs、src、scripts、outputs）塞在一个目录里。后续加入 LoRA、prototype、noise-filtering 等方法时会混乱。需要提升为多实验项目结构。

## New Structure

```
noise/                                    # 仓库根目录
├── data/                                 # 数据集（实际路径通过 config 指定，不提交）
├── common/                               # 所有实验共享的代码
│   ├── dataset.py                        # TrainImageDataset / TestImageDataset
│   ├── utils.py                          # load_config / set_seed / setup_logging
│   └── submission.py                     # pred_results.csv + submission.zip 生成
├── configs/                              # 每个实验一个 YAML
│   ├── baseline.yaml
│   └── (lora.yaml, prototype.yaml ...)
├── scripts/                              # 共享脚本（数据准备、检查）
│   ├── split_data.py
│   ├── check_data.py
│   └── check_submission.py
├── experiments/                          # 每个实验方法一个独立子目录
│   └── baseline/
│       ├── model.py                      # CLIPLinearClassifier
│       ├── train.py                      # 训练入口
│       ├── evaluate.py                   # 验证集评估
│       └── infer.py                      # 测试集推理
├── outputs/                              # 输出按实验隔离
│   └── baseline/
│       ├── checkpoints/
│       ├── logs/
│       ├── submissions/
│       └── splits/
├── requirements.txt
└── README.md
```

## Design Principles

1. **common/** — 不随实验变化的共享代码。Dataset 类、工具函数（config/seed/logging）、提交文件生成。所有实验 `from common.xxx import ...`

2. **experiments/<name>/** — 每个方法自包含：model.py + train.py + evaluate.py + infer.py。只依赖 common，不互相依赖。

3. **configs/<name>.yaml** — 一个实验一个 config，指向对应的 experiment 和 output 目录。

4. **outputs/<name>/** — 输出按实验隔离，避免互相覆盖。

5. **scripts/** — 数据准备和提交检查，与具体实验无关，直接 import common。

## Migration Steps

1. Git mv: 移动文件到新位置
2. Fix imports: 更新所有内部 import 路径（`from .xxx` → `from common.xxx` 等）
3. Fix configs: 更新 YAML 中的路径引用
4. Fix scripts: 更新脚本中的 import
5. Update .gitignore: 适配新 output 路径
6. Test: 跑一次完整的 data-check → split → train (2-epoch) → infer → submission → check 流程

## What Stays the Same

- 所有 Python 文件的内容逻辑不变
- 配置项名称不变
- CLI 接口不变（argparse 参数不变）
- requirements.txt 不变

## What Changes

- Import 路径（`from .dataset` → `from common.dataset`）
- 文件在仓库中的位置
- Config 中的相对路径（split_dir、save_dir 等）
- .gitignore 规则
