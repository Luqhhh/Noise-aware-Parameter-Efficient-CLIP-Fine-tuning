# RM_FT / RM_LT 严格本地比较材料

本交付只对已有 best checkpoint 导出预测，不重新训练、不选新 checkpoint、不调整测试推理。
所有路径相对仓库根目录 `/home/lux1/noise`。二进制与逐样本导出保存在本地 outputs，Git 保存脚本、说明和哈希清单。

## 配方与 checkpoint

| 项目 | RM_FT | RM_LT |
|---|---|---|
| 配置 | `configs/rematch750_ft.yaml` | `configs/rematch750_lt.yaml` |
| best checkpoint | `outputs/rematch750/RM_FT/seed42/checkpoints/best.pt` | `outputs/rematch750/RM_LT/seed42/checkpoints/best.pt` |
| checkpoint SHA-256 | `94a361e8672a0db67f43e1e35e91f22715964124ab59ee5157fd3d86c67be305` | `088f6464564e82c15a268e35e2aaef3b82699f2a0ca57c24a9013bb28bbd778e` |
| 训练代码 commit | `b975d3268c2b23512b9739135fa9e6b5d79e68fb` | `a63f7ad6d61c52400da963a819bfde564fb54dc8` |
| 采样 | 普通 shuffle | 每样本权重 `1/sqrt(n_class)`，有放回，每轮抽取 train_dev 样本数 |

共同配方：OpenAI CLIP ViT-B/32，750 类线性头；从同一 RM_LP best 独立初始化；视觉 full_finetune，但 conv1 与位置嵌入冻结。seed 42，8 epochs，batch 32，AdamW，head LR 1e-4、backbone LR 3e-6，weight decay 1e-4，cosine horizon 8、LR floor 1%、无 LR warmup，梯度裁剪 1.0。前 2 epoch CE，之后 GCE q=0.5，加权 2.0 的官方冻结参考特征 cosine distillation。增强为 224 随机裁剪（scale 0.70–1.0，ratio 0.85–1.15）和 p=0.5 水平翻转。无 MixUp、trust、loss reweighting 或 prior。AMP 初始 scale 128，growth interval 1e9。

每 2 epoch 验证一次，按 raw macro 最大、同值时 raw micro 选 best；两者均选中 epoch 8。验证/测试为官方 224 center crop、单 checkpoint、AMP、无 TTA。完整解析配置在各 run 的 `checkpoints/resolved_config.json`，初始化配置为 `configs/rematch750_lp.yaml`。

## 共用 split

目录：`artifacts/stages/repechage/20260921/`。训练 133,815，验证 14,880，验证覆盖 750 类；两份 split 的路径和 decoded-RGB 内容组交集均为 0。seed 42，按 decoded RGB SHA-256 分组划约 10% 验证，保护每类至少 4 个训练组（不足 4 组时保留所有）。保留原始含噪标签。

| 文件 | SHA-256 |
|---|---|
| `train_dev.csv` | `615d510481e9df610371e84cbdeeb0058cc9a176989786a8ccf4eb30235b92c6` |
| `val_dev.csv` | `d91106df365b62f9bf56eff51474c222925301546642fa427f6778258cb833ab` |
| `class_to_idx.json` | `6f6fdc3eef8e526d3cde89da6d98690603e26ee8cae5de7e7e90c743bf3a551c` |

## 指标口径

| 模型 | micro | macro | bottom-10% macro | 固定训练尾部 10% macro | best epoch |
|---|---:|---:|---:|---:|---:|
| RM_FT | 72.9032% | 71.8362% | 22.5567% | 55.5827% | 8 |
| RM_LT | 72.6949% | 71.8655% | 23.5166% | 58.1485% | 8 |

micro 为全部验证样本准确率；macro 为 750 类 recall 均值。bottom-10% 为各模型验证 recall 最低的 75 类的 recall 均值，两个模型的类别集合可能不同。固定训练尾部指标用 train_dev 样本数最少的同一组 75 类；并列按 class ID 排序。两种类别集合均记录在 manifest。这里的 bottom-10% 不等同于旧报告中 `<20` 样本的 support_tail（仅 2 类）。验证标签含噪，指标不是干净标签准确率。

## 逐样本记录与 logits

每个候选的导出目录：`outputs/rematch750/{RM_FT,RM_LT}/seed42/comparison/`。

- `val_prediction_records.csv`：14,880 行，字段 `image_path,label,prediction,correct,confidence,true_label_probability`；label/prediction 是 0-based class index。
- `val_logits.pt`：float32 `[14880,750]`，键 `logits,labels,paths` 等；paths 与 val_dev.csv 顺序一致，保留 `train/` 前缀。
- `test_logits.pt`：float32 `[37444,750]`，键 `logits,names,class_mapping,checkpoint_sha256,inference`；按 names 对齐测试图。
- `manifest.json`：文件 SHA-256、重算指标、类别集合、原验证逐类正确数与原 test 提交逐图预测的一致性检查。

第 j 列对应 `class_to_idx.json` 中 index=j 的类。概率可由 `payload['logits'].softmax(dim=1)` 得到，属于未校准 softmax 概率。

```python
import torch
p = torch.load('outputs/rematch750/RM_FT/seed42/comparison/val_logits.pt',
               map_location='cpu', weights_only=False)
probabilities = p['logits'].softmax(dim=1)
```

重放命令（仓库根目录，需 GPU）：

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -u scripts/export_rematch_comparison.py
```

导出脚本逐图校验源文件，验证重算的 750 类正确数与原 best_evaluation.json 完全相同；test argmax 必须逐图等于原提交 CSV。两组已有 `submission/pred_results.csv` 与 `submission/submission.zip` 均重新通过提交校验（37,444 图）。详细机器可读结果见 `results/rematch750_local_comparison_20260922.json`。
