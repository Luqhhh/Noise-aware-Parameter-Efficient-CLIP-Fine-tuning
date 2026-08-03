# F1_FLAT_MLP_LORA_FP32（2026-08-03）

## 结论

固定 3 epochs 训练完成，使用最终 `epoch_3.pt` 生成平台候选，不使用本地晋级门槛。
该实验冻结 FLAT 已学习的全部 attention-LoRA，在 12 个视觉 Transformer block
的两层 MLP 线性映射上增加 rank-4 LoRA。MLP-LoRA 共 368,640 个参数；加低学习率
线性头后总可训练参数为 625,140，attention 参数保持冻结。

初始 AMP 运行在首个训练步被有限梯度审计截停（visual gradient NaN），没有完成
有效更新。正式运行改为 FP32、batch 32，首步审计通过并跑满固定三轮。

## 复现

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train \
  --config configs/f1_flat_mlp_lora.yaml
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train \
  --config configs/f1_flat_mlp_lora.yaml \
  --resume outputs/F1_FLAT_MLP_LORA_FP32/seed42/checkpoints/last.pt
```

- 配置：`reproducibility/aegis_f1/configs/f1_flat_mlp_lora.yaml`
- 父 checkpoint：FLAT epoch 7，SHA-256
  `6543d93bd7bf3b52f70e30487f2d0be6d37dd9e922d678202503df0c61024b54`
- 最终 checkpoint：epoch 3，SHA-256
  `df6f385c9cc7c1c82d9a20bcf4792d408d11f07d0835f3ec56b7091b5aa2e2eb`
- 回归测试：`250 passed`
- 训练：seed 42、FP32、batch 32、固定 3 epochs；恢复包含 optimizer、scheduler
  与 RNG 状态。

## 训练记录

| 指标 | FLAT/epoch 0 | epoch 1 | epoch 2 | epoch 3 |
|---|---:|---:|---:|---:|
| global raw micro | 71.3164% | 71.2776% | 71.2970% | 71.4327% |
| global trusted macro | 81.1528% | 81.0861% | 81.1210% | 81.2691% |
| global proxy macro | 79.9475% | 79.8912% | 80.0001% | 80.1557% |
| global clean-core micro | 81.9261% | 81.8988% | 81.9670% | 82.1034% |
| flip agreement | 89.7829% | 89.8895% | 89.9089% | 90.1706% |

本地指标只用于记录数值健康、类别覆盖和训练轨迹，不作为是否生成平台包的条件。

## 候选提交

- 目录：`outputs/delivery/flat_mlp_lora_fp32_ep3_l040_f050`
- 推理：单 checkpoint，M1 crop160/top5/local0.40 + flip0.50，temperature 1.5，
  balanced prior 0.85
- prediction count：24,967；classes：500；corrupt images：0
- 相对 FLAT 平台最佳包改变预测：1,032（4.13%）
- CSV SHA-256：
  `b240209ea44d98818405f52884adec1014e4dd29f985a354ca43ea56b4d9b564`
- ZIP SHA-256：
  `afa7cf8863b15218b6dc2975874406f85dc599824c693f4a37ba97cc9ec6efc7`
- `aegis_clip.cli.audit_submission --allow-tta`：PASS
- 平台状态：`selected_audited_pending_platform`

## 同轮关闭的推理方向

- CVRG 五折嵌套交叉拟合：baseline raw 72.1888%，dynamic 72.1210%，
  下降 0.0679pp；未读取 test。
- 硬均衡配额：验证集约束每类 20/21 后 raw、trusted、proxy、clean-core 均下降
  约 0.9--1.1pp；未读取 test。
