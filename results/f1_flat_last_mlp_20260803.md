# F1_FLAT_LAST_MLP（2026-08-03）

## 结论

选择 epoch 2 生成平台候选。该实验从当前平台最佳 FLAT checkpoint 继续训练，
冻结继承的全部 attention-LoRA 和 attention 参数，只更新最后一个视觉 Transformer
block 的 MLP/`ln_2`、`visual.ln_post`、`visual.proj` 与低学习率分类头，共
5,375,220 个可训练参数。这样保持 M1 裁剪所使用的注意力图不变，同时允许最终
视觉表征适配噪声鲁棒目标。

全局预测略有退化，但固定 M1+flip 协议的 raw、trusted、proxy、clean-core 指标
全部超过 FLAT。候选相对已实测 67.7014% 的 FLAT 包改变 442/24,967 个预测。

## 复现

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train \
  --config configs/f1_flat_last_mlp.yaml
```

- 配置：`reproducibility/aegis_f1/configs/f1_flat_last_mlp.yaml`
- 父 checkpoint：FLAT epoch 7，SHA-256
  `6543d93bd7bf3b52f70e30487f2d0be6d37dd9e922d678202503df0c61024b54`
- 候选 checkpoint：epoch 2，SHA-256
  `e8cf821874a2ad0dee1cec02e5b98701e1fc7ee969ca9c620d9acbcbcf7c0422`
- 回归测试：`248 passed`
- 训练在 epoch 2 评估完成后人工终止；epoch 1 和 epoch 2 的 global promotion
  都未通过，因此没有继续运行计划中的 epoch 3。

## 固定指标

| 指标 | FLAT | epoch 1 | epoch 2 | epoch 2 vs FLAT |
|---|---:|---:|---:|---:|
| global raw micro | 71.3164% | 71.1904% | 71.1225% | -0.1939pp |
| global clean-core micro | 81.9261% | 81.6533% | 81.7487% | -0.1773pp |
| M1+flip raw micro | 72.2082% | 72.2470% | 72.4021% | +0.1939pp |
| M1+flip trusted macro | 81.6395% | 81.8895% | 81.8912% | +0.2517pp |
| M1+flip proxy macro | 80.1980% | 80.3831% | 80.4710% | +0.2730pp |
| M1+flip clean-core micro | 82.4853% | 82.6490% | 82.7718% | +0.2865pp |
| M1+flip clean-core macro | 83.0653% | 83.3686% | 83.3144% | +0.2491pp |

固定推理为 crop160/top5/local0.40/flip0.50/temperature1.5；测试推理继续使用
已在平台验证过的 balanced-prior strength 0.85。

## 候选提交

- 目录：`outputs/delivery/flat_last_mlp_ep2_l040_f050`
- 推理：M1 attention crop + horizontal flip，单 checkpoint，balanced prior 0.85
- prediction count：24,967；classes：500；corrupt images：0
- CSV SHA-256：
  `2e6dfae246cafe2d3a672e7974b6cc064d13f93c983e0f99733d261f2a134cc9`
- ZIP SHA-256：
  `583e9858f8700d3399ca4dfcfd38c70f783de6293fb3bc8fc6cc6f99a5c47692`
- `aegis_clip.cli.audit_submission --allow-tta`：PASS
- 平台状态：`selected_audited_pending_platform`
