# F1_A12CORR_ADAPTFORMER（2026-08-03）

## 结论

关闭，不生成测试提交。该实验保留当前平台最佳 A12_CORR 的全 12 层
attention-LoRA 并将其冻结，在末 6 层增加零初始化 AdaptFormer 分支，只训练
604,032 个 adapter 参数和单一线性分类头。

## 复现

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.train \
  --config configs/f1_a12corr_adaptformer.yaml
```

- 配置：`reproducibility/aegis_f1/configs/f1_a12corr_adaptformer.yaml`
- 父 checkpoint：A12_CORR epoch 8，SHA-256
  `ec6948c96aaa921df1034c78972dec8ca58e88aea04fd7ccbc58b143060a3238`
- 输出 checkpoint：epoch 2，SHA-256
  `ca0175cc3d71f12b0381246b70c690173e8396bee98d9cd0a55fa382d2000362`
- 回归测试：`247 passed`
- 训练命令正常退出，2 epochs，seed 42。

## 固定指标

| 指标 | A12_CORR epoch 0 | Adapter epoch 2 | 变化 |
|---|---:|---:|---:|
| global raw micro | 71.2098% | 71.1904% | -0.0194pp |
| global clean-core micro | 81.8306% | 81.8306% | 0.0000pp |
| M1+flip raw micro | 72.2373% | 72.1501% | -0.0872pp |
| M1+flip clean-core micro | 82.6354% | 82.4035% | -0.2319pp |
| M1+flip trusted macro | 81.7414% | 81.5403% | -0.2011pp |
| M1+flip proxy macro | 80.3538% | 80.1699% | -0.1839pp |

固定推理为 crop160/top5/local0.40/flip0.50/temperature1.5。训练 promotion
为 FAIL（selector gain 0），固定 M1+flip 的四项指标全部低于 A12_CORR，因此不读
test、不生成包。最终候选继续使用已完成审计的 FLAT/A12_CORR 包。
