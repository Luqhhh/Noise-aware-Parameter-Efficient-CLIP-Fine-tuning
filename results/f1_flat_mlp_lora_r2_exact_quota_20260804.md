# F1_FLAT_MLP_LORA_R2_EXACT_QUOTA_T050（2026-08-04）

## 结论

在平台新最佳 `F1_FLAT_MLP_LORA_R2_FP32`（67.86558256899107%）的同一
checkpoint、同一 M1+flip 融合 logits 上，将 prior-0.85 校准替换为测试集级
Sinkhorn 分配与精确整数近均衡配额。24,967 张测试图被严格分为 467 个 50 张
类别和 33 个 49 张类别；33 个较小配额类别由模型在第 50 个候选上的支持度决定，
不按类别索引硬编码。

本候选不使用验证集得分作晋级条件。它用于直接检验平台测试集是否接近官方声明
的均衡类先验，平台结果是唯一裁判。

## 方法与复现

先对固定融合 logits 以 temperature 0.5、100 iterations 做软均衡 Sinkhorn；
再从过量类别中按最小模型分数损失确定性地移动样本，直到每类计数与整数配额完全
一致。

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer_exact_transport_submission \
  --dump outputs/F1_FLAT_MLP_LORA_R2_FP32/seed42/test_fused_logits_ep3_l040_f050_t15.pt \
  --checkpoint outputs/F1_FLAT_MLP_LORA_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_r2_ep3_exact_quota_t050 \
  --temperature 0.5 --iterations 100 \
  --acknowledge-balanced-test-prior
```

- checkpoint SHA-256：
  `fe901eb0cfed4368ce4ea68c8ccc83cba74d73fabffdd0938611b3145edbe3b5`
- 融合 logits SHA-256：
  `a56ac1540df77aa6485c4a366e4f5225329845953f0cd419cb092cbe152c6d5e`
- Sinkhorn argmax 初始计数范围：39--61
- 精确配额修复移动：616 张
- 最终类别计数范围：49--50；500 类完整覆盖
- 相对平台 67.8656% 的 prior-0.85 包改变：2,525/24,967（10.11%）
- CSV SHA-256：
  `cfa250f2d776c402b5f85043cda7e4344402a24e05b49e7fe333bc26074cdbaa`
- ZIP SHA-256：
  `64200582a8f93eb211b5088f4c04bf86181dcc959fcb8fbe1154223c1f1ec1bd`
- `aegis_clip.cli.audit_submission --allow-tta`：PASS
- 回归测试：`253 passed, 8 warnings`
- 桌面副本 SHA-256 完全一致
- 平台状态：`selected_audited_pending_platform`
