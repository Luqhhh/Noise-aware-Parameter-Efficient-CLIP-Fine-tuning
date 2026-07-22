# P1 方案：单模型均衡类别先验校准

日期：2026-07-19

## 发现

赛题说明第 2 页明确写明最终测试集“类别分布均衡”。初赛官方训练集也接近严格均衡：500 类各 201–213 张。与之相比，F1 的 24,967 张测试预测虽然覆盖全部类别，但单类预测数为 2–170，标准差约 24.2，存在明显的模型类别偏置。

## 方法

P1 不训练新模型，也不修改 F1/F2 权重。它收集同一检查点在完整测试集上的 logits，通过一个全局类别 bias 做迭代比例拟合，使平均 softmax 分布接近赛题公开的均匀先验：

`p'(y|x) = softmax(logit(x) + alpha * bias)`

其中 `alpha=0` 等于原模型，`alpha=1` 为完整先验匹配。最终仍对每张图独立取 Top-1，提交产物仍来自一个模型、一次推理流程，不使用测试标签，也没有任何模型参数更新。

## 合规边界

- 支持依据：赛题说明明确宣布测试集类别均衡，并允许“单一模型或单一推理流程”。
- 风险：规则同时规定测试数据只能用于测试，不得参与自监督或无监督训练。P1 属于推理校准而非训练，但它使用整批测试预测的边际分布，仍应先由队长向赛方确认口径。
- 工程默认关闭；启用时必须显式传入 `--acknowledge-balanced-test-prior`，并在提交清单中记录强度、迭代次数、校准前后边际误差及类别计数范围。
- 不应直接用平台分数扫描多个强度。先在开发验证集上固定 `alpha`，平台只评一个预注册候选。

## 验证顺序

1. 显卡空闲后导出 F1 验证集的裸推理与 Flip TTA logits。
2. 只在预先定义的 `alpha = {0.25, 0.5, 0.75, 1.0}` 上比较 raw、clean-core、trusted 三组指标及类别计数偏差。
3. 使用 clean-core micro 为主指标，clean-core macro 与 trusted macro 为并列裁决；不使用平台结果选强度。
4. 只有在本地通过且队长确认规则后，才生成一个 P1 提交包。

## 已完成结果

预注册扫描选择 `alpha=0.5`。相对原始 F1 Flip TTA：

- clean-core micro：81.5441% -> **81.6522%**（+0.1081pp）
- clean-core macro：82.0553% -> **82.2334%**（+0.1781pp）
- trusted micro：80.4773% -> **80.5376%**
- trusted macro：80.6580% -> **80.7687%**
- raw micro：71.0353% -> **71.0353%**（持平）

测试集候选改变 3,093/24,967 条预测（12.3884%），类别预测数从 2–170 收敛到 8–97；仍覆盖全部 500 类。提交包审计通过，但在队长确认赛方对整批推理校准的口径前不得上传。

- 预测 CSV SHA-256：`d9c6023e90ae29e481ec2cf198be3f836e61d1ce9b9d0fa55d0592418032943a`
- ZIP SHA-256：`5492237ccf81882c803f36a8d37fd7f9c070906099df7214726de302dc629c52`

预注册本地扫描命令：

```bash
python -m aegis_clip.cli.sweep_online_tta_fusion \
  --config configs/f1_visual_lora_clean_core.yaml \
  --checkpoint outputs/F1_VISUAL_LORA_CLEAN_CORE/seed42/checkpoints/best.pt \
  --output outputs/F1_VISUAL_LORA_CLEAN_CORE/seed42/analysis/prior_alignment_sweep.json \
  --prior-mode mean_probabilities --prior-temperature 0.5 \
  --prior-strengths 0.25,0.5,0.75,1.0
```

## F2 固定参数迁移候选

F2 全量训练完成后，只迁移 F1 已经预注册选出的配置，不使用 F2 的重叠诊断集重新选择参数：水平翻转、概率均值、`T=0.5`、`alpha=0.5`、50 次先验拟合。审计结果为：

- checkpoint SHA-256：`7904312e7ca13b5ea6ea01d47dca5b3e59df64c374f4bdb69af9c7672a0042b6`
- 推理模式：`horizontal_flip:mean_probabilities:t=0.5:balanced_prior=0.5`
- 校准前后平均边际 L1：0.339466 -> 0.183831
- 类别预测数：2–161 -> 8–96，仍覆盖全部 500 类
- 相对 F2 TTA 改变 3,042/24,967 条预测（12.1841%）
- 预测 CSV SHA-256：`3be990602fff78f811db192cffe994a99cf6a448e6903a8959f0349e73d07a85`
- ZIP SHA-256：`e961069e34200a0fc775a7d8135033947b4a8e8fd7506d9c395d4765003758a7`

该产物通过工程审计，但合规状态与 F1 P1 相同：在队长获得赛方对“使用整批测试预测边际做单模型推理校准”的明确答复前，不上传、不用平台分数继续搜索强度。
