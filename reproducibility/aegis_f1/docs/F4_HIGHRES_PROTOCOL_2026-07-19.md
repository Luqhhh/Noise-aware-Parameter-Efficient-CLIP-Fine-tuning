# F4：同骨干 288px 高分辨率协议

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan/run
- Origin Date: 2026-07-19
- Verification Status: ANALYZED — STOP
- Version Label: f4_highres_protocol_v1

## 假设

OpenAI CLIP ViT-B/32 在 224px 下只有 7×7 个视觉 patch。保持官方 ViT-B/32
骨干、官方权重、分类头和 F1 LoRA 权重不变，将输入固定为 288px，并把官方
7×7 视觉位置编码以二维双三次插值为 9×9，可能改善细粒度局部辨识。

## 预注册约束

- 唯一候选分辨率：288px；不扫描 256/320/336/384。
- 固定检查点：`F1_VISUAL_LORA_CLEAN_CORE/seed42/checkpoints/best.pt`。
- 固定推理：水平翻转 TTA、概率均值、温度 0.5。
- 固定验证集、clean-core 阈值 0.70、seed 42，不使用测试集选择。
- 224px 基线来自既有 `online_tta_sweep.json` 的已选定 F1 TTA 方案。

## Phase 1 门槛：仅改变推理分辨率

| 指标 | 224px 基线 | 288px 通过条件 |
|---|---:|---:|
| clean-core micro（主指标） | 0.8154408 | ≥ 0.8169408（+0.15pp） |
| trusted macro（护栏） | 0.8065804 | ≥ 0.8060804（最多 -0.05pp） |
| raw macro（弱护栏） | 0.7099994 | ≥ 0.7089994（最多 -0.10pp） |

三项须同时满足。若失败，停止 F4，不训练、不生成测试集提交；若通过，才允许
以 F1 检查点为起点进行预先限定的 2 epoch 288px 适配，并以同一门槛判断是否
进入全量重放。

## Phase 1 实测结果

固定 F1 检查点与预注册 TTA 参数于 2026-07-19 完成一次 288px 评估：

| 指标 | 224px 基线 | 288px 实测 | 变化 | 判定 |
|---|---:|---:|---:|---|
| clean-core micro | 0.8154408 | 0.8081396 | -0.7301pp | FAIL |
| trusted macro | 0.8065804 | 0.8007475 | -0.5833pp | FAIL |
| raw macro | 0.7099994 | 0.7070132 | -0.2986pp | FAIL |

三项均未达到预注册条件。F4 按协议停止，不进行 288px 训练、不生成测试集提交，
也不继续扫描其他分辨率。结果文件为
`outputs/F4_HIGHRES_288_EVAL/seed42/analysis/f1_checkpoint_eval.json`。

## 合规边界

F4 不更换骨干、不增加视觉层、不引入外部数据或额外模型，所有可学习参数仍来自
官方 OpenAI CLIP ViT-B/32 与本工程单模型微调。位置编码插值是确定性的输入尺寸
适配，但赛规未逐字说明是否允许改变官方 224px 推理尺寸；因此即使本地通过，
生成平台包前仍需在合规文档中标记这一解释边界，并建议由队长/主办方确认。

## 预定命令

```bash
.venv/bin/python -m aegis_clip.cli.evaluate \
  --config configs/f4_highres_288_eval.yaml \
  --checkpoint outputs/F1_VISUAL_LORA_CLEAN_CORE/seed42/checkpoints/best.pt \
  --tta horizontal_flip \
  --tta-fusion mean_probabilities \
  --tta-temperature 0.5 \
  --output outputs/F4_HIGHRES_288_EVAL/seed42/analysis/f1_checkpoint_eval.json
```
