# 初赛 75 分冲刺 v9：原图回采样 local 推理

计划 ID：`PRELIM75_V9_20260920`

固定审阅提交：`cbea37b03b75b91436b536cacd646f0e53a792e2`

状态：**实现与只读资产 preflight 已完成，33 项合成 CPU 测试通过；尚未执行 D0 或 GPU 阶段。**

## 决策与边界

v8 的 B0=67.8816%、B1=62.0659% 均低于匹配的无 prior 对照，本轮不再改变偏置强度、正则、边界或监督。v9 保留 L1 无 prior 69.2794% 为唯一模型，只检验在 attention 框和输出分辨率不变时，local 像素从 native 224 canvas 或原始解码 RGB 重采样的影响。

| 对象 | local 像素来源 | 输出 | 作用 |
|---|---|---:|---|
| 归档 L1 | 224 normalized tensor 内裁剪，`F.interpolate` | 224 | 69.2794% 回退与完整重放基线 |
| R0 | native 224 PIL RGB canvas，Pillow BILINEAR float box | 224 | 量化、kernel 与 ROI 边缘路径对照 |
| R1 | 原始解码 RGB 上的逆映射 float box，Pillow BILINEAR | 224 | 原图回采样候选 |

R0/R1 共享 L1 的 global tensor、global logits、attention 和 native 框；不重算框、不平均候选、不加 prior。R0 不与原生 `F.interpolate` 宣称像素等价，R1−R0 也不作为纯分辨率因果分解。

固定资产：

- L1：`outputs/prelim75_v7_20260918/L1/candidate.pt`，SHA-256 `cd485c7beed2781ac13d08fbd33d63b8f1c46971be008966f7280c67cec1fcee`。
- 回退 ZIP：SHA-256 `55d25f1fea9c57ba3431164d53d21f34d20baa2a49c805ad87cca188b5634c5a`。
- 回退 CSV：SHA-256 `5277b2c33f63b951ea2236ae2e50b8c574ef9f8d2f989e8239fcce7b065c0171`。
- 类别映射：SHA-256 `3edfd9e48e2f171b5d452577c52e0d189ae99dbfed3759d3e4f164fc090a6647`。

ViT-B/32、输入 224、top-k 5、尺度 112/128/144/160、尺度权重 0.2/0.3/0.4/0.1、global/local 0.6/0.4、Flip 0.5、温度 1.5、FP32、batch 64、O3/PTA 与无 prior 均固定。全部参数 `requires_grad=False`，不创建 optimizer。

## D0 与逐图门槛

D0 只使用完整训练清单和内容组映射。每组取字典序最小 canonical path，按 `sha256("PRELIM75_V9_20260920/42/" + canonical_path)` 排序取前 2,048 组。只解码尺寸并核对部署版 Resize/CenterCrop，不运行模型、不读取标签、trust、置信度或测试尺寸。

至少 205/2048 组满足 `source_gain=min(W/Wr,H/Hr)>=1.5` 才继续。真实推理逐图使用同一个固定门槛；低于门槛时 R0/R1 均直接复用原生 local tensor，逐元素不变。D0 失败则状态为 `closed_insufficient_source_support`，保留 L1，不降低门槛。

## 几何与像素契约

部署 transform 必须是 OpenAI CLIP 的 Resize(224, bicubic)+CenterCrop(224)+RGB+ToTensor+Normalize。运行时从实际 transform 得到 resized 尺寸并核对：CenterCrop 偏移使用 `int(round((length-224)/2.0))`。

Flip 框先在 224 坐标系解除翻转：`(224-x1,y0,224-x0,y1)`，再映射到未翻转原图：

```
u0 = (left + x0) * W / Wr
v0 = (top  + y0) * H / Hr
u1 = (left + x1) * W / Wr
v1 = (top  + y1) * H / Hr
```

边界保留浮点，使用 `Image.resize((224,224), BILINEAR, box=..., reducing_gap=None)`；Flip local 在采样后水平翻转，只接 native RGB/ToTensor/Normalize tail，不再次 Resize/CenterCrop。禁止 EXIF transpose、ICC 校色、锐化、超分、去噪或新的透明通道处理。

## 执行与停止规则

固定队列为 `preflight -> d0 -> smoke -> diagnostic -> baseline_replay/R0/R1 infer -> check -> register`。smoke 检查 global tensor、logits/features/attention、共享框、低 gain 回退及全部权重/缓冲区不变。10,316 行重叠诊断只拒绝相对本次 baseline replay 下降超过 2pp 的明显退化，不用于挑参数或泛化声明。

完整测试 baseline 必须按 basename 与归档 L1 CSV 的 24,967 行全部一致；任一不一致即停止交付。R0/R1 与已有预测集完全相同时只复用同预测集反馈，不制造新实测分数。最多两个唯一新候选，每份 CSV 覆盖全部测试 basename，标签为 `0000`–`0499`，ZIP 只含 `pred_results.csv` 且包内外字节一致。

GPU 阶段累计上限 7,200 秒；训练、拟合和 optimizer update 均为 0。不扫描门槛、kernel、尺度、融合比例或 G0。达到预算即停止，不缩减样本清单。

平台选择以 L1 69.2794% 为匹配基线：两者不超过 L1 则关闭本轮；R0 最好则保留 R0 但不宣称原图有效；R1 最好则保留 R1 并报告 R1−R0、R1−L1；R1 超过 `max(L1,R0)` 至少 0.30pp 只表示达到工程投入门槛，仍不自动扩展实验。同分优先 L1；R0/R1 同分且均超过 L1 时保留更简单的 R0。

72.4677% 的 G0+legacy test-batch prior0.9 继续作为不同来源协议的单列绝对最高，不参与同条件机制归因。本轮不读取或移植该偏置。

## 实现入口

- `configs/prelim75_v9.yaml`
- `reproducibility/aegis_f1/aegis_clip/source_recrop.py`
- `reproducibility/aegis_f1/aegis_clip/prelim75_source_recrop.py`
- `reproducibility/aegis_f1/aegis_clip/cli/infer_prelim75_v9.py`
- `scripts/run_prelim75_v9_queue.py`
- `reproducibility/aegis_f1/tests/test_prelim75_v9.py`

正式命令：

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -u scripts/run_prelim75_v9_queue.py --config configs/prelim75_v9.yaml --execute
```

运行前必须在 `main`、工作树干净且 `origin/main` 已同步。队列不自动上传平台、不自动 push、不覆盖旧输出。
