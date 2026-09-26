# L05 三位置裁剪 + Flip（预注册）

实验 `L05_THREECROP_20260926`；分支 `codex/l05_threecrop`。
2026-09-26 fetch 后逐一检查所有分支及 main 历史：F05 A0 为注意力引导的放大局部图；
letterbox 引入缩小/填充；RM-FT/L05 几何诊断是低通缩图；L05 已测的是同一个中心裁剪的
Flip。本轮保持相同缩放尺度，补回 center crop 舍弃的长边两端，属于尚未测量的空间覆盖机制。
不以旧的闭环失败配方派生参数扫描。

## 固定协议 l05_threecrop_v1

同一个冻结 L05 checkpoint、无训练/梯度/权重变化。先用原生 CLIP bicubic Resize 将
短边变为 384；取中心、左上端、右下端三个 384×384 方形窗口（短边无位移，因此实际沿
长边平移）。每窗口加水平翻转，共六视图。方图三个窗口相同，保持既有 Flip 行为。
每路 softmax(z/1.4)，六路等权平均，再 log；val 上拟合 uniform prior，迭代固定 50 次、
强度固定 0.60。温度、权重、视图数、crop size、prior 强度均不扫描。

这是显式登记的**新推理协议**，执行入口 `scripts/run_l05_threecrop.py`；只接受配置中的
协议 id、冻结 checkpoint SHA、复赛 manifest/split/mapping SHA。复用现有 CLIP 模型、
均匀 prior 实现和同 checkpoint TTA 授权。保持原始 dataset_manifest，不修改/清空第一轮
CLI 闸门，也不把新协议冒充为第一轮 global-only。测试推理必须等验证门通过后再实现并冻结，
不能从测试预测反向选择配方。

验证集 14,880 张固定为旧缓存顺序，逐图检查文件 SHA；首 32 张重新执行原生 center+Flip，
要求两路 top1 全一致且最大 logit 差 ≤0.02。比较六视图和现役 L05 两视图，报告无 prior 与
固定 prior 两种读数，正式预筛：解码 macro ≥+0.30pp 且 micro 不下降。未过门立即关闭，
不派生扫描；无新测试候选，本轮交付已冻结 L05 保底包复核。

若过预筛门，继续做固定三折条件校准：按 validation content_group 的
SHA256("42:"+group) 前 8 bytes 大端整数 mod 3 分折；各折的 prior 仅用另外两折 logits 拟合。
候选与基线共享折。pooled macro 相对基线 ≥+0.20pp 且 micro 不下降才进入出包。
这只是移除 prior 的本折拟合泄漏；checkpoint 和 T/prior 强度此前已在整份 val 选过，
不能声称是独立无偏的全链路 OOF。若通过则锁定全部 val 拟合的 bias，测试仅单次确定性推理。

## 规则合规性

- Backbone: CLIP ViT-B/32
- Pretrained weights: OpenAI official
- External data: No
- Test data used for training/adaptation: No
- Cross-stage data/checkpoint reuse: No
- Multi-model ensemble: No（六视图全来自同一 checkpoint）
- Manual cleaning required: No
- Fully reproducible: Yes
- Rule risk: None；COMPETITION_RULES_AGENT.md §18.1/7 已记录同 checkpoint 多 crop/Flip 合规

固定配置 `configs/l05_threecrop/fixed.json`。独立输出 `outputs/codex/l05_threecrop/`。
复现：`python3 scripts/run_l05_threecrop.py --config configs/l05_threecrop/fixed.json`。


## 实测结果

| 固定协议 | 无 prior macro / micro | prior=0.60 macro / micro |
|---|---|---|
| 现役 L05 center+Flip | 75.4271% / 76.4651% | 75.8265% / 76.6532% |
| 三位置裁剪+Flip | 75.7463% / 76.7675% | 75.9645% / 76.7339% |
| 差（pp） | +0.3192 / +0.3024 | **+0.1380 / +0.0806** |

共 693 张预测改变，纠正 174、破坏 162。预注册主判据是现役完整解码口径，macro 未过
+0.30pp 门，故**关闭本固定协议，不派生 crop/融合权重/温度/prior 扫描，不运行晋级后的
交叉校准，不出新测试候选、不使用平台名额**。无 prior 的 +0.3192pp 是诊断，不能事后替换
正式判据；且其绝对 macro 仍低于现役 prior 解码。结果说明补充空间覆盖有小幅本地收益，
但不足以在既有 pipeline 中晋级，不能据此声称平台会涨或所有多 crop 方案均无效。

实施：新增独立推理协议入口和固定配置，复用同 checkpoint / prior 组件；原生 CLI 闸门、
模型权重、训练划分、旧提交包均未修改。测试图未用于该验证实验。平台最佳仍为既有
**66.94797564362783%**，70 分目标未达。

## 验证与重放

- 首 32 张原生 center/Flip logits 对旧缓存最大绝对差均 0、top1 全一致。
- 四个新增视图覆盖全部 14,880 张；15 个分块的路径、血缘和文件 SHA 均复核。
- 独立 NumPy float64 重算六路 softmax→平均→log→冻结 bias，14,880/14,880 最终预测一致，
  macro/micro 复现到 1e-12；最大 logit 差与完整耗时见结果 JSON。冻结 checkpoint 的运行后
  SHA 与运行前相同。
- 裁剪几何、方图等价性与非法融合输入测试 **6 passed**。
- 现役保底包 **9/9** 提交校验通过，37,444 行。

```bash
cd /home/lux1/noise-worktrees/l05_threecrop
python3 -m pytest tests/test_l05_threecrop.py -q
python3 scripts/run_l05_threecrop.py --config configs/l05_threecrop/fixed.json
python3 scripts/audit_l05_threecrop.py --config configs/l05_threecrop/fixed.json
python3 scripts/check_submission.py --test_dir /home/lux1/noise/test --class-mapping /home/lux1/noise/artifacts/stages/repechage/20260921/class_to_idx.json --csv /home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/pred_results.csv --zip /home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/submission.zip
```

完整输出在 `/home/lux1/noise-worktrees/l05_threecrop/outputs/codex/l05_threecrop/`。
重跑使用新的独立 output 目录，完成结果拒绝覆盖。实现提交 `9d8110c` 已在执行前推送。
机器可读指标、缓存 SHA、脚本 SHA、独立审计与保底交付信息：
[results/l05_threecrop_20260926.json](../results/l05_threecrop_20260926.json)。

本轮保底 CSV/ZIP（既有赢家，非新生成）在
`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/`。
CSV SHA `51e0efe7178528d23993a44351069d2829b0cd3669c195a77d8d879e66776a75`；
ZIP SHA `e788f07636b80abb61685335cd8df108803863ea36ffbde8b353f8e8317fcd7f`。

改动文件：`scripts/run_l05_threecrop.py`、`scripts/audit_l05_threecrop.py`、`configs/l05_threecrop/fixed.json`、
`tests/test_l05_threecrop.py`、本预注册、结果 JSON、README 状态段、
`docs/current_execution_plan.md`、`.gitignore`。
