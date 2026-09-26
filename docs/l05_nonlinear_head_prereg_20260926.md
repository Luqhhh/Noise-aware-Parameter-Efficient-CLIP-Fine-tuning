# L05 冻结视觉骨干的非线性适配（预注册）

分支 `codex/l05_nonlinear_head`；独立 worktree `/home/lux1/noise-worktrees/l05_nonlinear_head`。
2026-09-26 已 fetch 并核对所有本地/远端分支。上一段分辨率诊断已关闭；NEW01/02/03
最多 +0.0615pp 未晋级。F05 的 D1/O3 使用局部特征；HEAD_L2SP 是线性头权重锚定；
本轮研究 L05 全局特征上的非线性表达能力，不重复这些方案。已有空白同名 worktree 继续使用。

## 固定设计

配置 `configs/l05_nonlinear_head/fixed.json`。冻结已发布 L05 视觉骨干，从同一训练划分
133,815 张的确定性 center-crop 特征训练：NH00 线性头重拟合；NH01 在单分类头前接
已有 `ResidualFeatureAdapter`（512→128→512，GELU，残差 scale=0.1，up 零初始化）。
两者均从 L05 classifier 开始，初始函数一致；AdamW lr=1e-4、WD=1e-4、GCE q=0.5，
batch=1024、20 epoch cosine、seed42。两臂训练顺序完全配对，不增加增强、LR/WD/维度扫描。
视觉骨干不更新；不声称这些训练特征是 OOF。

缓存先验证原生验证 logits 与已冻结 L05 缓存的 top1 一致率和最大误差，32 张每路
top1 全一致且 max abs ≤0.02 方可训练。逐图核验官方 manifest 文件 SHA，训练与验证路径
及 content_group 不重叠，缓存绑定 parent/split/mapping/hash。分块原子写入，重启复用已核验块。

每 epoch 测 center raw macro，以 macro（平手 micro）选单一 checkpoint，epoch0 也可胜出。
达到 −2pp macro 或 micro 的臂立即止损。最终按原 L05 固定 flip、mean_probabilities、T=1.4、
val 拟合均匀 prior、strength=0.60 评估（明确这是拟合集内读数）。
NH01 须同时超过 L05 与 NH00 的固定解码 macro ≥0.30pp，且 micro 不下降才晋级；
NH00 若单独过 L05 +0.30pp 且 micro 不下降，可作为线性续训候选，不归因于非线性。
未过门关闭，不派生参数扫描。晋级后单独审计重载、独立分组校准读数再决定提交建议。
测试集不参与选模/拟合/阈值选择；仅冻结赢家后才生成新候选包。

## 规则合规性

- Backbone: CLIP ViT-B/32
- Pretrained weights: OpenAI official
- External data: No
- Test data used for training/adaptation: No
- Cross-stage data/checkpoint reuse: No
- Multi-model ensemble: No（适配器是单网络内串行模块、一个分类头；比较两臂不融合）
- Manual cleaning required: No
- Fully reproducible: Yes
- Rule risk: None；同 checkpoint Flip 沿用已批准协议

若两臂关闭，交付已冻结 L05_T14_P060 保底包的重新校验记录，不将旧平台成绩归因于本轮。
运行：`python3 scripts/run_l05_nonlinear_head.py --config configs/l05_nonlinear_head/fixed.json --phase cache`
以及同入口 `--phase train`。完整命令、结果与包校验将在运行结束回填。

## 训练前审计

原生验证前 32 张、两路 logits 与冻结缓存最大绝对差均 0。全 14,880 张将 head
从 AMP 改成 FP32，center/flip 的 max abs 分别 0.008713/0.008394，top1 分别改变
8/6 张；编码器仍 AMP。两臂同用 FP32 head；最终比较保留原生 L05 为基线，报告
epoch0 的差异，不把精度差异归因为学习收益。模型/checkpoint 定向测试 23 passed。
预注册与实现已推送 `d0edf94`，运行输出独立保存。

## 实测结果（两臂均训练满 20 轮）

| 模型 | 选中 epoch | center macro / micro | 固定解码 macro / micro | 解码相对 L05 macro / micro |
|---|---:|---|---|---|
| L05 原生参照 | 16 | 75.2497% / 76.2970% | 75.8265% / 76.6532% | — |
| NH00 线性头重拟合 | 5 | 75.3545% / 76.3978% | 75.8552% / 76.6801% | +0.0287pp / +0.0269pp |
| NH01 非线性适配 | 8 | 75.4042% / 76.3777% | 75.8251% / 76.4919% | −0.0014pp / −0.1613pp |

两臂均未触发 −2pp 止损，但均未过 +0.30pp 晋级门。NH01 相对 NH00 的解码 macro
也下降 0.0301pp；纠正/破坏为 188/212，线性对照为 12/8。本固定非线性配方关闭，
不派生参数扫描，不生成新测试候选、不占平台名额，不替换现役 L05。结论只涉及该固定
缓存特征与训练配方，不能推广为所有非线性结构无效。

两臂 epoch0 的 center macro/micro 均为 75.2506% / 76.2970%，初始函数一致。
FP32 head 的初始 macro 与原生 L05 只差 +0.000883pp；该数值差异不足以改变不晋级结论。
解码 prior 只在验证集拟合并在同一验证集读数，因此不是独立无偏的全流程评估；
本轮未过预筛门，不再做晋级后的 conditional cross-fit，也不声称平台收益。

实现复用 Aegis 已有 ResidualFeatureAdapter，未修改骨干或公共训练器。训练 checkpoint
内包含完整视觉权重、单头及适配器（NH01）、固定 prior、refit 配置和数据血缘；
不依赖父模型参与推理。父模型的生成代码保留在 `focus/f05-four-lines@f050ecb` 的
`scripts/run_l05_local_retrain.py`，其训练配置同时保存在冻结 parent checkpoint 中。

### 可重放命令

```bash
cd /home/lux1/noise-worktrees/l05_nonlinear_head
python3 scripts/run_l05_nonlinear_head.py --config configs/l05_nonlinear_head/fixed.json --phase cache
python3 scripts/run_l05_nonlinear_head.py --config configs/l05_nonlinear_head/fixed.json --phase train
python3 scripts/run_l05_nonlinear_head.py --config configs/l05_nonlinear_head/fixed.json --phase audit
PYTHONPATH=reproducibility/aegis_f1 python3 -m pytest reproducibility/aegis_f1/tests/test_model.py reproducibility/aegis_f1/tests/test_checkpoint.py -q
python3 scripts/check_submission.py --test_dir /home/lux1/noise/test --class-mapping /home/lux1/noise/artifacts/stages/repechage/20260921/class_to_idx.json --csv /home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/pred_results.csv --zip /home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/submission.zip
```

重放需新独立输出目录（配置修改 `output`；缓存 identity 因而改变），已有结果拒绝覆盖。
原始训练/验证图、父模型只读。缓存 37 块覆盖 133,815 train / 14,880 val，均通过逐图文件 SHA；
缓存耗时 1250.9 秒，硬件为本机 RTX 4070 Laptop 8GB。日志和两臂 checkpoint 位于
`/home/lux1/noise-worktrees/l05_nonlinear_head/outputs/codex/l05_nonlinear_head/`。
机器可读结果、逐轮曲线、全部块 SHA 与执行源码 SHA 见
[结果 JSON](../results/l05_nonlinear_head_20260926.json)。

### 可提交保底产物

本轮复核现役 `L05_T14_P060` 的既有包（不是本轮新模型），37,444 行，9/9 提交检查通过。
路径：`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/{pred_results.csv,submission.zip}`。
CSV SHA-256：`51e0efe7178528d23993a44351069d2829b0cd3669c195a77d8d879e66776a75`；
ZIP SHA-256：`e788f07636b80abb61685335cd8df108803863ea36ffbde8b353f8e8317fcd7f`。
既有平台 66.94797564362783% 来源于 focus 分支的
`results/f05_focus_l05_tta_prior_platform_20260925.json`，不归因于本轮。
70 分以上目标仍未达到。

### 改动文件

- `scripts/run_l05_nonlinear_head.py`：分块缓存、配对训练、完整 checkpoint 和原图重载审计。
- `configs/l05_nonlinear_head/fixed.json`：唯一固定配方和本机数据/父权重路径。
- `docs/l05_nonlinear_head_prereg_20260926.md`：事前判据、结果与重放命令。
- `results/l05_nonlinear_head_20260926.json`：实测结果、逐轮曲线、血缘、校验和保底交付。
- `README.md`、`docs/current_execution_plan.md`：检查点与关闭结论。
- `.gitignore`：本方案独立输出目录排除。


### 最终审计

NH00 与 NH01 的完整 checkpoint 重载后，视觉权重与冻结 parent **逐位一致**；
各自重新读取 14,880 张验证图、运行原图及翻转两路，**29,760/29,760 top1 均与缓存一致**。
原图重载最大 logit 差分别 `2.0980835e-05` / `3.2424927e-05`，完整解码指标复现到 1e-12。
缓存特征经 Aegis 原生 `model(features=...)` 重载同样 top1 全一致。
模型与 checkpoint 定向测试 **23 passed**；保底包 **9/9** 提交校验通过。
本段已具备验证结果和可提交保底产物；仅关闭本配方，平台 >70 的总目标仍未完成。
