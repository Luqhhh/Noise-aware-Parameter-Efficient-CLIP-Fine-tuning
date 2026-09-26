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
