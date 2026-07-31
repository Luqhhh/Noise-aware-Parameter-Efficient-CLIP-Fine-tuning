# J1：A2 保留集上的可信梯度投影视觉适配

日期：2026-07-20

## 假设

A2 的平台优势来自谨慎删除而非大规模改标，但它完全冻结 ViT-B/32，可能没有学到自然动植物细粒度所需的局部特征。F6 证明在严格门控子集上加入 LoRA 有正向趋势，但样本仅约 9k。J1 使用 A2 实际保留的 90,204 张图学习视觉表征，同时通过 trust 连续权重降低困难样本影响，并用可信锚点梯度投影阻止主梯度破坏高可信方向。该方向对应赛事官方参考的 TrustCLIP 思路，但不声称复现其不可获得的完整实现。

## 严格配对

- J0：A2 epoch48 + 视觉 LoRA + trust 加权，无梯度投影；
- J1：与 J0 完全相同，仅每 8 step 对 `clean_probability >= 0.999` 的锚点启用冲突梯度投影；
- 训练集严格重建自 A2 purification manifest，只保留 `sample_weight=1` 的 90,204 张，不重新纳入 991 张删除样本，不做标签修正；
- 验证使用 A2 原始 `d3_strict` 10,322 张未见验证集，路径重叠必须为 0；
- 两组均固定 2 epoch、seed42、weak RRC + flip、GCE `q=0.5`、batch64、head LR `5e-5`、LoRA LR `2e-5`、4 个末层 rank8 Q/V/out LoRA、feature distillation weight 2.0。

## 预注册门槛

1. J1 clean-core micro 相对 J0 至少 `+0.30pp`；
2. J1 clean-core micro 相对 A2 初始点至少 `+0.50pp`；
3. J1 trusted macro 不低于 A2 初始点，raw micro 不得下降超过 `0.20pp`；
4. J1 必须实际触发梯度投影；若没有负冲突梯度，则机制门槛失败；
5. 任一条件失败即停止，不追加 epoch、不扫描投影间隔或锚点阈值、不生成平台包。

## 合规

仅使用官方当前阶段训练数据；测试集不参与训练、选择或自监督。最终仍为单一 OpenAI CLIP ViT-B/32、单个线性头及一组 LoRA 权重，单路推理，不含模型集成。

## 冻结输入审计

- A2 train source：91,195；保留：90,204；排除：991；500 类均保留，单类 36--192 张；
- 未见验证集：10,322；canonical path overlap：0；
- `a2_kept_train.csv` SHA-256：`e3d9e4c5cb0d2ce3db9b083b0ea4cef1c5d63c1fed594e753fb157030086c3b2`；
- A2 purification manifest SHA-256：`ed67ad12b631bf9f3e4156a6739979f434430a0de3d5461193e34da1c2608787`；
- A2 validation CSV SHA-256：`607e019165912bb0639efb456b7e8dea122b3e8579a2344dedb8109798921eae`。

启动审计发现 A2 清洗 CSV 的路径前缀为 `train_dedup/`，因此在线图像根目录必须是团队仓库根而不是其 `train/` 子目录。首轮 J0 在首个训练 batch 加载前即失败，模型没有参数更新；两组共同修正路径根后按原协议重启，全部数据、优化和门槛不变。

## 严格门控结果

J0 与 J1 均在 epoch2 达到最佳，结果完全相同：raw micro `69.7442%`（相对 A2 `+0.2906pp`）、trusted macro `80.7065%`（`+0.3334pp`）、clean-core micro `82.8389%`（`+0.2634pp`），flip prediction agreement 从 `87.3668%` 增至 `88.0740%`。J1 两个 epoch 的 `projection_count` 均为 0，J0/J1 最终模型张量逐元素完全相同。

因此 J1 相对 J0 为 `+0.0000pp`，相对 A2 clean-core 为 `+0.2634pp`，且机制实际触发门槛失败；未达到 `+0.30pp` / `+0.50pp` 主门槛。按协议不追加 epoch、不扫描投影间隔或锚点阈值、不生成平台包。实验同时保留一个正向诊断：广覆盖 A2 保留集上的 trust 加权视觉 LoRA 在全部主要本地指标上同步改善，但当前证据不足以晋级平台。

- J0 `best.pt` SHA-256：`3d271d4906a3ae47377207f9dc95c0908e54f8d1e923a81e6a32ff6cbd3c1382`
- J1 `best.pt` SHA-256：`9f270d52649405f42f29b06d792b0cde60ddbe5bf6901c0ec6676e70a73af6ff`
