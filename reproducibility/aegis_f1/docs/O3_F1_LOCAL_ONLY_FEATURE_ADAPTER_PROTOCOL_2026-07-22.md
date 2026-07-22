# O3：冻结 F1 全局路径的局部专用特征适配器

日期：2026-07-22

状态：**实现与 CPU 数据审计完成；尚未运行 GPU 缓存与训练**。

## 平台证据与研究问题

F1+M1 在平台达到 `63.3276`，比 F1 Flip 高 `2.2269pp`，说明注意力局部视图是目前最强、且已经跨检查点复现的方向。O2L 又给出更细的因果证据：局部监督使纯局部 clean-core micro 相对严格控制提高 `+1.8929pp`，O2L+M1 的 clean-core micro、trusted macro、raw micro 分别提高 `+0.2704pp`、`+0.2905pp`、`+0.3102pp`；但共享 LoRA/分类头令 center clean-core micro 下降 `-0.6355pp`，因此 O2 按预注册门槛关闭。

O3 检验的问题是：**能否保留 O2 已证明有效的局部可学习性，同时从结构上禁止局部梯度改写 F1 全局分支？**

这一路径也与近期研究观察一致：microCLIP 指出 CLIP 的粗粒度全局特征会限制细粒度分类，并用轻量 coarse-fine token fusion 补充局部线索（<https://arxiv.org/abs/2510.02270>）；CVPR 2026 的训练期异构 image-patch-text graph 工作表明，局部 patch 监督可以只作为训练期辅助知识而不替换最终基础适配器（<https://openaccess.thecvf.com/content/CVPR2026/html/Mohammad_Training-Only_Heterogeneous_Image-Patch-Text_Graph_Supervision_for_Advancing_Few-Shot_Learning_Adapters_CVPR_2026_paper.html>）。O3 只迁移“受限局部适配、保留稳定全局路径”的原则，不使用论文中的文本、外部模型或外部数据。

## 与失败的 N1/O2 的结构差异

- N1 学习自由的 `512→500` 局部 logit 残差，共 `256,500` 个参数，直接加到全局 logits，最终 raw micro 相对固定 M1 下降 `1.0463pp`；
- O2L 让局部与全局共享同一组 LoRA 和分类头，局部能力提高，但全局表示被共同梯度拖动；
- O3 只有约 `34,336` 个局部 bottleneck 参数，使用 F1 原有且冻结的唯一分类头；全局前向完全绕过 O3，因此从计算图上不能发生回退。

## 固定模型与公式

共同父模型为平台已验证的 F1 checkpoint：

- 路径：`/home/x28639/projects/AegisCLIP-Noise-Robust/outputs/F1_VISUAL_LORA_CLEAN_CORE/seed42/checkpoints/best.pt`；
- SHA-256：`7da95e427b959e85cbbf37c99d47d9909b941032e836fc219aaea8e690d72cc4`；
- OpenAI CLIP ViT-B/32，最后 4 个视觉 block 的 Q/V/out rank-8 LoRA；
- F1 视觉参数和唯一线性分类头 `C` 在 O3 全程冻结。

对原图 `x`：

1. `z_g = C(F1(x))`，与平台 F1 整图路径逐位相同；
2. 用 F1 最后视觉 block 的 12-head 平均 CLS→patch attention，固定 top-5 加权中心，裁剪 `160×160` 后放大到 `224×224`，得到 `x_l`；
3. `f_l = F1(x_l)`；
4. `f'_l = f_l + 0.25 * U(Dropout(GELU(D(LN(f_l)))))`；为保证零初始化时逐位相同，不对已经归一化的 F1 特征做第二次归一化；
5. `D:512→32`，`U:32→512`，`U` 的 weight/bias 全零初始化，dropout `0.1`；
6. `z'_l = C(f'_l)`，仍使用同一个冻结分类头；
7. 最终 `p = 0.5*softmax(z_g) + 0.5*softmax(z'_l)`。

因此 O3 epoch 0 必须与现有 F1+M1 完全一致；训练后也只有局部视图发生变化。

## 固定数据与优化

- 训练源：F1 原始 train split 的 `92,902` 张图，不含固定 validation；
- 训练选择：cross-fitted trust `clean_probability >= 0.70`，预审计为 `65,473` 张、覆盖全部 `500` 类、每类 `26–185` 张；
- 固定高可信清单 SHA-256：`a4a47bcc54bdbf1afce6713815d6c39c2d9b34a905f06553b80b4d21f5e6c6bb`；
- validation：固定 `10,316` 张，从不用于 O3 梯度；
- train/validation cache 均由同一 F1 checkpoint、同一 `crop=160/top5` 路径生成，局部特征保存为 float32；
- 损失：`GCE_q=0.5(fused) + 0.25*GCE_q=0.5(local) + 2.0*||f'_l-f_l||²`；验证时同时记录 cosine drift 与特征范数偏移；
- optimizer：AdamW，LR `5e-4`，weight decay `1e-4`；
- batch `1024`，最多 `20` epoch，cosine decay，patience `5`；
- 缓存后的 adapter 训练默认只用 CPU，不占用团队 GPU；
- 不扫描 bottleneck、scale、dropout、LR、损失权重、crop、top-k 或融合比例。

## 预注册选择与门槛

epoch 0 为固定 F1+M1 基线。候选 epoch 只有同时满足下列安全条件才可参与 clean-core micro 选择：

1. trusted macro 不低于 epoch 0；
2. raw micro 不低于 epoch 0 超过 `0.10pp`；
3. 局部特征平均 cosine drift 不超过 `1.0%`；
4. 预测覆盖全部 500 类且所有损失/指标有限。

最终晋级测试推理还必须同时满足：

1. clean-core micro 相对 F1+M1 至少 `+0.20pp`；
2. trusted macro 相对 F1+M1不下降；
3. raw micro 相对 F1+M1不得下降超过 `0.10pp`；
4. F1 global cache 与既有 F1 center cache 最大绝对差 `0`、预测一致率 `100%`；
5. epoch-0 fused cache 与既有 F1+M1 cache 最大绝对差 `0`、预测一致率 `100%`；
6. 任一门槛失败则不读取测试集、不生成提交包、不进行事后参数扫描。

## 合规性

最终模型是一个复合 checkpoint：一套 F1 CLIP ViT-B/32、一个共享线性分类头和一个仅对确定性 attention-local 视图激活的轻量特征 adapter。不存在第二个 backbone、第二个分类头、模型投票、外部数据、类别文本、测试时训练或测试标签使用；测试图像只在门控通过后做一次固定单模型推理。

## 队长/JJT 可执行交接命令

以下命令均在独立 Aegis 工作树运行，不修改团队工程。GPU 缓存前必须先检查团队进程；若团队正在占卡，不启动或改由明确空闲的 GPU 执行。

### 1. CPU：生成固定高可信训练清单

```bash
cd /home/x28639/projects/AegisCLIP-F6-A2LoRA
PYTHONPATH=$PWD /home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
  -m aegis_clip.cli.prepare_high_clean_split \
  --source-csv /home/x28639/projects/AegisCLIP-Noise-Robust/artifacts/stages/preliminary/seed42/train.csv \
  --validation-csv /home/x28639/projects/AegisCLIP-Noise-Robust/artifacts/stages/preliminary/seed42/val.csv \
  --trust-bundle /home/x28639/projects/AegisCLIP-Noise-Robust/artifacts/trust/cvt_v1.pt \
  --output-csv artifacts/o3_f1_local_only_adapter/seed42/train_clean070.csv \
  --threshold 0.70 --expected-selected 65473 --expected-classes 500
```

### 2. GPU：顺序生成 train/validation 缓存

```bash
cd /home/x28639/projects/AegisCLIP-F6-A2LoRA
PYTHONPATH=$PWD /home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
  -m aegis_clip.cli.cache_local_adapter_features \
  --checkpoint /home/x28639/projects/AegisCLIP-Noise-Robust/outputs/F1_VISUAL_LORA_CLEAN_CORE/seed42/checkpoints/best.pt \
  --split-csv artifacts/o3_f1_local_only_adapter/seed42/train_clean070.csv \
  --output outputs/O3_F1_LOCAL_ONLY_FEATURE_ADAPTER/seed42/cache/train.pt \
  --batch-size 64 --num-workers 4 --crop-size 160 --top-patches 5

PYTHONPATH=$PWD /home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
  -m aegis_clip.cli.cache_local_adapter_features \
  --checkpoint /home/x28639/projects/AegisCLIP-Noise-Robust/outputs/F1_VISUAL_LORA_CLEAN_CORE/seed42/checkpoints/best.pt \
  --split-csv /home/x28639/projects/AegisCLIP-Noise-Robust/artifacts/stages/preliminary/seed42/val.csv \
  --output outputs/O3_F1_LOCAL_ONLY_FEATURE_ADAPTER/seed42/cache/validation.pt \
  --batch-size 64 --num-workers 4 --crop-size 160 --top-patches 5
```

### 3. CPU：训练、选择并生成门控文件

```bash
cd /home/x28639/projects/AegisCLIP-F6-A2LoRA
PYTHONPATH=$PWD /home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
  -m aegis_clip.cli.train_local_feature_adapter \
  --parent-checkpoint /home/x28639/projects/AegisCLIP-Noise-Robust/outputs/F1_VISUAL_LORA_CLEAN_CORE/seed42/checkpoints/best.pt \
  --train-cache outputs/O3_F1_LOCAL_ONLY_FEATURE_ADAPTER/seed42/cache/train.pt \
  --validation-cache outputs/O3_F1_LOCAL_ONLY_FEATURE_ADAPTER/seed42/cache/validation.pt \
  --output-dir outputs/O3_F1_LOCAL_ONLY_FEATURE_ADAPTER/seed42/checkpoints \
  --center-reference outputs/M1_F1_TRANSFER/seed42/f1_center.pt \
  --m1-reference outputs/M1_F1_TRANSFER/seed42/f1_attention_local_global.pt \
  --expected-train-samples 65473 --device cpu
```

权威判定文件为 `outputs/O3_F1_LOCAL_ONLY_FEATURE_ADAPTER/seed42/checkpoints/gate.json`；只有其中 `passed=true` 才进入测试推理。

## 结果

CPU 前置审计已通过：source `92,902` 张，选择 `65,473` 张，覆盖 `500` 类，每类 `26–185` 张，与固定 validation 重叠 `0`；trust bundle SHA-256 为 `52e59a991a5eb3c57abdfabee5647423726f51fbdd3da2ce377467664d173608`。GPU cache 与 adapter 训练待队长/JJT 或明确空闲 GPU 执行。
