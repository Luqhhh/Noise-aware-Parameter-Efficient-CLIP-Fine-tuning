# R1：F1+M1 局部 Part-Token 残差适配

日期：2026-07-22

状态：**协议、实现与 CPU 单元测试完成；尚未生成 GPU cache，尚未训练，尚无平台分数。**

## Material Passport

- 实验编号：`R1_F1_M1_PART_TOKEN_RESIDUAL`
- 研究问题：在逐位保留平台最佳 F1+M1 基线的前提下，局部 crop 内的 patch-token 细节能否纠正局部 CLS 丢失的细粒度证据？
- 数据边界：比赛官方 train；固定 validation 只做选择；test 仅在门控通过后推理一次。
- 父模型：单一 OpenAI CLIP ViT-B/32 F1 checkpoint。
- 当前验证级别：实现已测试；科学假设尚未运行验证。

## 1. 为什么不再继续堆损失或类别规则

当前平台最佳 `F1+M1=63.3276`，相对 F1 Flip 的 `61.1007` 提高 `+2.2269pp`；A2 上 M1 也相对 Flip 提高 `+1.4619pp`。这是迄今唯一在两个不同 checkpoint 上复现的强平台信号。相反，clean-core 错误分散在大量类别与混淆对，现有 center/M1/M3 的 oracle 上限也不足以解释与榜首之间的巨大差距，继续做类别表、先验校正或更多视图组合不会解决表示瓶颈。

O2L 提供了更直接的局部学习证据：纯局部 clean-core 相对严格控制提高 `+1.8929pp`，但共享 LoRA/分类头令 center 下降 `-0.6355pp`。因此新实验必须同时满足：

1. 只学习局部证据；
2. 全局 F1 计算图不可被局部梯度修改；
3. epoch 0 严格等于已经平台验证的 F1+M1；
4. 新增表示必须来自同一 OpenAI CLIP 的同一次局部前向。

[microCLIP](https://arxiv.org/abs/2510.02270) 及其[官方实现](https://github.com/sathiiii/microCLIP)表明，CLIP 的全局 CLS 表示可能遗漏细粒度局部信息，coarse-fine token fusion 是有潜力的结构方向；[CVPR 2024 的 part-level noisy-label 工作](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Estimating_Noisy_Class_Posterior_with_Part-level_Labels_for_Noisy_Label_CVPR_2024_paper.html)也说明局部/部件证据可帮助噪声标签学习。R1 **不复制**其中不合规的类别名称、文本描述、外部模型、外部权重或额外数据，只迁移“同一视觉主干中保留 coarse CLS、补充 fine patch”的结构原则。

## 2. 唯一预注册模型

父 checkpoint 固定为：

```text
/home/x28639/projects/AegisCLIP-Noise-Robust/outputs/F1_VISUAL_LORA_CLEAN_CORE/seed42/checkpoints/best.pt
SHA-256: 7da95e427b959e85cbbf37c99d47d9909b941032e836fc219aaea8e690d72cc4
```

对原图 `x`：

1. `z_g=C(F1(x))`，保持平台 F1 原生全局路径；
2. 用最后一个视觉 block 的 12-head 平均 CLS→patch attention，固定 top-5 加权中心，裁剪 `160×160` 并放大到 `224×224`，得到 M1 局部图 `x_l`；
3. 对 `x_l` 只执行一次原生 F1 前向；forward hook 只读捕获同一次 transformer 输出，不改变 CLS 的运算顺序；
4. 得到原局部 CLS 特征 `f_l∈R^512`，以及经过同一 `ln_post` 和 `visual.proj` 的 49 个 patch 特征 `P_l∈R^(49×512)`；
5. 用 `f_l` 与每个 patch 的 cosine similarity 稳定排序，固定取 top-8；相似度除以温度 `0.07` 后 softmax 加权，得到归一化 part 摘要 `f_p`；相似度相同时按原 patch 顺序稳定裁决；
6. 只学习差值 `d=f_p-f_l` 上的残差：

   `f'_l=f_l+0.25·U(Dropout(GELU(D(LN(d)))))`

7. `D:512→32`，`U:32→512`，dropout `0.1`；`U` 的 weight/bias 全零初始化；
8. 使用 F1 唯一且冻结的线性头权重做锚定残差：`z'_l=z_l+W(f'_l-f_l)`；其中 `z_l` 是同一次原生/AMP 局部前向的 logits，线性头 bias 在差值中抵消；该式在实数运算上等价于 `C(f'_l)`，同时保证零残差时不因 CPU/GPU matmul 精度重新计算而偏离缓存基线；
9. 最终输出固定为 `0.5·softmax(z_g)+0.5·softmax(z'_l)`。

R1 只有一个视觉 backbone、一个线性头和一个约 34k 参数的局部残差器。零初始化时 `f'_l` 与 `f_l` 逐位相等，因此 epoch 0 必须与 F1+M1 完全一致；part-token 计算本身无法改变基线 logits。

## 3. 固定数据、缓存和优化

| 项目 | 唯一固定值 |
|---|---|
| 训练选择 | F1 train 中 cross-fitted `clean_probability>=0.70` 的 `65,473` 张 |
| 类别覆盖 | 500 类；每类 26–185 张 |
| validation | 固定 `10,316` 张；与训练清单交集 0 |
| 图像 batch | train/validation cache 均固定 `128`，匹配已评分 F1+M1 数值布局 |
| M1 定位 | last block / mean-12-head / top-5 / crop 160 |
| part pool | `cls_cosine_topk_v1` / top-8 / temperature 0.07 |
| Adapter | `512→32→512` / scale 0.25 / dropout 0.1 / up 零初始化 |
| 损失 | fused GCE `q=0.5` + `0.25×` local GCE + `2.0×` feature L2 anchor |
| 优化 | AdamW，LR `5e-4`，weight decay `1e-4` |
| 训练 batch | 1024，CPU |
| 轮数 | 最多 20，cosine decay，patience 5 |
| 参数扫描 | 禁止；没有第二组 top-k、温度、bottleneck、LR、loss 或融合权重 |

train 与 validation cache 必须拥有相同的父 checkpoint SHA-256、part-pool spec 和 batch size；训练/验证路径不得重叠。缓存只保留每张图的 global logits、local CLS、local logits、pooled part feature 与 trust 元数据，不保留测试信息。

## 4. 训练前与 epoch-0 复现门

任何参数更新前必须同时满足：

1. validation global cache 对正式 F1 center 参考：最大 logit 差 `0`、argmax 一致率 `100%`；
2. validation F1+M1 重算对正式参考：最大 logit 差 `<=4e-6`、argmax 一致率 `100%`；
3. 新 Adapter epoch 0 对其缓存基线：最大 logit 差 `0`、argmax 一致率 `100%`；
4. 任一不满足立即关闭 R1，不改变 batch、AMP、top-k、温度或容差重跑。

## 5. 选择规则与晋级门槛

epoch 只有同时满足下列安全条件才可参与 clean-core micro 选择：

- trusted macro 不低于 epoch 0；
- raw micro 不低于 epoch 0 超过 `0.10pp`；
- 局部特征平均 cosine drift `<=1%`；
- 不产生空预测类；所有损失和指标有限。

最终只有同时满足下列条件才允许读取 test 并生成提交：

- clean-core micro 相对 F1+M1 `>=+0.20pp`；
- trusted macro `>=0pp`；
- raw micro `>=-0.10pp`；
- drift `<=1%`；
- 三项复现门全部通过；
- 记录 clean-core corrected/harmed 数量与比值，避免只报告净变化。

失败时保存 `history.json`、`gate.json` 和零初始化复合 checkpoint，但不得生成测试包，也不得事后扫描参数。

## 6. 固定执行命令

以下命令只允许在独立工作树运行；启动 GPU cache 前先确认团队无人占卡。

### 6.1 生成 train cache（GPU，尚未运行）

```bash
cd /home/x28639/projects/AegisCLIP-F6-A2LoRA
PYTHONPATH=$PWD /home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
  -m aegis_clip.cli.cache_part_token_adapter_features \
  --checkpoint /home/x28639/projects/AegisCLIP-Noise-Robust/outputs/F1_VISUAL_LORA_CLEAN_CORE/seed42/checkpoints/best.pt \
  --split-csv artifacts/o3_f1_local_only_adapter/seed42/train_clean070.csv \
  --output outputs/R1_F1_M1_PART_TOKEN_RESIDUAL/seed42/cache/train_bs128.pt \
  --batch-size 128 --num-workers 4 --crop-size 160 --top-patches 5 \
  --part-top-patches 8 --part-temperature 0.07
```

### 6.2 生成 validation cache（GPU，尚未运行）

```bash
PYTHONPATH=$PWD /home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
  -m aegis_clip.cli.cache_part_token_adapter_features \
  --checkpoint /home/x28639/projects/AegisCLIP-Noise-Robust/outputs/F1_VISUAL_LORA_CLEAN_CORE/seed42/checkpoints/best.pt \
  --split-csv /home/x28639/projects/AegisCLIP-Noise-Robust/artifacts/stages/preliminary/seed42/val.csv \
  --output outputs/R1_F1_M1_PART_TOKEN_RESIDUAL/seed42/cache/validation_bs128.pt \
  --batch-size 128 --num-workers 4 --crop-size 160 --top-patches 5 \
  --part-top-patches 8 --part-temperature 0.07
```

### 6.3 CPU 训练与门控（缓存通过后）

```bash
PYTHONPATH=$PWD /home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
  -m aegis_clip.cli.train_part_token_adapter \
  --parent-checkpoint /home/x28639/projects/AegisCLIP-Noise-Robust/outputs/F1_VISUAL_LORA_CLEAN_CORE/seed42/checkpoints/best.pt \
  --train-cache outputs/R1_F1_M1_PART_TOKEN_RESIDUAL/seed42/cache/train_bs128.pt \
  --validation-cache outputs/R1_F1_M1_PART_TOKEN_RESIDUAL/seed42/cache/validation_bs128.pt \
  --output-dir outputs/R1_F1_M1_PART_TOKEN_RESIDUAL/seed42/checkpoints \
  --center-reference outputs/M1_F1_TRANSFER/seed42/f1_center.pt \
  --m1-reference outputs/M1_F1_TRANSFER/seed42/f1_attention_local_global.pt \
  --expected-train-samples 65473 --expected-cache-batch-size 128 --device cpu
```

权威判定文件为：

```text
outputs/R1_F1_M1_PART_TOKEN_RESIDUAL/seed42/checkpoints/gate.json
```

只有 `passed=true` 才能执行 `aegis_clip.cli.infer --tta attention_part_token_adapter_global --batch-size 128 --acknowledge-tta-risk`。

## 7. 合规判断

- 主干仍是官方 OpenAI CLIP ViT-B/32，父 F1 权重固定；
- 不使用类别名称、文本塔、GPT 描述、MetaCLIP、外部教师或外部图像；
- patch token、CLS 与 attention 全部来自同一个 checkpoint 和同一输入；
- 最终只有一个复合 checkpoint、一套共享分类头和一条确定性推理流水线；
- 不做多模型 ensemble、投票、test-time training 或测试集先验估计；
- M1 属于同模型固定多视图推理，仍保留赛事 TTA 文字解释风险，提交前需队长/组委会确认。

## 8. 当前实现验证

- 新增模块、缓存器、CPU 训练器、推理模式与 fail-closed 校验；
- 针对性 CPU 测试：`14 passed`；R1 合并时独立 Aegis 完整回归：`188 passed`、团队整合快照：`189 passed`；纳入后续 T1/U0 增量后的最新团队完整回归：`201 passed`；
- 已验证零初始化逐位回退、稳定 tie-break、cache 非有限值拒绝、复合 checkpoint 生成和 native forward hook 不改变 logits；
- 真实 F1 checkpoint 的 CPU 冒烟审计：独立两次原生前向的 logits/CLS 均逐位一致，patch shape 为 `(2,49,512)`，归一化范数最大误差 `1.7881e-7`；
- 真实 F1 的完整 M1 对 R1 epoch-0 端到端审计：500 类融合 logits 逐位一致，最大绝对差 `0`、预测一致率 `100%`、局部适配前后特征逐位一致；
- 两张官方训练图的 CPU 端到端 cache 冒烟测试通过：`global_logits=(2,500)`、`local_features=(2,512)`、`local_logits=(2,500)`、`part_features=(2,512)`，part 特征范数最大误差 `0`；
- 尚未使用真实 F1 图像运行 cache/训练，因此不得把 R1 描述成有效候选或平台提升。
