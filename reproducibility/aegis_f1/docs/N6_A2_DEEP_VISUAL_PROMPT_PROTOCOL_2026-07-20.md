# N6：A2 深层视觉 Prompt 门控

日期：2026-07-20

## 方向选择

N6 检验项目此前未覆盖的标准 Deep VPT。它在每个 ViT block 的 CLS token 后插入
5 个可学习视觉 token；该层输出后丢弃旧 prompt，下一层换入自己的 prompt，图像 patch
与 CLS 则连续传递。12 层总计只增加 46,080 个视觉参数，CLIP 原权重保持冻结。

该方案不等同于文本 Prompt。ICCV 2023 的噪声鲁棒文本 Prompt 依赖固定语义类别名，
而本赛题类别只有 `0000--0499`，关键假设不成立，因此不采用。Deep VPT 不需要类别名；
其原论文在 CUB、NABirds、Flowers、Dogs、Cars 等细粒度数据上专门验证。CVPR 2025
Prompt-CAM 进一步说明视觉 prompt 可以学习面向类别差异的局部 trait，但其一次加入全部
类别 token 的形式在 500 类、8GB GPU 上代价过高，N6 只采用标准 5-token Deep VPT。

主要依据：

- Visual Prompt Tuning（ECCV 2022）：https://arxiv.org/abs/2203.12119
- 官方实现：https://github.com/KMnP/vpt
- Prompt-CAM（CVPR 2025）：https://openaccess.thecvf.com/content/CVPR2025/html/Chowdhury_Prompt-CAM_Making_Vision_Transformers_Interpretable_for_Fine-Grained_Analysis_CVPR_2025_paper.html
- PTNL 噪声机制边界（ICCV 2023）：https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Why_Is_Prompt_Tuning_for_Vision-Language_Models_Robust_to_Noisy_ICCV_2023_paper.html

## 冻结配置

- 父模型：平台 61.2128% 的 A2 epoch 48；
- 12 层各 5 个独立 prompt token，prepend after CLS，层间替换，dropout 0；
- 初始化：VPT 的 patch-aware Xavier uniform，范围
  `sqrt(6 / (3*32*32 + 768))`；
- 分类头从 A2 精确加载；CLIP 所有原生参数冻结；
- A2-kept 严格训练集、N3 相同 trust、GCE 和特征蒸馏；
- AdamW，prompt LR 1e-3，head LR 5e-5，4 轮、1 轮 warmup；
- 不扫描 token 数、层数、初始化、学习率或 dropout。

## 工程门与效果门

训练前必须通过真实 batch 64 反向审计：只有 46,080 个 prompt 参数和 256,500 个
分类头参数可训练；所有 CLIP 原生参数不得出现梯度；全部 prompt 张量必须获得有限非零
梯度。随机 prompt 会改变初始 logits，因此不要求像 N3 一样 epoch 0 严格恒等，但必须
记录初始退化并在训练后完整恢复。

进入平台候选必须同时满足：

1. 相对 A2 center，clean-core micro 至少 `+1.00pp`；
2. trusted macro 至少 `+0.50pp`；
3. raw micro 至少 `+0.25pp`；
4. flip agreement 不低于 A2 的 87.37%；
5. 平均 feature drift 不超过 1%；
6. 最佳轮次必须比随机 prompt 初始点高至少 1.50pp clean-core。

失败即关闭 N6，不追加轮次、不改 prompt 数或学习率。通过时只允许同一 N6 检查点的
裸推理与固定水平翻转；attention-local 路径尚未实现 prompt token 的注意力审计，不得沿用
M3。

## 合规性

N6 仍是官方 OpenAI CLIP ViT-B/32，只有视觉输入 token 和单个线性头可训练，不使用
外部数据、类别名称、文本塔、测试时训练或模型集成，属于可审计的单模型 PEFT。
