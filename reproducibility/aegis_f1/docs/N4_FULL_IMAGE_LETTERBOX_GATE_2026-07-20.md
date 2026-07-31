# N4：保持全图的 Letterbox 输入门控

日期：2026-07-20

## 假设与新证据

N4 不再重复 F4 的整图高分辨率插值。F4 已证明：把中心裁剪后的整图从 224px
直接放大到 288px 会使 clean-core、trusted macro 和 raw macro 同时下降。

本轮检验的是另一种几何问题。OpenAI CLIP 的标准推理先把短边缩放到 224，随后截取
224×224 中心区域。严格验证集 10,322 张图像的原始长宽比统计显示：80.42% 的图像
长宽比大于 1.25，21.16% 大于 1.50；标准中心裁剪平均只保留原缩放图像 73.83%
的面积，最差 10% 只保留不超过 64.07%。这可能在训练分布与干净平台测试分布之间
造成不可见的主体截断。

N4 将较长边缩放到 224px，保持纵横比，并用 CLIP 像素均值填充到 224×224。
填充区域归一化后接近零，不增加视觉内容，也不修改模型权重。

进一步的训练前因果检查按原始长宽比分箱。N3 center 在 clean-core 上的准确率分别为：
`<=1.25: 83.22%`、`1.25--1.50: 82.86%`、`1.50--2.00: 83.92%`、
`>2.00: 80.72%`。只有最后一组明显偏低，但仅 83 张。这限制了 N4 的潜在总体收益，
因此先运行一次廉价的裸 center 分支预筛，不直接执行三分支 M3 全量评估。

## 冻结协议

- checkpoint：N3 epoch 2 `best.pt`；
- 模型：同一官方 OpenAI CLIP ViT-B/32、同一 AdaptFormer 和同一线性头；
- Phase 0 候选：`clip_letterbox + center`；对照为既有 N3 `clip_center_crop + center`；
- Phase 1 候选：`clip_letterbox + complementary_flip_local_global`；对照为既有 N3
  `clip_center_crop + complementary_flip_local_global` 缓存；
- crop size、top patches、三个分支融合公式和 batch size 全部保持不变；
- 不读取测试图像，不使用平台分数选择几何模式，不扫描填充值、缩放尺度或融合权重。

## 预注册门槛

Phase 0 只有在 clean-core micro 至少 `+0.20pp`、trusted macro 不下降且 raw micro
不下降超过 `0.10pp` 时，才允许执行 Phase 1。Phase 1 候选只有同时满足以下条件
才可进入测试推理：

1. clean-core micro 相对 N3+M3 至少提高 0.50pp；
2. trusted macro 不下降；
3. raw micro 不下降超过 0.10pp；
4. clean-core 的 corrected 必须至少是 harmed 的 1.5 倍；
5. 结果缓存必须覆盖 10,322 张严格验证图像，路径、标签、信任证据和 checkpoint
   SHA-256 与对照完全一致。

若失败，N4 立即关闭，不扫描 warp resize、padding 颜色、尺度或融合系数。中心分支、
flip 分支与 local 分支的单独结果只用于机制解释，不用于改选提交。

## 合规边界

Letterbox 是确定性的单模型输入预处理，不使用外部数据、不训练测试集、不改变骨干，
最终仍为一个 CLIP ViT-B/32 检查点的同模型多视图推理。由于 M3 已属于团队当前接受的
同模型 TTA 口径，N4 不扩大模型融合范围；若进入提交，清单必须显式记录
`input_resize_mode=clip_letterbox`。

## 执行结果：Phase 0 失败并关闭

同一 N3 epoch-2 checkpoint、同一 10,322 张严格验证集，仅改变输入几何后：

| 输入 | clean-core micro | trusted macro | raw micro |
|---|---:|---:|---:|
| 标准 center crop | 83.1162% | 80.8799% | 69.8508% |
| full-image letterbox | 74.3831% | 72.2287% | 61.6644% |
| 差值 | **-8.7330pp** | **-8.6512pp** | **-8.1864pp** |

letterbox 改变 2,775/10,322 个预测，并产生 1 个空预测类别。三个隔离指标方向一致且远低于 Phase 0 门槛，说明在 ViT-B/32 的 7×7 patch 网格上缩小主体的伤害远大于保留边缘内容的收益。N4 状态固定为 `CLOSED`：不执行 Phase 1、不扫描填充/缩放变体、不生成测试提交。机器结果见 `outputs/N4_FULL_IMAGE_LETTERBOX_GATE/seed42/phase0_comparison.json`。
