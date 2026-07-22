# F7：A2 低学习率固定全量重放

日期：2026-07-20

## 假设

A2 的水平翻转 TTA 已获平台 61.2128%，但它只训练了严格开发划分中的
91,195 张图。F7 不改变模型结构和主要噪声鲁棒配方，而是在 A2 epoch 48
强基座上，用接近原余弦调度末端的学习率吸收此前未参与训练、且独立 OOF
判断为高可信的保留样本。目标是减少因开发留出 10% 数据造成的覆盖损失。

## 数据和 MixUp 安全

- 原 A2 strict-train 91,195 张中，991 项三方共识黑名单继续物理删除；
- 原 A2 validation 10,322 张中，仅 `clean_probability >= 0.70` 者加入；
- 新增样本的 trust 来自与标签隔离的 Aegis OOF bundle；
- 黑名单和低可信样本不只设为零权重，而是从训练 CSV 物理移除，防止
  MixUp 把其噪声标签重新混入有效样本；
- 审计 A2 strict-train / validation 的内容 SHA-256 组；发生重叠的 held-out
  图像必须从新增监督中物理排除，并在 manifest 中显式计数。

## 固定训练协议

- 父模型：A2 `NR_CL_KNN_DROP` epoch 48；
- 单一 OpenAI CLIP ViT-B/32，视觉骨干完全冻结，只训练线性头；
- GCE q=0.5；像素 MixUp alpha=0.2、概率=0.2；确定性 CLIP crop；
- batch 128，head LR 5e-5，weight decay 1e-4，固定 4 epochs；
- LR 量级对应 A2 原始 5e-3 余弦调度的 1% 末端，不重新进行大步搜索；
- 历史 A2 validation 已部分加入训练，所有指标仅用于崩塌诊断；
  `selection_policy=last_epoch` 强制选择 epoch 4，禁止重叠验证选模。

## 冻结资产审计

- A2 strict-train：91,195；物理删除 reject 后：90,204；
- A2 validation：10,322；排除 50 个内容冲突后，OOF-clean 新增：7,173；
- 固定 full-fit 总量：97,377，覆盖 500/500 类；
- full-fit CSV SHA-256：`86840b10b2ba8ca2fbbcefee9b8fecf40876b802b6d7b6ca8f74ae1da7e0b5aa`；
- 资产只作为后备低风险对照；结构性高收益方案验证完成前不启动训练。

## 运行前门禁

1. 训练 CSV 无重复路径，500/500 类均有监督；
2. 991 项 A2 reject 与全部低可信 held-out 不出现在训练 CSV；
3. MixUp 触发时走真实像素前向，未触发时允许复用同一官方 CLIP 缓存；
4. epoch 0 logits 与 A2 完全一致；
5. 77 项以上自动测试、配置闭合和真实资产覆盖全部通过；
6. GPU 无团队计算进程才可启动。

## 停止与发布

- 非有限 loss/梯度、类别塌缩、训练准确率异常下坠立即停止；
- 不因重叠 validation 的某个 epoch 更高而改变固定 epoch 4；
- 完成后只有 raw / trusted / clean-core 均无明显崩塌才生成 Bare 与固定
  horizontal-flip TTA；
- 平台首次结果只登记，不据此扫描 epoch 或学习率。

## 反方审查

最强反对意见是：平台收益可能来自 A2 的早停点，而继续训练会让线性头
重新记忆噪声；新增 held-out 虽经过 OOF 筛选，分布仍可能与干净测试不同。
因此 F7 采用末端量级低 LR、物理删除噪声、固定短预算和 last-epoch，且不
引入 LoRA。它仍需要一次平台评测才能判断是否真正优于 A2；本地重叠指标
不能被表述为泛化证据。

## 合规性

只使用官方训练集和官方 OpenAI CLIP ViT-B/32；测试集只做最终推理；
无人工清洗、外部数据、多模型融合或测试时适应。团队仓库与 A2 资产只读。
