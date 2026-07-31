# M3：A2 Flip 与 M1 的互补双分支单模型 TTA

日期：2026-07-20

## 已有独立证据

- A2 的既有平台候选使用 center/水平翻转 logits 1:1 平均，平台得分 `61.2128`；
- M1 使用 center/attention-local 概率 1:1 平均，在 J0/F1/A2 三个检查点上分别获得 `+0.8525 / +0.9597 / +0.6588pp` raw micro 本地增益；
- 两条证据来自不同的局部变换：水平翻转保持全局视野，attention-local 放大模型自定位的细节区域。

## 冻结输入

- A2 exact checkpoint SHA-256：`1e2c1a4a274c5e466b716ded41ccf58bebf167a2fadbeba75693d08bdb4f039c`；
- A2 online center cache SHA-256：`1ab0a42f64d49315aa7202bc099eb76c796f599355dd9388c58c537eb35bacd1`；
- A2 M1 cache SHA-256：`cea5f7651bf0d24828b7da8b252cb47b52288fa2cb7638a1c6d32359c0c92698`；
- validation CSV SHA-256：`607e019165912bb0639efb456b7e8dea122b3e8579a2344dedb8109798921eae`。

## 冻结推理公式

对同一张 224×224 图像只使用同一个模型：

1. `z_c`：原生 center logits；
2. `z_f`：输入沿宽度水平翻转后的 logits；
3. `z_l`：M1 固定 attention top-5 加权中心、160×160 裁剪的 local logits；
4. A2 Flip 分支：`p_flip = softmax((z_c + z_f) / 2)`；
5. M1 分支：`p_m1 = (softmax(z_c) + softmax(z_l)) / 2`；
6. M3：`p_m3 = (p_flip + p_m1) / 2`，最终分数为 `log(p_m3)`。

不使用温度、置信度门控、类别先验或可调权重；不扫描 logits/probability 融合变体。

## 实现审计

1. M3 center logits vs A2 online center：最大绝对差 `0`、预测一致率 `100%`；
2. M3 attention-local logits vs A2 M1 local：最大绝对差 `0`、预测一致率 `100%`；
3. M3 内部 `p_m1` vs A2 M1 最终 logits：最大绝对差 `0`、预测一致率 `100%`；
4. 任一项失败即判实现无效，不读取效果门控。

## 增量效果门槛

M3 必须同时对照 A2 M1 与本次缓存中严格复现的 A2 Flip：

1. clean-core micro 至少比两者中较高者再高 `+0.10pp`；
2. trusted macro 不得比两者中较高者下降超过 `0.05pp`；
3. raw micro 不得比两者中较高者下降超过 `0.10pp`；
4. M3 相对 M1 与 Flip 均至少改变 `1%` 的验证预测；
5. 任一门槛失败即关闭：不扫描权重/温度/融合空间，不运行测试集，不生成提交包。

## 合规性

M3 是同一个 OpenAI CLIP ViT-B/32 检查点的确定性多视图推理，无第二模型、投票、测试集训练或测试先验估计。团队仓库保持只读，开发产物全部位于独立工作树。

## 正式结果

三条实现审计全部通过：

- center path vs A2 online center：最大绝对 logit 差 `0.0`，预测一致率 `100%`；
- attention-local path vs A2 M1 local：最大绝对 logit 差 `0.0`，预测一致率 `100%`；
- nested M1 path vs A2 M1 final：最大绝对 logit 差 `0.0`，预测一致率 `100%`。

| 指标 | A2 Flip | A2 M1 | M3 |
|---|---:|---:|---:|
| clean-core micro | 82.9221% | 83.1716% | **83.4350%** |
| clean-core macro | 82.8096% | 82.9844% | **83.2619%** |
| trusted micro | 80.9862% | 81.1766% | **81.4889%** |
| trusted macro | 80.6949% | 80.8197% | **81.1606%** |
| raw micro | 69.7152% | 70.1124% | **70.4030%** |
| raw macro | 69.7157% | 70.1100% | **70.4000%** |

M3 相对两个基线中逐指标较高的 M1：clean-core micro `+0.2634pp`、trusted macro `+0.3409pp`、raw micro `+0.2906pp`，均通过增量门槛。M3 相对 Flip 的对应增益为 `+0.5129 / +0.4657 / +0.6878pp`。

M3 相对 M1 改变 `783 / 10,322` 个预测（`7.5857%`），相对 Flip 改变 `513 / 10,322` 个预测（`4.9700%`），均满足至少 1% 的非平凡作用门槛。相对原 A2 center，M3 raw micro 总增益为约 `+0.9494pp`。

产物哈希：

- M3 validation cache：`51bb51a550fc7e67cfdcc32d9ff1c205836233c47bdc995144ab9eddef1dcb3a`；
- M3 evaluation JSON：`b9059e5e3076a5b0a84a150881f1f0a31899036dc00ec6a06f2d9bac54915f95`；
- 测试：`129 passed`。

## 门控结论

M3 通过全部实现与增量效果门槛，结论为 **PROMOTE TO TEST INFERENCE**。只批准预注册的双分支固定 1:1 概率融合；不批准权重、温度或置信度扫描。

## 正式测试提交产物

- submission ZIP：`outputs/M3_A2_FLIP_M1_COMPLEMENTARY/seed42/submission/submission.zip`
- ZIP SHA-256：`8f757c6590e9d92ce7655e716d72eb36397d8f302e14c94f691b45e5e184ef4b`
- prediction CSV SHA-256：`6d53c6699fb24cbf59a0f9319aaffa90536ae701fd89675b6e7693c267d92322`
- manifest SHA-256：`5b5f019bcd2f382b94478a6b145b4327c7a3fc2a6ac40cad60f234c853131372`
- prediction count：24,967；classes：500；corrupt images：0；audit：passed；
- balanced test prior：未使用；test adaptation：未使用。

M3 相对 A2 M1 测试包改变 `2,672 / 24,967` 个预测（`10.70%`）。测试预测类别计数 CV 从 M1 的 `0.4976` 降到 M3 的 `0.4813`；M3 类别预测计数范围为 `4–196`，M1 为 `3–180`。这些仅用于产物多样性与异常审计，不用于参数选择。

平台验证优先级更新为：M3 第一、M1 第二。M1 是 M3 的必要消融对照；若提交名额有限，应优先 M3。

## 平台结果与结论修正（2026-07-22）

M3 平台得分为 **62.0259**：

- 相对 A2 Flip `61.2128`：`+0.8131pp`，说明注意力局部分支仍提供有效信号；
- 相对 A2 + M1 `62.6747`：`-0.6488pp`，说明额外加入 Flip 分支稀释了更强的 M1 局部证据；
- 相对 F1 + M1 `63.3276`：`-1.3017pp`。

因此，上述“本地 M3 第一”的预平台优先级已被真实平台结果否定。当前结论修正为：**保留 M3 作为有效消融，不再将其作为最佳提交；后续新检查点优先迁移纯 M1，而不是默认迁移 M3。** 这也是一个重要的选择器证据：本地 M3 虽比 A2 M1 高 `0.2634pp`，平台却低 `0.6488pp`，本地噪声验证不能可靠排序多视图推理方案。
