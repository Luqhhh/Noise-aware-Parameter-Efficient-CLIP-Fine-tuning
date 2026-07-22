# N2：高可信局部类别原型分类器

日期：2026-07-20

## 假设

N1 证明 attention-local feature 含有额外细粒度信号，但自由的 `512→500` 残差会逐步接管 A2 logits，并对残余噪声与裁剪偏差过拟合。N2 检验另一条机制：不再通过梯度拟合 256,500 个自由参数，而是把每类高可信训练样本的局部特征聚合为一个受强约束的类别原型，使局部分支只能表达“该局部区域与该类训练分布有多接近”。

这是 N1 失败后的新机制检验，不修改 N1 的超参数或选择规则。

## 固定输入

- base checkpoint：A2 exact converted checkpoint，SHA-256 `1e2c1a4a274c5e466b716ded41ccf58bebf167a2fadbeba75693d08bdb4f039c`；
- train cache：N1 固定高可信子集 `66,929` 张，`clean_probability >= 0.70`，覆盖 `500` 类；
- validation cache：固定 D3 validation `10,322` 张；
- train/validation 路径重叠 `0`；
- local feature：与 M1/N1 完全相同的 top-5 attention centroid、160×160 crop、L2-normalized 512 维特征；
- 不读取测试图像，不使用外部数据、类别名称或文本提示。

## 唯一冻结构造

对类别 `c`：

`prototype_c = normalize(sum_i(clean_probability_i × local_feature_i) / sum_i(clean_probability_i))`

将 A2 原分类头第 `c` 行的 L2 范数记为 `s_c`：

`prototype_weight_c = s_c × prototype_c`

局部原型 logits 复用 A2 的原 bias：

`local_prototype_logits = local_feature × prototype_weight^T + A2_bias`

最终 N2 采用唯一、固定的 mean-logits 组合：

`logits_N2 = (global_A2_logits + local_prototype_logits) / 2`

这样无需温度、融合权重、学习率、正则或 epoch。原型方向来自高可信局部训练分布，尺度与类别偏置由单个 A2 线性头确定。N2 仍为一个 CLIP ViT-B/32、一个由官方训练集确定的类别头和固定双视图推理，不进行模型集成或测试时适配。

## 预注册审计与门槛

1. 每类必须至少有一个高可信训练样本，全部原型有限且 L2 norm 在数值误差内为 `1`；
2. A2 global logits 必须与既有 online-center cache 最大绝对差 `0`、预测一致率 `100%`；
3. 只报告两个预注册端点：local-prototype-only 作为机制诊断，N2 mean-logits 作为唯一候选；
4. N2 相对 A2+M1：clean-core micro 至少 `+0.25pp`；trusted macro 不得下降；raw micro 不得下降超过 `0.10pp`；
5. 任一门槛失败即 CLOSE，不改变 clean threshold、原型权重、特征、尺度、bias 或融合公式，不运行测试集、不生成提交包；
6. 全部门槛通过后，才允许另行预注册 N2 与水平翻转的互补组合。

## 执行结果与结论

状态：**CLOSE（机制假设失败）**。

所有构造审计通过：500 类均有样本；每类 trust mass 为 `30.0027–184.0`；原型方向 norm 为 `0.9999998–1.0000001`；与 A2 classifier 行范数最大绝对差 `1.53e-5`；train/validation overlap 为 `0`。

| 端点 | clean-core micro | trusted macro | raw micro | empty classes |
|---|---:|---:|---:|---:|
| A2 global | 82.5755% | 80.3732% | 69.4536% | 0 |
| local prototype only | 24.6604% | 23.2162% | 18.8336% | 260 |
| N2 mean logits | 67.9928% | 65.4020% | 54.8537% | 43 |
| A2 + M1 门槛基线 | 83.1716% | 80.8197% | 70.1124% | 0 |

N2 远低于全部门槛，因此不运行测试集、不生成提交包，也不修改尺度、bias、融合权重或 trust threshold。结果表明 M1 的单个 attention-local crop 可以作为同一 A2 分类头的互补视图，但跨样本并不形成稳定对齐的类别部件空间；不同图像的裁剪内容在类内差异过大，直接求类中心会形成严重的类别塌缩。后续不再沿“单 attention crop 的自由分类头或类别 centroid”继续调参，应回到能改变训练表征和区域一致性的上游方案。
