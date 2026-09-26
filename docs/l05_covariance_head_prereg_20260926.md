# L05 共享协方差判别头（预注册与实测，已关闭）

实验 `L05_COVARIANCE_HEAD_20260926`，独立分支 `codex/l05_covariance_head`，
基于 main `04fc74b`。开始前已 fetch 并核对所有本地/远端分支；V4 H01–H24、V5 H01–H12
均为优化训练的 linear/cosine head，NH00/NH01 为 GCE 线性/非线性适配；未找到共享协方差
判别头的配置、实现或结果。本轮检验生成式类内距离假设，不改变已关闭 NH 配方参数。

## 固定机制及依据

只读复用本阶段冻结 L05（SHA256 `35d17c0c3b9f281e13353959d098c2713def6b7d508d334de5d3a7e769cd2397`）
的133,815张 train_dev中心特征，14,880张 val_dev中心/Flip特征。
加载前验证原配置身份、父checkpoint、stage manifest、内容组不交叠、每个分片的路径/标签。
原缓存编码器为AMP、存储float32，分类头在float32执行；不会重编码/覆盖源缓存。

在CPU float64中对训练特征估计每类均值，以及减去各自类均值后的共享协方差。
固定两个臂：CH00使用 `trace(S)/d * I` 的各向同性对照，CH01使用OAS自动收缩的完整协方差。
OAS按训练样本计算，使用sklearn公开实现的大维度近似公式；不在val选择收缩系数。
所有类使用均匀先验。判别函数为 `xᵀΣ⁻¹μc − 0.5μcᵀΣ⁻¹μc`，直接编译为一个线性头。
不与原分类头加权、不集成、不新增骨干。无额外温度拟合或噪声筛选。

公式依据：[LDA官方文档](https://scikit-learn.org/stable/modules/lda_qda.html)、
[OAS作者论文](https://arxiv.org/abs/0907.4698)。
**L05类内协方差是否能改善本任务分类是待测假设；不假定特征服从高斯。**

## 预注册判据与交付

固定Flip/mean_probabilities/T1.4、各候选自身val拟合prior0.60，与现役L05相同解码。
CH01须同时超过L05和CH00 macro ≥0.30pp，micro不下降；CH00可独立超过L05同门槛。
中心视图macro或micro较L05下降≥2pp则工程止损，不进入额外图像推理/平台候选流程。
失败后关闭，不派生协方差、收缩、温度或先验强度扫描。成功才做全val图像重放、
条件交叉校准及单checkpoint测试集CSV/ZIP；平台上传由用户执行。
失败交付既有L05_T14_P060保底CSV/ZIP并重跑9/9校验，记录路径/哈希。

先做OAS独立实现、直接Gaussian距离与线性头等价、类别置换、缺类拒绝测试；
正式缓存结果再以独立NumPy解码核验，并原生重载完整checkpoint验证头及骨干。
缓存复核不是独立图像重放，本地收益不是平台收益。70分必须有平台回执才算完成。

```bash
python3 -m pytest tests/test_l05_covariance_head.py -q
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 scripts/run_l05_covariance_head.py \
  --config configs/l05_covariance_head/fixed.json
```

代码、配置、正式输出均在本worktree的独立目录。禁止覆盖已存在结果；
只拟合训练集均值/协方差、验证集解码prior；测试集不用于任何拟合或选择。
CLIP ViT-B/32/OpenAI官方权重、单checkpoint、无外部/跨阶段资产、无测试时训练。


## 实测结果与交付

CPU正式拟合完成；训练133,815张、验证14,880张，750类全部覆盖，训练每类4–223张。
训练侧OAS收缩系数0.0009290332；没有扫描或修改预注册公式。

| 候选 | center macro/micro | 固定解码 macro/micro | 对L05解码 Δmacro/Δmicro |
|---|---|---|---|
| CH00 各向同性 | 68.7317% / 69.4220% | 69.0352% / 69.5901% | −6.7913pp / −7.0632pp |
| CH01 共享协方差OAS | 70.6992% / 71.4449% | 71.1367% / 71.6935% | −4.6898pp / −4.9597pp |

两臂中心视图均触发−2pp工程止损，未晋级；虽CH01较CH00好，仍不能替换L05。
关闭该固定配方，不派生收缩/温度/类先验扫描，也不进行测试集候选推理。
原型均值与高斯假设在该特征/标签条件下没有达到目标；不据此推断所有协方差方法无效。

4项测试通过，包括与sklearn OAS独立实现的一致性、直接Gaussian距离与线性头等价。
正式两臂各14,880个解码预测经NumPy float64独立重算与Torch完全一致；原生完整checkpoint
重载后各29,760个分支预测一致，logits最大差0，视觉权重逐位未变。
这些是**缓存验证与模型重载证据**，未重跑全部图像推理，未产生平台成绩。

实现新增2个脚本、1份固定配置、4项测试及本记录。实现提交`9bb7fc3`已在开跑前冻结并推送。
全部分片SHA、源配置/数据/父模型身份、checkpoint SHA、实测数值和交付校验输出见
[机器可读结果](../results/l05_covariance_head_20260926.json)。本地运行输出在
`/home/lux1/noise-worktrees/l05_covariance_head/outputs/codex/l05_covariance_head/`。

本轮交付既有L05_T14_P060保底包（不是本轮候选），37,444行，9/9校验通过：

- CSV：`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/pred_results.csv`
- ZIP：`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/submission.zip`
- CSV SHA256：`51e0efe7178528d23993a44351069d2829b0cd3669c195a77d8d879e66776a75`
- ZIP SHA256：`e788f07636b80abb61685335cd8df108803863ea36ffbde8b353f8e8317fcd7f`

校验确切命令记录在结果JSON的`delivery.validation_command`。
平台最佳仍为66.94797564362783%，70分目标未达。本轮验证后提交/推送方案分支，
main集成目录按自动模式pull、merge、复核、push，随后停在交付检查点。
