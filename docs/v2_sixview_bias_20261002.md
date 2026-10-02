# V2_SIXVIEW_BIAS_20261002

用户明确要求：448、512、576三种尺度，各含原图和水平翻转，再加固定均衡bias。
此前本机四视图bias推理已按该要求停止，未生成候选；
[停止记录](v2_fixed_prior_delivery_20261002.md)保留。本段承接生成提交包的授权。

## 输出前固定配方

使用已下载并核验的V2 full第5轮raw单checkpoint：
SHA256 `3b9fcce3a4a2a113f2eb2ad3a2028c7f224fd1b11f1bdaadd6641ee1ccc85138`。
原冻结plan、750类映射、当前阶段数据清单、逐张测试图像素、模型源码和原始包血缘仍全部核验。
原包及完整6份权重已交付Windows，13:57服务器执行关机且三次SSH离线；本段仅使用本机CUDA。

六视图依次为`scale448/scale448_flip/scale512/scale512_flip/scale576/scale576_flip`。
每种尺度先bicubic短边resize至round(尺度×1.14)，再中心裁剪至对应448/512/576输入；
翻转视图在相同裁剪后做确定性水平翻转，沿用CLIP归一化，BF16前向、batch16。
输入实际尺寸随尺度变化，位置编码由原V2模型按网格插值。

沿用V2的概率合并：六份float32 softmax等权平均，取float64 `log(max(p,1e-30))`，
固定200次均衡bias、damping1、strength1、目标1/750。无新训练、参数更新、温度或强度搜索。
这是单checkpoint的新六视图确定性解码，不与其他模型或checkpoint组合。

同一完整概率缓存生成六视图raw与bias两包，均做37,444行/9项正式提交检查，
再独立NumPy/float64复算bias、逐张决策和CSV/ZIP字节。
六视图raw与服务器原四视图raw的变化只记为不同解码的比较；
原四视图数值重放的0.1%/0.01阈值不适用于用户指定的解码变化。
同缓存raw argmax与取log后的argmax必须一致，独立float64校正决策必须全量一致。
没有测试真值或平台反馈，不将预测变化或先验均衡当作准确率提升。

## 归属、实现和执行命令

分支`codex/v2_sixview_bias_20261002`，worktree
`/home/lux1/noise/worktrees/v2_sixview_bias_20261002`，从最新`origin/main`的`43442f9`创建。
已fetch并检查全部V2/六视图/bias分支、近期main和CUDA进程；不存在重叠V2六视图实现或正式任务。
复用既有固定prior入口，仅增加严格固定的六视图profile；原四视图profile继续受原门限制。
原模型/训练代码、源plan、原始包及下载关机服务不修改。

配置：[v2_sixview_bias_20261002.json](../configs/v2_sixview_bias_20261002.json)。
在本段worktree执行：

```bash
export PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONPATH=reproducibility/aegis_f1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python3 -u scripts/run_v2_fixed_prior.py \
  --config configs/v2_sixview_bias_20261002.json \
  --output outputs/codex/v2_sixview_bias_20261002/candidate --action run
```

61项检查通过（含8项新六视图检查和原profile/模型回归），3.78秒，
见[实施校验记录](../results/v2_sixview_bias_20261002/implementation_checks.json)。
真实不对称图像验证三个实际尺寸及逐像素翻转配对；模拟输出区分概率平均与logits求和，
合成完整交付验证新解码相对原包变化不阻断同缓存独立float64复算。
此前测试导入问题已修正；CPU模拟的多进程fork阻塞改为单进程测试加载，正式CUDA仍为2个worker。
这些检查不是测试集候选或平台分，正式候选尚未交付。核验后Windows目标为
`C:\Users\lqh22\Downloads\v2_continuation_20261001\submission_bias_sixview\submission.zip`，
原始无bias四视图包保留在`submission\submission.zip`。

## 规则范围

CLIP ViT-B/32/OpenAI官方初始化；当前阶段数据和checkpoint；单full raw模型；
固定多尺度/翻转TTA，无梯度或模型参数更新，无平台上传。
均衡bias在用户转述官方允许范围内，见[规则确认](v1_test_prior_authorization_20261001.md)。
现役平台最高74.41512658903963%来自独立V1 768方案，未被本段替换；
现役包及校验见[平台配对记录](v1_768_bias_platform_20261002.md)。
