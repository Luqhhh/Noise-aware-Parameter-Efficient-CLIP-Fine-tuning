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
这些检查不是测试集候选或平台分。14:31:03 CST实际启动后台服务
`noise-v2-fixed-prior-delivery-20261002.service`，PID323230；该通用交付服务名在旧任务停止后复用，
实际工作目录、配置和输出均为本六视图方案，不写旧四视图candidate。
已完成完整本地回执、源码及checkpoint血缘与37,444张测试图逐张SHA校验，进入`scale448`推理；
[实际启动binding](../results/v2_sixview_bias_20261002/launch.json)。运行期间按20分钟检查，正式候选已完成。
六视图及固定校正/双包检查实测1,236.336271秒，随后独立复算46.446627秒。
服务Result=success/ExecMainStatus=0并退出；
[真实结果](../results/v2_sixview_bias_20261002/report.json)、
[独立复算](../results/v2_sixview_bias_20261002/independent_verification.json)。
37,444行两包均通过9项检查，独立float64拟合后逐张校正预测全量一致，bias最大差4.776505e-7；
全部输入/代码摘要及CSV/ZIP字节复算一致，概率零值为0。
六视图raw相对原四视图raw改变2,760张；bias相对同缓存六视图raw改变3,077张。
两者分别是解码变化与先验校正变化，不是已证明的净正确修正；平台分仍未知。

| 包 | CSV SHA256 | ZIP SHA256 |
|---|---|---|
| 六视图raw | `446165ca7d862431ab921d9de0a06fcd33e0d3011a59397a12831b2afa220f8e` | `4228a78a390e34744e42fb9a8ef861c6807d37b4e28dedc61bc3984c98353c6c` |
| 六视图bias | `194c0a0a4da379e6f340e6b078ae48439d00978f3924feb03df9806ed67d2e90` | `7164b26d9662a8c0e78c9c0636393efe0927739cb84df8dc88ed33e35f47659a` |

本机两包位于本段worktree的`outputs/codex/v2_sixview_bias_20261002/candidate/submission_{raw,bias}`。
用户需要的bias包、CSV、manifest、源回执和校验记录已复制到Windows，逐文件SHA再次一致；
[Windows复制回执](../results/v2_sixview_bias_20261002/windows_copy_receipt.json)、
[原样复制脚本](../results/v2_sixview_bias_20261002/copy_to_windows.py)。实际Windows文件为
`C:\Users\lqh22\Downloads\v2_continuation_20261001\submission_bias_sixview\submission.zip`，
806,198 bytes；原始无bias四视图包保留在`submission\submission.zip`。
在本worktree执行`python3 results/v2_sixview_bias_20261002/copy_to_windows.py`可核对/补拷同一固定产物；
不同现存目标文件会拒绝覆盖。两个检查日志在本结果目录`submission_{raw,bias}/submission_check.log`。
后续[配对补交付](v2_sixview_pair_handoff_20261002.md)已将同缓存六视图raw原样复制到
Windows同根目录`submission_raw_sixview/submission.zip`，两包正式检查通过；
需要评估纯bias贡献时使用这个六视图raw，原四视图包作为历史产物保留。
本段交付后暂停，不自动追加解码、bias强度或训练搜索；用户自行提交并回填平台结果。

## 规则范围

CLIP ViT-B/32/OpenAI官方初始化；当前阶段数据和checkpoint；单full raw模型；
固定多尺度/翻转TTA，无梯度或模型参数更新，无平台上传。
均衡bias在用户转述官方允许范围内，见[规则确认](v1_test_prior_authorization_20261001.md)。
现役平台最高74.41512658903963%来自独立V1 768方案，未被本段替换；
现役包及校验见[平台配对记录](v1_768_bias_platform_20261002.md)。
