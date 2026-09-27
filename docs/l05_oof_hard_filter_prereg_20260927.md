# L05 当前阶段 OOF 低可信硬筛：CPU 前置与训练预注册

状态：CPU 筛样与训练协议接入已完成。用户 2026-09-27 要求开启实验；本机筛后 LP 已完成，筛后 FT 正在运行。尚无 FT 结果或新提交包，实验未完成。

## 假设与历史边界

旧阶段 A2 的高精度 CL+kNN 共识仅删除 991/91,195 张，而类内扩大筛除曾使平台 TTA 比对照低 0.76pp。当前 750 类的 OOF 连续降权实验整体 macro 下降 0.1311pp，不能视为硬筛的证据。因此本实验只检验一个保守的、事先固定的约 1% 硬筛规则；不复用旧阶段任何数据、置信度、原型或 checkpoint。

## CPU 置信度与固定筛除规则

只读当前 `20260921` 阶段的 `train_dev.csv` 和已冻结的官方 CLIP ViT-B/32 特征缓存。验证集、测试集均不参与置信度拟合或阈值选择。按 `content_group` 完整分组、类别分层，seed 42，5 折。每折仅用另外 4 折训练图计算两个判别器，再给留出折打分：

1. 每类归一化平均特征的余弦相似度（centroid）。
2. 同一类平均特征经训练折特征协方差的 ridge 变换后再归一化（ridge-whitened centroid），固定正则 10.0。

二者均取 argmax。只有二者一致预测**同一个非原标签类**、且各自与原标签的相似度差都严格大于零，才进入可疑集合。以两项差的较小值降序、路径字典序破平局。最多删除 `floor(1% × 133,815)` 张；每类最多删除 `floor(5% × 原类样本数)` 张；原类不足 20 张一律不删；任何类别至少保留 4 张。该量是折外标签支持度，不是校准后的“干净概率”。这些数字在看筛除清单前固定，不因 CPU 结果或验证集调参。

输出独立目录 `outputs/codex/l05_oof_hard_filter/cpu_prepare/`：逐样本两路折外分数、被删行、实际训练 CSV、带 SHA-256 的 manifest。严格检查特征缓存与当前阶段数据血缘、组不跨折、train/val 内容组隔离、全部 750 类保留、删前删后计数守恒。原始数据只读。

## GPU 训练的唯一比较

候选从**筛后数据**重新训练线性探针 RM-LP，再用 L05 固定 384px/16 轮配方微调；正常对照是现役未筛样 L05。两者均只使用同一当前阶段资产与 seed42；唯一实验变量是训练集的预冻结筛样清单。必须对筛后训练 CSV 建立显式血缘，不能绕过 `rematch_assets` 的原始 `train_dev.csv` 闸门，也不能误用未筛样 RM-LP 父 checkpoint。

每偶数轮验证，按中心视图 macro 选单个 checkpoint，固定 Flip/T1.4/prior0.60 解码。最终 macro 相对现役 75.8265% 至少 +0.30pp，且 micro 不低于 76.6532%，才进入独立解码、内容组条件交叉校准；后者仍需 +0.20pp 且 micro 不退化。通过后才能生成 37,444 行 CSV/ZIP 并做 9/9 校验，由用户决定平台上传。任一可比偶数轮 macro 或 micro 较匹配未筛样 L05 低 2pp 则止损。失败关闭，不调整筛除比例、ridge 正则、fold 数或训练预算。

CPU 产物不证明模型受益；FT 与最终解码完成前，不声称实验完成或平台可用。

## CPU 实测（2026-09-27）

预定规则实际从 133,815 张 `train_dev` 中移除 **1,338 张（0.9999%）**，保留
132,477 张、750/750 类；有 485 个类各删除至少 1 张，最大类内删除比例 5%，
最小保留类样本数 4。两个折外判别器同指向非原标签的可疑样本共 27,019 张，
但预注册全局/类内上限仅选前述 1,338 张；入选样本的较小相似度差范围
0.10146–0.26483。训练与验证内容组交集 0，计数、预算、类覆盖和双路一致性
均由独立脚本复查通过。两个定向测试通过。

输出路径和 SHA-256 见[CPU 结果记录](../results/l05_oof_hard_filter_cpu_20260927.json)。
原框架的筛样血缘门禁现已在独立的固定 runner 中接入，见下节。

## 训练协议接入（CPU 已验证，GPU 未启动）

`scripts/run_l05_oof_hard_filter.py` 从固定代码提交 `f050ecb59e0a885da0b43cdc6c8c316953fb5e7d`
重建 L05 框架到本方案独立输出目录，覆盖一份带当前阶段硬筛血缘检查的 V4/V5 共享协议，
生成独立的 `RM_LP_HF01` 和 `RM_V5_L05_HF01` 两份配置。LP 用筛后数据及当前阶段
官方冻结特征重新训练；FT 沿用本地 L05 CUDA 配方（384px、16 轮、有效 batch 1024、
从第 5 轮启用局部监督），只允许以本次筛后 LP 为父。两份 checkpoint 绑定筛后 CSV
与筛样 manifest 的 SHA-256，原始 RM-LP 或现役 L05 权重不能充当本次父 checkpoint。

入口会逐项复核原始 train/val、特征缓存、筛样配置、逐样本分割、删行双路一致性、750 类覆盖、
训练与验证内容组隔离及官方权重。CPU `prepare` 校验真实资产已通过；两个协议测试验证
正确 LP 父血缘可接受、旧 LP ID 或错误训练集 SHA 均被拒绝，筛后 CSV 也不可换成别的路径。

用户解除暂停且 GPU 空闲时，执行：

```bash
python3 scripts/run_l05_oof_hard_filter.py run --device cuda:0
```

脚本先做 CPU 预检，再依次训练 `RM_LP_HF01` 与 `RM_V5_L05_HF01`；若 LP 已成功而 FT
需要单独启动，可执行 `python3 scripts/run_l05_oof_hard_filter.py train-ft --device cuda:0`。
2026-09-27 已在本机执行筛后 LP 与 FT；实际结果和最终提交仍须按前述门槛完成。

## 启动记录（2026-09-27）

远端 RTX 4090 的 LP 曾启动，但其 SSH 端点持续关闭连接，图像传输停于 235/750 类；远端 LP 最终状态无法读取。用户要求开启后，改用当时空闲的本机 RTX 4070 Laptop GPU，沿用同一固定筛样清单、协议和配置。`prepare` 的真实资产血缘检查再次通过。筛后 LP 用上方命令完成 20 轮、10,360 次成功更新，最佳 epoch 20，中心视图 raw macro/micro 为 62.6255%/63.6223%。最佳 checkpoint SHA-256 为 `ae472dd173705145c63b94c14589ae3239fe01084bfe68015a0b4d65e7148b90`，其绑定文件的筛后 CSV 与 manifest 哈希均匹配预注册。随后执行 `python3 scripts/run_l05_oof_hard_filter.py train-ft --device cuda:0`，首步梯度审计通过；FT 正在运行，最终分数与提交包尚未产生。机器可读记录见[本机启动记录](../results/l05_oof_hard_filter_local_launch_20260927.json)。

沙箱内 `screen -ls` 曾误报这个外部 `l05_hf01_local` 会话为 dead，造成一次重复的前台 FT 启动；该第二进程已退出，原 screen 进程继续运行，2026-09-27 22:26 到第 40 次成功更新。两进程短暂共用日志目录，最终必须独立重载 checkpoint 和验证，不能只依赖训练日志。

## 复现命令

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 scripts/prepare_l05_oof_hard_filter.py \
  --config configs/l05_oof_hard_filter/fixed.json
python3 scripts/run_l05_oof_hard_filter.py prepare
```
