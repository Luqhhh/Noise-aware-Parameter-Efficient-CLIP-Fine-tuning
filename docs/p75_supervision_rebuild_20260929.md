# P75：监督重建优先，混淆边界重学条件备用

本轮按用户给定方案实现 P0、R1、R2 的独立入口。**当前实验段是 P0；R1/R2 的实现准备不等于训练完成。** 按仓库段边界约定，P0 交付、推送和 main 集成后暂停，不在同一检查点后自动启动训练。用户已明确授权空闲 GPU 做快照及后续固定试跑；本机没有 NPU。已有 Full SAM 任务不由本轮重启或停止。

## 已有工作比对

2026-09-29 fetch 后检查全部本地/远端分支、worktree 和近期 main。
从 `78deb3a` 创建独立分支 `codex/p75_supervision_rebuild_20260929`，目录
`/home/lux1/noise-worktrees/p75_supervision_rebuild_20260929`。
复用 `P75_SUPPORTED_CE` 的资产校验、双视图 Dataset 和固定验证解码；其
0.5 CE/GCE 少量恢复不作为本次 R1。Patch Readout、MixStyle、Fourier、RSC、
随机深度、校准等关闭状态保持不变。

## P0 固定证据与边界

当前阶段 train_dev 133,815 / val_dev 14,880，750 类。只读当前阶段资产，
复用官方 OpenAI 224px 冻结特征和已有五折原型/ridge 输出；不重训教师或 OOF。
按全量训练及验证行查询，但**近邻库只有 train_dev**；排除查询所属整个内容组，
20 个独立组各一票。组内标签冲突不投任何类别，查询自身组冲突也不授予类别通道。
余弦相同按训练行号排序；分块计算，不创建完整 N×N 相似度矩阵。

L05 epoch16 用原始 384px center / 同张量 Flip、T=1、无 prior、各自 FP32 softmax
做一次训练侧快照。逐图核验已审计字节摘要。与历史训练期记录严格区分：逐样本
历史 p_y、p_max、局部准入没有保存，CSV 对应栏为空；快照门限栏只表示当前
全局最大概率是否达到 0.7，不表示历史或实际局部准入。GCE 梯度栏仅记当前
`sqrt(p_y)`，不据此宣称训练期间持续压制。

诊断固定首点（不是标签真值）：

- 原标签通道：20 个组外近邻中至少 12 票支持原标签，且两路已有 OOF 都支持它。
- 软目标通道：至少 16 票支持同一替代类别、两路 OOF 同意替代类别、L05 两个
  弱视图也同意；证据冲突则归不确定。软分布来自完整组外投票分布，不硬重标。
- 不确定通道：其余样本。缺证据与被证实噪声不是同义词。
- 可信难样本：原标签获上述支持且 L05 center 错分或原标签概率 <0.3。
- 混淆边：L05 固定解码同一类对双向各至少 2 个错误、合计至少 6 个；连通分量
  仅用于展示，机制切片仍要求**直接类对边**，不把传递连通当作可靠负例关系。
- 可信混淆切片还要求验证原标签至少 12 票、两弱视图 top1 一致，两端训练类别
  各至少 20 个获支持独立组。训练侧没有使用验证图作近邻。
- 困难监督关联切片：验证原标签至少 12 票，且同类训练至少 5 个可信难内容组。
  这是类级关联，不是逐错误的因果解释或可恢复收益上限。
- 独立内容组不足 20 的类别单列。**固定筛选后可信参考稀少**另记 coverage deficit，
  不把筛选器缺证据自动归因为数据缺监督；不能借此吃掉未知错误。
- 若超过 5% 类别在原标签通道不足 5 个独立组，则判本次监督构建失败，拒绝生成
  可启动训练配置；不放宽阈值重扫或重复采样凑覆盖。

CPU 近邻预算 7,200 秒、GPU 快照预算 5,400 秒，完成标记和哈希最后写出；部分
输出不能当成证据。弱视图和 OOF/近邻仍可能共同出错。切片来自已用验证集，
不是新独立测试集。错误规模、重叠、未知、2,139 张旧代理集合的联证都写入摘要。
最终只交付一张逐样本证据表和一页主线选择；JSON、切片清单及日志是复现附件。

## R1 已实现的固定训练入口（受 P0 门禁约束）

从官方 OpenAI ViT-B/32 新建分类头。第一步用已有官方 224px 冻结特征训练头
20 轮，batch1024、head LR0.01、末轮；不加载 RM-LP 或 L05 的学生分类器。
没有独立弱视图缓存的未知样本在这一步不产生分类梯度；视觉表示冻结。
随后只把这个学生 head checkpoint 作为视觉微调起点：384px、micro4×累积256、
有效1024、head LR0.0004、backbone LR0.000012、anchor2.0、16轮cosine周期，
其余沿现有 L05 配方。代码拒绝旧 trust / mixup 同时混入。

原标签通道全程 CE；软通道 soft CE；不确定通道不施加类别目标，视觉阶段使用
同一增强图及其 Flip 的特征余弦一致性，权重1。教师为前一轮边界的学生副本，
全程 eval/no_grad，epoch1 来源是新学生的官方视觉＋新头。软目标保留75%固定
组外投票，最多25%滞后教师；教师概率先限制在固定参考支持的类别内，不能
增加支持类别或改变通道。它仅用于训练，正式评估和提交仍是单一学生。

局部几何保持 L05，第5轮启用。原标签可信样本通过有限值和空间变化检查后即可
使用局部CE，不依赖已达到0.7自信。软目标仍需0.7门限及相同有效性检查；未知
不产生局部类别梯度，回退到全局一致性。空间标准差≥0.05只是排除近空白，
不宣称它能识别有效对象。

每轮记录三个通道实际样本数、一致性项和局部准入汇总。固定通道表按哈希绑定，
不允许模型越学越自信就扩大“可信”集合。逐类覆盖在 P0 表内可审计。

## R2 独立备用入口

只有 P0 可信稳定混淆切片关联至少1,000个验证错误才生成R2配置；不叠加R1。
从原RM-LP父模型开始独立视觉训练，原全局/局部GCE监督保留。每两个microbatch
一个常规随机批、一个主要混淆类对的 `2类×2独立内容组` 批；只有原标签通道
获支持的训练样本可进入有组织配对。可信同类不同组为正样本，只有预冻结直接
混淆边上的可信异类可为负样本。SupCon温度0.1、权重0.1，作用于现有归一化
视觉特征；不增加投影头，不把普通异类近邻自动设为负例。

同内容组不能充当两个独立正样本；没有可信负例/正例的anchor不参与对比。
运行器和损失入口分别检查。固定R2失败后不扫温度、margin、维度或比例。

## 试跑、延续与提交

显式阶段，无自动候选队列。试跑固定6轮，保留16轮调度；6轮内最多21,600秒。
因此覆盖第5轮局部启用后的两轮。保留epoch4/6逐图缓存，比较候选自身4→6的
冻结目标切片与非目标部分，再与L05第6轮center raw macro作相近进度的有限参照。
不把6轮低于L05最终模型当作充分否证。

预设完整训练信号：目标切片先关联至少1,000个L05验证错误；4→6目标切片净修正≥50，非目标净退化≤50，全量净修正>0，
第6轮center raw macro不低于L05同进度超过2pp。它是一次探索的资源门，不能单独
证明因果；报告同样披露两条学习过程不同。延续从原配置的last checkpoint恢复
优化器和调度到16轮，滞后教师在轮边界从学生重建。禁止另起学习率或更换规则。
延续预算57,600秒。结果没有显著机制信号就关闭固定路线，不扩展参数搜索。

最终使用固定epoch16单学生，在全部val按Flip/T1.4/val prior0.60报告macro/micro、
修正、退化；每个冻结切片与标签争议组也完整报告。优先提交门：macro与micro
各至少+1.5pp，目标机制切片净改善。过门才允许生成测试包并自动9项校验；不读
测试预测分布调参、不集成、不上传平台。平台分由用户回填，代码没有外推比例。

## 可重放命令

以下在方案worktree执行。前四类为本段实际动作；后续R1/R2仍需P0门禁且在段检查点之后。

```bash
python3 scripts/run_p75_supported_ce.py --phase prepare
python3 scripts/p75_supervision_evidence.py --phase neighbors
# 复用原框架的模型/资产校验；新入口额外绑定原始 L05 YAML
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
PYTHONPATH=outputs/codex/p75_supported_ce_20260929/framework/support_runtime:outputs/codex/p75_supported_ce_20260929/framework/reproducibility/aegis_f1 \
timeout --signal=TERM 5400s python3 -u scripts/p75_supervision_snapshot.py --execute-gpu
PYTHONPATH=outputs/codex/p75_supported_ce_20260929/framework/reproducibility/aegis_f1 \
python3 scripts/p75_supervision_evidence.py --phase diagnose \
  --snapshot outputs/codex/p75_supervision_rebuild_20260929/snapshot
python3 -m pytest tests/test_p75_supervision_evidence.py tests/test_p75_supervision_runtime.py tests/test_p75_supported_ce_snapshot.py -q

# 条件性后续入口，prepare本身不启动训练，覆盖失败即拒绝
python3 scripts/run_p75_supervision_rebuild.py --route R1 --phase prepare
python3 scripts/run_p75_supervision_rebuild.py --route R1 --phase head
python3 scripts/run_p75_supervision_rebuild.py --route R1 --phase pilot --execute-gpu
python3 scripts/run_p75_supervision_rebuild.py --route R1 --phase evaluate-pilot --execute-gpu
python3 scripts/run_p75_supervision_rebuild.py --route R1 --phase full --execute-gpu
python3 scripts/run_p75_supervision_rebuild.py --route R1 --phase evaluate-full --execute-gpu
python3 scripts/run_p75_supervision_rebuild.py --route R1 --phase deliver --execute-gpu
# R2必须先满足自己的混淆证据门，不执行head阶段，不与R1并行排队。
```

首次复用旧快照入口在模型加载前因保存配置缺 `_config_path` 停止；新入口从原始
YAML恢复路径并核对checkpoint binding，未绕过校验。一次无快照的CPU组装检查
暴露numpy整数JSON序列化问题，已修复；该中间目录没有完整manifest，不作为
最终结论。失败日志与最终运行日志均归档，未覆盖其他人的产物。

## 交付落点

正式P0结果见 `results/p75_supervision_rebuild_20260929/`，包含压缩逐样本CSV、
一页选择结论和机器可读摘要。大体积近邻、双视图概率及私有框架保留在方案目录
`outputs/codex/p75_supervision_rebuild_20260929/`。构建代码、运行时覆盖范围、
测试与真实诊断分别记账；无新学生checkpoint、候选提升或平台分。

现役可提交包为：
`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/{pred_results.csv,submission.zip}`。
本段复检37,444行及9项校验，包内外CSV字节相同。它是现役保底包，不是R1/R2产物。
