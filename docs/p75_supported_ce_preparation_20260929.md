# P75_SUPPORTED_CE：固定支持集合的原标签 CE 恢复

> **后续结论已更新（2026-09-29）**：以下为当时方案/结果记录；现行决定见[已有证据拆分复核](p75_evidence_recheck_20260929.md)。不再以旧2,139张代理支持困难监督恢复；207是双向类对门槛后的覆盖，不是全部混淆规模。P75_SUPPORTED_CE与旧三通道R1均不启动，缺证据不能自动撤销类别监督；R2无新启动许可。

> **2026-09-29 搜索纪律补充**：本文件记录 CPU 实现准备，不构成完整训练启动许可。
> 后续快照/筛选先登记有限诊断预算；支持集合非空或通过 +0.30pp 本地出包门，均不能
> 替代[主要误差规模与可恢复收益门禁](p75_error_budget_policy_20260929.md)。缺少能支撑
> 数个百分点收益的机制证据时，交付诊断与未知项，不自动启动完整训练或放宽阈值扫描。

状态：**实现、真实资产 CPU 预检和离线检查完成；遵照本轮用户指令，未启动 GPU 快照、训练、验证推理或测试推理。** 支持集合规模、候选验证指标和平台分均未知。方案分支为 `codex/p75_supported_ce_20260929`，独立工作目录为 `/home/lux1/noise/worktrees/p75_supported_ce_20260929`；从最新 `origin/main` 的 `2a9243a` 开始。

## 问题与范围

本候选只检验：对训练侧另有组外近邻支持、而固定 L05 仍不确定的训练图，恢复部分原标签分类监督是否有用。它不是原 `P75_HARD_SUPPORT` 的分类监督与局部门控双重恢复方案，不宣称支持集合标签已经确认干净。未通过筛选的样本也不称为噪声。

已核对全部本地/远端分支和近期 main：现有 Hard Support 只有准备代码，缺少同一训练行的 L05 弱视图与组外近邻证据；OOF hard filter 是删除样本后训练 LP/FT 的另一方向。本候选复用现有混合损失和支持 CSV 检查，不重复 OOF、教师训练或删样本方案。既有 Full SAM 的执行记录保留，本轮不启动或干预它。

L05 的 `confidence_gate=0.7` 是全局最大 softmax 概率门限，低于时局部分类损失回退到全局分类损失；不是原标签正确概率。`local_quality()` 的裁剪空间标准差只能识别近空白图，不是正确对象检测。本候选不调用它来放开门控。

## 固定资产与实现

运行源码从 L05 的 focus 提交 `f050ecb59e0a885da0b43cdc6c8c316953fb5e7d` 导出到本方案 `outputs/codex/p75_supported_ce_20260929/framework/`，只在该副本施加严格唯一锚点补丁。不修改 focus、SAM 或其他方案工作目录。私有 Python 路径只含本适配器和该冻结框架，排除共享 `p75_runtime/sitecustomize.py` 的另一候选模型挂钩。

资产绑定当前复赛 `repechage/20260921`，核对训练/验证内容组不交叉、类别映射、父权重、L05 权重、官方 OpenAI 权重及缓存哈希。完整绑定见 [机器可读记录](../results/p75_supported_ce_preparation_20260929.json)。核心文件为：

- [固定首点规则](../configs/p75_supported_ce_20260929/fixed.json) 与 [待支持集合的配置](../configs/p75_supported_ce_20260929/P75_SUPPORTED_CE.pending.yaml)。后者 `enabled:false`、支持哈希为空，**不能作为已绑定训练配置**。
- [CPU 准备、筛选与来源校验](../scripts/p75_supported_ce.py)、[训练侧快照](../scripts/p75_supported_ce_snapshot.py)、[显式阶段运行器](../scripts/run_p75_supported_ce.py)。
- [分类监督与局部恢复独立开关](../scripts/p75_runtime/p75_hard_support.py) 和 [冻结源码补丁](../scripts/p75_hard_overlay.py)。
- [全部 val_dev 的配对与分组报告](../scripts/p75_supported_ce_report.py)。

## 一次训练侧快照

将来用户授权且现有单卡任务释放资源后，读取固定 L05 `best.pt`，以 eval / no_grad / 冻结参数导出每张 train_dev 图的 384px 中心视图及该张量的水平翻转视图。沿用 L05 的 AMP 数值设置；概率均为独立分支 `softmax(logits.float())`，温度为 1，不做 TTA 融合或 prior 修正。逐图校验原始文件字节哈希。只加载训练行对应的图像，不读验证或测试图像。

产物保存在独立的 `snapshot/`：

| 文件 | 内容 |
|---|---|
| `rows.csv` | 原训练行序的 image_path / label / content_group |
| `center_probabilities.npy`、`flip_probabilities.npy` | 两路完整 float32 原始概率，均为训练样本数 × 类别数 |
| `manifest.json` | checkpoint 的确切 epoch、完整资产身份、视图与 AMP 协议、产物 SHA |

概率用磁盘映射分批写出；只有所有行完成后才写最终 manifest。已完成或中断的快照目录都不允许覆盖。这个快照只描述 L05 最终选中 checkpoint 的状态，不能恢复过去 16 轮的学习轨迹，不能证明任何训练图一直被双重压制。

RM-LP 使用的冻结 OpenAI 特征缓存可复用；已核对当前阶段来源、官方权重哈希、完整路径索引行序及 tensor 哈希。共享缓存包含官方训练池，筛选时按 train_dev 行序抽出查询库，并丢弃全量缓存张量；验证行不进入近邻库。没有重新训练教师、读取旧阶段资产或读取已有 OOF 记录。

## 唯一筛选点

阈值只来自已冻结 `fixed.json`，CLI 不提供网格或阈值覆盖参数：

| 条件 | 固定值 |
|---|---|
| 两路各自的原标签概率 | 均严格 `<0.30` |
| 两路各自的最大概率 | 均严格 `<0.70` |
| 视图稳定 | 两路 top-1 相同；不要求等于原标签 |
| 组外近邻 | train_dev 冻结特征余弦距离最近的 20 个不同内容组，排除自身完整内容组 |
| 原标签支持 | 至少 16 个组支持原标签 |

邻居中一个内容组只投一票，以该组最相近图像代表；相似度相同时按原训练行序确定顺序。组内原标签有冲突时记录完整标签集合，该组不计作任何原标签的支持。这是“组支持”的保守定义，不是新增的可信标签判定器。

只对通过弱视图条件的子集查询。每次查询至多 16 行 × train_dev 库，不建立全量 N×N 相似度矩阵。记录完整两路分布的 L1 与自然对数 JSD，不以它们新增筛选阈值。

`selection/diagnostics.csv` 包含每张训练图的两路原标签概率、最大概率、top-1、分布差异、是否查询、近邻组数和支持票数；`neighbors.jsonl` 记录被查询行的邻居图像、组、原标签集合、相似度和支持票。`support.csv` 严格保留原训练行序和原标签，仅权重为 0 或 0.5。

若不能得到 20 个组或支持不足，该样本不加入；这不说明其原标签错误。若集合为空，保存 `closed_empty_support`，不生成可启动的候选 YAML，运行器拒绝训练；不为凑样本数放宽规则。即使集合非空，稳定的 L05 错误和近邻共同错误仍可能存在。

## 分类损失与训练边界

独立开关固定为 `ce_restore_enabled:true`、`local_restore_enabled:false`。支持文件的存在不会自动打开任意机制。前两轮 CE warmup 直接返回原损失对象；之后支持样本分类损失为 `0.5*GCE + 0.5*CE`，其余保持 GCE。分类混合应用于原全局项及**原门控已经允许的局部分类项**；低置信局部仍回退到全局损失，不为支持样本增加局部准入。裁剪、局部权重和 0.7 置信门限完全沿用 L05。

按单标签 `q=0.5` 的 GCE，梯度相对 CE 缩放为 `sqrt(p_y)`，混合后为 `0.5+0.5*sqrt(p_y)`；`p_y=0.01` 时从 0.10 变成 0.55。CPU 梯度测试核对这一数值。它既恢复难例学习信号，也会放大错选标签的梯度，不能由这个公式推出支持标签干净。

完整训练从同一 RM-LP 父权重重新开始，运行器没有接续 L05 / SAM 的 resume 入口。保留 384px、16 轮、seed42、microbatch4 × 累积256 = 有效 batch1024、AMP=true、head LR 0.0004、backbone LR 0.000012、原 cosine horizon、feature anchor 2.0、第5轮开始的局部安排、原增强与原 center raw_macro / raw_micro 选模规则。配置逐字段与 L05 保存配置比对，除方案身份、独立输出目录和支持选项外的漂移均拒绝。标签、样本规模和结构均不改变。

训练每轮额外写 `logs/supported_ce_epoch_<epoch>.json`，记录固定支持集合的原标签概率、准确数和分类 base/CE/blended 损失；只是训练拟合诊断，不作泛化证据，不与不同增强的弱视图快照冒充同条件比较。

## 固定解码、报告与出包

验证主指标始终为全部 14,880 张 val_dev：单 checkpoint、Flip、mean_probabilities、T1.4、val 拟合 prior0.60，与现役同协议。候选评估先使用冻结框架的缓存血缘校验，再独立重算预测和 macro/micro；逐张记录修正、退化、前后预测。原验证标签来自含噪训练池，不能改称干净测试准确率。

按每类获得 CE 恢复的**训练图数**固定分组：0、1–4、5–19、≥20；报告每组涉及的类数、验证数、macro/micro recall、召回变化和修正/退化数，另有逐类 CSV。所有类别和验证图都进入报告，不筛选模型认为可信的验证子集，也不因局部改善自动晋级。

出包门：总体 macro 相对现役至少 +0.30pp、micro 不下降。平台优先门：macro 至少 +1.00pp、micro 不下降。运行器重新检查评价、配对记录、checkpoint、缓存、配置和支持集合的哈希，重算门槛后才允许测试推理；没有未过门出包覆盖开关。出包沿用冻结脚本，生成 CSV/ZIP 并自动跑 9 项提交校验；不上传平台。

支持类别与总体同时改善才支持继续；训练拟合改善而验证不改善不能算提分；少数类别收益被其余退化抵消也不晋级。一次本地结果只用于候选筛选，不证明稳定性或平台收益。

## 可重放命令

以下在本方案 worktree 根目录执行。本轮实际调用了 CPU 准备/预检与测试，**没有调用后面的 GPU 命令**。

```bash
python3 scripts/run_p75_supported_ce.py --phase prepare
python3 scripts/run_p75_supported_ce.py --phase preflight
python3 -m pytest tests/test_p75_supported_ce.py tests/test_p75_supported_ce_snapshot.py tests/test_p75_hard_support.py tests/test_p75_runner_logs.py tests/test_p75_patch_readout.py tests/test_p75_sam.py -q
PYTHONPATH=outputs/codex/p75_supported_ce_20260929/framework/reproducibility/aegis_f1 python3 -m pytest outputs/codex/p75_supported_ce_20260929/framework/reproducibility/aegis_f1/tests/test_losses.py outputs/codex/p75_supported_ce_20260929/framework/reproducibility/aegis_f1/tests/test_config.py -q
PYTHONPATH=outputs/codex/p75_supported_ce_20260929/framework/reproducibility/aegis_f1 python3 scripts/p75_supported_ce_cpu_audit.py
```

以后经用户授权，单卡任务释放资源后按顺序执行；没有自动队列、后台启动、占卡或改动共享环境。每个 GPU 阶段启动前检查原 CUDA 设备空闲，并尊重 `CUDA_VISIBLE_DEVICES` 映射。

```bash
# GPU，一次最终 L05 训练侧快照
python3 scripts/run_p75_supported_ce.py --phase snapshot --execute-gpu
# CPU，固定一次；空集到此停止
python3 scripts/run_p75_supported_ce.py --phase select
python3 scripts/run_p75_supported_ce.py --phase preflight --require-support
# GPU，同 RM-LP 父权重完整训练和最终验证缓存
python3 scripts/run_p75_supported_ce.py --phase train --execute-gpu
python3 scripts/run_p75_supported_ce.py --phase cache --execute-gpu
# CPU，固定解码及按训练恢复数量分组的配对报告
python3 scripts/run_p75_supported_ce.py --phase evaluate
# GPU，仅达到已登记出包门后生成与校验测试包
python3 scripts/run_p75_supported_ce.py --phase deliver --execute-gpu
```

真实绑定 YAML 将由非空筛选结果生成于 `outputs/codex/p75_supported_ce_20260929/configs/P75_SUPPORTED_CE.yaml`；它引用支持 CSV 的确切 SHA。快照、筛选与结果不可覆盖，阶段日志在 trainer 所拥有的 `seed42/` 目录外，不会阻止 fresh run。

## 本轮验证与交付边界

44 项 P75 CPU 测试通过，涵盖真实冻结 trainer 的补丁编译、关闭/CE warmup 恒等、0.7 门控边界、原标签梯度、图像字节检查、CPU toy 的完整双视图导出、固定 16/20 票边界、排除重复组/冲突标签、空集停机、全 val 分组、GPU 显式开关和可见设备检查。另有私有冻结框架原损失/配置回归 44 项通过，合计 88 项；结果登记于机器可读记录。

CPU 实际读取并核对 train_dev 特征库，未进行任何真实训练图的邻居查询；现役验证缓存自比较重现既定 macro/micro，修正和退化均为 0。该审计只是现役自身重放，不是候选实测。

现役保底包 `/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/{pred_results.csv,submission.zip}` 再通过 37,444 行提交校验，内部 CSV 字节与外部一致。它是已有可提交产物，**不是本候选新包**。现役平台分仍为用户已登记的 66.94797564362783%；本轮未占平台名额。

本轮用户明确只要求完整实现且暂不启动 GPU，因此在实现准备提交、推送及 main 集成检查点暂停，不以常规实验闭环要求擅自启动快照或训练。快照/支持规模/候选 checkpoint/候选指标/新提交包全部待后续 GPU 授权。
