# V1_TASK_NEIGHBOR_PROBE_20261002：任务训练表征的邻居恢复预算

状态：CPU准备已验证；以下检索、群体与投入门在新邻居输出前冻结。尚无GPU邻居结果。
从`origin/main@732bd4c`新建`codex/v1_task_neighbor_probe_20261002`和独立worktree。
已fetch并核对全部本地/远端分支及近期main，无相同任务训练768邻居诊断的既有结果或在途工作。
A的head/LoRA配对、B的增强配对及v2固定续跑保持，不使用NPU/远端或子agent，不改共享依赖。

## 可观察问题与可证伪假设

四条DEV路线共同错误2,875张；原LR512的3,458错中任一原图/镜像原类top5有1,881张。
低细节固定配对、较大镜像分歧入口和0.05 cosine-margin候选已各自关闭；不扫描其参数或恢复训练。
它们未判断**已经任务训练的特征中，同类邻域是否可支持分类头错例**。
先前512/768 teacher与k16检查用官方冻结224特征，不能替代当前已任务训练的448特征检查。

假设：若已学习视觉表征中的内容组独立邻居能稳定纠正普通768头的大块错误，
局部支持仍是可研究机制；若固定多数支持也不能提供足够净修正，关闭本邻居入口。
未知：图库原标签含噪、训练父已经见过train_dev、head用v1去噪监督而邻居用原标签，
因此不是纯head结构因果实验，也不是折外训练噪声鉴定或平台收益证明。

以kNN检查特征判别信息可参考[DINO官方评估代码](https://github.com/facebookresearch/dino/blob/main/eval_knn.py)。
本轮采用一个无距离权重的k16规则，不复现其指数温度投票，不读外部模型/数据或下载权重。

## 固定源、图库与检索

[配置](../configs/v1_task_neighbor_probe_20261002.json)。只读已交付`V1_CRT768_DEV_20261001`缓存：
148,695张当前阶段官方训练侧图像，448单中心、`ln_post_pre_projection`原生768，
视觉/LoRA为133,815 train_dev训练的DEV v1 SWA4–12；特征提取本身没有参数更新。
缓存SHA256`26230ec2f0236607ff72bfbea0ebbb7122425b304802ac20c7ea49825009228c`。
原普通768头checkpoint、原DEV父、缓存和sidecar、官方权重及当前split/映射摘要均核对。
原普通头14,880条448 native预测逐图重放一致，micro/macro76.5188/75.6027%。
val的14,658个内容组与train_dev无交集；缓存包含val，只用于无梯度query，绝不加入图库。

图库只用原train_dev：每个只有一个原标签的decoded内容组保留路径字典序第一张；
跨标签内容组全部排除，不决定哪一标签干净。133,815行/131,723组中排除1,705冲突组/3,527行，
另去掉270同标签重复行，固定图库130,018行、750类均覆盖。
这是诊断图库规则，不删除或重标任何训练数据，也不把余下原标签视作干净。

一个点：k16、每个内容组等票；各768向量在CPU转float64并归一化，GPU只做float64余弦检索。
cosine固定舍入9位小数，检索同分按图库路径升序；类别票数同分按固定class index升序。
query128/gallery8192分块，稳定排序合并出全图库top16；没有近似检索、温度或k扫描。
原普通768头固定为native参照，不依据本轮结果切换到均衡头或LR512参照。

先真实64 query成本检查，丢弃其检索输出；完整固定14,880 query，每张搜索相同130,018图库。
不以时长中止，数值/资产/资源错误封存、不自动重试。6项相关测试与CPU准备通过。
无需新图像编码、不训练、不生成test预测、提交候选或head/邻居融合预测。

## 结果前冻结的群体与门

全量、四候选共同错误、tail75、小图分别保留native与邻居macro/micro及修正/退化。
另报告至少12/16票支持、补集、支持且与native不同、共同错误内支持。
这些支持组只由邻居原标签票数/模型输出定义，不按val真值选图；判断正确仍用完整原标签。
高支持不证明标签干净，16个不同内容组也不证明16个不同真实来源。

进入后续投入复核须同时满足：支持且不同组native错≥200、修正≥200、净≥150、
修正≥1.5×退化，以及共同错误内支持修正≥100。
未达关闭这个固定任务邻居入口，不降一致票数、换k/权重/图库或派生参数网格；
通过也只支持机制与成本复核，不自动训练、恢复NDCW Drop/Relabel、teacher阈值、监督阶梯或full。
本段无融合预测文件，不以“用邻居覆盖head后的分数”冒充合规单checkpoint候选。

## 精确命令与交付

工作目录`/home/lux1/noise/worktrees/v1_task_neighbor_probe_20261002`：

```bash
env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -m pytest reproducibility/aegis_f1/tests/test_v1_task_neighbors.py -q
env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u scripts/probe_v1_task_neighbors.py prepare --config configs/v1_task_neighbor_probe_20261002.json --output outputs/codex/v1_task_neighbor_probe_20261002/probe_r1
env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u scripts/probe_v1_task_neighbors.py cost --config configs/v1_task_neighbor_probe_20261002.json --output outputs/codex/v1_task_neighbor_probe_20261002/probe_r1 --execute
env PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u scripts/probe_v1_task_neighbors.py run --config configs/v1_task_neighbor_probe_20261002.json --output outputs/codex/v1_task_neighbor_probe_20261002/probe_r1 --execute
```

原文件拒绝覆盖，重放使用新output路径。日志同方案`logs/{prepare,cost,run}_r1.log`。
逐张图库/邻居/票数NPZ保留在private输出；聚合报告、独立核验与本记录入Git。
独立核验将用NumPy64完整重搜seed42的64张随机query、重算全部238,080返回相似度、
14,880票数/组成员与完整指标配对。64张全图库重搜不能声称14,880张最优近邻均经独立CPU重搜。
验证后立刻方案commit/push，main自动autostash pull、合并、复验、push，并在检查点暂停。

本段为诊断交付，现役full v1 SWA仍为用户报告70.98600576861446%，不生成新提交包。
现役路径`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/{pred_results.csv,submission.zip}`；
ZIP SHA256`1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`，
[既有9项检查](../results/v1_full_swa_platform_20261001/submission_check.log)、
[包核验](../results/v1_full_swa_platform_20261001/artifact_verification.json)。
