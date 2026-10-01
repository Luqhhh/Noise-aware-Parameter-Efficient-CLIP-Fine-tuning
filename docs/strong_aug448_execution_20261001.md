# STRONG_AUG448_20261001 本机执行记录

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-01
- Verification Status: UNVERIFIED（正式两臂尚在运行；工程与预检已核验）
- Version Label: strong_aug448_execution_v1

## 最新恢复状态

本文件保留初次执行记录。后续弱增强第1–3轮完整完成，第4轮写draws时磁盘满中止；
用户已明确「继续」。现从完整epoch3在D盘恢复第4轮，再接强增强四轮，
原绑定/冻结协议保持，未保存的旧第4轮更新仅归档。
命令、实际目录、18项恢复验证及最新动态入口见[授权恢复记录](strong_aug448_resume_20261001.md)。

## 授权与初次执行状态

用户在本会话指定机器B的弱增强与固定强增强配对，**每臂固定四轮**。
执行范围沿用[machine_b任务单](team_exploration_20261001/machine_b.md)与
[冻结协议](../configs/team_exploration_20261001/machine_b.json)：不设墙钟截止，质量止损保留。
本文件区分工程交付、零更新重放、探针与正式结果；没有训练结果时不声称候选完成。

独立分支为`codex/strong_aug448_20261001`，起点`58d3532`。
本机为RTX 4060 Laptop 8GB，使用现有独立Python 3.12 / PyTorch 2.6.0+cu126 /
torchvision 0.21.0+cu126环境。Windows WDDM将桌面应用显示为C+G上下文，入口核对PID与
本机进程名，记录已知桌面应用；遇到其他/未知GPU工作即退出，未停止任何其他进程。

## 资产和迁移核验

交接包`D:/2863949663/team_exploration_assets_20261001.zip`中26份白名单文件的字节数、
SHA256、原manifest均通过Git白名单核验，ZIP SHA256为
`35396b7b6482b2fd3cbf2cf8f807bf6744edabeda67f79948db8aa88950dc5cd`。

父为原train_dev v1 SWA EMA4–12，448px/512维头；原checkpoint与sidecar保留，
SHA256 `01caedaa4996d25fa9dfc213d4b2376c147bb99ebdc23dacd93e6e1ff2d6fa20`。
监督目标原SHA256为`7df09ad425164707cf1f2a55b9c24c45054d268cbd0be98bd6e8d3e29c4c60b2`。
原133,815张train_dev中119,074张有效样本保持，验证14,880张；原标签、可靠性权重、
软目标、类别映射均未修改。逻辑batch32，最后批保留；每臂每轮3,722次更新，四轮14,888次。

本机`D:/codex_b448_20260929`中数据manifest与原机manifest仅图像根路径不同，
全部split/decoded内容组清单摘要一致。官方初始化文件名为`ViT-B-32-hf.pt`，实际字节
严格匹配OpenAI官方SHA256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`。
同目录另一个`ViT-B-32.pt`的摘要不匹配，未使用。

本机224px/512维缓存独立编码，tensor摘要
`c28554017af7852d7050d11aa66a97ec6e0e8e464796faa2ee92b2a44d2cdc1f`；它与原机缓存字节不同。
入口验证其独立的当前阶段数据/官方权重/精度/预处理/路径顺序绑定，单独记录local location map，
从不将本机缓存的摘要写入原父绑定。父模型由交接包中真实历史classifier构造；当前主线增加的
pre-projection选项不参与。复用的原算法、权重平均、数据读取和提交组件进行AST等价核对。

## 独立入口与固定行为

新增`reproducibility/aegis_f1/strong_aug_pair`，旧v1/continuation schema和共享transform未改。
control为RRC[0.8,1.0]/flip/RA2,7；candidate为RRC[0.35,1.0]/flip/
ColorJitter(0.5,0.5,0.5,0.125)/RA2,7/ToTensor/Normalize/RandomErasing(p=0.3,value=random)。
两臂共享有效人口、逻辑batch、每轮抽样顺序、Mixup、监督质量、LR轨迹。每张图每轮使用
局部增强随机作用域，抽样和Mixup用独立随机流；保存全部更新的逐批记录供两臂复算。

全量零更新448中心预测须与交接包的原DEV父512维头逐图完全一致，六视图不替代此门。
本机首次全量重放已通过14,880/14,880逐图一致，178.469秒，macro75.1645263756151%、
micro76.11559139784946%，正确11,326张。增强配对正式训练当时尚未启动。
低训练相似度分组仅用train_dev gallery，训练查询排除相同decoded内容组，
采用float32分块最大cosine与numpy linear第10百分位，val严格小于阈值为目标组。
所有train/val标记和SHA在正式训练前冻结；目标组父错误不足75张即关闭固定机制门。
本次阈值0.8656789064407349，目标1,543张、父错分653张，补集13,337张，尾75类1,003张。
分组文件共148,695行，SHA256 `d7fc83e50d3a69c68112d01d369ef0db5c84df823018c6165bd45e757b269e36`。
分块计算及落盘共6.141秒，全部分位数/逐图标记/父错分数已用独立计算复核，未加入val到gallery。

两臂各做5次真实逻辑更新和完整验证的成本探针；保留探针证据、丢弃探针权重，
正式训练从原父冷加载。成本包含完整验证、存盘、两臂六视图出包和检查；不按耗时取消。
两臂成本探针均已通过5次更新，峰值allocated显存分别2,233,984,512与1,874,206,208字节；
平均更新计算0.478/0.4686秒，完整验证153.812/186.141秒。五步墙钟均值还包括worker启动，
因此cost.json中的30.25小时外推是有明显启动开销的保守记录，不能当稳态耗时或时间截止。
正式两臂已由原父重新冷加载启动，当前先执行control，candidate随后执行；尚无完整四轮结果。
训练复用完整逻辑批质量归一化与已验证AMP同批重算，四轮EMA2–4为唯一主导出。
原−2pp/数值/资产质量止损保留；中止不自动恢复或冒充完整候选。

主判据为相对control与父均净+75，目标相对control净+25，修正/退化比均≥1.25；
“其他大组不集中退化”在plan中明确冻结为补集和尾75类相对control/父均净不下降。
达到只记录supports_review；不自动full、派生配方、平台上传。各臂独立冷加载单checkpoint，
报告448中心/六视图、flip一致性、末轮raw/EMA诊断，固定六视图无bias各出CSV/ZIP并检查9项。
独立脚本重计分位数、逐图修正/退化、macro、抽样/Mixup/LR和两包校验。

## 复现与进度

在独立worktree根目录、PowerShell、现有本机环境中：

```powershell
$env:PYTHONPATH = 'reproducibility/aegis_f1'
$python = 'D:/codex_b448_20260929/venv312/Scripts/python.exe'
& $python -m strong_aug_pair.runtime prepare --locations outputs/codex/strong_aug448_20261001/locations.json --output outputs/codex/strong_aug448_20261001/<new_run_id>
& $python scripts/run_strong_aug_pair.py --plan outputs/codex/strong_aug448_20261001/<new_run_id>/plan.json
```

本次唯一实际启动的目录为
`C:/Users/28639/Documents/New project 3/research_73/strong_aug448_20261001/outputs/codex/strong_aug448_20261001/run_20261001_2115/`。
此前`run_20261001_2100`与`run_20261001_2110`只生成过准备plan，未启动探针/训练，已被新版计划替代。
runner在本次运行存活期间保持本机系统唤醒，退出恢复；没有关闭桌面应用或使用远端资源。
`status.json`为阶段状态，`progress.json`为逐批进度，`run.log/run.err`为日志；
成功后有两臂权重/包、`report.json`、独立`validation.json`；失败则保存中止证据。

## 工程验证及环境限制

新增12项有意义的测试通过，覆盖固定增强、val确定性、增强RNG隔离、乱序取图重放、
完整逻辑batch与不整除micro累积梯度一致、内容组排除、train-only gallery、配对计数、
四轮/无时间截止门以及判据；CPU小模型还完整执行实际四轮优化器循环，逐tensor核对主导出
等于第2/3/4轮EMA算术平均，并核对最后不整除批保留。复用v1的21项和continuation的16项通过。

原v1的一项符号链接测试在Windows缺少创建链接权限（WinError1314）；
原continuation的一项toy测试用numpy默认int32标签导致one_hot报错。
这些限制未改写共享组件或旧测试；实际本机监督labels/targets均核对为int64。
工程测试不表示正式四轮已完成或产生平台收益。

现役仍为full v1 SWA、用户回填平台70.98600576861446%。本机本段没有替换或上传现役包。
现役既有提交包与检查依据沿用[三机分工](three_machine_exploration_20261001.md#今晚平台名额)。

## 仓库集成

GitHub已明确提示团队仓库迁移到`https://github.com/Luqhhh/cli.git`，本地origin已改为该地址。
旧长名称是此前remote配置，不是另一个实验仓库。
工程及父重放检查点已推送方案`a85511b`，独立main集成目录使用自动autostash模式pull，
合并后12项新增测试再次通过，推送main为`39975b29afd475ba07822f6bb96da4f1e5e0608a`。
未切换或修改其他队员的实验目录。正式结果仍需等两臂四轮、出包和独立复算。
