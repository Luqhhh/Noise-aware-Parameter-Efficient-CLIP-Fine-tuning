# 机器B执行固定强增强配对

本机独立入口、资产迁移核验和实际进度见[执行记录](../strong_aug448_execution_20261001.md)。
原DEV父14,880张448中心预测已逐图重放一致；工程交付不等于正式四轮完成。

把本文件交给本机agent执行。用户已授权三台本地机器各自拉代码探索，本机承担B；租用4090继续现有v2。先读CLAUDE.md、当前入口及[三机分工](../three_machine_exploration_20261001.md)。本轮只比较固定增强配方，不恢复WFT、监督阶梯或旧L05风格增强。收益未知，具备输入、复现和数值条件后自行实施这个有限配对。

## 接手与父权重

fetch并检查全部分支、main近期历史，从最新origin/main新建`实际成员名/strong_aug448_20261001`分支及独立worktree。协议为`configs/team_exploration_20261001/machine_b.json`，需要本机agent实现独立入口；不是旧CLI可直接执行的配置。

先verify交接包。仅用`assets/common/dev512/selected.pt`及原sidecar、`assets/common/targets.pt`：train_dev v1 SWA EMA4–12、448输入、官方ViT-B/32、全12块rank32/alpha64 LoRA、512 cosine头。不能以看过val的full v1/full768作父，不借用A的训练结果。

移植时保留原artifact binding并建立新location map。用原父模型全量val逐图重放原DEV父512中心结果，macro75.1645%、micro76.1156%，不能拿六视图父结果代替中心结果；原预测来源见CRT报告和post768关闭核对。父模型构造与官方权重校验复用已验证代码。

## 固定两种增强

两臂同一父、同一监督/权重、同一采样、逻辑batch和优化器。control用原v1的RRC[0.8,1.0]、水平flip、RA2/7；candidate用RRC[0.35,1.0]、水平flip、ColorJitter亮度/对比度/饱和度0.5、hue0.125、RA2/7、RandomErasing概率0.3。固定顺序为RRC、flip、ColorJitter、RA、ToTensor、Normalize、RandomErasing；RRC比例范围仍为[3/4,4/3]、插值BICUBIC，Erasing填充值为random，其他参数保持torchvision默认并记录版本。Mixup0.2、LS0.1保持两臂相同，不加入CutMix、dropout改动、采样均衡或标签变更。

这是一个整体增强配方对照，不能将收益单独归因于crop/Jitter/Erasing。两臂各4轮，448输入、LoRA LR5e-5、head LR2.5e-4，AdamW LoRA WD0/head WD0.01，OneCycle pct_start0.1/cos、EMA0.999、grad clip1；逻辑batch32/micro8/worker2、seed42。主输出固定EMA2–4算术平均，末轮raw/EMA仅诊断。分离增强与采样/Mixup RNG，确保强增强多消费随机数不会改变后续样本和混合参数。

## 先冻结本干预的代理分组

假设：更强的训练扰动能减少对训练外观相似性的依赖，改善低训练相似度样本；也可能裁掉目标或放大标签噪声。WFT的平台下降尚不能证明来源漂移，本分组不能称测试替身或真实来源。

仅为本次新干预复用官方当前阶段已审计224px/512缓存，不扩旧内容审计。train_dev每张图对其他decoded内容组的train_dev取最大cosine，固定训练分布第10百分位为阈值；val对train_dev取最大cosine，低于阈值为目标组。gallery中不得包含val，不按candidate或测试图分布改阈值。优先复用已有合法近邻结果，否则一次分块计算并记录吞吐，不设计算时间上限；缺合法缓存就交付该机制未知，不重新编码全数据或改组。

保存阈值、缓存摘要、算法/分位数实现、全部逐图group标记及SHA，在任何正式训练输出前冻结。目标组父原标签错误若小于75张，关闭本次强增强机制门，不放宽10%阈值凑数。这是成块误差存在的探索入口，不证明能够修正75张。

前5个真实逻辑更新与完整验证测两臂成本，计入分组、存盘、六视图两包及9项校验的预计投入；不设时间上限、不因耗时中止。探针后丢弃权重，正式两臂重新从原父开始。不自行删减或追加轮数；不抢占已有CUDA任务。

## 结果与交付

完整14,880张448中心无bias报告macro/micro、父与control与candidate逐图结果。分别计算candidate对control及父的总体/目标组/补集/尾75类修正和退化，列重叠；另报训练前后同图flip一致性作为诊断，不把一致性改善当识别收益。主输出达到分工文档净+75、目标净+25和修正/退化比条件时仅supports_review；总体涨但目标组不改善，也不能支持该泛化假设。正式结果之前冻结所有判据。

代码复用v1 classifier、weighted_mixup_loss、WeightAverage与已验证的micro累积/AMP溢出逻辑。新增独立recipe/入口，保持旧计划schema及旧绑定；不直接修改共享image_transform默认行为影响别人的任务。验证增强只作用train、val不随机、共享人口/Mixup一致、父重放、全逻辑batch的监督质量、两臂冷加载单checkpoint。

正常四轮完成后两臂各自用固定448/512/576和flip六视图、无bias出CSV/ZIP，跑9项检查与独立配对复算。输出放`outputs/实际成员名/strong_aug448_20261001/唯一run_id/`。数值/资产错误或总体相对父下降超过2pp停止，保留中止证据，不伪称完成候选。

立即提交推送方案；main集成采用自动autostash模式，pull之后合并、重新校验再push；报告命令、结果、产物路径、分支/SHA与合并状态，在该检查点暂停。不自动full、增强强度/组件网格、监督改造、多seed、平台上传或子agent。
