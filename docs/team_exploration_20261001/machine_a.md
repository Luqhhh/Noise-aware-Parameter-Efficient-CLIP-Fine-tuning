# 机器A执行768头后继续LoRA

把本文件交给本机agent执行。用户已授权三台本地机器各自拉代码开展本轮探索；本机承担A，租用4090上的v2保持原固定运行。先读仓库CLAUDE.md、当前执行入口与[三机分工](../three_machine_exploration_20261001.md)。本任务是一次固定四轮的实现与配对执行，不设时间上限，已有768 full不重训；不要另行等待一般性开训确认。

## 接手与输入

先fetch、检查全部分支及重叠工作，从最新origin/main建立`实际成员名/lp_lora768_20261001`分支和独立worktree。协议为`configs/team_exploration_20261001/machine_a.json`，性质为implementation_required；不能直接塞给旧CLI。

交接包先verify，再读`assets/a_parent/selected.pt`与原sidecar。这是普通CRT768末轮头，视觉塔只在train_dev训练，768头使用同一train_dev监督。父模型为frozen_visual_head_last20，48处LoRA，768→750 cosine头；使用`assets/common/targets.pt`和原sidecar，共119,074条正权重。不要使用balanced头、full768、full v1或重新做teacher/伪标签。

原始路径与确切SHA在asset_sources.json和ZIP内manifest。重放现有普通768头14,880张中心视图，预测应与入库`results/v1_post768_close_20261001/closure_verification.json`和CRT报告一致，macro75.6027%、micro76.5188%；以原逐图预测为严格重放依据，不能只比四舍五入总分。数据、类别、split、权重不一致就先解决输入，禁止放宽门禁。

## 唯一配对干预

两臂从**同一个已拟合普通768头+同一LoRA父权重**初始化：control只更新头，candidate更新原有全12块rank32/alpha64 LoRA和头。其他视觉权重均冻结；唯一干预是是否继续视觉LoRA学习。两臂保留同一v1原标签/伪标签、权重、LS0.1、Mixup0.2、RRC下界0.8、flip及RA2/7；不均衡重采样、不引入额外监督。

两臂各固定4轮，逻辑batch32/micro8/worker2/seed42，AdamW LoRA LR5e-5、head LR2.5e-4；LoRA WD0、head WD0.01，OneCycle pct_start0.1/cos，grad clip1，EMA0.999。control没有LoRA优化组。保存完整独立模型状态，包括control中保持冻结的LoRA；EMA/平均导出不能漏掉这些参数。主输出固定EMA2–4单轨迹平均，末轮raw/EMA作诊断，不选最佳轮。

共同冻结人口顺序、逻辑batch和Mixup随机数。分离采样、增强、Mixup与模型dropout RNG，不能因control少消费视觉随机数而改变后续batch。首128图与前5个逻辑更新分别测CUDA、完整验证和存盘成本，只记录显存、吞吐与预计投入，不设置时长门或超时终止。探针不算正式轮，开训重新载入原父权重。设备在忙时不抢任务，记录状态并等资源；资产、数值或显存预检失败则交付原因，不改轮数。

## 目标问题与评价

假设：768头已经拟合后，视觉LoRA还能改善细粒度混淆，并超过仅继续拟合头的收益。已用未训练父模型的入库中心预测冻结前10个无序混淆类对，见`results/three_machine_exploration_20261001/a_target_groups.json`；对各机本地重放核对该排序、成员与SHA，不重新选择类对。目标组全体417张、父原标签错误173张，仅为诊断预算；它们是诊断代理，不是语义真值。完整结果产生前冻结分组，不能看candidate再选类对。

报告父、control、candidate的全量macro/micro；candidate分别相对父和control的修正/退化；前10对涉及类别的全部样本、尾75类、其余样本及分组重叠。目标组预算用**父模型原标签错误数**，不把train数量或top-k命中数叫可恢复错误。目标组若小于75张父错误，保留总体验证但关闭这次主机制投入门。

主输出达到分工文档的净+75/目标净+25/修正退化比1.25等条件，仅标supports_review；否则关闭固定配方。论文的LP后微调结论不能充当本轮实际收益；候选相对head-only的改善才检验继续LoRA，不能声称证明768普遍优于512。

## 实现与交付

复用`aegis_clip.v1_strategy`的classifier、weighted_mixup_loss和WeightAverage，以及`v1_continuation.runtime`经过验证的逻辑batch累积/AMP溢出重试。新增本方案入口，不修改旧WFT/LR512冻结schema或父artifact字节。Windows用sys.executable、Path与POSIX清单键；不更改共享环境。

开训前完成真实父权重零更新重放、control视觉参数不变、candidate确有LoRA更新、逻辑batch Mixup/可靠质量等价、冷加载单模型输出检查。测试须覆盖这些实际风险，不用玩具测试冒充真实父重放。

正常完成两臂后，各用单checkpoint固定六视图解码，无bias；各生成CSV/ZIP并跑scripts/check_submission.py的9项检查，保存37,444行、内外CSV字节一致和SHA。主评价始终是独立val中心视图，不用测试预测调参。输出、日志、checkpoint放`outputs/实际成员名/lp_lora768_20261001/唯一run_id/`。

独立复算全量与分组配对，记录确切命令/配置、指标、文件、分支与commit；立即推送方案，在main集成目录先自动模式pull --rebase --autostash、合并、重新校验、push，随后暂停。不自动启动full、重建768监督、第二seed、窗口或LR搜索；不平台上传，不启动子agent。
