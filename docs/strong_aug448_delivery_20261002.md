# STRONG_AUG448_20261001：固定四轮配对完成交付

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-02
- Verification Status: VERIFIED（训练、配对重计、两包检查完成；平台分未知）
- Version Label: strong_aug448_delivery_v1

## 完成状态与固定结论

机器B弱增强/固定强增强配对于北京时间2026-10-02 04:05完成独立核验。
两臂各完整四轮、14,888次有效更新；主权重均为同一轨迹EMA2/3/4算术平均。
448中心、无bias的冻结主评估下，强臂相对弱臂修正349/退化400/净−51，
低训练相似度目标组修正58/退化65/净−7；相对原父全量净+2、目标净−8。
六项预注册复核条件均未通过，结论为`closed_fixed_recipe`。
本固定强增强配方关闭，不扩full、续训或搜索强度；两包未上传平台。

固定六视图下强臂相对弱臂修正377/退化336/净+41，目标净+17。
这是原协议要求报告的辅助解码结果，与中心主评估方向不同，完整保留；
不以结果后改换主解码来替代原复核门，也不从本地结果声明平台提升。
即使只看这份六视图配对，全量+41、目标+17、修正/退化比1.1220仍低于
原+75/+25/1.25阈值；不作新的推进许可。

## 主平均权重的完整验证指标

验证人口14,880张、750类；下表百分数，中心解码为固定448px。
弱/强父为原DEV v1 SWA EMA4–12，448px、512维头，原父14,880张逐图重放一致。

|权重与解码|macro|micro|正确张数|
|---|---:|---:|---:|
|原父，448中心|75.164526|76.115591|11,326|
|弱，EMA2–4，448中心|75.529074|76.471774|11,379|
|强，EMA2–4，448中心|75.164924|76.129032|11,328|
|弱，EMA2–4，固定六视图|75.472655|76.485215|11,381|
|强，EMA2–4，固定六视图|75.797121|76.760753|11,422|

六视图为短边448/512/576各中心crop448及flip，单checkpoint sum logits、无bias。
中心/flip top1一致率为弱93.111557%、强92.869622%。

### 中心主评估逐图配对

|比较|组|人口|修正|退化|净修正|
|---|---|---:|---:|---:|---:|
|强−弱|全量|14,880|349|400|−51|
|强−弱|低相似度目标|1,543|58|65|−7|
|强−弱|补集|13,337|291|335|−44|
|强−弱|尾75类|1,003|28|27|+1|
|强−父|全量|14,880|350|348|+2|
|强−父|低相似度目标|1,543|55|63|−8|
|强−父|补集|13,337|295|285|+10|
|强−父|尾75类|1,003|27|23|+4|
|弱−父|全量|14,880|294|241|+53|
|弱−父|低相似度目标|1,543|46|47|−1|
|弱−父|补集|13,337|248|194|+54|
|弱−父|尾75类|1,003|19|16|+3|

强−弱修正/退化比0.8725，强−父1.0057，均低于1.25；全量未达到相对两者净+75，
目标未达到相对弱净+25，补集相对弱净下降。冻结分组阈值仍为0.8656789064407349，
分组SHA仍`d7fc83e50d3a69c68112d01d369ef0db5c84df823018c6165bd45e757b269e36`。
低相似度是train-only、排除相同decoded内容组的代理，不等于干净标签或真实来源迁移。
单seed、原含噪标签的有限配对不证明所有强增强都无效。

### 六视图辅助配对（强−弱）

|组|修正|退化|净修正|
|---|---:|---:|---:|
|全量|377|336|+41|
|低相似度目标|64|47|+17|
|补集|313|289|+24|
|尾75类|28|21|+7|

### 末轮诊断（不替换预注册主平均）

|权重|macro|micro|
|---|---:|---:|
|弱末轮raw|75.465542|76.411290|
|弱末轮EMA|75.501020|76.438172|
|强末轮raw|75.345537|76.344086|
|强末轮EMA|75.319366|76.310484|

## 产物、核验和复现

实际输出根：`D:/codex_strong_aug448_20261001/run_resume_20261001_2338`。
每臂的主权重为`arms/<arm>/ema_swa_2_4.pt`；CSV/ZIP在`arms/<arm>/submission/`。
`<arm>`弱为`original_augmentation`，强为`strong_augmentation`。
两包各37,444行、9项提交检查与独立重复检查通过，ZIP内仅CSV且与外部CSV字节一致。
原父、监督、plan、分组绑定未重写，test不入训。

|臂|checkpoint SHA256|CSV SHA256|ZIP SHA256|
|---|---|---|---|
|弱|`9b14d6314835dc36b415dc03ddaee2596d1866a87f110c2b8d78d013706790cd`|`690be1d7dd4d40a639ca4946c67325f866b14ec867f4d9c3a7387a1f29f94343`|`e08785c90fc9e4025ce003775300331de15bd2a1e9a9b7521749b1f068dff981`|
|强|`39ad62b514d117e03f2ff0e30d31fd6182792dc750ebc17d55390b8b0b3766db`|`13905096017cd8daeaa9eb257e8c4be8ee05501c141bcb1c9d9580d7bfe9ef98`|`4bc500233740a715c11bd1fb3fa99325abd422d07d0df26502e4d8fc7f904567`|

两臂14,888条draws的indices/permutation/Mixup/LR/监督质量逐批完全一致。
原独立验证器核对报告产物SHA、训练分位数、冻结成员、预测路径/标签/logit argmax、
完整修正/退化/macro、每臂四轮和两包9项检查，写出`validation.json=completed_verified`。
交付时另核对59,520条中心/六视图logits与预测、两种解码配对、两份checkpoint/CSV/ZIP SHA，
并检查ZIP/CSV字节。未另跑训练或搜索解码。

完整汇总证据入库`results/strong_aug448_resume_20261001/delivery/`：
原字节report、独立validation、plan、恢复manifest、两臂history/diagnostics/status/sidecar、
出包manifest/检查日志、追加`delivery_verification.json`及复制文件摘要。
其中`report.json`的status仍为独立核验之前的原快照，最终状态以独立validation/status为准。
`.gitattributes`保留复制证据原字节；checkpoint、CSV/ZIP、draws和大logits仍在上述独立D盘输出，
未将图片、模型或大缓存加入Git。

确切恢复命令、环境和冻结配方见[恢复记录](strong_aug448_resume_20261001.md)及
[原执行记录](strong_aug448_execution_20261001.md)。原独立验证命令为：

```powershell
$env:PYTHONPATH = 'C:/Users/28639/Documents/New project 3/research_73/strong_aug448_20261001/reproducibility/aegis_f1'
& 'D:/codex_b448_20260929/venv312/Scripts/python.exe' 'C:/Users/28639/Documents/New project 3/research_73/strong_aug448_20261001/scripts/verify_strong_aug_pair.py' --plan 'D:/codex_strong_aug448_20261001/run_resume_20261001_2338/plan.json'
```

交付后的只读重计入口（不重新训练或改写原完成证据）：

```powershell
& 'D:/codex_b448_20260929/venv312/Scripts/python.exe' scripts/verify_strong_aug_delivery.py --delivery results/strong_aug448_resume_20261001/delivery
```

独立方案分支仍为`codex/strong_aug448_resume_20261001`；恢复入口提交`9894e5e`已集成，
本次只新增最终交付证据、只读复核入口及当前状态，不改变训练引擎。

弱臂在C盘第4轮日志磁盘满后，按用户「继续」从epoch3恢复；旧1,260条未保存更新只归档。
首次D盘预检因Steam界面上下文退出，未进入训练；经明确PID身份核验的下一次启动完成本次运行。
恢复尝试和原中止日志保留，不把重算更新计作额外轮次。
本次D盘运行到报告16031.266秒（4小时27分11秒），包括弱第4轮、强四轮、评估和两包，
不包括旧C盘前三轮、丢失的部分第4轮及迁移准备；不设置墙钟截止。
18项配对/恢复工程测试已在方案和main集成时通过，CPU恢复位级一致不代表CUDA位级一致。

交付后停止本固定实验。当前平台最高与其他队员任务按
[当前执行入口](current_execution_plan.md)维护；本段平台分未知，没有替换或上传任何包。
