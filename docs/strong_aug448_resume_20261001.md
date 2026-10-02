# STRONG_AUG448_20261001：磁盘中止后的授权恢复

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-01
- Verification Status: 工程/迁移已核验；两臂四轮与出包独立核验已完成，详见交付记录
- Version Label: strong_aug448_resume_v1

## 最新完成状态

本页保留恢复时的记录。两臂各四轮及两份CSV/ZIP于2026-10-02 04:05完成独立核验；
中心主评估强−弱净−51、目标净−7，原固定配方关闭。六视图辅助配对净+41，
完整指标、预注册门、包路径与校验见[最终交付记录](strong_aug448_delivery_20261002.md)。
后文「尚未完成」描述恢复启动时的历史状态，不代表当前仍在训练。

## 恢复范围和启动时结果

用户在获知磁盘中止及弱增强三轮结果后明确要求「继续」。任务仍为
[machine_b](team_exploration_20261001/machine_b.md)的弱增强与固定强增强配对，
每臂固定四轮；增强、原父、监督、抽样/Mixup/LR、EMA2–4平均和原止损/复核门保持。
没有墙钟截止，不派生配置、不自动full或平台上传。

旧运行在写逐批日志时出现`OSError: [Errno 28] No space left on device`。
弱增强已完整完成第1/2/3轮；第4轮有1,260条更新记录，但没有对应持久化权重，
强增强尚未开始。第3轮EMA验证macro75.60290205589666%、micro76.55913978494624%，
相对原父全量净+66、冻结低相似度目标组净−1；这是中间诊断，不是主平均候选结果。

可恢复检查点为`epoch_03.pt`：epoch3、updates11,166，scheduler last_epoch11,166/
total_steps14,888、EMA count11,166/decay0.999、算术平均count2（EMA2、3）。
raw、AdamW、scheduler、GradScaler、EMA和平均状态均存在，原SHA/sidecar绑定通过。
模型没有启用dropout；每图增强、每轮抽样、每批Mixup使用原独立确定性随机流。
恢复从第3轮末状态执行完整第4轮，旧第4轮未保存的1,260次更新只保留为中止证据，
不进入最终轨迹，不将恢复计为第5轮。强臂仍从共同原父执行四轮。

## 实际目录与复现

独立分支`codex/strong_aug448_resume_20261001`从`origin/main`的`9f763da`创建。
恢复worktree为`D:/codex_strong_aug448_20261001/worktree_resume`。
原字节冻结的实现仍从
`C:/Users/28639/Documents/New project 3/research_73/strong_aug448_20261001`
加载，避免Windows新checkout换行变化导致原plan代码SHA失效；原代码与资产未改写。

原中止运行：
`C:/Users/28639/Documents/New project 3/research_73/strong_aug448_20261001/outputs/codex/strong_aug448_20261001/run_20261001_2115`。
当前恢复运行：`D:/codex_strong_aug448_20261001/run_resume_20261001_2338`。
其中`original_abort_archive`为旧运行逐文件SHA一致的完整拷贝；旧目录保持只读。
新运行的plan、冻结组、原父重放、探针和epoch1–3均来自该拷贝，保留原绑定和sidecar。
`resume_manifest.json`另行绑定迁移列表、epoch3、恢复入口和经检查的循环源，
没有重写原checkpoint绑定。有效draws前缀仅11,166条，epoch4将重新追加。

```powershell
$python = 'D:/codex_b448_20260929/venv312/Scripts/python.exe'
$runDir = 'D:/codex_strong_aug448_20261001/run_resume_20261001_2338'
$env:TEMP = "$runDir/temp"
$env:TMP = "$runDir/temp"
$env:CUDA_CACHE_PATH = "$runDir/cuda_cache"
$env:PYTHONDONTWRITEBYTECODE = '1'
& $python scripts/strong_aug_resume.py run --manifest "$runDir/resume_manifest.json"
```

以上是本次已启动命令记录，不可对已运行目录再次执行。新恢复先以独立request执行
`scripts/strong_aug_resume.py prepare --request <request.json>`，需新D盘目录及明确继续授权。
本次启动为北京时间2026-10-01 23:37:29，launcher PID37860；实际训练PID以status为准。
实际训练PID14956，已观察到恢复后epoch4的462次有效更新。
与旧中止日志重叠的462批，indices/Mixup/LR/监督质量和AMP缩放/重算记录均逐批一致。
CUDA梯度范数不全位级一致，重叠区最大绝对差0.02472543716430664；
不声称CUDA位级轨迹重现，CPU完整状态/平均恢复的位级相同与此区分。
启动证据和manifest摘要见`results/strong_aug448_resume_20261001/launch_verification.json`。
输出、日志、TEMP/TMP及CUDA缓存放D盘；启动前D盘可用约327GB，不删除其他文件、
不更改系统分页设置。运行期间维持系统唤醒，退出恢复。

首次D盘启动于23:34:01，在资源预检因新Steam图形上下文退出，未进入训练、没有status。
其证据保存在`previous_preflight_attempt`；该目录不是另一次正式训练。
Windows CIM独立核对PID18332确为`D:/steam/bin/cef/cef.win64/steamwebhelper.exe`，
参数含`--type=gpu-process`。恢复预检仅认可这个明确PID/名称/路径/参数的界面上下文，
仍限制G/C+G并拒绝其他未知GPU任务，每次预检重新核对身份，不停止任何其他进程。

## 实现与验证

恢复入口不修改原训练模块。使用必须唯一匹配的源码锚点，只调整弱臂已有目录、
日志追加、epoch4范围并注入完整状态恢复；训练更新、增强、EMA、检查点保存和评估表达式
直接复用原函数。正式调度复用原两臂训练/出包/配对/判据/摘要收尾，仅以原冻结证据
替换重复父重放/分组/成本探针。恢复代码快照与适配后源码SHA均保存在新manifest。
强臂训练与两臂导出未改，最后调用原独立验证器。

18项相关测试通过（原配对12项、恢复6项）。CPU小模型实际运行原完整四轮循环，
模拟第4轮中断后恢复epoch3：末轮raw/EMA、EMA2–4平均、AdamW/scheduler/scaler、
全部逐批draws与不中断执行逐tensor完全一致；保留不可整除micro的最后逻辑批。
另核对错误边界、遗漏draw、源码漂移及仅单PID界面例外的拒绝行为。
正式checkpoint恢复与原绑定核验独立于toy测试；工程测试不证明CUDA位级重现或平台收益。

```powershell
$env:PYTHONPATH = 'C:/Users/28639/Documents/New project 3/research_73/strong_aug448_20261001/reproducibility/aegis_f1'
& $python -m pytest reproducibility/aegis_f1/tests/test_strong_aug_resume.py reproducibility/aegis_f1/tests/test_strong_aug_pair.py -q
```

当前仍须等待弱臂第4轮、强臂四轮、两包和独立复算。最终每臂14,888次有效更新，
主checkpoint为同一轨迹EMA2/3/4算术平均；各自单checkpoint固定六视图无bias出包，
37,444行CSV/ZIP各检查9项，独立复算完整配对、macro、分位数和draws一致性。
成功写`report.json`及`validation.json`；异常保存证据并退出，无自动重试。
动态阶段读取当前D盘`status.json`、`progress.json`、`run.log`、`run.err`。

现役仍为full v1 SWA，用户回填平台70.98600576861446%；未替换或上传现役包。
既有现役提交包路径及校验依据见[三机分工](three_machine_exploration_20261001.md#今晚平台名额)。
