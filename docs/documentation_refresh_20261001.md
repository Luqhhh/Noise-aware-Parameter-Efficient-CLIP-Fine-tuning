# DOCS_REFRESH_20261001：项目文档更新

用户要求更新项目内过时文档，README保持不变。本段只整理文档，未启动训练、推理或平台上传，
没有新候选或新实测分数。基线为已同步的`origin/main`：`f474ff17e6d48b3330d031861d0ba58d7119950c`。
开始前已检查全部本地/远端分支、worktree及近期main；此前09-29资源归档已合入，不重复实现旧清理。
WFT448/LR512现有工作目录不切分支、不修改，其未归档准备不写成训练结果。

## 修改范围

- 将[当前执行入口](current_execution_plan.md)整理为现行状态、平台结果、冻结项、资源和交付入口；
  整理前全文保存在[历史快照](history/execution_plan_before_docs_refresh_20261001.md)，只调整归档相对链接。
- 同步v1的固定SWA/full交付，更新v2现役包引用与v3完成状态；旧暂停/恢复页标明时间范围。
- 更新搜索预算为full SWA用户报告70.98600576861446%、约1,503张净修正缺口；
  L05的3,015张保留为历史参照，full含val的本地指标保留训练内诊断边界。
- 同步NDCW规模门关闭、SAM由用户停止及E6配对净−11；区分工程准备、授权、实测和平台分。
- 为旧团队分工、旧OOF/SAM计划及Aegis原协议加历史用途说明；
  初赛旧合规矩阵的CSV格式改为当前逗号后一个空格，并链接根规则。
- 限定历史经验的适用范围，避免把cosine head、增强、EMA或SWA历史负结果视为跨阶段禁令；
  修复A2平台结果文档乱码，保留四个原分数和包路径。
- 所有文件名包含README的已跟踪文件逐字节对照基线，保持不变；既有历史日志与比赛原始证据保留。

## 核验与证据

核验脚本检查修改范围、全部README字节、修改文档的相对路径/标题锚点、原执行入口归档完整性、
full SWA平台预算与v3配对数字、旧A2分数及Git空白错误。
指标来源为[full平台包复核](../results/v1_full_swa_platform_20261001/artifact_verification.json)和
[v3独立配对结果](../results/v3_final_delivery_20261001/comparison.json)，不通过文案推定新实验成绩。
记录见[本段校验](../results/docs_refresh_20261001/validation.json)，可在任意本仓库worktree根目录复核：

```bash
python3 results/docs_refresh_20261001/validate.py
git diff --check f474ff17e6d48b3330d031861d0ba58d7119950c HEAD
```

首次核验通过：32份Markdown文档、276处相对路径/锚点、15份README字节一致；
整理前执行入口全文还原一致，平台预算/v3配对数字及空白检查通过。

首次未提交编辑核验使用：

```bash
python3 results/docs_refresh_20261001/validate.py --include-working-tree \
  --output results/docs_refresh_20261001/validation.json
```

本段为纯文档修改，不运行模型测试或生成新包。现役可提交产物引用
`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/{pred_results.csv,submission.zip}`；
桌面为`/mnt/c/Users/lqh22/Desktop/v1_full_swa_submission.zip`。
既有37,444行、9项通过及ZIP内外CSV一致的依据为
[正式校验日志](../results/v1_full_swa_platform_20261001/submission_check.log)与上方包复核；本段没有重验包。

## Git交付

方案分支`codex/docs_refresh_20261001`，独立目录
`/home/lux1/noise/worktrees/docs_refresh_20261001`。
核验通过后提交并`git push -u origin codex/docs_refresh_20261001`。
main集成采用自动模式`git pull --rebase --autostash origin main`，随后合并、重新核验、推送；
Git自动恢复临时stash，不另行stash pop，保留原main工作区改动及未跟踪产物。
确切方案/集成SHA及推送状态在本次交付回复登记，完成后在文档检查点暂停。
