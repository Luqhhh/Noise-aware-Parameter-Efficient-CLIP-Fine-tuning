# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 指引位置

本仓库面向 agent 的工作指引**统一维护在 [CLAUDE.md](CLAUDE.md)**，不再在此处重复，以免两份文件互相漂移。请先阅读它，内容涵盖：

- 项目概述与**当前阶段**锚点（阶段相关事实只在那一个区块维护）
- 代码布局（注意 `reproducibility/aegis_f1/` 才是当前主线）
- 常用命令与提交校验
- 比赛硬约束
- 关键设计细节与配置系统
- 协作约定（Git 工作流 + 段边界纪律）

## 协作约定（摘要，以此处为准的完整版见 [CLAUDE.md § 协作约定](CLAUDE.md)）

每个实验段都必须遵守：

- **动手前**：确认分支、工作区状态、`origin/main` 是否最新；检查本地与远端全部分支及近期 main 历史，若已有重叠的实验名 / 算法 / 配置 / 结果报告 / 在途工作，先比对、协调或关闭，**不要重复实现**。
- **收尾时**：立刻提交并推送已验证的改动到 `origin/main`，报告实验标识、确切命令/配置、结果指标、改动文件、commit SHA、push 状态，**并在该检查点暂停**，让协作者能复用结论或做淘汰比较。
- 报告必须可复现，并区分「实现改动」与「实测结果」；没有验证输出前不得声称该段完成。
- 暂停前必须交付到可提交产物（预测 CSV/ZIP + 提交校验 + 路径记录）。只有代码或只有单测的段不算完成。
- 直接在 `main` 上工作，不开 feature 分支；**禁止 `git push --force`**；已验证的推送不得推迟到后续实验段。
- Git 工作流两种模式不可混用，下指令时必须说明是哪种：
  - 手动：`git stash push -u -m "..."` → `git pull --rebase origin main` → `git push origin main` → `git stash pop`
  - 自动：`git pull --rebase --autostash origin main`（Git 自动恢复临时 stash，**不要再执行一次 `git stash pop`**）
