# CLAUDE.md

本文件为 Claude Code 提供在本仓库工作的指引。

## 项目概述

面向噪声标签数据的细粒度图像分类。骨干固定为 CLIP ViT-B/32，权重限用 OpenAI 官方版本。这是比赛参赛项目，规则要求**单模型**交付：禁止任何形式的集成，禁止测试时训练或自适应。

**本文件只写阶段之间不变的东西。** 类别数、数据规模、当前最优成绩、下一步做什么，一律不写在这里 —— 见下方「当前阶段」。

## 当前阶段 ← 唯一随阶段更新的区块

**阶段切换时只改这一节。**

- 当前阶段、权威状态与下一步：[docs/current_execution_plan.md](docs/current_execution_plan.md)
- 本阶段详细的执行记录与固定配方：见该文件顶部「当前执行入口」指向的当轮文档

代码与超参数经验可以跨阶段迁移；**数据、checkpoint、特征缓存、伪标签、拟合出的 prior 一律不可跨阶段复用**。

## 代码布局

```
common/                      稳定骨架：数据集、配置/随机种子/日志、提交生成、PEFT、噪声鲁棒工具
experiments/                 早期「一个方法一个子目录」的实验骨架（现多为薄封装）
configs/                     实验 YAML
scripts/                     数据准备与提交校验，实验无关
reproducibility/aegis_f1/    当前主线实验框架（Aegis），绝大多数近期工作在这里
results/                     每轮实验的结果记录（json/csv/md）与提交登记表
docs/                        方案、预注册、执行记录
outputs/                     产物
```

**注意**：`reproducibility/aegis_f1/` 是本仓库现在真正的前沿，`common/` + `experiments/` 是上一阶段的骨架。读代码时先确认自己看的是哪一层。

Aegis 自带独立的 `pyproject.toml`、测试（`reproducibility/aegis_f1/tests/`）与 CLI（`reproducibility/aegis_f1/aegis_clip/cli/`）。所有入口都需要 `PYTHONPATH=reproducibility/aegis_f1`。

## 常用命令

全部在仓库根目录执行。

### 当前主线（Aegis）

```bash
# CLI 总入口；子命令：prepare / verify / cache / train / infer / run / record / prepare-full
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.rematch --help

# 完整命令序列以当轮执行文档为准：见 docs/current_execution_plan.md 顶部的「可重放命令」
```

### 提交校验

```bash
# 注意：--num-classes 或 --class-mapping 二者必给其一，否则脚本 sys.exit(2)
python3 scripts/check_submission.py \
  --test_dir <测试集目录> \
  --class-mapping <class_to_idx.json> \
  --csv <pred_results.csv> \
  --zip <submission.zip>
```

校验 9 项：文件名、行数、字段数、图片名存在、重复、覆盖、标签格式、标签范围、zip 内容。

### Aegis 测试

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -m pytest reproducibility/aegis_f1/tests -q
```

**已知失败**：`test_scope_protocol.py::test_scope_asset_gate_verifies_every_frozen_asset` 因上一阶段的 train/test 图像根目录已被删除而抛 `ScopePreflightError: train/test image root is missing`（2026-09-21 实测 1 failed / 4 passed）。判定"是否通过"时先确认失败项是否为这一类。

### 早期骨架（历史，仅在需要读旧代码时使用）

```bash
python3 scripts/check_data.py --config configs/<name>.yaml
python3 scripts/split_data.py --config configs/<name>.yaml
python3 -m experiments.baseline.train --config configs/<name>.yaml
python3 -m common.submission --raw <pred_raw.csv> --out_dir <目录>
```

这些命令依赖已删除的上一阶段数据资产，通常无法直接重放。

## 比赛硬约束

完整规则见 [COMPETITION_RULES_AGENT.md](COMPETITION_RULES_AGENT.md)。要点：

- 骨干固定 **CLIP ViT-B/32**，权重限 **OpenAI 官方**（代码内硬校验：`common/clip_utils.py` 的 `ALLOWED_MODEL_NAME` / `ALLOWED_PRETRAINED_SOURCE`）
- **禁止跨阶段复用**：数据、checkpoint、特征缓存、伪标签、原型、拟合的 prior 均不可带入新阶段
- **测试集只读**：禁止测试图入训、无监督自适应、TTT/梯度更新、用测试集预测分布调参
- **禁止集成**：多模型、多 checkpoint 投票、多 seed 平均、骨干融合、多头融合、logits/概率加权组合均不允许。最终结果 = 单个训练好的 checkpoint + 单份确定性推理脚本 + 一份 `pred_results.csv`
- 多尺度 + Flip TTA 已裁定合规（前提：单 checkpoint + 单确定性流程）
- 提交格式：`文件名, 0001`（逗号 + 空格 + 4 位补零），ZIP 内只含 `pred_results.csv` 且与外部字节一致
- 平台**每日提交次数有上限**，具体按当轮文档

项目内部自设门槛（贯穿所有实验）：**+0.30pp** 迭代投资回报门、**−2pp** 本地止损、**+0.20pp** 筛选晋级门。未达门槛的轮次一律关闭，不自动派生参数扫描。

## 关键设计细节

**CLIP dtype**：`conv1` 在 CUDA 上以 fp16 加载而其余参数为 fp32。`build_model()` 用 `.float()` 把视觉编码器统一为 float32，否则前向报 "Input type and weight type should be the same"。

**类别映射**：按目录名字典序排序 → 索引 `0..N-1`。`class_to_idx.json` / `idx_to_class.json` 由划分脚本生成，训练与推理共用。**永远不要硬编码类别索引或类别数。**

**提交格式**：见上文命令一节。

**样本权重接口**：`TrainImageDataset` 返回 `(image, label, image_path)`，并提供 `get_sample_weights()`。这是有意的 —— 噪声鲁棒方法可借 `image_path` 赋逐样本权重，无需改数据集代码。

**坏图处理**：`_safe_load_image()` 失败返回 `None`，调用方以零张量兜底，训练不中断。注意兜底张量尺寸是写死的，若改动输入分辨率需同步检查。

## 配置系统

YAML 由 `common/utils.py:load_config()` 加载。

主 schema 顶层键：`experiment` / `data` / `model` / `train` / `eval` / `output`（`common/config_schema.py` 对未知顶层键**fail-closed**）。

另有 `train` / `loss` / `mixup` / `sample_weighting` / `peft` / `head_ema` / `posthoc` / `cache` / `soft_targets` / `hyper_search` 等可选段。

**注意**：`configs/` 下同时存在**多套互不兼容的 schema**。部分文件（早期 `prelim75_v*.yaml`、`a2_lora_*.yaml`、`oof_cache.yaml`）使用扁平的 plan schema，路径带 `../` 前缀、假定不同的 cwd，且会被主 schema 校验拒绝。使用前先确认属于哪一套。

## 协作约定

本项目多人协作，以下约定用于避免重复劳动和丢失经验。**每个实验段都必须遵守。**

**开始新实验段之前：**

1. 确认当前分支、工作区状态、`origin/main` 是否为最新。混合工作区用 `git pull --rebase --autostash origin main`。
2. 检查全部本地与远端分支、以及近期 main 历史，看是否已有**重叠的实验名、算法、配置、结果报告或在途工作**。若有相似方向，先比对 / 协调 / 关闭，再决定是否动手，不要重复实现。

**实验段结束之后：**

3. **立刻提交并推送**已验证的改动到 `origin/main`，然后报告：实验标识、确切命令/配置、结果指标、改动文件、commit SHA、push 状态。**在该检查点暂停**，让协作者能复用结论或做淘汰比较。
4. 报告必须可复现，并**区分「实现改动」与「实测结果」**。没有验证输出前不得声称该段已完成。
5. 暂停前必须交付到**可提交产物**：生成预测 CSV/ZIP、跑完提交校验、记录路径与校验结果。只有代码或只有单测的段不算完成，要继续做完。

**Git 操作细则：**

- 直接在检出的 `main` 上工作，**不开 feature 分支**
- 混合工作区两种模式**不可混用**，下指令时必须说明用的是哪种：
  - 手动：`git stash push -u -m "..."` → `git pull --rebase origin main` → `git push origin main` → `git stash pop`
  - 自动：`git pull --rebase --autostash origin main`（Git 自动恢复临时 stash，**不要再执行一次 `git stash pop`**）
- **禁止 `git push --force`** 解决 non-fast-forward
- 已验证的推送不得推迟到后续实验段

## 历史

上一阶段（初赛）的完整过程记录在 [docs/current_execution_plan.md](docs/current_execution_plan.md) 的历史部分与 `results/`；阶段经验提炼见 [docs/lessons_learned.md](docs/lessons_learned.md)。

上一阶段的权重与二进制缓存已删除，不可重放。
