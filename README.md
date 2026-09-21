# Noise-Aware Parameter-Efficient CLIP Fine-Tuning

面向噪声标签数据的细粒度图像识别。初赛数据为 500 类、约 103K 训练图；2026-09-21 已将本地数据替换为复赛数据：**750 类、148,695 张训练图、37,444 张测试图**。数据来源、路径、完整性校验和使用前准备见[复赛数据集元信息](docs/rematch_dataset_20260921.md)。复赛数据准备、RM-LP、RM-FT 与 RM-LT 已完成；独立验证 FT macro **71.8362%** / micro **72.9032%**，LT macro **71.8655%** / micro **72.6949%**。提交包已校验并复制到 Windows 桌面 submission 目录；当前以策略选择为主，RM-FT 为正式 baseline、RM-LT 为采样对照；RM-FULL 已取消；见[当前复赛执行记录](docs/rematch750_execution_20260921.md)。下文既有实验结果属于历史数据阶段。

## 当前状态

**上面那段是本项目约定的状态落点：每完成一次交付就在原地更新它**（完成后做了什么、独立验证指标、是否有平台成绩）。阶段相关的一切都以它为准，不要在别处重复。

细节与下一步：

- **→ [docs/current_execution_plan.md](docs/current_execution_plan.md)** —— 当前执行入口 + 历史记录
- 当轮详细方案与固定配方：该文件顶部「当前执行入口」所指向的文档
- 提交登记表：[results/rematch_submission_registry.csv](results/rematch_submission_registry.csv)

本文件的其余部分描述阶段之间不变的内容。

## 快速上手

```bash
# 当前主线（Aegis 框架）的总入口
PYTHONPATH=reproducibility/aegis_f1 python3 -m aegis_clip.cli.rematch --help
# 子命令：prepare / verify / cache / train / infer / run / record / prepare-full
# 完整可重放命令序列见 docs/current_execution_plan.md 顶部所指向的当轮执行文档

# 提交校验（--num-classes 或 --class-mapping 二者必给其一）
python3 scripts/check_submission.py \
  --test_dir <测试集目录> --class-mapping <class_to_idx.json> \
  --csv <pred_results.csv> --zip <submission.zip>

# Aegis 测试
PYTHONPATH=reproducibility/aegis_f1 python3 -m pytest reproducibility/aegis_f1/tests -q
```

## 项目结构

```
common/                      稳定骨架：数据集、配置/随机种子/日志、提交生成、PEFT、噪声鲁棒工具
experiments/                 早期「一个方法一个子目录」的实验骨架（现多为薄封装）
configs/                     实验 YAML（注意：存在多套互不兼容的 schema，见 CLAUDE.md）
scripts/                     数据准备、提交校验、审计，实验无关
reproducibility/aegis_f1/    当前主线实验框架（Aegis），绝大多数近期工作在这里
results/                     每轮实验的结果记录与提交登记表
docs/                        方案、预注册、执行记录
outputs/                     产物（*.pt 与 cache/ 不入 git）
```

`common/` + `experiments/` 是上一阶段的骨架；`reproducibility/aegis_f1/` 是当前前沿，自带独立的 `pyproject.toml` 与测试套件。

## 比赛约束

完整规则见 [COMPETITION_RULES_AGENT.md](COMPETITION_RULES_AGENT.md)。要点：

- 骨干固定 **CLIP ViT-B/32**，权重限 **OpenAI 官方**
- **禁止跨阶段复用**数据、checkpoint、特征缓存、伪标签、拟合的 prior
- **测试集只读**：禁止测试图入训、TTT、用测试分布调参
- **禁止集成**：多模型 / 多 checkpoint / 多 seed / 多头融合均不允许。最终 = 单 checkpoint + 单确定性推理脚本 + 一份 `pred_results.csv`
- 多尺度 + Flip TTA 已裁定合规（单 checkpoint + 单确定性流程）
- 提交格式：`文件名, 0001`（逗号 + 空格 + 4 位补零），ZIP 只含 `pred_results.csv` 且内外字节一致

## 经验与方法论

上一阶段（初赛）跑了几十轮实验，有大量可迁移的结论 —— 哪些方向有效、哪些被证伪、以及若干会让人得出**错误结论**的陷阱（本地验证与平台反相关、单 seed 不可靠、split 谱系不一致会伪造 +8pp 假信号等）。

**→ [docs/lessons_learned.md](docs/lessons_learned.md)**

建议在任何一轮新实验开始前先读一遍，尤其是「陷阱」一节。

## 文档地图

| 文档 | 用途 |
|---|---|
| [docs/current_execution_plan.md](docs/current_execution_plan.md) | **当前状态权威入口** + 历史记录 |
| [docs/lessons_learned.md](docs/lessons_learned.md) | 可迁移经验与方法论教训 |
| [docs/README.md](docs/README.md) | 文档索引与结果状态约定 |
| [COMPETITION_RULES_AGENT.md](COMPETITION_RULES_AGENT.md) | 比赛规则全文 |
| [CLAUDE.md](CLAUDE.md) | 面向 agent 的工作指引（含协作约定） |

## Git 策略

- ✅ 跟踪：`.json` / `.csv` / `.log` / `.yaml` / `.md` 结果与记录文件
- ❌ 忽略：`.pt` 检查点、`cache/`、`train/` `train_dedup/` `test/`

协作约定（提交推送纪律、避免重复劳动）见 [CLAUDE.md § 协作约定](CLAUDE.md)。
