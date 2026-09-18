# R2 复现依赖补齐与合成完整链演练

源提交 `f87ba68052bd841fc75ac86c9b995fd192f58bbd`，main。运行 ID `20260911_reproduction_dependencies_r1`。
本段为用户要求继续 R2/R3 的工程工作，不启动新的正式初赛机制实验。

## 变更与原因

- 生产者必须属于依赖祖先；节点排序不再代替 depends_on 声明。
- operation 必须匹配实际 CLI，拒绝重复参数覆盖已登记参数。
- 初始化谱系检查读取的父 train/val CSV 也必须纳入输入绑定。
- 特征提取必须绑定配置，推理必须绑定 checkpoint 和所用 prior 文件；拒绝测试批内 prior 拟合参数。

保留合成 scope 限制。以上不是完整来源真实性验证，不解锁正式训练。

## 实际执行

新的输出目录运行 `aegis_clip.reproduction.run_recipe(recipe, repository_root)` 默认审计，
然后 `execute=True` 执行，再以 `execute=True, resume=True` 检查完成节点复用。
recipe 为 `outputs/stage_readiness/20260911_reproduction_dependencies_r1/recipe.json`；实际每节点 cwd/argv、输入/输出哈希记录在
`outputs/stage_readiness/20260911_reproduction_dependencies_r1/executed/reproduction_run.json`。三个节点均 checks_passed：
OpenAI 固定特征提取 → 现有训练 CLI 的合成 1 epoch → 现有推理 CLI。
这是 5 类、20 张训练源图像、5 张测试图像的合成软件演练；不把指标当正式模型精度。
记录执行耗时 21.14 秒，完成节点恢复检查通过。

合成预测包 `outputs/stage_readiness/20260911_reproduction_dependencies_r1/executed/submission/submission.zip`，提交检查退出码 0，
覆盖全部 5 张图像、标签范围 0000–0004，ZIP 仅含 CSV 且内外字节一致。
ZIP SHA-256 `cb29d8d92cb1c93977abbb42b3cb0526ecbab4d28987becd4fe2649a666a3b3f`。
此包不是官方候选，不复制替换桌面正式包，不上传平台。

## 验证与仍缺项

根套件 414 passed；最终 AEGIS 套件 415 passed，退出码均 0。
定向复现测试 11 passed。源 diff 空白检查通过。

历史 full-ft 配置 10 个输入字段中 9 个对应文件存在（包含重复引用）。
`features.tensor_path` 的 features.pt 缺失；其 paths/manifest 存在不等于完整缓存可用。
R3 初始化 checkpoint 和 trust 包可读，但完整生产谱系还没有证明。
冻结特征是特征蒸馏参考，不能与教师伪标签包混称。
完整私有清单为 `outputs/stage_readiness/20260911_reproduction_dependencies_r1/fullft_input_audit.json`。

下一段仍需补正式输入/官方初始化的完整绑定及真实生产命令闭合，再推进 R3 隔离和 R5 校准来源。
本段没有完成方案一或二整体；没有新增官方平台成绩。
变更为 reproduction.py、对应测试、复现入口说明及状态文档。本地提交，可供代码同步；未推送。

## 2026-09-18 存储状态更新

本文记录的合成链检查结果与哈希继续有效，但 `outputs/stage_readiness/20260911_reproduction_dependencies_r1/` 的本地演练输出在完整复现退出当前计划后已按用户要求清理，共 `1,054,713,716` bytes。该目录不含正式比赛候选；需要再次检查字节时必须重新运行合成演练，不能把文档中的哈希当作当前文件仍存在。
