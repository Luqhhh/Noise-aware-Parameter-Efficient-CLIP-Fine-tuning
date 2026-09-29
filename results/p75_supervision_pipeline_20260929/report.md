# P75_SUPERVISION_PIPELINE 工程检查点

实验状态 **proposal**；工具实现 **engineering_ready**。未启动实验训练/续训/平台上传。

- 针对性工程测试：21 passed。覆盖 CE/GCE 全局/局部分类屏蔽、anchor 保留、未命中梯度与原分母、相同 LP 初始化/末端、原 cosine 下限、授权绑定、超时标记、固定分组、E6选模副本不改变已评估checkpoint。
- 两臂实际 E4 输入：epoch4、524 次更新；模型/优化器/scheduler/AMP/RNG/generator 完整，与归档已评估 E4 的完整状态一致。原框架恢复协议检查两臂通过。恢复复制是字节一致的新目录输入，原文件和旧预算未改。
- 冻结 train mask 1,246 张/423类；750类均保留样本。逐类剩余内容组见 class_coverage.json。val132/其余14748，23类419张，其中B主导332张。
- A 还缺新授权和 local真实步时/完整验证成本；B还缺新授权和已有绑定flip特征。prepare没有借探测之名启动训练或重编码。
- A/B配对表为待运行模板，识别指标空值；没有新accuracy、完整视觉候选、CSV/ZIP或平台分数。旧E4+8不冒充新段结果。

复现命令与所有冻结边界见 ../../docs/p75_supervision_pipeline_20260929.md。manifest.json绑定源码、配置、输入、准备目录；archive_sha256.json绑定本记录。工程测试仅用小型合成张量，不占实验训练预算。历史现役包及校验引用见任务文档，本段未重验。
