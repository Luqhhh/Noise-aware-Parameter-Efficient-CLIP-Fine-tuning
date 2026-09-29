# P75_SUPERVISION_PIPELINE：独立工程交付，训练仍为 proposal

用户任务单的 `proposal_only` 边界有效。本段只实现和验证工程：没有启动 A/B 实验训练、续训旧任务、改变旧预算或上传平台。`engineering_ready` 只描述工具代码；实验状态仍为 `proposal`。没有新模型识别结果、可提交包或平台成绩。

## 冻结问题与边界

阶段 20260921，750 类，133,815 train_dev / 14,880 val_dev / 37,444 test。现役 L05_T14_P060 平台 66.94797564362783%，25,068 张正确；75% 需 28,083 张，缺口净 3,015。绝对本地—平台差不是固定换算，也不证明本地排序完全无效。

H1：冻结内容代理上的错误类别监督可能在 local 启用后损伤其他图识别。H2：从 LP 开始屏蔽可能优于只在 FT 屏蔽。均待检验；1,246 张、423 类和最终 logit 梯度份额不是测试收益上限。代理不等于已确认噪声。

沿用 `sN-max(sB,sA)>=0.05`、生物标本/解剖/绘图保护和原始标签，不加名单、外部图片或阈值扫描。剩余 132,569 张保持原监督。不恢复 HARD_SUPPORT、SUPPORTED_CE、旧 R1/R2 或 L05 邻域扫描，不新增拒识类、均匀目标、OE、伪标签、噪声转移矩阵。SAM 只受其原授权约束，本段未操作。

## 已实现入口

- `scripts/p75_pipeline.py prepare|verify`：固定源码物化到新目录，恢复输入核验、原配置血缘、分组冻结、每类剩余样本/内容组核对、A/B 独立 manifest、空指标配对表、结论模板。拒绝覆盖已有目录。
- `scripts/run_p75_pipeline_a.py`：显式授权后才可能运行的 A 调度器。E4→E5 两臂，然后 E5→E6 两臂；每个边界恢复两臂，记录顺序摘要、成功/尝试更新、LR、全局和局部实际命中。使用独立源码副本，不修改共享训练器。
- `scripts/p75_pipeline_lp.py`：CPU 缓存分类头配对。读取真实 RM-LP 父 checkpoint 的配置，保留 20 轮、batch256、AdamW(lr=.005, wd=.0001)、原 cosine 的 0.01 下限和梯度裁剪。两头相同随机初始化、相同 batch 顺序和更新数；只把选中图 CE 置零，除以原 batch 大小。保持 750 输出。CPU 使用 float32，关闭 AMP/worker，明确不声称复现旧硬件逐位轨迹。共同比较训练预算末端 E20，而不是旧收敛头对新头；每臂保存完整最终头状态及原始初始化。
- `scripts/p75_pipeline_report.py`：验证缓存与 checkpoint 绑定、各自拟合 prior、固定解码、全量与重叠分组、F/D/净修正和各组逐类结果、保守投入结论。

两个训练入口都要求独立授权 JSON，绑定本次主 manifest 的 SHA256、实验 ID、新预算及用户明确指令。当前不生成 `approved=true` 授权。这个入口用于执行将来的明确授权，不把代码存在当作授权。没有 C1/C2 自动训练或上传入口。

## A：E6 有限配对

输入为经验证的各臂 E4 `last.pt`。核对其与旧报告绑定的 `epoch_4.pt` 的完整状态一致，而非假设两个独立 torch 序列化文件的字节摘要相同。保留原始 checkpoint 和 sidecar；新目录内复制字节一致的 E4 输入，单独记录续接配置绑定、原配置 SHA 和原 checkpoint 路径。仅允许实验标识、输出路径和字节相同的选择表路径变化，训练配方不同则拒绝。

A0 原监督；A1 全局/局部分类屏蔽。仍为 384px、4×256、有效 batch1024、16 轮 LR 日程、GCE q=.5、feature anchor2、E5 local、confidence gate=.7。不是 6 轮 cosine。E5 保存完整状态及分组训练统计，不做完整验证；E6 统一生成 center+flip 验证缓存及报告。原 E4 归档与预算不变。

恢复含模型、AdamW、scheduler、AMP、Python/NumPy/Torch/CUDA RNG、数据 generator。DataLoader worker 状态无法恢复，双方使用相同进程边界；不声称与连续旧 L05 视图逐位相同。主对照仅同轮次两臂，不能以 E6 对 L05 E16 作因果比较。

新预算提案 21,600 秒，训练/缓存/验证合计。启动前必须提供局部分支真实更新耗时、两臂完整验证成本和开销的测量来源；`524*每更新秒数+完整验证秒数+开销 <= 0.8*21600` 才放行。旧 E4 速度不能充当 local 实测。缺测量或不能完成两臂时拒绝开跑，不降低分辨率/数据量。运行中按剩余预算给自有子进程期限；训练在期限前保存 `incomplete.pt` 并退出。半轮状态标记不可恢复为完整 epoch，硬终止时只承认最后原子落盘状态，不伪称已保存未落盘梯度。超时为 `incomplete`，不判机制无效，不重置预算。

## B：缓存 LP

仅现有同阶段 OpenAI 冻结图像特征，不重新编码、不改变 train/val，不删除命中行、不重加权其他行。读取原 RM-LP checkpoint 时使用已有 CPU 存储兼容函数；没有连接或调用 NPU 设备。

启动需要现有 center 特征和可验证的 val flip 特征，才能完成约定的固定解码。标准阶段目录当前只发现 center 缓存；flip 缓存尚未落实，因此是明确运行缺项，不能拿中心分冒充最终结果。支持的 flip 文件为 CPU torch 字典：`features[14880,512]`、有序 `paths`、`binding`（data_version=20260921、pretrained=openai、backbone=ViT-B/32、feature_manifest_sha256、val_csv_sha256、tta=horizontal_flip）。授权记录还必须绑定该文件 SHA256；不通过重编码补缺。

预算提案 3,600 秒，包含读取缓存、两头更新和验证；1 小时不是时长承诺。超时存未完成记录，无 GPU 任务、不抢占在途 CUDA。B1 不叫干净 LP，不中途插入 A1。B 负结果不能单独否定 FT 干预，B/A 增益不相加。这里只生成分类头状态，不能当成单 CLIP 可提交 checkpoint。

## 评估与投入判定

固定 center+horizontal flip / mean probabilities / T1.4 / prior .60。每臂只在自己的全部 val logits 上用既有程序拟合 prior，不读取测试统计或 L05 偏置；保存 logits、bias、输入/权重/输出身份。复用过的 val 不是独立验证集。

配对表包含 all14880、selected132、remaining14748、训练命中率≥5%的23类419张，以及 B 主导全部、23类内332张、其余类 B 主导。各组重叠，不相加。宏指标只对组中出现的类别平均；逐类 F/D/净值完整保存。23类在旧 E4 之后产生，对 E4 只能称事后；本段运行前冻结也不使其成为独立验证集。未命中/B主导都不是干净真值。

- 仅置信度降低、全量/未命中无识别改善：关闭本配方，不做 16 轮。
- 未命中净 1–44：有界弱信号，不能自动扩训或为名额出包。
- E6 未命中净≥45、该组 macro 正、全量 micro/macro 不退化：进入候选审查，仍须检查逐类集中度、状态重载和成本；45 不是自动许可。
- 全量下降但未命中净≥45/macro正且 B主导同向：标记 `proxy_supported_review_required`，可讨论有限完整候选/一次平台例外，不宣称干净准确率上升。
- 只有23类小切片上升、其余类别抵消：不能作为冲75主线晋级。代码交付全部逐类表，结论始终要求人工复核集中度。

C1：A有成块收益且B无额外可信优势，讨论 A1 E6→E16，原日程/选模/父不变。C2：B有留出识别证据且A/其他证据支持干预，新建以B1为父、从FT E1贯穿CE/GCE/local屏蔽的候选，先同E6比较再决定E16。两条不默认都跑，下一段最多一个完整视觉候选。A 报告从绑定的 E6 中心缓存应用原 raw_macro/raw_micro 选模，生成独立 continuation_E6.pt 并维护原 best 记录，保持被评估 epoch_6.pt 的身份不变；它仍不是完整 E16 候选。无足够信号就关闭，不因75未到开始新网格。

一般完整候选优先信号为固定解码 macro约+1pp且micro不降，0.3–1pp保留；上述内容监督 proxy_supported 例外不能泛化到任意低分模型。一个新模型只提交一个冻结解码，首次明确正平台反馈后再推独立改动。70/72/75仅是真实平台里程碑。最终仍需单 OpenAI CLIP ViT-B/32 checkpoint、37,444行、映射/文件名及ZIP内外字节一致。所有上传留给独立明确授权。

## 可重放工程检查

在方案工作目录执行（新 prepare 输出目录必须不存在）：

```bash
python3 -m pytest tests/test_p75_pipeline.py tests/test_p75_text_page_runtime.py tests/test_p75_semantic_pair.py -q
python3 scripts/p75_pipeline.py prepare --out outputs/codex/p75_supervision_pipeline_20260929_final
python3 scripts/p75_pipeline.py verify --out outputs/codex/p75_supervision_pipeline_20260929_final
```

训练入口仅供以后获授权时使用，当前不执行：

```bash
python3 scripts/run_p75_pipeline_a.py --out <已冻结目录> --authorization <A的新授权.json>
python3 scripts/p75_pipeline_lp.py --out <已冻结目录> --authorization <B的新授权.json>
```

授权格式参见 `configs/p75_supervision_pipeline_20260929/authorization.example.json`，默认拒绝运行。工程验证记录见 `results/p75_supervision_pipeline_20260929/`；指标空值表示尚未运行，不是0收益。状态序列明确区分 proposal / engineering_ready / running / incomplete / local_result / package_ready / platform_measured；格式/梯度测试不等于提分。

现役包引用原记录：`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/{pred_results.csv,submission.zip}`；既有37,444行、9/9校验来源为 `results/l05_prior_tta_two_slot_20260927.json` 和 `results/p75_supervision_rebuild_20260929/submission_check.log`。本段未重建/重验该包，也不因路径出现在记录中声称文件已检查可用。
