# P75_TEXT_PAGE_MASK_V1：固定纯文字页代理的取消分类监督试验门禁

## 假设与范围（扫描前固定）

已有图像核验观察到纯文字、地图、表格及粗粒度冲突，不再重复200例核验。标签物种无法确定不等于可见内容无法判断；这些观察尚未证明污染主导平台差距。

本次只允许一条检测规则与一个配对试跑。受外部模型/类别文本约束，先以无模型的像素连通区域识别**近单色、密集规则文字行且几乎无非文字墨迹的页面**。这是很窄的内容代理，不是一般开放集检测器；地图、彩色表格、商品、闭集错标均不在覆盖承诺内。非自然照片、低置信度或近邻不支持均不是屏蔽条件。无法确定的图保留原GCE。

规则完整记录在 `configs/p75_text_page_mask_20260929/detector.json`，运行前复制规则及源码哈希到协议文件。选择函数只接受图像和固定参数，不接受类别、L05、OOF或200例人工观察。共享数据只读，检查字节哈希，不读测试图、不装依赖、不下载模型。

## 固定预算与停止条件

- CPU图像扫描一次，4进程，最多3600秒；训练/验证同步冻结相同像素规则。只复用P0最终快照检查选中图的原标签概率、sqrt(p_y)及 `2 sqrt(p_y)(1-p_y)`（logit梯度L1，不是参数梯度或错误影响量）。历史轨迹仍缺失。
- 从**自动命中的训练图**按种子20260929抽取至多16张，仅检查检测器是否明显误伤有效生物外观；这不扩展200例错误归因核验。任何明确误伤就否定整条固定规则，不手工排除该图后继续，不改阈值重扫。
- 无命中、扫描不完整、数据不匹配、某类被清空、明确误伤时，不生成正式权重、不启动训练。此时交付实测门禁失败，不能称干预完成。少量样本/未知收益本身不构成失败门槛。
- 通过门禁才落实一次 control/masked 配对：同一合法RM-LP父权重，从视觉微调第1轮重训到第6轮，保持原16轮LR日程，覆盖第5轮局部分支启用后的两轮；合计最多6 GPU小时（含验证），预算不足则报告试跑未完成，不能判机制无效。不启动全量训练或参数扫描。

## 配对的唯一变化

全局与局部分类的逐样本损失在命中图置零，包括前2轮CE暖身及之后GCE。保留原有效权重分母，不重新归一化；保留采样、更新次数、feature anchor、局部几何与pmax门控、优化器、批大小、随机种子及推理解码。control执行同一训练接入与日志但不开屏蔽。不是从L05最终权重短续训。共同RM-LP父权重的分类头已经接触过旧监督，因此这是视觉微调阶段取消分类项的试验，不是检验整个学习过程中所有污染影响。训练期记录selected/remaining组实际视图的py、pmax、sqrt(py)、原分类损失；不能用最终快照回填历史。

## 固定评价

比较两臂相同第4、6轮（第4轮仅观察，第6轮主判断），固定中心/翻转、T=1.4、prior=0.6的原解码。报告全量原标签macro/micro、逐张修正与退化，另列检测命中与未命中代理分组、类覆盖。未命中组可能仍含大量污染，绝不称“干净/有效目标真值”。L05最终仅作现役参照，不替代同进度control。

总分下降不是一票否决；未命中组明确净改善而下降集中在命中组时保留候选解释，但这种低召回代理的证据有限。仅命中图置信度降低、未命中组无净改善不支持完整训练。单次试跑结果必须结合代理局限判断；不因代码或集合非空自动扩预算。实际训练段须完成预测CSV/ZIP和校验；未训练的检测门禁段引用现役包，不冒充新候选。

## 执行命令与实测

固定扫描已结束；扫描结果没有反过来改动上述规则。

| 项目 | 实测 |
|---|---:|
| 训练扫描 / 命中 | 133,815 / 0 |
| 验证扫描 / 命中 | 14,880 / 0 |
| CPU扫描用时 | 1,115.27秒 |
| 原P0验证错误核对 | 3,474，一致 |
| 被清空类别 | 0 |
| 新训练 / GPU / 新候选 | 均无 |

状态：`failed_empty_selection`。没有命中图，因此没有新增图像核验、没有可估计的“命中污染图梯度分布”、没有配对训练。空集合的梯度总量为0只是算术结果，不表示污染梯度已被GCE消除。未生成训练权重或训练配置；损失接入仅完成实现验证，不能称干预完成。

本次规则过窄，未形成可用入口。它不能排除污染的存在、危害或解释平台差距的可能性，也不支持复活旧困难CE/三通道R1。关闭这个固定点，不通过改阈值或手工名单补足样本；在该检查点暂停。

结果：[report.md](../results/p75_text_page_mask_20260929/report.md)、[report.json](../results/p75_text_page_mask_20260929/report.json)、[逐图像素结果](../results/p75_text_page_mask_20260929/pixels.jsonl.gz)、[完整文件哈希与现役交付路径](../results/p75_text_page_mask_20260929/manifest.json)。逐图结果仅为自动几何测量，不能当成生物/非生物标签。

从仓库根目录可重放（输出目录必须不存在）：

```bash
python3 scripts/p75_text_page_detector.py --stage /home/lux1/noise/artifacts/stages/repechage/20260921 --image-root /home/lux1/noise --rule configs/p75_text_page_mask_20260929/detector.json --out outputs/codex/p75_text_page_mask_20260929 --workers 4
python3 scripts/p75_text_page_report.py --scan outputs/codex/p75_text_page_mask_20260929 --evidence results/p75_supervision_rebuild_20260929/sample_evidence.csv.gz --p0-summary results/p75_supervision_rebuild_20260929/summary.json --image-root /home/lux1/noise --out outputs/codex/p75_text_page_mask_20260929/linked
python3 -m pytest tests/test_p75_text_page_detector.py tests/test_p75_text_page_runtime.py tests/test_p75_text_page_report.py -q
```

12项检查通过：CE/GCE与局部分支/回退屏蔽、原分母、feature anchor、冻结源码插入、输入篡改拒绝、快照联证与预览的官方中心裁剪一致性。真实命中为0，所以没有真实图像误伤率可报告；合成反例不能证明高精度或真实召回。

现役可提交包（未新生成）：`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/{pred_results.csv,submission.zip}`。既有9/9校验、37,444行见`results/p75_supervision_rebuild_20260929/submission_check.log`。本轮没有新平台结果。

方案分支：`codex/p75_text_page_mask_20260929`；独立目录：`/home/lux1/noise-worktrees/p75_text_page_mask_20260929`。集成使用自动模式`git pull --rebase --autostash origin main`，合并后重跑检查再推送；确切commit见交付回复及Git历史。

成本参考：现役原始日志第6轮累计训练约10,115.95秒，两臂约5.62小时（未计验证，且不保证当前主机同速）；6小时为硬上限，不能为完成而自动超支。

已知检测局限（规则冻结后记录）：一张用本机 DejaVuSans 12px 渲染的600×768纯文字页（固定39行）满足背景、字形数量与规则行条件，但墨迹落在行内比例0.93593，低于固定0.95要求，因而漏检。该开发合成样本不参与训练，不据此调整阈值。矩形字形合成正例通过仅证明代码路径可运行，不证明真实文字页召回或无目标检测可靠。渲染参数及字体哈希保存到 `synthetic_probe.json`。
