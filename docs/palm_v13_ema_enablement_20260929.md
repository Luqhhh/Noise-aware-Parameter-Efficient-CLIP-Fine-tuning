# V13 策略：按用户指令启用训练侧 EMA

实验标识 `P75_RESOLUTION_LADDER_EMA_ENABLE`，更新对象 `P75_RESOLUTION_LADDER`。
用户在确认EMA未被明确判定违规后，明确要求“启用ema”。本次落实该选择，交付范围为代码与CPU检查；
`engineering_ready / not_started`，GPU、真实项目训练、新预测包、local/platform指标均未启动或产生。
本次用户选择不声明获得了新的组委会裁定，原规则文档不改写。

## 当前行为

- `ema_enabled=true`、`ema_decay=0.9995`，每次完整逻辑批次的optimizer更新后更新EMA一次。
- 每段继承选中的一份父权重，再重新建立optimizer、scheduler和EMA。EMA从该父权重开始，计数重置为0。
- dev阶段分别记录固定holdout上的raw/EMA。按macro选择状态（相等选EMA），按该状态的`macro + 0.5×micro`选择epoch；不融合两路预测。
- checkpoint记录raw、EMA、衰减率和实际更新数；EMA结构、衰减或更新终点不匹配时拒绝作为父模型。
- EMA验证采用已有`using_weights`上下文，异常或预算耗尽时恢复raw参数，避免将EMA误保存为raw。
- 全量段仍没有独立holdout，不做验证或选模。最后固定第5轮raw `last.pt`用于单模型确定性推理。
- CSV使用`文件名, 0001`，ZIP仅含相同字节的`pred_results.csv`。

EMA累加与权重切换复用现有`aegis_clip.b448_strategy.WeightAverage / using_weights`，不重复新增平均算法。
新plan同时绑定该共享源码的SHA256。配方保留显式EMA开关；关闭时需同时选择`best_raw_on_holdout`。
旧准备目录仅为历史，不继续用旧plan或旧授权文件执行当前代码。

## 已验证的复现入口

方案分支 `codex/palm_v13_ema_20260929`，从最新`origin/main`的
`505e73158c4422cd8a6ca4cb95596967df3ed47b`创建；独立目录
`/home/lux1/noise/worktrees/palm_v13_ema_20260929`。在该目录执行：

```bash
PYTHONPATH=reproducibility/aegis_f1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python3 -m pytest \
  tests/test_palm_v13_strategy.py reproducibility/aegis_f1/tests/test_model.py -q
PYTHONPATH=reproducibility/aegis_f1 CUDA_VISIBLE_DEVICES='' python3 -m palm_v13.plan prepare \
  --output outputs/codex/palm_v13_ema_20260929/prepared
PYTHONPATH=reproducibility/aegis_f1 CUDA_VISIBLE_DEVICES='' python3 -m palm_v13.plan verify \
  --plan outputs/codex/palm_v13_ema_20260929/prepared/plan.json
```

实测**58 passed**（39项策略检查+19项既有模型检查）。合成CPU检查分别执行EMA开/关的小模型阶段链，
验证更新次数、阶段重置、EMA选中后逐张量精确继承、raw验证/保存状态、全量段无验证、异常恢复及CSV精确格式。
受控的合成选模分数只用于覆盖代码分支，不是项目准确率；没有使用项目图像训练，CUDA未初始化。

当前计划仍绑定148,695张复赛官方训练图及133,815/14,880固定内容组划分，四段共38,266次逻辑更新。
完整配置、plan SHA256及现役提交包引用见[核验JSON](../results/palm_v13_ema_enablement_20260929.json)。
现役包仍引用`worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/{pred_results.csv,submission.zip}`，
其37,444行、9/9校验及ZIP/CSV一致性来源见[前次核验](../results/palm_v13_rules_review_20260929.json)。

GPU成本和显存尚未测量，默认执行授权仍为false；未来dev成本需包含完整raw+EMA两路验证和双状态checkpoint写入。
“启用ema”没有撤销此前“不启动GPU实验”的指令。后续训练依照当前探索/成本门禁执行。
方案校验后立即推送，在独立main集成目录以**自动模式**`git pull --rebase --autostash origin main`、
合并、重新校验并推送；不另行`stash pop`，在工程检查点暂停。
