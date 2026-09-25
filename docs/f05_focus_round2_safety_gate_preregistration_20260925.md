# F05 Focus Round-2 安全门修复预注册（2026-09-25）

## Material Passport

- artifact_type: protocol amendment / preregistration
- protocol: `F05_FOCUS_SEVEN_UNIT_STAGE1`，仅涉及 Round 2 的 D1（O3）与 D2（PTA）
- branch: `focus/f05-four-lines`
- base commit: `1f53c56`
- amended clause: §6 统一晋级制度中，**逐 epoch 选择安全门**与**最终 `passed` 门**里的 `prediction_empty_classes` 条件
- status: `registered_before_rerun`
- platform submission: no
- test data use: none；本修订不触碰测试集

## 0. 修订的边界：改哪一条，不改哪一条

**只改一条**：`prediction_empty_classes` 由**绝对等于 0** 改为**不劣于 F05 + 同一 local view（M1）基线**。

**不动**：

- 晋级阶梯：`promote` Δmacro ≥ +0.30pp 且 Δmicro ≥ −0.10pp；`strong` ≥ +0.50pp；`breakthrough` ≥ +1.00pp；
- 最终门的 `clean_core_micro_delta_pp >= 0.20`、`trusted_macro_delta >= 0.0`、`raw_micro_delta >= -0.10`、`local_feature_drift <= 0.01`；
- 参照审计：center 逐位 0、M1 融合重算 ≤ 4e-6、epoch-zero 逐位 0、global 路径逐位；
- 固定配方、父模型、`clean070.csv`、局部视角（320 → crop160 → top5）、seed 42、epoch/patience；
- 不派生任何 LR / WD / bottleneck / dropout / temperature / epoch 扫描。

**为什么这不是「把门槛调低到能过」**：被改的这条门与分数无关，它只决定「一个 epoch 有没有资格进入候选池」。真正决定晋级的是上面那组一个都没动的阈值。修订前后，任何候选仍必须靠自己拿到 Δmacro ≥ +0.30pp 才能晋级。

## 1. 缺陷：该门在当前局部视角下**形式不可满足**

旧条件要求候选融合预测的 `prediction_empty_classes == 0`。但候选的对照基线 **F05 + M1 自身就有 4 个空类**。我直接从 M1 参照 cache 复算（14,880 行 val、750 类）：

```text
F05 + M1 基线        raw_macro 0.7370553612709045  raw_micro 0.7482526898384094
prediction_empty_classes = 4        （counts == 0 的类数）
相对 center 改变预测 1,531 张
```

代理模型（`adapter=None`，即「什么都不做」）在这个门下 **必然不合格**。一个连「恒等」都拒绝的门，不再是在筛风险，而是在禁止整个单元产生任何结果：

- D1（O3）：5 个 epoch 全部 `safety_eligible=false`，空类 4/4/5/5/5；
- D2（PTA）：5 个 epoch 全部 `safety_eligible=false`，空类 5/5/5/6/6。

两单元的 `best_record` 因此落回零初始化 adapter，delta 全 0.0。这是门的定义缺陷，与 adapter 是否学到东西无关——两单元的 feature drift 都在单调上升（D1 0.0024→0.0032，D2 0.0027→0.0037），说明梯度路径是通的。

## 2. 结果披露（本预注册不是盲测）

写这份文档时，D1/D2 在旧门下的结果**已经被观察到**：两者 `passed=false`、三项 delta 均为 0.0pp、best 落回 epoch 0。记录见 `results/f05_focus_d1_o3_result_20260925.json`、`results/f05_focus_d2_pta_result_20260925.json`。

之所以仍可预注册，是因为缺陷的论证不依赖这两个结果：第 1 节只需要「M1 基线自身有 4 个空类」这一个事实，就能证明该门对恒等候选不可满足。换句话说，**即使 D1/D2 从未运行，这条门也应当被判为定义错误**。此处显式披露，以免后续读者误以为修订发生在盲态。

## 3. 修订后的门（精确谓词）

`baseline_metrics` = 同一 validation cache 上 `adapter=None` 的评估（即 F05 + M1，与参照审计同一对象）。

**逐 epoch 选择安全门**（`train_local_feature_adapter.py`、`train_part_token_adapter.py`）：

```text
finite
AND prediction_empty_classes(candidate) <= prediction_empty_classes(baseline)
AND trusted_macro(candidate)  >= trusted_macro(baseline)
AND raw_micro(candidate)      >= raw_micro(baseline) - 0.001
AND local_feature_drift(candidate) <= 0.01
```

**最终 `passed` 门**：把其中的 `prediction_empty_classes(candidate) == 0` 同样替换为 `<= prediction_empty_classes(baseline)`，其余合取项逐字不变。

**为什么取「不劣于基线」而不是别的写法**：

- 保留原意。原门想拦的是「适配器把预测边际弄塌」。参照物本来就该是冻结基线，而不是一个基线自己都达不到的绝对 0。
- 无新增魔数。不引入「允许 5 个空类」这类需要事后辩护的容差；阈值由基线自身给出，随视角自动成立。
- 可审计。单个整数比较，`gate.json` 里同时记录候选与基线的空类数。
- 与选择规则不冲突。`best_record` 仍要求 `selector > best_selector`（严格优于基线），最终门仍要求 `clean_core_micro_delta >= +0.20pp`。因此「什么都不做」永远不可能 `passed`——恒等候选只会成为可比较的参照点，不会成为结果。

**已考虑但未采用**：更严的「候选不得引入基线没有的新空类」（集合包含而非计数）。语义上更贴原意，但需要逐类计数集合而非标量，审计面更大；本轮采用标量版，并把该备选记在此处，供后续协议复用。

## 4. 重跑范围

只重跑 **D1、D2 的 adapter 训练**：

- 输入逐字节复用：`D1_train_cache_crop160.pt`、`D1_val_cache_crop160.pt`、`D2_train_cache_crop160.pt`、`D2_val_cache_crop160.pt`、`f05_val_logits.pt`、`f05_val_m1_crop160.pt`；
- 不重建 cache、不重跑 A0/P0、不改任何固定参数；
- 命令与 `deploy/f05_focus_gpu0/run_focus_queue.sh`、`deploy/f05_focus_gpu0/run_d2_adapter.sh` 中的训练段一致，仅训练器代码含本修订。

报告内容：每个 epoch 的 `safety_eligible` 与读数、`best_record`（epoch 与指标）、`gate.json`、相对 M1 的 delta（pp）。

## 5. 判定与止损

- 若修订后仍**没有任何 epoch 严格优于 M1**：D1、D2 均判 **FAIL**，Round 2 局部 adapter 方向关闭，**D3 不生成**，不追加 seed、不派生参数扫描。
- 若有 epoch 入选且达标：按 §6 原阶梯判 `weak` / `promote` / `strong`，并按 §5.3 判断是否满足生成 D3 的前提（D1 > M1 **且** D2 > M1）。
- 任一参照审计失败即停，不调容差、不重试。

## 6. 合规边界

- 测试集只读；本修订不产生、也不授权任何测试集推理或提交包。
- 单模型、单 checkpoint；D1/D2 只训练冻结父模型之上的 adapter，global 路径逐位不变。
- 七单元不产生平台候选；平台线仍由独立登记的 L05 负责。

## 7. 交付物

```text
docs/f05_focus_round2_safety_gate_preregistration_20260925.md   # 本文件
reproducibility/aegis_f1/aegis_clip/cli/train_local_feature_adapter.py  # 门修订
reproducibility/aegis_f1/aegis_clip/cli/train_part_token_adapter.py     # 门修订
configs/f05_focus/manifest.json                                  # safety_gate 写入机器可读契约
outputs/f05_focus/D1_O3/gate.json                                # 重跑后
outputs/f05_focus/D2_PTA/gate.json                               # 重跑后
results/f05_focus_d1_o3_result_20260925.json                     # 重跑后更新
results/f05_focus_d2_pta_result_20260925.json                    # 重跑后更新
results/f05_focus_summary.csv                                    # 重跑后更新
```
