# 复赛（1500 类长尾）准备工作 — 2026-08-31

目标：把初赛代码与算法经验迁移到复赛（1,500 类 / 297,282 训练 / 74,896
测试，含长尾分布），同时严格做到「只传代码，不传数据/伪标签/原型/已拟合
参数」。本文件记录本轮落地的组件、用法与验证结果。

## 规则边界（COMPETITION_RULES_AGENT.md）

- 复赛只能使用当前阶段官方数据；初赛 checkpoint、特征缓存、伪标签、类别
  原型、OOF manifest、拟合好的 prior 强度一律不得跨阶段复用。
- 禁止使用测试预测分布优化类别先验：balanced-prior 的类别偏置必须只在
  当前阶段验证集上拟合，测试集只做确定性应用。
- 多尺度 + Flip TTA 已获组委会答复为**合规**（2026-08-31，经用户确认；
  前提：单一 checkpoint + 单一确定性推理流程，不构成多模型集成）。仓库仍
  保留 `--acknowledge-*-risk` 显式确认门，作为推理 manifest 的审计痕迹。

## 本次实现

### 1. 长尾训练基础设施（reproducibility/aegis_f1）

- 新增 `aegis_clip/longtail.py`：采样、损失重加权、训练期 balanced-softmax
  logits 修正三类开关，全部按类别计数动态计算，无任何固定类别数假设。
- `trainer.py` 接入 `longtail` 配置段：

```yaml
longtail:
  sampler_mode: class_balanced        # none|class_balanced|sqrt_class_balanced|balanced_oversample
  loss_reweighting: inverse_frequency # none|inverse_frequency|sqrt_inverse_frequency|effective_number
  effective_number_beta: 0.9999
  balanced_softmax_tau: 0.5           # 缺省时回落到 loss.class_prior_adjustment_tau
```

采样器与 shuffle 互斥处理、损失重加权与 trust 权重在 MixUp 前合并、
`_promotion_decision` 的 `required_predicted_class_count` 缺省改为当前
`num_classes`（原先硬编码 500）。

### 2. head / medium / tail 三段评估

- `aegis_clip/evaluation.py::longtail_segment_metrics`：按训练频率把类别等分
  为三段（500/500/500 或 166/167/167），输出每段 micro/macro 与类数，
  `evaluate()` 在传入 `class_counts` 时自动附带，trainer 四个评估点均已接入。
- 根仓库 `common/logit_adjustment.py`：新增 `compute_class_counts` 与
  `head_medium_tail_metrics`，`_compute_metrics_from_logits` /
  `sweep_logit_adjustment` 支持 `class_counts`；`scripts/evaluate_tta.py`
  自动读取 `train.csv` 输出 baseline/TTA 的三段 macro。

### 3. balanced-prior 验证集拟合 + 冻结推理

- `aegis_clip/prior_alignment.py` 拆分为 `fit_prior_bias`（IPF 拟合，仅验证
  集）与 `apply_prior_bias`（确定性应用，测试集可用）。`align_logits_to_prior`
  保留为离线扫描的兼容入口。
- 新增 `aegis_clip/cli/sweep_prior_strength.py`：读验证集 logits 缓存 →
  拟合一次偏差 → 按验证集指标（默认 `clean_core_micro`）选强度 →
  写 `prior_config.json`（含 `test_data_used: false` 声明）。
- `aegis_clip/cli/infer.py` 新增 `--prior-config`：测试推理只应用冻结偏差，
  不再对测试分布做任何拟合；与旧的 `--prior-alignment-strength`（测试批内
  拟合路径）互斥，后者仅保留给离线研究。

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.sweep_prior_strength \
  --validation-logits <val_logits.pt> --output-dir <prior_dir> \
  --strengths 0.0,0.25,0.5,0.75,0.85,0.9,0.95,1.0

PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint <best.pt> --output-dir <submission_dir> \
  --prior-config <prior_dir>/prior_config.json
```

### 4. 硬编码清理

- `common/cache.py`：`expected_num_classes` 缺省不再回落 500，改为缺配置即报错。
- `scripts/build_master_split.py`：`expected_num_classes=500` 改为按实际类别
  目录数推导，新增可选 `--expected-num-classes` 作为硬校验。
- `scripts/real_dry_run.py`：两处 `range(500)` 改为 `range(num_classes)`。
- `common/logit_adjustment.py::compute_class_priors`：`num_classes` 改为必填。
- `scripts/check_submission.py`：标签范围检查新增 `--num-classes` /
  `--class-mapping`，不再写死 `[0000, 0499]`。

### 5. 1500 类合成 dry-run

- 新增 `scripts/build_synthetic_stage.py`：生成长尾 1500 类训练目录 + 平铺
  测试目录（确定性噪声 JPEG），供全链路演练。
- 示例配置 `reproducibility/aegis_f1/configs/dryrun_repechage_1500.yaml`。

### 6. 噪声流程阶段内重建链

- 新增 `aegis_clip/stage_pipeline.py` + `aegis_clip/cli/stage_pipeline.py`：
  一个 manifest 驱动的六步链 `split → features → folds → oof → trust →
  final_train`，所有阈值（OOF 的 epoch/lr/q、trust 的 folds/probe/correction
  cap 等）都从 manifest 传入，不再带任何初赛拟合值。
- `folds` 步骤新增内容分组感知的分层 OOF 折分配（StratifiedGroupKFold +
  SHA-256 内容组），保证同内容图不跨折。
- 每步产物与 SHA-256 记录进 `pipeline_run.json`。

```bash
PYTHONPATH=$PWD python3 -m aegis_clip.cli.stage_pipeline --manifest pipeline.json
```

## Dry-run 验证结果（RTX 4070 Laptop, 2026-08-31）

| 项 | 结果 |
|---|---|
| 合成数据 | 1500 类，18,274 训练（min 12 / max 80）+ 3,000 测试 |
| prepare_stage | 通过；分层 + SHA-256 分组 split 正常 |
| 特征缓存吞吐 | 18,274 张约 32s ≈ 565 img/s → 复赛 297,282 张约 9 min；fp32 缓存 ≈ 608MB |
| 长尾训练 | 1 epoch 通过；class-balanced 采样 + 逆频率重加权 + tau=0.5 生效 |
| 分段指标 | 评估输出 head/medium/tail 各 500 类 micro/macro 正常 |
| prior 工作流 | 验证集拟合 → 强度扫描 → `prior_config.json`（test_data_used=false）→ 冻结推理 |
| 提交校验 | 3,000 行、四位标签 `[0000, 1499]`、ZIP 单文件全部通过 |
| 阶段重建链 | 六步全通（约 92s，3 折 OOF + 3 折 trust），final_train 覆盖 18,274 样本 |

## 测试状态

- aegis 隔离线：`315 passed`（新增 longtail / prior fit-apply / sweep /
  三段指标测试）。
- 根仓库：`408 passed, 2 failed`；两个失败均为 2026-07-30 起已记录的既有
  问题（`_SingleProcessDataLoaderIter requires timeout == 0` 与 OOF soft
  target 的 float32 精确相等断言），与本轮改动无关。

## 复赛数据到位后的执行顺序

1. 用官方复赛数据目录替换路径，跑 `prepare_stage`（写清 expected_classes /
   expected_samples）。
2. `stage_pipeline` 一键重建 features / OOF / trust；阈值只在 manifest 里调。
3. 用新验证集重扫 prior 强度，训练期间用三段指标盯 tail。
4. 先交基线包拿反馈，再决定长尾开关组合。
