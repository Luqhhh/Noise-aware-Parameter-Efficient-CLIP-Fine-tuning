# F05 Focus 七单元一阶段方案（设计冻结稿）

## Material Passport

- artifact_type: experiment protocol / implementation scaffold
- protocol: `F05_FOCUS_SEVEN_UNIT_STAGE1`
- branch: `focus/f05-four-lines`
- base: `codex/rematch750_search_v5 @ 5ec6784`
- status: `design_only_not_yet_run`
- platform submission: no
- test-data use in this document: none; test data remains inference-only

## 0. 硬基线

所有新实验只与同一严格基线比较：

| metric | value |
|---|---:|
| F05 Macro | 74.4307% |
| F05 Micro | 75.4435% |
| F05 platform | 65.4711% |

F05 固定配方：

- 320px、`weak_rrc_flip`；
- 16 epochs、effective batch `256 x 4 = 1024`；
- head LR `4e-4`、backbone LR `1.2e-5`、WD `1e-4`；
- CE warmup 2 → GCE `q=0.5`、feature anchor `2.0`；
- seed 42、raw macro 选模、raw micro 破平局；
- 不引入 LR/WD/SAM/warmup/GCE-q/epoch 新扫描。

## 1. 第一阶段 7 个有效实验单元

| ID | 方向 | 改动 | 成本 | 第一阶段目的 |
|---|---|---|---|---|
| C0 | CUDA control | 原样重跑 F05 | 高 | 建立 GPU 对照 |
| N1 | 高精度噪声 | F05 recipe + strict drop | 高 | 迁移初赛 CL+kNN drop |
| A0 | attention-local | F05 checkpoint + M1 | 低 | 验证局部互补是否仍成立 |
| P0 | prior | F05 + val-fitted prior | 极低 | 恢复初赛大收益校准线 |
| D1 | Dual Adapter | F05 + O3 | 中低 | 验证 local adapter |
| D2 | Dual Adapter | F05 + PTA | 中低 | 验证 part-token adapter |
| D3 | Dual Adapter | F05 + O3 + PTA | 低 | 两者都有效后组合 |

不允许从这 7 个单元派生 LR/WD/SAM/warmup/GCE-q/epoch 搜索；禁止 N × A × D × P 全排列。

## 2. GPU 使用方式

两张卡作为两个独立 worker，不做 DDP、不做跨卡梯度聚合。每个 run 必须保持：

```text
effective batch = microbatch x grad_accum = 1024
```

如果显存不足，只降低 microbatch 并同步提高 `grad_accum`，例如：

```text
64 x 16 = 1024
```

统一队列（以 F05 结果为 baseline）：

| Round | GPU0 | GPU1 |
|---:|---|---|
| 0 | A0 F05-M1 probe | P0 val-prior calibration |
| 1 | C0 F05 CUDA control | N1 high-precision drop |
| 2 | D1 O3 adapter | D2 PTA adapter |
| 3 | D3 compose/eval | 当前 strongest 的第二个 seed |
| 4 | strongest seed3407 | strongest seed2026 |
| 5 | 只组合已独立过门机制 | 同左 |

Round 0 必须在长时间 full training 之前完成。

## 3. Round 0

### 3.1 A0 — F05 + attention-guided local

A0 不训练。直接加载 F05 checkpoint，做：

```text
F05 native global
+ attention-guided local crop 224 / top-5
```

固定两个融合点，不扫 crop size、top-k、flip：

- `A0-50`: global 0.50 + local 0.50
- `A0-40`: global 0.60 + local 0.40

局部几何为：

```text
input 320 -> attention crop 224 -> topK=5
```

强制审计：

```text
native F05 global logits
==
attention pipeline global logits
```

至少要求 prediction agreement = 100%、max abs logit diff ≈ 0。脚本使用
`scripts/run_a0_m1_probe.py`，默认 tolerance `1e-5`，低于门禁即 fail-closed。

晋级门：

```text
ΔMacro >= +0.30pp
ΔMicro >= -0.10pp
```

`ΔMacro >= +0.50pp` 记为 strong，直接进入平台候选观察，但平台提交仍需队长决定。

输出：

```text
outputs/f05_focus/A0_M1/probe.json
artifacts/f05_focus/f05_attention_cache.pt   # --cache-output
results/f05_focus_summary.csv
```

### 3.2 P0 — validation-fitted uniform prior

P0 不训练、不重新推理模型。仅使用 `prior_alignment.py` 的合规接口：

```python
bias, report = fit_prior_bias(validation_logits, target_prior=None)
corrected = apply_prior_bias(test_logits, bias, strength=s)
```

禁止旧入口：

```python
align_logits_to_prior(test_logits)
```

做法：

1. 在 F05 validation logits 上拟合 uniform `1/750` 的 IPF bias；
2. 只在 validation 上扫 `s in {0.25, 0.50, 0.75, 1.00}`；
3. 选择规则：raw macro 优先，raw micro 破平局；
4. test 阶段只读入 test logits 并 apply 冻结 bias，不再统计 test soft marginal。

输出：

```text
outputs/f05_focus/P0_PRIOR/probe.json
artifacts/f05_focus/prior_bias.pt
results/f05_focus_summary.csv
```

Test 只读应用（不重新拟合）：

```bash
PYTHONPATH=reproducibility/aegis_f1 \
python3 scripts/apply_frozen_prior_to_test_logits.py \
  --test-logits artifacts/f05_focus/test_logits.pt \
  --bias artifacts/f05_focus/prior_bias.pt \
  --class-mapping artifacts/stages/repechage/20260921_npu/class_to_idx.json \
  --output-dir outputs/f05_focus/P0_PRIOR/test_apply
```

该脚本不做 test soft-marginal 统计、不做 test fitting；最终 CSV/ZIP 仍需走仓库正式的
提交校验流程。

## 4. Round 1

### 4.1 C0 — CUDA F05 control

C0 不是可选点。C0 与 N1 必须只差 `sample_weight / reject mask`，其他全部一致：

- parent、split、augmentation、loss、LR、epoch、batch、scheduler、seed、resolution；
- effective batch 1024；
- 320px、16 epochs、CE 2 → GCE q=0.5、feature anchor 2.0；
- head LR 4e-4、backbone LR 1.2e-5、WD 1e-4；
- seed 42、same selection policy。

CUDA 侧配置把 `optimizer_impl` 改为 `foreach`、`gradient_norm_impl` 改为 `foreach`，
这是硬件执行实现绑定差异，不是新搜索变量。`init_checkpoint` 在运行前必须指向真实
RM-LP parent；仓库默认值来自历史 `/workspace/...` 路径，执行机应显式覆盖。

配置：

```text
configs/f05_focus/C0_cuda_control.yaml
```

### 4.2 N1 — High-Precision Noise Drop

N1 不做 soft repair、hard relabel、consistency。只做 `weight = 0`，标签不改。规则复用初赛
`NR_CL_KNN_DROP` 高精度共识：

```text
confident-joint issue
AND oof_top1 != original_label
AND knn_top1 != original_label
AND oof_top1 == knn_top1
AND top1_margin >= class 75th percentile
AND knn_agreement <= 0.20 + eps
```

并额外加入：

```text
cross-label exact duplicate conflict -> reject
```

不追求固定 1% reject rate；只审计 reject count、reject rate、per-class reject rate、
head/mid/tail reject rate、cross-label conflicts removed。若自动规则删 0.6%，就保留 0.6%。

配置：

```text
configs/f05_focus/N1_hp_noise_drop.yaml
```

N1 manifest 构建：

```bash
PYTHONPATH=reproducibility/aegis_f1 \
python3 scripts/build_hp_noise_manifest.py \
  --sample-quality <current-stage-sample-quality.csv> \
  --oof-logits <current-stage-oof-logits.pt> \
  --output artifacts/f05_focus/hp_noise_manifest.csv
```

配置中的 `trust.sample_weight_path` 必须精确指向该文件。若出现任意类别被删到零 clean，
manifest builder fail-closed。

## 5. Round 2 — Dual Adapter

D1/D2 父模型均为冻结 F05 checkpoint；global path 不减、不训练。第一轮样本集不能使用 N1
的新筛选结果，否则 adapter gain 与 cleaning gain 耦合。固定使用 current-stage quality
asset：

```text
clean_probability >= 0.70
```

生成同一个 `artifacts/f05_focus/clean070.csv` 供 D1、D2 使用。

### 5.1 D1 — O3 Local Feature Adapter

固定参数：

- `BottleneckLocalFeatureAdapter`
- feature dim `512`
- bottleneck `32`
- residual scale `0.25`
- dropout `0.1`
- local geometry `320 -> crop160`，top-patches `5` 不扫
- backbone frozen，仅训练 adapter

### 5.2 D2 — Part Token Adapter

固定参数：

- `PartTokenResidualAdapter`
- bottleneck `64`
- residual scale `0.25`
- dropout `0.1`
- `part_top_patches = 8`（不按 patch grid 放大到 16）
- `part_temperature = 0.07`
- local geometry 与 D1 使用同一固定局部视角
- backbone frozen，仅训练 adapter

### 5.3 D3 — Compose

D1、D2 都完成后分别测：

```text
F05 + M1
F05 + O3
F05 + PTA
```

baseline 必须统一为 `F05 + same local view`，不能把 M1 自身提升算到 adapter 上。
只有：

```text
D1 > M1
D2 > M1
```

才生成 `D3 = F05 + O3 + PTA`。两个 adapter namespace 独立，不重新训练 backbone。

## 6. 统一晋级制度

| 等级 | 条件 | 动作 |
|---|---|---|
| FAIL | Macro ≤ baseline | 关闭 |
| weak | 0 < ΔMacro < +0.30pp | 记录，不派生 |
| PROMOTE | ΔMacro ≥ +0.30pp 且 ΔMicro ≥ -0.10pp | 保留 |
| strong | ΔMacro ≥ +0.50pp 且 ΔMicro ≥ 0 | 平台候选 |
| breakthrough | ΔMacro ≥ +1.00pp | 最高优先级 |

训练型候选 N1、D1、D2、D3 在 seed42 strong 之后才用两张卡并行跑 seed3407、seed2026；
要求至少 2/3 seeds 同方向。不要所有实验一开始就三 seed。

## 7. Greedy 组合规则

只有独立过门的机制才允许组合，禁止全排列。例如假设：

```text
P0 +0.80
A0 +0.65
D3 +0.45
N1 +0.35
```

只跑：

```text
F05
F05 + P
F05 + P + A
F05 + P + A + D
F05_HP + P + A + D
```

每增加一个组件要求 incremental ΔMacro ≥ +0.20pp，否则立即回退。

## 8. 目录与产物

```text
configs/f05_focus/
  C0_cuda_control.yaml
  N1_hp_noise_drop.yaml
  A0_m1.yaml
  P0_prior.yaml
  D1_o3.yaml
  D2_pta.yaml
  manifest.json

outputs/f05_focus/
  C0_F05_CUDA/
  N1_HP_DROP/
  A0_M1/
  D1_O3/
  D2_PTA/
  D3_DUAL/
  P0_PRIOR/

artifacts/f05_focus/
  hp_noise_manifest.csv
  clean070.csv
  prior_bias.pt
  f05_val_logits.pt
  f05_attention_cache.pt

results/f05_focus_summary.csv
```

`results/f05_focus_summary.csv` 字段固定为：

```text
experiment_id,parent_sha,seed,macro,micro,delta_macro,delta_micro,
changed_predictions,selected_epoch,reject_rate,platform_score,status
```

其中 `delta_macro`、`delta_micro` 使用百分点（pp）。

## 9. 建议执行命令

Round 0：

```bash
export F05_CHECKPOINT=/path/to/F05/seed42/checkpoints/best.pt

# P0 需要中心全局 validation logits。可以先独立生成，之后 A0 也可以复用。
PYTHONPATH=reproducibility/aegis_f1 \
python3 -m aegis_clip.cli.cache_validation_logits \
  --checkpoint "$F05_CHECKPOINT" \
  --view-mode center \
  --output artifacts/f05_focus/f05_val_logits.pt \
  --batch-size 128 --num-workers 4

PYTHONPATH=reproducibility/aegis_f1 \
python3 scripts/run_a0_m1_probe.py \
  --checkpoint "$F05_CHECKPOINT" \
  --device cuda:0 \
  --crop-size 224 \
  --top-patches 5 \
  --cache-output artifacts/f05_focus/f05_attention_cache.pt \
  --output-dir outputs/f05_focus/A0_M1 \
  --summary results/f05_focus_summary.csv

PYTHONPATH=reproducibility/aegis_f1 \
python3 scripts/run_p0_prior_probe.py \
  --validation-logits artifacts/f05_focus/f05_val_logits.pt \
  --strengths 0.25,0.50,0.75,1.00 \
  --output-dir outputs/f05_focus/P0_PRIOR \
  --bias-output artifacts/f05_focus/prior_bias.pt \
  --summary results/f05_focus_summary.csv
```

若 A0 已先完成，`f05_attention_cache.pt` 同时包含 `logits = native F05 global`
与 labels/quality 字段，也可直接作为 P0 的 validation logits 输入。

Round 1：

```bash
PYTHONPATH=reproducibility/aegis_f1 \
python3 -m aegis_clip.cli.rematch train \
  --config <runtime-cuda-c0-config> \
  --device cuda:0

PYTHONPATH=reproducibility/aegis_f1 \
python3 -m aegis_clip.cli.rematch train \
  --config <runtime-cuda-n1-config> \
  --device cuda:1
```

Round 2 的 adapter cache / 训练命令必须使用同一 `clean070.csv`、同一 F05 checkpoint、同一
local view，并在运行前打印 epoch-zero audit；未过审计不得训练。

## 10. 合规边界

- 测试集只读；本方案所有拟合与选择都只发生在 validation 或 train split 上。
- P0 禁止 test-batch prior fitting；test 只 apply 冻结 bias。
- N1 不修改标签，不创建 pseudo-label，不做 soft repair。
- Adapter 使用冻结 F05 parent，global path 不变。
- 禁止多模型集成、跨 seed 平均、多 checkpoint 加权。
- 未产出预测 CSV/ZIP 与提交校验前，不得声称平台候选或可提交。

## 11. 当前交付状态

**2026-09-25 更新：七单元中 C0、N1、A0、P0、D1、D2 已实际执行；D3 与两个 seed 轮未触发。**

| 单元 | 结果 | 证据 |
|---|---|---|
| C0 | 完成，作为其后所有单元的冻结父模型（`best.pt` SHA-256 `f8205b5c683d23dd08a2df380e7ee99b242715351943d436caf440ef0fea3c2a`） | `outputs/f05_focus/C0_F05_CUDA/` |
| N1 | **fail**，Δmacro −0.3755pp（reject rate 4.33%） | `results/f05_focus_summary.csv` |
| A0 | **fail**，Δmacro −0.0351pp / Δmicro +0.0470pp | `outputs/f05_focus/A0_M1/probe.json` |
| P0 | **weak**，Δmacro +0.2615pp / Δmicro −0.1344pp（strength 0.75） | `outputs/f05_focus/P0_PRIOR/probe.json` |
| D1（O3） | **fail**，epoch-zero 与 global 路径逐位精确，但 5 个 epoch 全部 `safety_eligible=false`，无 epoch 可入选，best = 零初始化 adapter，delta 0.0 | `outputs/f05_focus/D1_O3/gate.json`、`results/f05_focus_d1_o3_result_20260925.json` |
| D2（PTA） | **fail**，同上；5 个 epoch 空类为 5/5/5/6/6 | `outputs/f05_focus/D2_PTA/gate.json`、`results/f05_focus_d2_pta_result_20260925.json` |

D1、D2 的晋级基线统一为 **F05 + 同一 local view（M1）**：raw macro `0.7370553613`、raw micro `0.7482526898`。两者 delta 均为 0.0，按 §6 判 **FAIL**。**D3 要求 D1、D2 同时 > M1，未满足，不生成**；两个 seed 轮要求 seed42 先达 strong，未触发。

**安全门已修订（2026-09-25 预注册）。** 原条件 `prediction_empty_classes == 0` 形式不可满足：M1 基线自身就有 **4 个空类**，连恒等候选都会被拒。修订把该条件改为**不劣于基线**（`candidate <= baseline`），其余阈值、特征漂移界与全部参照审计逐字未动。修订已实现并重跑，但**判定不变**：修订后 D1 epoch 1–2 已满足空类门，真正阻塞项变成 `trusted_macro(candidate) >= trusted_macro(baseline)`——两单元每个 epoch 都低于冻结基线 0.08–0.27pp，而这条是相对且可满足的条件（恒等候选即满足），属实质性负结果，故不再放宽。Round 2 局部 adapter 方向据此关闭。完整预注册、披露与逐 epoch 分解见 `docs/f05_focus_round2_safety_gate_preregistration_20260925.md`，机器可读契约为 `configs/f05_focus/manifest.json` 的 `safety_gate` 块。

执行过程中修复了另外两处实现缺陷，均只改实现、不改判据与容差：

1. D1 的两个 cache 与参照 cache 批大小不一致（64 vs 128），见 `results/f05_focus_d1_reference_audit_failure_20260925.json`；
2. O3 评估路径改用与 PTA 相同的 `anchored_classifier_residual_logits` 残差写法，使 epoch-zero 门按构造精确成立，见 `results/f05_focus_d1_o3_result_20260925.json`。

本七单元之外另有一条已闭环的平台线：本阶段 CUDA 本地重训 **L05** 取得平台 **66.94797564362783%**（TTA T=1.4 + prior 0.60），见 `results/f05_focus_l05_tta_prior_platform_20260925.json`。七单元中没有任何单元产生平台提交包。

**2026-09-25 补充（保护 + 绑定 + 固定配方诊断）：** `L05_T14_P060` 已登记为不可覆盖产物
（`results/f05_focus_l05_protected_packages.json`），CSV/ZIP/manifest 哈希复核通过，未重训、未重交。
`run_l05_local_retrain.py` 补上真正生效的 `--train-seed`（不动 split seed 42、RM-LP parent SHA 与配方），
`build_l05_tta_prior_submission_final.py` 默认拒绝覆盖，新增
`aegis_clip/tta_prior_binding.py` 把校准缓存绑定到 checkpoint / class mapping / split / 内容组 /
分辨率精度 / TTA 与 prior bias SHA。在既有缓存上按固定配方（T=1.4、prior=0.60）做内容组 3 折条件交叉拟合：
pooled OOF Macro **0.7556778**、Micro **0.7639113**，相对 flip-TTA 无 prior 基线
**+0.1407pp / −0.0739pp**，纠正 144 / 破坏 155，逐折 ΔMacro 全为负。因此平台 +0.5582pp
不能归因于 prior。完整记录见
[L05 校准绑定与固定配方诊断](f05_focus_l05_calibration_binding_20260925.md)。

**2026-09-26 补充（单卡新候选搜索，不做复验版）：** 从 L05 checkpoint 独立续训三个候选
（seed 42、最多 3 个新增 epoch、`parent_kind=same_split_continue`，不重划数据、不重跑
A0/N1/O3/PTA、不补 seed3407/2026、不扫参），全部完成并落表
`results/l05_new_candidates.csv`。固定协议（flip + `mean_probabilities` + T=1.4 + 本阶段验证集
prior 0.60）下 decode macro 相对 L05 winner（0.7582651）的差：
NEW01 原图 ROI **−0.0335pp**（fail）、NEW02 融合目标 **+0.0144pp**（weak）、
NEW03 冲突组集合监督 **+0.0615pp**（weak，唯一同时提高 micro 的，+0.0470pp）。
三者均未过 +0.30pp 晋级门，**不替换 L05 winner、不生成提交候选、未上传平台**。
按预注册口径只记为「阶段二续训配方收益」，不归因于单个新组件。实现与完整记录见
[L05 单卡新候选搜索实现记录](l05_new_candidates_implementation_20260925.md)。
