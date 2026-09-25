# L05 TTA+prior：交付保护、种子/覆盖/绑定修复与固定配方条件交叉拟合诊断

- artifact_type: execution record + implementation change
- branch: `focus/f05-four-lines`
- date: 2026-09-25
- platform submission in this round: **none**（不重训、不重提交、不生成测试候选）
- test data use: none（本文件所有脚本均不读取测试图；提交包只做只读哈希/格式校验）

本文件对应三项要求：①保护已有最高分包；②修复训练种子、出包覆盖、校准缓存绑定三个执行问题；
③在已有缓存上按固定配方（T=1.4、prior=0.60、`mean_probabilities`、horizontal-flip）做条件交叉拟合诊断，
不重新扫描温度/强度、不生成测试候选。

## 1. 已保护的最高分包

| 项 | 值 |
|---|---|
| candidate | `L05_T14_P060` |
| 仓库路径 | `outputs/f05_focus_l05/L05_T14_P060` |
| 平台分 | **66.94797564362783%**（center-only 66.38980878111312%，+0.5582pp） |
| `pred_results.csv` SHA-256 | `51e0efe7178528d23993a44351069d2829b0cd3669c195a77d8d879e66776a75` |
| `submission.zip` SHA-256 | `e788f07636b80abb61685335cd8df108803863ea36ffbde8b353f8e8317fcd7f` |
| `manifest.json` SHA-256 | `3712f796bf8736207c5dc28c38b0b4364248f4ba32d09ffaf4a475a8762ef79b` |

核对动作（只读，未改动任何候选目录）：

- `scripts/check_submission.py` 对既有 CSV/ZIP 重跑 9/9 通过（37,444 行，覆盖、标签格式、ZIP 内容均通过）；
- 三件文件的实际 SHA-256 与 `manifest.json` 中的记录一致；
- 训练侧身份核对：checkpoint `best.pt` SHA-256 `35d17c0c3b9f281e13353959d098c2713def6b7d508d334de5d3a7e769cd2397`
  （与 `manifest.json` 一致），训练配置 `L05_RM_V5_L05_CUDA_LOCAL_cuda_0_mb4.yaml` SHA-256
  `70576eddbd625b20ba942bc5fa74e3687eafe5092d425a9027a68a7899018c44`（与 `manifest.json` 一致）；
- 常量登记：`results/f05_focus_l05_protected_packages.json` 把该 candidate 标为
  `protected_do_not_overwrite`；`scripts/build_l05_tta_prior_submission_final.py` 读取该登记表并直接拒绝
  同名 candidate；
- 机器可读审计：`results/f05_focus_l05_binding_audit_20260925.json`。

**没有因为换单卡而重训或重交**：checkpoint、配置、CSV/ZIP 全部保持原字节。

## 2. 三个执行问题的修复

### 2.1 训练种子必须真正改变（`scripts/run_l05_local_retrain.py`）

原脚本没有 seed 参数，运行配置也没有覆盖逻辑，`project.seed` 始终取自 `L05.yaml` 的 42。现在：

- 新增 `--train-seed`，写入 `config["project"]["seed"]`。这个值被 trainer 用于
  `set_seed()`、shuffle generator、worker seed，并决定 `.../seed{seed}` 运行目录，因此训练随机性真的改变；
- 运行配置与 preflight 记录的文件名包含 seed：`..._mb4_seed3407.yaml` / `.preflight.json`，
  每个 seed 独立，不覆盖已审计的 `..._mb4.yaml`；
- 数据划分**不重划**：driver 仍指向冻结的 `artifacts/stages/repechage/20260921/{train_dev,val_dev}.csv`；
- 新增 fail-closed 前置检查（写入 `.preflight.json`）：
  - `dataset_manifest.json` 记录的 split seed 必须等于 `SPLIT_SEED=42`，且 manifest `files` 中每个
    资产的 SHA-256 必须与磁盘一致（train/val 内容组划分不变）；
  - RM-LP parent 的 SHA-256 必须等于审计值
    `d5cb8f5265754fd900d3efde23e24fefbcf616c747f2cab13e4dc2201fdd689b`；
  - 训练配方不变：把 `project.seed` 归一化后与参考配置
    `.../L05_RM_V5_L05_CUDA_LOCAL_cuda_0_mb4.yaml` 逐字段比较，必须 `recipe_diff == []`。

实测（dry-run，未训练）：

```bash
python3 scripts/run_l05_local_retrain.py --dry-run --device cuda:0 \
  --microbatch 4 --grad-accum 256 --train-seed 3407 \
  --stage-dir /home/lux1/noise/artifacts/stages/repechage/20260921 \
  --train-root /home/lux1/noise/train --test-root /home/lux1/noise/test \
  --rm-lp-checkpoint outputs/f05_focus/local_parent/RM_LP/seed42/checkpoints/best.pt \
  --experiment-id RM_V5_L05_CUDA_LOCAL \
  --output-root outputs/f05_focus_l05 --runtime-dir outputs/f05_focus_l05/_runtime_configs
```

结果：`recipe_diff: []`；seed3407 与 seed42 配置在把 `seed` 行归一化后 `diff` 为空；
`run_dir = outputs/f05_focus_l05/RM_V5_L05_CUDA_LOCAL/seed3407`。
反向测试：传入错误的 `--expected-parent-sha256` 立即报错退出。

> 注：这里**只用 dry-run 验证配置与前置检查**。seed3407/seed2026 的完整训练尚未启动，
> 是否占用空闲 GPU 由队长决定；本轮不消耗训练预算。

### 2.2 出包默认拒绝覆盖（`scripts/build_l05_tta_prior_submission_final.py`）

原实现 `if destination.exists(): shutil.rmtree(destination)` 会毁掉已有最高分包。现在：

- 目标 candidate 目录存在即 `SystemExit`，除非显式 `--allow-overwrite`（默认关闭）；
- `--tag` 命中保护登记表直接拒绝（反向测试：`--tag L05_T14_P060` → 拒绝）；
- `create_submission(..., overwrite=...)` 与桌面目录都改为 `exist_ok=False`，桌面目录不存在时 fail-closed
  而不是静默新建路径；
- 新 candidate 使用新 tag，避免一次复验破坏既有产物。

### 2.3 校准缓存绑定到模型（新模块 `aegis_clip/tta_prior_binding.py`）

新增 fail-closed 绑定链，所有字段都从磁盘资产**重新计算**（因此旧缓存无需重写也能核验）：

| 绑定项 | 来源 |
|---|---|
| checkpoint SHA | checkpoint 文件 + trainer 写的 `best.binding.json` |
| class mapping SHA | `class_to_idx.json`，并与 checkpoint binding 交叉核对 |
| validation 样本顺序 | 缓存 `paths` vs `val_dev.csv` 逐行顺序 |
| split/group SHA | `dataset_manifest_sha256` + `dataset_fingerprint` + `content_groups.csv` + 内容组集合 SHA |
| 分辨率/预处理/精度 | checkpoint binding 的 `input_resolution` / `preprocessing` / `*_precision` / `autocast` |
| TTA 融合与 temperature | 缓存记录的 `tta_fusion` + 声明的固定配方 |
| prior bias SHA | 拟合出的 bias 张量规范字节 SHA-256，并记录所依据的 cache SHA |

`validation_cache_identity()` 会拒绝：checkpoint 不符、class mapping 变化、val CSV 变化、
样本顺序/标签不符、split seed 不是 42、内容组缺失、融合方式未知。
`fit_bound_prior()` 产出的记录带有 `cache_sha256`、`checkpoint_sha256`、`bias_sha256`、
`fit_scope`、`test_data_used=false`；`verify_prior_record()` 在任一字段不符时抛错。
因此**新 seed checkpoint 无法在未核验的情况下复用旧模型的 prior bias**。

`scripts/cache_validation_tta_logits.py` 现在写出 `format_version=2` 与 `provenance` 块（未来缓存自带绑定）；
旧 `format_version=1` 缓存仍可被上述函数核验（实测旧缓存全部字段重算一致）。

## 3. 固定配方条件交叉拟合诊断

脚本：`scripts/diagnose_l05_fixed_recipe_crossfit.py`
结果：`results/f05_focus_l05_fixed_recipe_crossfit_20260925.json`

- 复用既有缓存 `artifacts/f05_focus_l05/l05_val_branch_logits.pt`
  （SHA-256 `7bfcfb08458430ba339f086cb7abb1799f70c8431f77b582ed842509f40672dc`）；
- 复用既有 14,880 个验证样本的内容组划分（`content_groups.csv`，14,658 个内容组），
  用仓库既有 `analysis.oof.build_folds.assign_group_stratified_folds` 生成 3 折、
  seed 42、内容组不跨折；折分 SHA-256 `14748868b6d707c60e4ae9468bb73c71b5b5556c83600429330b4ef4fe665139`；
- 固定配方 T=1.4、prior=0.60、`mean_probabilities`、horizontal-flip，`--allow-recipe-drift` 未开启
  （任何偏离直接报错）；**不重扫温度、不重扫 prior 强度、不读取测试集**；
- 对每一折，只用其余两折拟合 bias，再把 bias 应用到该折。

### 结果

| 口径 | Macro | Micro | 纠正数 | 破坏数 | 变更数 |
|---|---:|---:|---:|---:|---:|
| center（单视图） | 0.7524971 | 0.7629704 | — | — | — |
| flip-TTA，无 prior | 0.7542707 | 0.7646505 | — | — | — |
| 全验证集拟合 prior（生产口径，重建） | 0.7582651 | 0.7665322 | 138 | 110 | 604 |
| **条件交叉拟合 pooled OOF** | **0.7556778** | **0.7639113** | **144** | **155** | **703** |

pooled OOF 相对“flip-TTA 无 prior”基线：**Macro +0.1407pp / Micro −0.0739pp**，
且破坏数（155）多于纠正数（144）。

逐折（每折 bias 只用另外 9,9xx 个样本拟合，评估该折）：

| fold | 留出样本 | 拟合样本 | 留出类数 | raw Macro | 交叉拟合 Macro | ΔMacro | ΔMicro | 纠正 | 破坏 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4,953 | 9,927 | 749 | 0.7613210 | 0.7608664 | −0.0455pp | 0.0000pp | 45 | 45 |
| 1 | 4,973 | 9,907 | 746 | 0.7476401 | 0.7470914 | −0.0549pp | −0.0201pp | 60 | 61 |
| 2 | 4,954 | 9,926 | 745 | 0.7593408 | 0.7566614 | −0.2679pp | −0.2019pp | 39 | 49 |

（逐折 Macro 只在“该折出现过的类”上取平均，类集合各折不同；pooled OOF 在全 750 类上计算，
两者不能直接互相比大小。逐折 Δ 与各自 raw 基线同口径。）

### 必须保留的限制

checkpoint、温度 1.4、强度 0.60 都是在**整个验证集**上选出来的；本诊断只把“拟合 bias 时不得读取被评估折”
这一条补上，**不能撤销**此前的全验证集选模。因此它只能叫
**固定配方的条件交叉拟合诊断（fixed-recipe conditional cross-fit diagnostic）**，
不能当作完整流程的无偏验证，也不能用事后切分包装成无偏结论。

### 读数

- 全验证集口径的 +0.5768pp（相对 center）里，flip-TTA 本身贡献约 +0.177pp，
  剩余部分包含 prior 与“全验证集选 T/强度”的选择效应；
- 条件交叉拟合后，prior 的净 Macro 收益降到 +0.14pp（Micro 为负，纠正少于破坏），逐折 ΔMacro 全为负；
- 因此 L05 T14 P060 的平台 +0.5582pp **不能归因于 prior**；原包按“观测到的平台收益”继续保护，
  但方法学上应记为 TTA + 选择的组合收益。

## 4. 复现命令

```bash
# 绑定与保护包审计（只读，不跑模型）
python3 scripts/verify_l05_tta_prior_binding.py \
  --checkpoint outputs/f05_focus_l05/RM_V5_L05_CUDA_LOCAL/seed42/checkpoints/best.pt \
  --config outputs/f05_focus_l05/_runtime_configs/L05_RM_V5_L05_CUDA_LOCAL_cuda_0_mb4.yaml \
  --val-branch-cache artifacts/f05_focus_l05/l05_val_branch_logits.pt

# 固定配方条件交叉拟合诊断（只需缓存，约 33 秒，CPU）
python3 scripts/diagnose_l05_fixed_recipe_crossfit.py \
  --checkpoint outputs/f05_focus_l05/RM_V5_L05_CUDA_LOCAL/seed42/checkpoints/best.pt \
  --config outputs/f05_focus_l05/_runtime_configs/L05_RM_V5_L05_CUDA_LOCAL_cuda_0_mb4.yaml \
  --val-branch-cache artifacts/f05_focus_l05/l05_val_branch_logits.pt \
  --output results/f05_focus_l05_fixed_recipe_crossfit_20260925.json

# 单测
PYTHONPATH=reproducibility/aegis_f1 python3 -m pytest \
  reproducibility/aegis_f1/tests/test_tta_prior_binding.py -q
```

## 5. 本轮改动文件

- `reproducibility/aegis_f1/aegis_clip/tta_prior_binding.py`（新增）
- `scripts/diagnose_l05_fixed_recipe_crossfit.py`（新增）
- `scripts/verify_l05_tta_prior_binding.py`（新增）
- `reproducibility/aegis_f1/tests/test_tta_prior_binding.py`（新增）
- `scripts/run_l05_local_retrain.py`（种子 + 前置检查）
- `scripts/build_l05_tta_prior_submission_final.py`（防覆盖 + 绑定核验 + 固定配方）
- `scripts/cache_validation_tta_logits.py`（写出 provenance 块）
- `results/f05_focus_l05_protected_packages.json`（保护登记表）
- `results/f05_focus_l05_fixed_recipe_crossfit_20260925.json`（诊断结果）
- `results/f05_focus_l05_binding_audit_20260925.json`（绑定审计）
