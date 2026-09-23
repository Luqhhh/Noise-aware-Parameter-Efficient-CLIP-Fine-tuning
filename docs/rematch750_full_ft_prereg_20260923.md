# REMATCH750 全量数据对照（FULL00 → FULL01）：预注册与 NPU 交接

日期：2026-09-23

方案分支：`clairvoyanttt/rematch750-full-ft`

状态：实现完成，**等待 NPU 执行**。本文件不声明实验有效、不声明有收益。

## 1. 研究问题

把 V3 获胜配方 `RM_V3_B1024_E16_LR4` 的训练数据从 133,815 张（`train_dev.csv`）换成全部
148,695 张（`full_train.csv`），**在平台分上是否可辨识地更好？**

这是复赛现存数据里**唯一还没喂给模型的 11%**。同一阶段官方数据、同一划分谱系、同一条内容
分组，规则上完全合法；它不是新方法，是一次**数据口径**的对照。

## 2. 为什么现在做，以及诚实的收益预期

支持做的理由：

1. 这是**唯一未测的数据轴**。机制轴（MixUp、弱锚点、WD、LoRA、采样、分辨率、SAM…）都已有
   点位或在队友的 V4 批次里；数据量这一维此前被"RM-FULL 已取消"一直封着。
2. 取消它的现实约束是**算力**。现在 NPU 单点约 34 分钟，约束已不成立。
3. val 已无法裁决平台策略（本地/平台差 11.90pp），留一整个 val 做选模的价值随之下降。

必须同时写明的反向证据 —— **先验收益很小**：

- FT 这一族**已贴自身天花板**：FT 60.9657% 与 LT 60.8856% 在平台上只差 0.080pp，而本地 LT
  的 macro 反而更高。
- 本地/平台 **11.90pp** 的差距已被两条独立测量排除掉「输入几何」（推理侧几何探针）与
  「映射/头/编号缺陷」（T0 类别覆盖审计）两种解释，剩下的候选是**来源漂移**。**加 11%
  同源数据不会改善来源漂移。**
- 因此本轮的合理预期是落在**空档（±0.20pp）**，它买的是**信息**（关掉最后一条数据轴），
  不是分数。**它要花掉一个平台名额**，这一点必须在决策时明说。

**结论**：这是一次"把最后一条数据轴关掉"的对照，而不是一次期待涨分的冲刺。若名额机会成本
高于"关轴"的价值，应当延后而不做。

**2026-09-23 机会成本更新（用户当日指令）**：上式是按本机算力稀缺时的口径写的。现在队友 NPU 单点 ≈34.3 分钟且**可并行**，本包（FULL00 分钟级 + FULL01 ≈34 分钟）的算力机会成本**≈0**；并且**跑不跑、要不要占名额一律由队友裁决**，本机只提供判据与候选。因此本包现在是**最便宜的一条关轴动作**，唯一真实代价是那一个平台名额。

## 3. 两阶段结构，以及为什么是 **2 次训练而不是 1 次**

GPT-6 的方案里 P1 记为"1 次"。**按代码核对，这一步实际需要 2 次训练**，原因是 CM 血缘绑定：

- `rematch_assets.checkpoint_binding()`（`rematch_assets.py:93-100`）把 **`train_csv_sha256`**
  与 `feature_manifest_sha256` 一起写进 checkpoint 血缘。
- `validate_checkpoint(path, config, parent=True)`（:103-112）要求
  `meta['binding'] == checkpoint_binding(config)`，并额外要求父 checkpoint 的
  `experiment_id ∈ ('RM_LP', 'RM_FULL_LP')`。

全量 FT 的 `data.train_csv` 必须是 `full_train.csv`（sha `142d255e…`），而 dev RM-LP 绑定的是
`train_dev.csv`（sha `615d5104…`）。两者不等 ⇒ **dev RM-LP 不能作为全量 FT 的父**，
`validate_training()`（`trainer.py`）会在开训前直接拒绝。因此必须先有一个**全量数据的 LP**。

这条结论已机器验证，不是注释里的声明：

```bash
PYTHONPATH=reproducibility/aegis_f1 python -m pytest \
  reproducibility/aegis_f1/tests/test_rematch750_full_ft.py -q
# test_dev_lp_is_rejected_as_parent_for_full_ft:
#   同一个真实 RM-LP checkpoint，在 dev 口径下 validate_checkpoint(parent=True) 通过；
#   在全量口径下抛 ValueError('checkpoint data lineage mismatch')。
```

**代价几乎为零**：FULL00 是 `peft_mode: frozen` + `use_cached_training: true`，即冻结特征上的
线性探针，算力是分钟级；总时长仍由 FULL01 的 16 轮（约 34 分钟）主导。

| 阶段 | experiment_id | 角色 | 模式 | 轮数 |
|---|---|---|---|---|
| FULL00 | **`RM_FULL_LP`** | 全量线性探针；FULL01 唯一合法父 | `frozen` + 缓存特征 | 20（原 LP horizon） |
| FULL01 | `RM_FULL` | V3 配方跑全量；产出提交包 | `full_finetune` | 16（V3 horizon） |

`RM_FULL_LP` 这个名字**不是命名风格，是硬性要求** —— `validate_checkpoint(parent=True)` 只接受
`RM_LP` / `RM_FULL_LP` 两个 id。

## 4. 固定血缘

| 资产 | SHA-256 / 值 |
|---|---|
| 全量 train CSV | `142d255e86aadc2be4f4f1bee832948b30416c54bdc904159ff096ce8eb278ad` |
| dev train CSV（对照） | `615d510481e9df610371e84cbdeeb0058cc9a176989786a8ccf4eb30235b92c6` |
| val CSV | `d91106df365b62f9bf56eff51474c222925301546642fa427f6778258cb833ab` |
| 阶段内父（dev） | `RM_LP`，`d5cb8f5265754fd900d3efde23e24fefbcf616c747f2cab13e4dc2201fdd689b` |
| seed | 42 |
| V3 参考 macro / micro | 0.7234723568 / 0.7341398001 |
| 平台基准（RM-FT） | **0.6096570879179575**（22,828 / 37,444） |

注意 FULL00 **不是** dev RM-LP 的子节点，而是它的**同配方兄弟**：同样是"从官方权重重建分类头 +
冻结特征线性探针"（`peft_mode: frozen`，无 `init_checkpoint`），只是训练集换成全量。

执行前仍须由 NPU 机器复核全量 train CSV、val CSV、class_mapping 的真实 SHA；任一不符立即停止。

## 5. 判据

**主判据只有一个：`RM_FULL` 的平台分相对 RM-FT `0.6096570879179575` 的差。**

| 平台差 | 判定 | 后续动作 |
|---|---|---|
| ≥ **+0.30pp** | 达内部投资回报门 | `RM_FULL` 取代 RM-FT 成为正式单模型交付 |
| +0.20 ~ +0.30pp | 晋级门内，记为正信号 | 因只有单点、无重复，**不立即定稿**；由用户决定是否补一次确认 |
| **±0.20pp** | 空档 | **数据量这条轴关闭**，不再派生任何全量变体 |
| ≤ −0.20pp | 全量入训有害 | 关闭并回退 RM-FT |

**本地读数在本轮不存在，且禁止使用。** val 是训练集的子集（`val_dev ⊂ full_train`），
所以 `evaluation.py` 的 `raw_macro` / `raw_micro` 已被污染。流程会在两处自动标记这一点：

- `aegis_clip/rematch_assets.py:51` 要求 `validation_overlap_with_training == full_training`，
  声明不符直接拒绝；
- `trainer._validate_train_val_overlap(allow_overlap=True)` 要求 `val ⊆ train`，否则抛错 —— 即
  `validation_overlap_with_training: true` **不是注释，是让本轮能跑起来的必要条件**；
- `cli/rematch.py:65,86` 把 `class_support.csv` 的 `validation_status` 与
  `selected_report.json` 的 `validation_interpretation` 写成 `overlapping diagnosis`。

选点固定为 `last_epoch`（`trainer._checkpoint_is_selected` 对 `last_epoch` 恒返回 True，
故 `best.pt` == 最后一轮），`tiebreak_metric: null`。这与 `rematch_registry.full_configs()` 的
既有约定一致，其 docstring 原文即为："``last_epoch`` is intended for a full-data replay whose
epoch count was fixed by a preceding development run. Its overlapping validation metrics remain
diagnostic only and cannot silently choose the submitted epoch."

噪声口径：单 seed 噪声约 **0.90pp** 量级，故 <1pp 的平台差不过度解读；+0.30pp 门本身就是为噪声设的。

不使用测试集选择任何东西（测试集只用于 `infer` 出包）。

## 6. NPU 执行

在独立 worktree 拉取本分支；原始数据共享只读；输出固定到
`outputs/clairvoyanttt/rematch750_full_ft/`，不会覆盖 V3/V4、`codex/`、`xjn/` 或 `rematch750_npu/`。

```bash
git fetch origin
git worktree add /workspace/noise-wt/clairvoyanttt-full-ft clairvoyanttt/rematch750-full-ft
cd /workspace/noise-wt/clairvoyanttt-full-ft

source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=<当时空闲的物理卡号>
export OMP_NUM_THREADS=8
export PYTHONPATH=reproducibility/aegis_f1

# 先只物化 + 预检 + 打印命令，不训练。预检会校验每个输入路径存在，
# 并当场跑一遍 provenance gate；任一项不过就整体 fail-closed，不写任何文件。
/workspace/noise-npu-venv/bin/python scripts/run_rematch750_full_ft.py --device npu:0

# 审核无误后执行：train FULL00 → train FULL01 → infer FULL01。
# 每个阶段独立写盘；任一步失败即 fail-closed，不自动跳过继续解释后续结果。
/workspace/noise-npu-venv/bin/python scripts/run_rematch750_full_ft.py \
  --device npu:0 --trials FULL00 FULL01 --execute
```

### 6.1 路径深度：本配置与 `configs/*.yaml` 差一层

`aegis_clip.config._resolve_paths()` 把相对路径解析为**相对 config 文件自身所在目录**
（`config.py:640-641`），不是仓库根。所以：

- `configs/rematch750_v3_b1024_e16_lr4.yaml`（一层）用 `../artifacts/...`；
- `configs/rematch750_full_ft/FULL0x.yaml`（两层）用 `../../artifacts/...`；
- 二者都落到**仓库根**。

把本目录的配置复制到别处、或改变目录层级，路径会静默指到别的地方 —— 失败会伪装成训练中途的
`FileNotFoundError`。runner 的预检就是为这件事加的：它会把每个解析后的绝对路径与存在性逐一打印，
并直接调用 `validate_dataset()`。

### 6.2 train 与 infer 必须用同一份 runtime YAML

`cli/rematch.py:114` 的 `infer` 会再次执行 `validate_checkpoint(checkpoint, config)`，其中
（`rematch_assets.py:109`）要求 `meta['training_config_sha256'] == sha256_file(config['_config_path'])`。
即 **infer 用的配置文件必须与 train 逐字节相同**。runner 已把两阶段都指向物化后的
`_runtime_configs/FULL01.yaml`；若是手工分两步跑，务必沿用同一路径，否则会在推理前被拒绝。

`infer` 完成后 `register()` 会向 `results/rematch_submission_registry.csv` 追加一行
（candidate = `RM_FULL`，status = `ready`），需由执行方提交推送。

## 7. 首轮验收

FULL00 第一个 optimizer step 后必须核对：

- 全量 train CSV / val CSV / class_mapping 的 SHA 与 §4 一致；
- 训练样本数为 **148,695**（不是 133,815），每轮 batch 数与 batch_size 256 相符；
- 参数组只有 head（`backbone_lr: 0.0` 冻结），`peft_mode: frozen`，走缓存特征；
- 有效 optimizer step 增长、loss/grad 有限、无 AMP overflow；
- 输出只位于 `outputs/clairvoyanttt/rematch750_full_ft/`。

FULL01 第一个 optimizer step 后必须核对：

- 父 checkpoint 解析到 `.../RM_FULL_LP/seed42/checkpoints/best.pt`，且其 `.binding.json` 的
  `train_csv_sha256` 等于全量 CSV 的 SHA（若等于 `615d5104…` 说明接错了 dev LP，立即停止）；
- 两组 LR 分别为 head 4e-4 / backbone 1.2e-5，`batch_size: 1024`，16 轮；
- `validation_interpretation` 已标为 `overlapping diagnosis`。

两点都通过后串行跑完，`infer` 产出 37,444 行 CSV/ZIP 并过 9/9 提交校验。
训练曲线、checkpoint SHA 与平台成绩由 NPU 执行方回填；在此之前本文件只声明 "ready for NPU"。

## 8. 已知风险

| 风险 | 处置 |
|---|---|
| 全量数据下 val 被污染，本地无任何可信读数 | 已在 §5 固定：判据只用平台分，禁止引用本地数字 |
| 花掉一个平台名额换一个大概率空档的结论 | 已在 §2 明写机会成本；**是否值得由队友裁决**（本机只提供判据与候选，不做名额决策） |
| 若 FULL00 未被先行产出，FULL01 会在开训前被血缘门拒绝 | runner 默认按 `FULL00 FULL01` 顺序执行；单独跑 FULL01 时预检要求父已存在 |
| `npu_fused_adamw` / 大 batch 在 NPU 上的既有崩溃记录（`cudaErrorUnknown`） | 沿用 V3 已跑通的 NPU 数值块；失败即停，不自动重启 |
| 全量训练集更大，首个 epoch 的 warmup 表现在 148,695 张上 | `ce_warmup_epochs: 2` 按 **epoch** 而非 step 计，比例不变 |
