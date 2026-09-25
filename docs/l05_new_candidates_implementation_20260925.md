# L05 单卡新候选搜索（不做复验版）实现记录

- artifact_type: implementation record
- branch: `focus/f05-four-lines`
- date: 2026-09-25
- status: **代码已实现、最小检查已通过、单卡 smoke 已跑通；3×3 epoch 正式续训队列已于
  2026-09-25 22:16 在本机 RTX 4070 上串行启动，结果尚未产出，无新分数，无平台提交**
- test data use: none（三个候选都不读取测试图；本文件不含任何新预测指标）

## 0. 已核对的代码基线

用户给出的 5 个 blob 全部逐一核对一致：

| 文件 | blob |
|---|---|
| `results/f05_focus_summary.csv` | `18836a4c35b2ceb2334ba7e368fb4651a0849b72` |
| `configs/rematch750_search_v5/L05.yaml` | `596eda88a888164f513a6aacd5c743c314e7c83e` |
| `aegis_clip/trainer.py` | `16421c1427e548b577c813ba640f536c3a4ca86a` |
| `aegis_clip/local_inference.py` | `2077d37b903dfdf2384df7037246cc975c3e3e49` |
| `analysis/noisy_labels/high_precision_drop.py` | `229410090454876300494214bb3b80a41470f9f8` |

## 1. 运行环境（重要）

本机为 WSL2，GPU 为 **NVIDIA GeForce RTX 4070 Laptop, 8 GiB（单卡）**。但 agent 的命令进程跑在
`bwrap`（bubblewrap）沙箱里，`/dev` 是精简挂载：`/dev/dxg`、`/dev/nvidia*` 不可见，
`nvidia-smi` 报 `GPU access blocked by the operating system`，`torch.cuda.is_available()` 为 `False`。
只有让命令脱离该文件沙箱（`danger-full-access`）才能看到 GPU。因此：

- 所有训练/缓存/评估/推理命令必须在该沙箱之外执行；
- 本仓库的 GPU 作业串行，同一时刻只有一个 GPU 作业（本轮由 `scripts/run_l05_new_candidates.py` 保证）。

## 2. 三个候选的实现

三个候选都从同一个已完成的 L05 CUDA checkpoint **独立**初始化，互不为父模型；只跑 seed 42；
不重跑 A0/N1/O3/PTA；不新增 LR/WD/q/阈值网格。

### NEW01 —— 从原图提取局部细节（`roi_source: original`）

原路径在 `trainer.py` 里对**已经缩小到 384 的模型输入张量**调用 `attention_guided_crop`，
再用 `affine_grid/grid_sample` 裁剪放大。NEW01 改为从原始 RGB 提取同一空间区域，只缩放一次到 384。

- `aegis_clip/data.py`：新增 `GeometryAwareImageDataset`。它用与 `weak_rrc_flip` 完全相同的配方
  （`RandomResizedCrop(384, scale=(0.7,1.0), ratio=(0.85,1.15), BICUBIC)` → `RandomHorizontalFlip(0.5)`
  → CLIP `ToTensor+Normalize`）生成 global 视图，同时记录
  `geometry = (H0, W0, i, j, h, w, flip, S)`（原图高宽、RRC 在原图上的 top/left/crop_h/crop_w、
  翻转标志、模型输入边长）。`original` 返回原始 RGB（uint8），`collate_geometry_batch` 只对**当前
  batch** 的原图做零填充并附带真实尺寸，padding 不参与 ROI 边界。
- `aegis_clip/original_roi.py`（新）：`attention_crop_box` 逐字复刻 `attention_guided_crop` 的
  topk→加权质心→`[half, size-half]` 钳制，返回 **global 输入坐标**下的框；`global_box_to_original`
  做逆翻转 + 逆缩放（`x = j + u*w/S`，翻转时先 `u' = S-u`）并钳到 `[0,W0]×[0,H0]`；
  `original_roi_batch` 在**原图**上裁剪、单次 bicubic 缩放、同方向翻转、归一化。
  **注意**：`original_roi_batch` 内部已经调用逆映射，调用方必须传 global 坐标的框（不要二次逆映射）。
- `trainer.py`：`attention_local_training.roi_source: original` 时走新路径；global 输入与 global 分支
  不变；box/top-k 仍 `detach`；推理协议不变（不先加测试局部视图）。
- 边界：本候选只尝试**保住原图已有的细节**，不生成细节，也不找回被 global crop 排除的区域；
  原图分辨率低时可能没有收益。

### NEW02 —— Flip 融合输出参与训练（`loss.flip_fusion`）

对同一 global 视图 `x` 与 `F(x)`：

```
p_o = softmax(z(x)/1.4);  p_f = softmax(z(F(x))/1.4);  p_mix = (p_o+p_f)/2
l_q(p,y) = (1 - p_y^q)/q,  q = 0.5
L_global_new = 0.5 * [l_q(p_o,y)+l_q(p_f,y)]/2 + 0.5 * l_q(p_mix,y)
```

- `aegis_clip/candidate_losses.py`（新）：`probability_generalized_cross_entropy`、
  `flip_fusion_global_loss`（返回 loss 与 `p_o/p_f/p_mix/term_o/term_f/term_mix`）。
- `trainer.py`：`loss.flip_fusion.enabled` 时，global 分类项替换为 `L_global_new`；两个分支都反传，
  不做 stop-gradient；local 分支、局部门控、global/local 总权重与 feature anchor 保留。
- prior bias 不进入训练梯度；不扫 TTA 温度。
- 历史依据是初赛 PRELIM75 v5 的融合分类目标（同条件平台仅 +0.0641pp），本候选不承诺大幅收益。

### NEW03 —— 跨标签重复图用候选标签集合监督（`loss.set_supervision`）

N1 把 `suspect OR duplicate_conflict` 的样本权重置零。NEW03 只作用于 `train_dev` 内按解码 RGB
与尺寸分组的**跨标签 exact duplicate group**，把该组官方原标签的去重集合 `S_g` 作为监督：

```
r_g = sum_{c in S_g} softmax(z)_c;   L_set = (1 - r_g^q)/q, q = 0.5
```

- `aegis_clip/duplicate_sets.py`（新）：`build_candidate_label_sets` 写出 sidecar（原始 CSV/标签/
  内容组不动），`load_candidate_label_sets`，`conflict_group_report`。只读 `train_dev.csv`，
  fail-closed 拒绝 val/test 文件名。
- `aegis_clip/candidate_losses.py`：`set_generalized_cross_entropy` + `build_padded_candidate_tensors`
  （每实例权重 `1/m_g`，使一个组合计权重为 1）。
- `trainer.py`：global 与 local 分类目标**使用同一个集合**；局部门控沿用；非冲突样本为单元素集合，
  退化为普通 GCE。
- 真实统计（只读扫描，未写 sidecar）：`num_groups=131723`、`num_conflict_groups=1705`、
  `num_conflict_samples=3527`（占 train_dev 2.64%）。局限：依赖“真实标签在给定集合内”的工作假设；
  若整组全错仍可能有害；不解决集合内哪个类正确。

## 3. 阶段二续训入口（同一 train/val/mapping）

没有把 L05 伪装成 RM-LP，也没有关闭全部谱系校验。续训使用既有显式入口：

```yaml
parent_kind: same_split_continue
parent_experiment_id: RM_V5_L05_CUDA_LOCAL
```

`validate_checkpoint(parent=True)` 会核验 L05 的 `*.binding.json` 与由候选配置重算的
binding 一致（class mapping、dataset manifest、fingerprint、train_csv、official 权重、384px
预处理与精度），并要求 `experiment_id == parent_experiment_id`。运行前 preflight 还会：
冻结 split 的 manifest 记录 seed 必须为 42，且 `files` 中每个资产 SHA-256 与磁盘一致；
L05 parent 的 SHA-256 为 `35d17c0c3b9f281e13353959d098c2713def6b7d508d334de5d3a7e769cd2397`。

## 4. 共同训练预算

`configs/l05_new_candidates/{NEW01,NEW02,NEW03}.yaml`（由 `L05.yaml` 派生，保证 schema 一致）：

- seed 42；epochs 3 / schedule_epochs 3；3-epoch cosine；`lr_warmup_epochs=0`；
- backbone LR `3e-6`、head LR `1e-4`、WD / GCE q=0.5 / anchor 沿用；
- 384px、effective batch 1024（microbatch × grad_accum = 1024）；
- `ce_warmup_epochs=0` 且 `attention_local_training.start_epoch=1`：从阶段二第 1 轮就用 GCE 与
  local 监督，不继承 `start_epoch=5`，也不重跑两轮 CE warmup；
- 从 L05 权重继续优化，trainer 重新建优化器（不伪称恢复旧完整训练轨迹）。

## 5. 最小实现检查（已通过）

```bash
python3 scripts/check_l05_new_candidates.py            # CPU，结果 results/l05_new_candidates_checks.json
```

| 检查 | 结果 |
|---|---|
| NEW01 坐标合法（框在 `[0,W0]×[0,H0]`、非退化） | true |
| NEW01 ROI 与“对 384 输入再裁剪”实质不同（`max abs delta=1.0`） | true |
| NEW01 记录原图尺寸 / ROI 形状 / 激活样本数 | `[[800,800],[956,1300],[644,500]]` / `[N,3,384,384]` / 3 |
| NEW02 两个分支梯度均非零、`p_mix` 为两路概率均值 | true |
| NEW03 单元素集合退化为普通 GCE | true |
| NEW03 同组副本共享同一候选集合、集合去重有序 | true |
| NEW03 实质受影响组数 / 样本数 | 1705 / 3527 |

注意 `results/l05_new_candidates.csv` 只有表头，**没有数据行**——正式续训尚未完成。

队列状态：2026-09-25 22:16 已按上面的命令在本机 RTX 4070（单卡、串行）启动，先跑 NEW01。
每个候选约需数小时（384px、133,815 张、有效 batch 1024），队列结束后
`results/l05_new_candidates.csv` 会自动写入三行；在那之前本文件不报告任何候选分数。

## 6. 单卡串行队列

```bash
# 正式（3 epoch/候选，串行，同一张卡，一个时刻一个作业）
python3 scripts/run_l05_new_candidates.py \
  --candidates NEW01,NEW02,NEW03 --device cuda:0 \
  --microbatch 4 --grad-accum 256 --num-workers 2
```

每个候选：preflight → train → 缓存本候选自己的 validation flip-TTA branch logits →
`scripts/evaluate_l05_candidate.py` 报告 center 与固定协议（flip、`mean_probabilities`、T=1.4、
本阶段验证集拟合 prior、strength 固定 0.60）的 macro/micro/纠正数/破坏数 → 追加
`results/l05_new_candidates.csv` 一行。**不自动上传平台**，不重扫温度/强度，不生成其它测试候选。

`results/l05_new_candidates.csv` 字段：
`candidate_id,base_checkpoint_sha,train_seed,training_recipe,selected_epoch,center_macro,center_micro,decode_macro,decode_micro,delta_vs_l05,affected_samples,corrections,regressions,status,platform_score`。

### smoke

正式 3 epoch 之前先做实现 smoke：`--smoke-max-steps 3 --grad-accum 8`（只跑 3 次优化更新，
`train.max_steps` 是仅用于实现的开关，日志会打印 WARNING 标注这不是预算内运行）。
smoke 使用真实 train/val、真实 L05 父模型、真实 GPU，用于确认三个开关端到端接通；
smoke 分数不进入 `results/l05_new_candidates.csv`，也不作为方法结论。

实测（RTX 4070 Laptop 8 GiB，真实 train_dev/val_dev，真实 L05 父模型）：

| 候选 | smoke 结果 | 说明 |
|---|---|---|
| NEW01 | 通过，`selector=0.746876` | first-step audit 通过；原图 ROI 路径实际执行 |
| NEW02 | 通过，`selector=0.746432` | 翻转分支参与反传 |
| NEW03 | 通过，`selector=0.746862` | `Set supervision active \| affected_samples=3527 \| max_set_size=7` |

**这三行只是 3 次优化更新后的 smoke checkpoint 指标，不是方法结论、不进入结果表。**
smoke 过程修掉了两处实现缺陷：①NEW01 的 ROI 调用方曾对 box 二次逆映射（`original_roi_batch`
内部已逆映射）；②NEW03 的候选集合张量留在 CPU，与 GPU 上的 `batch_indices` 索引冲突。


## 7. 本轮不做

- 不重跑 A0/N1/O3/PTA，不为组合而组合失败组件；不补 seed3407/2026、不补纯复现实验；
- 不做 epoch 拉长 / warmup / LR / WD / q / prior 强度 / crop-size 网格；
- 不使用初赛图片、权重、trust 或伪标签资产；不读取测试图训练或拟合 prior；
- 不把有监督 encoder 提取的 train 特征称为全链路 OOF；不用本地提升承诺平台分数。

## 8. 改动文件

新增：
`aegis_clip/candidate_losses.py`、`aegis_clip/duplicate_sets.py`、`aegis_clip/original_roi.py`、
`tests/test_candidate_losses.py`、`tests/test_duplicate_sets.py`、`tests/test_original_roi.py`、
`tests/test_l05_new_candidate_wiring.py`、`scripts/run_l05_new_candidates.py`、
`scripts/evaluate_l05_candidate.py`、`scripts/check_l05_new_candidates.py`、
`configs/l05_new_candidates/{NEW01,NEW02,NEW03}.yaml`、`results/l05_new_candidates.csv`（仅表头）、
`results/l05_new_candidates_checks.json`。

修改：
`aegis_clip/data.py`（geometry-aware dataset + collate）、`aegis_clip/trainer.py`（三个开关 +
`train.max_steps` 仅实现用开关 + 原图 ROI 接线）。

测试：完整 Aegis 套件 **2 failed / 745 passed / 8 skipped**；2 处失败均为
`tests/test_scope_protocol.py` 的冻结资产缺失（`ScopePreflightError`），与既有已知口径一致。
