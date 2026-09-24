# F05 Focus GPU1 部署包

第二台机器按本文件拉取同一分支，并用显式路径运行分配到的单元。所有实验必须使用
`focus/f05-four-lines` 的同一 commit；不要在 GPU1 上另开分支或修改配方。

## 1. GPU1 分配

| Round | GPU0 | GPU1 | GPU1 需要的数据 |
|---:|---|---|---|
| 0 | A0 | P0 | F05 `f05_val_logits.pt` |
| 1 | C0 | N1 | 训练/验证 split、feature cache、RM-LP parent、`hp_noise_manifest.csv` |
| 2 | D1 | D2 | F05 checkpoint、`clean070.csv`、D2 cache |
| 3 | D3 | strongest second seed | 对应 strongest 的可复现配置 |
| 4 | strongest seed3407 | strongest seed2026 | 同一个 strongest 方案的 runtime config |

Round 0 可以真正并行：GPU0 跑 A0，GPU1 跑 P0。P0 也不需要 GPU，可以在 CPU 上完成。

## 2. GPU1 获取代码

```bash
git clone git@github.com:Luqhhh/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning.git
cd Noise-aware-Parameter-Efficient-CLIP-Fine-tuning
git fetch origin focus/f05-four-lines
git checkout -B focus/f05-four-lines origin/focus/f05-four-lines
```

如果 GPU1 已有 repo：

```bash
git fetch origin focus/f05-four-lines
git checkout -B focus/f05-four-lines origin/focus/f05-four-lines
```

代码/配置/脚本通过 git 推送；`train/`、`test/`、`artifacts/features/`、checkpoint
等大文件不放进 git，使用 `sync_required_assets.sh` 或共享盘同步。

## 3. GPU1 环境变量

```bash
export REPO_ROOT="$PWD"
export STAGE_DIR="$REPO_ROOT/artifacts/stages/repechage/20260921"
export TRAIN_ROOT="$REPO_ROOT/train"
export TEST_ROOT="$REPO_ROOT/test"
export F05_CHECKPOINT=/path/to/F05/seed42/checkpoints/best.pt
export RM_LP_CHECKPOINT="$REPO_ROOT/outputs/rematch750_npu/RM_LP/seed42/checkpoints/best.pt"
export PYTHONPATH="$REPO_ROOT/reproducibility/aegis_f1"
export CUDA_VISIBLE_DEVICES=0   # GPU1 进程内逻辑卡按实际隔离修改
```

## 4. Round 0 — GPU1 P0

P0 只读 F05 validation logits。先确保：

```text
artifacts/f05_focus/f05_val_logits.pt
```

运行：

```bash
python3 scripts/run_f05_focus_unit.py \
  --unit P0 \
  --device cpu \
  --val-logits artifacts/f05_focus/f05_val_logits.pt \
  --output-root outputs/f05_focus \
  --summary results/f05_focus_summary.csv \
  --execute
```

它只拟合 validation、只在 validation 上选择 `s in {0.25,0.50,0.75,1.00}`，并输出
`artifacts/f05_focus/prior_bias.pt`。test application 单独执行，不做 test fitting。

## 5. Round 1 — GPU1 N1

N1 与 C0 唯一差异是 reject sidecar。开始前必须确认：

```text
artifacts/f05_focus/hp_noise_manifest.csv
```

如果 GPU1 上没有该文件，先在 GPU0 或已有 quality/OOF 资产的机器上构建，再同步。

运行：

```bash
python3 scripts/run_f05_focus_unit.py \
  --unit N1 \
  --device cuda:0 \
  --stage-dir "$STAGE_DIR" \
  --train-root "$TRAIN_ROOT" \
  --test-root "$TEST_ROOT" \
  --rm-lp-checkpoint "$RM_LP_CHECKPOINT" \
  --hp-noise-manifest artifacts/f05_focus/hp_noise_manifest.csv \
  --num-workers 16 \
  --output-root outputs/f05_focus \
  --summary results/f05_focus_summary.csv \
  --execute
```

`--execute` 默认不启用；先去掉 `--execute` 检查 runtime config 和 effective batch，确认
`batch_size x grad_accum_steps == 1024` 后再执行。

## 6. Round 2 — GPU1 D2

D2 不是单命令全自动流程。必须先准备：

1. `artifacts/f05_focus/clean070.csv`；
2. F05 validation center logits；
3. crop160/top5 的 F05 M1 reference logits；
4. `cache_part_token_adapter_features` 生成 D2 train/val cache；
5. 用 `aegis_clip.cli.train_part_token_adapter` 训练。

固定参数：

```text
F05 frozen parent
local geometry: input 320 -> crop160 -> top5
part_top_patches = 8
part_temperature = 0.07
bottleneck = 64
residual_scale = 0.25
dropout = 0.1
train subset = clean070.csv
```

cache 输出必须使用独立路径，避免与 D1 互相覆盖。

## 7. 数据同步

大文件不要提交到 git。推荐在 GPU0 上执行：

```bash
GPU1_HOST=user@gpu1-host \
GPU1_ROOT=/workspace/noise \
bash deploy/f05_focus_gpu1/sync_required_assets.sh
```

默认是 dry-run，只打印 rsync 命令；确认后加：

```bash
... bash deploy/f05_focus_gpu1/sync_required_assets.sh --execute
```

至少需要：

- `train/` 约 30G；
- `test/` 约 1.6G；
- `artifacts/stages/repechage/20260921/`，其中 `features/features.pt` 约 304M；
- `outputs/rematch750_npu/RM_LP/seed42/checkpoints/best.pt`；
- Round 0 的 `artifacts/f05_focus/f05_val_logits.pt`；
- N1 的 `artifacts/f05_focus/hp_noise_manifest.csv`；
- D1/D2 的 F05 checkpoint 和 cache。

## 8. GPU1 验证

```bash
bash deploy/f05_focus_gpu1/verify_gpu1.sh \
  --stage-dir "$STAGE_DIR" \
  --train-root "$TRAIN_ROOT" \
  --test-root "$TEST_ROOT" \
  --rm-lp-checkpoint "$RM_LP_CHECKPOINT" \
  --f05-checkpoint "$F05_CHECKPOINT"
```

验证脚本会检查 Python/CUDA、repo 分支、数据目录、feature cache、parent checkpoint
和 F05 checkpoint；缺失项 fail-closed。它不训练、不写 test 预测、不访问平台。
