# F05 Focus GPU1 部署包

第二台机器按本文件拉取同一分支，并用显式路径运行分配到的单元。所有实验必须使用
`focus/f05-four-lines` 的同一 commit；不要在 GPU1 上另开分支或修改配方。


## 0. 当前 GPU0 运行状态与 GPU1 缺口

GPU0 已在 RTX 4070 Laptop 8GB 上启动 C0：

```text
input 320px
batch_size 16 x grad_accum 64 = effective batch 1024
num_workers 8
outputs/f05_focus/C0_F05_CUDA/seed42
```

因此 GPU1 若跑 N1，必须使用同样的 `16 x 64` 组合，否则 C0/N1 paired comparison 失效。
GPU1 上若显存更大，也不要改成 256 x 4；effective batch 保持 1024 且 microbatch 配比与
C0 一致。

GPU1 代码拉到 `focus/f05-four-lines @ 867b982` 后，仍需要按它分配到的 unit 准备：

| GPU1 unit | 必需资产 | 当前状态 |
|---|---|---|
| P0 | `artifacts/f05_focus/f05_val_logits.pt` 或 F05 checkpoint | 待 GPU1 提供/生成 |
| N1 | `artifacts/f05_focus/hp_noise_manifest.csv` + RM-LP parent | manifest 未生成；parent 视 GPU1 现有 outputs |
| D2 | F05 checkpoint、`clean070.csv`、D2 cache | 需 F05 + quality asset |
| A0/D1 | F05 checkpoint | 需从 NPU 结果机同步 |

如果 GPU1 使用本地 `artifacts/stages/repechage/20260921`，而 RM-LP parent 是 NPU 格式
checkpoint，代码现已自动提供 `torch_npu` 兼容 shim；但 sibling binding 的
`dataset_manifest_sha256` / `feature_manifest_sha256` 若来自 `20260921_npu`，需要
与本地 manifest 做一次显式 rebase 后才能通过 V4 lineage gate。原始文件级
`train_csv_sha256` / `class_mapping_sha256` 未变，实验数据仍是同一 canonical split。

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

P0 只读 F05 validation logits。若 GPU1 上还没有该文件，可使用本机 F05 checkpoint
生成一次：

```bash
PYTHONPATH="$REPO_ROOT/reproducibility/aegis_f1" \
python3 -m aegis_clip.cli.cache_validation_logits \
  --checkpoint "$F05_CHECKPOINT" \
  --view-mode center \
  --output artifacts/f05_focus/f05_val_logits.pt \
  --batch-size 128 --num-workers 4
```

然后运行：

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
GPU0_ROOT=/home/lux1/noise \
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

---

## 9. C0 checkpoint 同步与路径覆盖（GPU1）

C0 完成后，GPU1 需要把 C0 checkpoint 放到自己的输出目录，并用
`scripts/make_gpu1_runtime_config.py` 生成一份路径已覆盖的 runtime YAML。

```bash
python3 scripts/make_gpu1_runtime_config.py \
  --checkpoint /path/to/C0_F05_CUDA/seed42/checkpoints/best.pt \
  --stage-dir /data/.../artifacts/stages/repechage/20260921 \
  --train-root /data/.../train \
  --test-root /data/.../test \
  --output configs/f05_focus_gpu1/C0_gpu1_runtime.yaml \
  --device cuda:0 \
  --experiment-id RM_V5_C0_GPU1 \
  --output-root outputs/f05_focus_gpu1
```

之后：

### A0

```bash
python3 scripts/run_a0_m1_probe.py \
  --checkpoint /path/to/C0_F05_CUDA/seed42/checkpoints/best.pt \
  --config configs/f05_focus_gpu1/C0_gpu1_runtime.yaml \
  --device cuda:0
```

### P0 validation logits

```bash
python3 -m aegis_clip.cli.cache_validation_logits \
  --checkpoint /path/to/C0_F05_CUDA/seed42/checkpoints/best.pt \
  --config-override configs/f05_focus_gpu1/C0_gpu1_runtime.yaml \
  --view-mode center \
  --output artifacts/f05_focus/f05_val_logits.pt
```

### D1/D2 adapter cache

生成 adapter cache 时传：

```text
--config-override configs/f05_focus_gpu1/C0_gpu1_runtime.yaml
```

这样 GPU1 可以使用自己的 `val_csv`、`train_root`、`features` 路径，而不依赖
GPU0 的绝对路径。

---

## 10. GPU1 直接生成 OOF/quality

GPU1 可以不等 GPU0 的 L05，直接先生成 OOF/quality 资产：

```bash
git fetch origin focus/f05-four-lines
git checkout -B focus/f05-four-lines origin/focus/f05-four-lines

TRAIN_CSV=<GPU1_TRAIN_DEV_CSV> \
CACHE_DIR=<GPU1_FEATURES_DIR> \
TRAIN_ROOT=<GPU1_TRAIN_ROOT> \
DEVICE=cuda:0 \
bash deploy/f05_focus_gpu1/run_oof_quality_gpu1.sh
```

产出：

```text
outputs/f05_focus/oof/sample_quality.csv
outputs/f05_focus/oof/oof_logits.pt
artifacts/f05_focus/hp_noise_manifest.csv
artifacts/f05_focus/clean070.csv
```

然后把 `hp_noise_manifest.csv` 和 `clean070.csv` 回传 GPU0，或直接供 GPU1 的
N1 / D1 / D2 使用。

## 11. 2026-09-25 GPU1 OOF 实测状态

GPU1 已在 `focus/f05-four-lines@ea53fc2` 完成 750 类、duplicate-aware 3-fold OOF：

- 133,815 个 train-dev 样本，每折 holdout 44,605，重复内容组零跨折；
- 三折固定 30 epochs，holdout accuracy 分别为 0.628382 / 0.630400 / 0.626253；
- 合并 OOF accuracy 0.628390，logits `133815 x 750` 且全部有限；
- N1 manifest reject 5,794（4.33%），strict suspect 1,981；
- clean070 保留 60,072（44.89%），750 类全覆盖、最小每类 2 条；
- 四个核心资产的 SHA 与 `artifacts/f05_focus/quality_chain_status.json` 一致。

完整数字与资产 SHA 见 `results/f05_focus_oof_gpu1_20260925.json`。运行中发现旧
`analysis.oof.run_oof` 会把 `oof_manifest.json` 的 parent 固定写成 500 类阶段的
`b2_gce05`；生成器现显式绑定 `REMATCH750_F05_FOCUS_OOF_3FOLD`、
`openai_clip_vit_b32_frozen_features`、750 类数据和 cache SHA。该修复只更正元数据，
不改变 logits、quality、N1 manifest 或 clean070 数值。

N1 仍必须等待与 GPU0 C0 完全相同的 RM-LP checkpoint（登记 SHA
`d5cb8f5265754fd900d3efde23e24fefbcf616c747f2cab13e4dc2201fdd689b`）。GPU1
本地另有 SHA `67a77e81...` 的 RM-LP 文件，未证明与 C0 parent 权重逐位一致，禁止
拿它替代后直接声称 paired delta。P0/D2 继续等待精确 F05/C0 checkpoint 或 val
logits。
