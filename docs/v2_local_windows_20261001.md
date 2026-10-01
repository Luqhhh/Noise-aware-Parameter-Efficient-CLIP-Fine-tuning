# v2 本机（Windows）可移植性修复与本机 dev 运行记录（2026-10-01）

本段性质：**工程与可移植性修复 + 本机 v2 dev 段运行记录**。
没有新平台成绩，没有为凑产物启动训练，没有生成或上传提交包。
报告区分「实现改动」与「实测结果」。

* 方案分支：`xjn/v2_windows_portability_20261001`，基点 `origin/main` = `40722ca`
  （"Integrate verified WFT448_FULL delivery"）
* 环境：Windows + DSH 受限沙箱，RTX 4060 Laptop 8GB，Python 3.12.7 / torch 2.6.0+cu126 / OpenAI `clip`
* 现役可提交包未变：`/home/lux1/noise/worktrees/rematch750_f05_focus/outputs/f05_focus_l05/L05_T14_P060/{pred_results.csv,submission.zip}`
  （37,444 行、9 项校验来源见既有 `results/p75_supervision_rebuild_20260929/submission_check.log`）；
  本段未重新生成、未重新校验该包。

---

## 1. 实现改动（3 个文件，共 +5/−3 行）

| 文件 | 改动 | 性质 |
|---|---|---|
| `reproducibility/aegis_f1/v2/core.py` | 清单匹配改为分隔符无关：`k.replace("\\","/").endswith("/train_manifest.csv")` | **真 bug**。原写法硬编码 `/`；Windows 下 `plan["inputs"]` 的键是 `\`，`check_checkpoint` 永远匹配不到清单，抛 `Checkpoint manifest identity mismatch` → 任何 checkpoint 血缘校验（含父权重加载）都必然失败 |
| `reproducibility/aegis_f1/v2/runtime.py` | 推理出包时调用提交校验器改用 `sys.executable`（并 `import sys`），原为 `"python3"` | Windows 无 `python3` 命令；POSIX 上等价 |
| `tests/test_v2.py` | 第 370 行 lineage 断言改为分隔符无关 | 原断言写死 `'s2_448/best.pt'`；**断言内容不变**，只是让 Windows 能跑 |

**不重复修改**：`v2/training_utils.py` 的坏 EXIF 缺陷已由 main 上的 `v1_continuations` 段修复
（`load_image` 的兜底元组加入 `SyntaxError`），本段沿用上游版本，未做二次改动。

## 2. 实测结果（本机）

### 2.1 CPU 验证链

| 步骤 | 结果 |
|---|---|
| `v2.plan prepare` | `prepared_no_gpu`；split **133,815 / 14,880**，`overlapping_content_groups = 0`；torch/torchvision/clip 均可用 |
| `v2.plan verify` | `verified` |
| `scripts/check_v2_cpu.py` | `cpu_forward_passed`；224 原生与插值差 **0.0**；384/448/576 → `[1,750]`；全参数 fp32；可训练参数 **88,233,966** |
| `pytest tests/test_v2.py tests/test_v2_swa.py reproducibility/aegis_f1/tests/test_model.py` | **67 passed**（含上游新增 9 项 SWA 测试） |

### 2.2 数据健康扫描（`scripts/scan_image_health.py`）

全部 **148,695** 张训练图（1,160 s）：

```json
{ "scanned": 148695, "bad_count": 1,
  "exception_kinds": { "builtins.SyntaxError: not a TIFF file (header b'IIU\\x00\\x18\\x00\\x00\\x00' not valid)": 1 } }
```

唯一致命文件：`0249/2daac9914a1a43918148491e37591b02.jpg`（label 249，800×800）。
其余全部只是 `UserWarning`（Corrupt EXIF data / Truncated File Read），不是异常。
像素层面无问题 —— 阶段审计 `20260921/decode_report.json` 记录 `train_checked: 148695`、`failures: []`。

### 2.3 成本 probe（`s1_384`，5 次逻辑更新）

| 指标 | micro batch 2 | micro batch 16 |
|---|---:|---:|
| 每次逻辑更新（含串行解码/增强与 DataLoader 等待） | 5.875 s | **3.781 s** |
| 一个 epoch holdout 验证（raw+EMA 两遍，14,880 行） | 646.3 s | 609.9 s |
| 一次 checkpoint 写入（含 sha256 回读） | 2.094 s | 2.672 s |
| setup（148,695 图 SHA256 复核 + 模型加载） | 876.3 s | 834.8 s |
| **峰值显存分配** | 2.30 GiB | **2.30 GiB** |

两条硬结论：
1. **显存与 micro batch 无关**（均为 2.30 GiB）—— 由参数 353 MB + 梯度 353 MB + AdamW 两份动量 706 MB 主导，
   梯度检查点下的激活只占极小部分。
2. **验证批次从 2 提到 16（7,440 → 930 batch）验证耗时几乎没变** → 验证是纯数据管线瓶颈，
   GPU 那一步可忽略：29,760 张图 ≈ 610 s = **20.5 ms/图硬下限**。

### 2.4 s1_384 dev 运行（本机，micro batch 16）

13,930 次更新全部完成，耗时 **62,236 s = 17.29 h**（估算 59,662 s，偏 +4.3%）。
`best.pt` = 第 9 轮 raw 状态：**accuracy 75.15% / macro 74.23%**（750 类 / 14,880 行独立 holdout）。

| 轮 | raw acc | raw macro | EMA acc | EMA macro | 选中 |
|---|---:|---:|---:|---:|---|
| 0 | 38.15 | 37.44 | 15.80 | 15.22 | raw |
| 1 | 55.66 | 54.82 | 54.76 | 53.69 | raw |
| 2 | 62.36 | 61.49 | 63.77 | 62.74 | EMA |
| 3 | 66.08 | 65.13 | 67.94 | 66.97 | EMA |
| 4 | 68.69 | 67.79 | 70.36 | 69.42 | EMA |
| 5 | 71.19 | 70.30 | 71.80 | 70.84 | EMA |
| 6 | 72.54 | 71.68 | 72.97 | 72.02 | EMA |
| 7 | 74.05 | 73.19 | 73.92 | 72.95 | raw |
| 8 | 74.91 | 73.98 | 74.68 | 73.75 | raw |
| 9 | **75.15** | **74.23** | 74.99 | 74.06 | raw |

EMA 在第 2–6 轮领先、第 7 轮起被 raw 反超，与 `ema_decay=0.9995`（窗口约 2,000 次更新）的预热行为一致。

### 2.5 与现役方案的对比（跨配方，非消融）

| 方案 | val accuracy | val macro |
|---|---:|---:|
| 现役 `L05_T14_P060`（平台 66.94797564362783%） | 76.6532 | 75.8265 |
| v2 **仅 s1_384**（四段第一段，最低分辨率） | 75.15 | 74.23 |
| 差距 | −1.50pp | −1.60pp |

**三点限定**：
1. 本流程**不是逐位可复现的**：同一 epoch（第 0 轮）两次运行 raw macro 分别为 37.44% 与 36.52%，差 **0.9pp**
   （v2 的 stage 配置没有 `deterministic`，`set_seed` 也未设 `cudnn.deterministic`）。
   所以「落后 1.6pp」与跑动噪声同量级，**单次结果不足以判断 v2 优劣**。
2. 这是**跨配方**对比，不是同配方消融。
3. 现役方案的 local micro 76.65 对应平台 66.95（差 9.7pp），**本地分不能外推平台分**；
   s1 的这个数字只说明「值得继续跑完阶梯」。

## 3. 两条可迁移的工程发现

### 3.1 stage 配置内嵌绝对输出路径 → 换目录重新 prepare 会让既有 checkpoint 全部失配

`v2.plan.prepare` 会把 `cfg["paths"]["project_root"]` 写成 `str(output)`，因此
**生成的 `configs/<stage>.json` 内容与 prepare 的输出目录强相关**，`stage_config_sha256` 也随之变化。

实测（本次踩到）：
* 在**新目录** `prepared_v2_swa` 重新 prepare 后，对既有 `runs/s1_384/best.pt` 调 `read_checkpoint`
  → `ValueError: Wrong stage configuration`；
* 逐行 diff 确认两份配置**只有 `paths.project_root` 一行不同**；
* 改回**原目录**重新 prepare 后，同一 checkpoint 通过校验：
  `epoch=9 global_step=13930 chosen=raw completed_epochs=10`。

结论：为绑定新代码而重新 prepare 时，**必须在原输出目录原地重建**
（先移走 `runs/`，把旧 plan 目录改名留档，再在原路径 prepare，最后把 `runs/` 移回）。
换目录 prepare 等于作废该 workspace 的所有 checkpoint。

### 3.2 受限沙箱禁止多进程 DataLoader → 数据管线成为瓶颈

`torch` DataLoader 的 worker 依赖 `multiprocessing` 命名管道，被沙箱拒绝
（建 worker 时 `PermissionError [WinError 5]`），只能 `num_workers=0`。
后果是可量化的：训练期间 **GPU 利用率在 15%~99% 之间跳动**（GPU 在等单进程解码），
验证吞吐存在 20.5 ms/图的硬下限。这是本机所有 GPU 实验的成本主因：
四段阶梯合计约 **64 h**，其中数据管线占相当比例。
若环境允许多进程加载，预计可压缩到约 35–45 h（未实测，属外推）。

## 4. 本机配方偏离（仅本机，未改仓库内 `configs/v2/recipe.json`）

| 字段 | 偏离 | 理由 |
|---|---|---|
| 4 个路径字段 | 指向本机数据/权重 | 原配方指向 `/home/lux1/...` |
| `num_workers` | 2 → **0** | 见 3.2 |
| `micro_batch_size` | 2 → **16** | `v2/core.py:logical_backward` 对每个 micro batch 计 `loss × (micro_rows / logical_rows)` 再 backward，**累加梯度与 micro batch 大小数学等价**（每样本恒为权重 1/N）；mixup/cutmix 在切分前、每逻辑批次只做一次。差异仅浮点累加顺序 → 结果等价但不逐位一致。依据：micro 2 峰值显存仅 2.30 GiB 而卡上可用约 6.3 GiB |

## 5. V2_FULL_LAST3_SWA 串行接线状态

上游已在 main 提供该候选的完整实现（`v2/swa.py`；`v2/runtime.py` 在 final 段保存
`epoch_{03,04,05}_raw.pt`；`infer --checkpoint` 为独立候选入口；原末轮 raw `last.pt` 提交策略不变）。

本机已完成的接线：
* 合并/对齐到最新 `origin/main`，CPU 测试 **67 passed**（含 `tests/test_v2_swa.py`）；
* 按**原配方、原输出目录**重新 prepare，使新快照代码进入绑定集合；
* 验证既有 `s1_384` 完成态 checkpoint **仍被接受为 `s2_448` 的父权重**（见 3.1 实测输出）。

**当前阻塞（未授权项）**：`V2_FULL_LAST3_SWA` 需要 `full_576` 的 `epoch_03/04/05_raw.pt`，
而这些快照只在跑完 `s2_448 → s3_576 → full_576` 后才会产生（估算合计 ≈47 h）。
`v2.swa` 在缺快照时按设计 **fail-closed 拒绝，且不重训凑快照**。本段未启动任何新 GPU 训练。

串行队列（沿用 `docs/v1_continuations_20261001.md` 的既定顺序）：
原 v2/v3 按既有授权协议收尾 → **v2 末段固定 SWA（快照齐全后）** → WFT448_DEV → LR512_DEV。

## 6. 复现命令（本机，仓库根目录）

```bash
# CPU 验证链（无需授权；prepare 拒绝覆盖已存在目录）
export PYTHONPATH=reproducibility/aegis_f1
python -m v2.plan prepare --recipe <local-recipe> --output <workspace>
python -m v2.plan verify  --plan    <workspace>/plan.json
python scripts/check_v2_cpu.py --recipe <local-recipe> --output <cpu_forward.json>
python -m pytest tests/test_v2.py tests/test_v2_swa.py reproducibility/aegis_f1/tests/test_model.py -q

# 数据健康扫描
python scripts/scan_image_health.py --manifest <workspace>/train_manifest.csv \
  --train-root <train_root> --tier exif --output <image_health_exif.json>

# 成本 probe / 训练（需授权文件：authorized=true + plan SHA 绑定 + 估算 ≤ 0.8×预算）
python -m v2.runtime probe --plan <workspace>/plan.json --authorization <auth.json> --stage s1_384 --steps 5
python -m v2.runtime train --plan <workspace>/plan.json --authorization <auth.json> --stage s1_384

# V2_FULL_LAST3_SWA（full_576 快照齐全后；导出为 CPU，推理需独立授权）
python -m v2.swa --plan <workspace>/plan.json --output <swa-export>
python -m v2.runtime infer --plan <workspace>/plan.json --authorization <infer-auth.json> \
  --checkpoint <swa-export>/selected.pt --output <submission>
```

## 7. 停止边界

* 未授权/未启动：`s2_448`、`s3_576`、`full_576` 训练，`infer`，平台上传。
* 未生成新的 `pred_results.csv` / `submission.zip`（v2 的交付边界是 `full_576`，`s1_384` 是 dev 段）。
* 本方案分支已推送，**未合并到 `main`**；main 集成按仓库约定在 main 集成目录完成。
* 环境备注：本机沙箱在本次会话中一度失效（会话临时目录被删）并降级为 ConstrainedLanguage 模式；
  训练进程未受影响并正常收尾。
