# REMATCH750_SEARCH_V5 实施规格与当前实现状态（2026-09-24）

本分支从 V4 最终结果（`origin/codex/rematch750_search_v4`，24b85fc）切出。
本轮 V5 的目标是把广搜从 V4 的 82 个无条件/弱条件点，收敛为 **68 个有明确语义、
可校验、可绑定、执行前 fail-closed 的条件槽位**；F05 仍是今日平台候选，本轮
不自动上传、不自动花费提交额度。

## 1. 基线

- 当前候选：`RM_V4_F05`
  - selected epoch 14
  - local macro `0.7443073987960815`
  - local micro `0.7544354796409607`
  - platform score 未回填，必须保持 `null`
- 历史锚点：`RM_V3_B1024_E16_LR4`
  - local macro `0.7234723567962646`
  - local micro `0.7341398000717163`
- 已知平台锚点：`RM_V4_F03 = 64.41352419613288% (24,119 / 37,444)`；
  `RM_FULL = 61.6360%`（重叠验证诊断，不参与独立 holdout 排名）

## 2. 本轮已落库

- `search_manifest.json`：V5 68 个条件槽位。
  - R 6、S 9、K 9、V 8、L 6、Q 6、N 4、H 12、X 8。
  - 每个槽位带有 `implementation_status`、`dependencies`、`config`、
    `config_sha256`、`micro_batch_size`、`effective_batch_size`、`status`。
- `configs/rematch750_search_v5/*.yaml`：68 个可加载、可 declaration 校验的配置。
- `reproducibility/aegis_f1/aegis_clip/rematch_search_v5.py`：
  - 未知 `project.search` 字段直接报错；
  - `validate_declaration(..., require_implemented=True)` 在执行前拒绝未实现机制；
  - 通过后复用 V4/V5 共用的数据、缓存、split、checkpoint lineage 校验。
- `scripts/build_rematch750_search_v5.py`：manifest 与配置的唯一生成入口。
- `scripts/cache_rematch750_v5_features.py`：从 F05 checkpoint 抽取 320px
  `full_train` 训练缓存与独立 `val_dev` 诊断缓存；测试集不读。
- `scripts/audit_rematch750_v5_candidate.py`：审计 checkpoint/config/CSV/ZIP、
  调提交检查器、并只在校验通过后做 hardlink 保留。
- `scripts/run_rematch750_search_v5_queue.py`：
  - 每个 trial/run 使用独占 `run_root` 与 `LEASE.json`；
  - 不移动、不删除任何运行中目录或已结束目录；
  - 写 `results/rematch750_search_v5/ledger.jsonl`；
  - 状态区分 `blocked_implementation`、`planned`、`failed_runtime`、
    `audit_incomplete`、`training_complete`；
  - 不调用平台接口，不自动上传。
- `config.py` / `trainer.py`：
  - V5 协议并入 rematch-search 配置校验；
  - `weak_rrc_flip` 训练几何现在读取 `rrc_scale_min/max` 与
    `rrc_ratio_min/max`，不再静默忽略 V 组面积下界。
  - V4 分辨率白名单保留；V5 允许 224/320/352/384/416/448。

## 3. 当前可执行与阻塞

可执行（`implementation_status=implemented`，共 15 个）：

- R01–R04：352/384/416/448 固定分辨率；声明的 microbatch 必须先在 NPU 上做
  smoke，再按 `micro_batch_candidates` 选择第一个通过的大小。
- S01/S04/S07：320/384/448 普通 FP32 AdamW 控制。
- K01/K02/K04/K05：320/384 固定 reference anchor 0 / 0.5。
- V01/V02/V05/V06：320/384 的 RRC 面积下界 0.5 / 0.8。

明确阻塞（不可静默执行）：

| 槽位 | 状态 | 依赖 |
|---|---|---|
| R05, R06 | `blocked_implementation` | 阶段式分辨率切换、位置嵌入与 optimizer state 迁移 |
| S02/S03/S05/S06/S08/S09 | `blocked_implementation` | effective-batch SAM（两遍完整 1024 有效 batch） |
| K03/K06 | `blocked_implementation` | anchor 权重 2.0→0.2 的显式退火 |
| K07–K09 | `blocked_implementation` | 同视图官方教师的两遍一致随机状态回放 |
| V03/V07 | `blocked_implementation` | 训练与推理共用的确定性 letterbox |
| V04/V08 | `blocked_implementation` | 仅末 4 轮 RRC [0.9,1.0] 的几何调度 |
| L01–L06 | `blocked_implementation` | 第 5 轮启用、面积/占比、置信回退与激活率账本 |
| Q01–Q06 | `blocked_implementation` | GSAM constant-rho + effective-batch 两遍框架 |
| N01–N04 | `conditional_pending_evidence` | 新的合法训练侧质量证据与 target/梯度活性 |
| H01–H12 | `pending_artifact` | F05 320px encoder 特征缓存与 binding |
| X01–X08 | `blocked_dependency` | 对应 component family 的胜者与兼容性重测 |

## 4. 尚未在本机执行

本工作区没有 NPU，也没有 V4 远端 checkpoints/缓存。下列事情**没有发生**：

- 没有启动 NPU 训练；
- 没有生成或修改 F05 的提交包；
- 没有上传平台；
- 没有填充 F05 checkpoint / CSV / ZIP 的真实 hash；
- 没有声称 70% 目标已达到。

## 5. 下一位执行者入口

```bash
# 重新生成 V5 manifest + configs
PYTHONPATH=reproducibility/aegis_f1 python3 scripts/build_rematch750_search_v5.py

# 只执行 manifest first_wave；阻塞点会写 ledger 并跳过
PYTHONPATH=reproducibility/aegis_f1 python3 scripts/run_rematch750_search_v5_queue.py \
  --device npu:0 --trials R02 R04 R01 R03 S01 S02 K02 K03 K07 V02 L01 H01

# 定向测试
PYTHONPATH=reproducibility/aegis_f1 python3 -m pytest \
  reproducibility/aegis_f1/tests/test_rematch750_search_v5.py -q
```

在 S02/S03/S05/S06/S08/S09 和 Q01–Q06 可执行前，必须先实现并验收
effective-batch SAM：真实 1024 有效 batch、两遍同一图像/增强/mix/RNG、只一次
optimizer/scheduler 更新、rho=0 退化同口径 AdamW、断点恢复完整。
