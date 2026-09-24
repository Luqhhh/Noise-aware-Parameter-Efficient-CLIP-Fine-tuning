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
  - 平台分已回填：`65.4711%`，当前已知 platform leader
  - 由 `65.4711%` 与 37,444 行推断正确数为 `24,515`；用户未单独给出正确数
  - 分数-only 回执见 `results/rematch750_f05_platform_20260924.json`
- 历史锚点：`RM_V3_B1024_E16_LR4`
  - local macro `0.7234723567962646`
  - local micro `0.7341398000717163`
- 平台对比：
  - `F05` 比 `F03` 平台高 **+1.0576pp / +396 张**（`65.4711%` vs `64.41352419613288%`）
  - 70% 目标需要 `26,211 / 37,444`；F05 还差 **1,696 张**，**未达 70%**
  - 本地 macro 高 0.3637pp，但平台观察差不能当作固定本地-平台换算率
- 其他已知平台锚点：`RM_FULL = 61.6360%`（重叠验证诊断，不参与独立
  holdout 排名）

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

可执行（`implementation_status=implemented`，共 32 个）：

- R01–R04：352/384/416/448 固定分辨率。
- S01–S09：320/384/448 的 FP32 AdamW 与 SAM rho=0.025/0.05；SAM 槽位走
  effective-batch 两遍累积路径，不是 microbatch-SAM。
- K01–K09：固定 anchor、anchor 2.0→0.2 退火、同视图官方 224px 教师。
- V01/V02/V05/V06：320/384 的 RRC 面积下界 0.5 / 0.8。
- L01–L06：第 5 轮启用、面积/占比/置信门控的 attention-local V5 语义。

等待工件（不可静默执行）：

| 槽位 | 状态 | 依赖 |
|---|---|---|
| H01–H12 | `pending_artifact` | F05 320px encoder 特征缓存与 binding |
| N01–N04 | `conditional_pending_evidence` | 新的合法训练侧质量证据与 target/梯度活性 |
| X01–X08 | `blocked_dependency` | 对应 component family 的胜者与兼容性重测 |
| R05/R06 | `blocked_implementation` | 阶段式分辨率切换、位置嵌入与 optimizer state 迁移 |
| V03/V07 | `blocked_implementation` | 训练与推理共用的确定性 letterbox |
| V04/V08 | `blocked_implementation` | 仅末 4 轮 RRC [0.9,1.0] 的几何调度 |
| Q01–Q06 | `blocked_implementation` | GSAM constant-rho + effective-batch 两遍框架 |

## 4. 远端执行状态

本工作区本身没有 NPU；V5 代码已部署到远端 `vllm-lqh-86` 的
`/workspace/noise-v5`，并从只读资产与 LP checkpoint 启动训练。

已启动并验证：

- R01/R02/R03/S01：固定分辨率与 FP32 AdamW 控制，均已越过 first-step audit；
- S02：first-step audit 后完成 effective-batch SAM 更新，step 20 已记录
  `successful_optimizer_updates=20`；
- K07：同视图官方教师 first-step audit 通过，step 20 已记录；
- remote ledger：`/workspace/noise-v5/results/rematch750_search_v5/ledger.jsonl`

仍未发生：

- 没有生成或修改 F05 的提交包；
- 没有上传平台；
- 没有填充 F05 checkpoint / CSV / ZIP 的真实 hash；
- F05 平台分只有 `65.4711%` 分数回执，没有 submission ID、带时区时间、
  reset-period 或包 hash；
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

S02 已用于验证 effective-batch SAM 的 1024 有效 batch 路径；Q01–Q06 仍需
GSAM constant-rho 实现。H01–H12 需先生成 F05 320px encoder 特征缓存。
