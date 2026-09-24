# REMATCH750 F05 Evidence-Transfer Sweep — 预注册与 NPU 交接

## Material Passport

- artifact_type: code experiment plan
- protocol: `REMATCH750_F05_EVIDENCE_TRANSFER`
- owner: xjn
- created: 2026-09-24
- status: ready for NPU; no training result; no submission package; no platform score
- baseline: same-revision `RM_HL00_F05_CONTROL`
- parent: `RM_LP`, SHA-256 `d5cb8f5265754fd900d3efde23e24fefbcf616c747f2cab13e4dc2201fdd689b`

## 1. 研究问题

F05 把 224px 配方提升到 320px 后，本地 macro 从 V3 的 72.3472% 提升到
74.4307%，平台也从 F03 的 64.4135% 提升到 65.4711%。本轮不再发散到新的
复杂机制，而是检验两类问题：

1. V4 在 224px 上出现过小幅正信号的标量因素，能否迁移到 F05 的 320px 配方；
2. F05 固定的 backbone/head 学习率和 cosine 动态，是否仍处于局部最优附近。

这是一个筛选实验，不预先声称任一因素有效。每个 trial 只改变一个声明变量，
共享同一父 checkpoint、split、seed、增强、有效 batch、优化器和选模规则。

## 2. 证据与候选

统一旧对照为 V3（macro 0.7234723568 / micro 0.7341398001）：

| trial | 唯一变量 | 旧证据或理由 | 优先级 |
|---|---|---|---:|
| ET01 | feature anchor 2.0 → 1.0 | V4 D03：macro +0.1293pp，micro +0.1142pp | 1 |
| ET02 | GCE q 0.5 → 0.7 | V4 B05：macro +0.0135pp，micro +0.0134pp | 1 |
| ET03 | Balanced Softmax τ=0.5 | V4 E01：macro +0.1126pp，micro +0.0134pp | 1 |
| ET04 | Balanced Softmax τ=1.0 | V4 E02：macro +0.1831pp，micro −0.0202pp | 1 |
| ET05 | backbone LR 1.2e-5 → 8e-6 | F05 局部下侧括点 | 2 |
| ET06 | backbone LR 1.2e-5 → 1.6e-5 | F05 局部上侧括点 | 2 |
| ET07 | head LR 4e-4 → 2e-4 | F05 局部下侧括点 | 2 |
| ET08 | head LR 4e-4 → 8e-4 | F05 局部上侧括点 | 2 |
| ET09 | LR warmup 0 → 1 epoch | 320px 优化稳定性探针 | 3 |
| ET10 | cosine 16 → 20 epochs | F05 最佳在 14/16，检验更慢余弦尾部 | 3 |

旧信号只用于确定搜索顺序，不当作本轮结果。ET05–ET10 是局部括点，不进行更细
网格，除非单点先达到晋级门。

## 3. 固定项与因果变量

固定：

- RM-LP 父权重与数据血缘；
- 320px、`weak_rrc_flip`；
- microbatch 256 × accumulation 4 = effective batch 1024；
- 16 epochs、head LR 4e-4、backbone LR 1.2e-5（除对应 trial）；
- CE warmup 2 → GCE q=0.5、feature anchor 2.0（除对应 trial）；
- AdamW WD、AMP、cosine、seed42；
- raw macro 选模、raw micro 破平局；
- 无 TTA、EMA、测试分布适配、模型融合或跨 checkpoint 平均。

ET10 的 epoch/schedule 同时改为 20，因为二者构成同一个“更慢 cosine 轨迹”处理，
不能只延长训练却沿用已经结束的 16-epoch scheduler。

## 4. 与现有工作的去重

- V5 已覆盖分辨率、SAM、feature-anchor 0/0.5/退火、same-view teacher、RRC、
  attention-local、cached heads、soft repair 与组合；本轮不重复这些点。
- Head L2-SP 已覆盖分类头锚定与 head WD；本轮保持普通线性头。
- WD_RELAX 已覆盖 backbone/head WD；本轮不修改 WD。
- FULL_DATA_CONTROL 改数据口径；本轮仍使用独立 train-dev/val-dev。
- V4 的 ET01–ET04 旧证据来自 224px；本轮检验的是 320px 交互，不是重复跑原点。

## 5. 晋级与止损

所有差值都相对**同版本、同代码、同机器口径的 HL00**计算，历史 F05 只作锚点：

- 晋级：macro ≥ HL00 +0.30pp，且 micro 不低于 HL00 −0.10pp；
- 止损：macro 或 micro 低于 HL00 2.00pp，立即关闭该因素；
- 单 seed 达线后只复核该因素的 seed 3407、2026；至少 2/3 seed 同方向才保留；
- 组合只允许由已经独立达线的因素派生，禁止提前跑无依据笛卡尔积；
- 平台包与是否占提交名额由队长决定，本运行器不会上传平台。

## 6. NPU 执行

```bash
git pull --ff-only origin main
export PYTHONPATH=reproducibility/aegis_f1

# 先校验生成器与配置，不训练。
python3 scripts/build_rematch750_f05_transfer.py --check
/workspace/noise-npu-venv/bin/python scripts/run_rematch750_f05_transfer.py \
  --device npu:0 --trials ET01 ET02 ET03 ET04 ET05 ET06 ET07 ET08 ET09 ET10

# 每张空闲物理卡可拆一个或多个 trial；逻辑设备仍写 npu:0。
ASCEND_RT_VISIBLE_DEVICES=<physical_card> \
/workspace/noise-npu-venv/bin/python scripts/run_rematch750_f05_transfer.py \
  --device npu:0 --trials ET01 --execute
```

建议先并行 ET01–ET04，再填充 ET05–ET10；资源充足时十点可全部独立并行。

## 7. 首步门禁与结果要求

- tracked config 保持 `npu:UNASSIGNED`，运行时配置才绑定逻辑卡；
- config SHA 与 `manifest.json` 逐项一致；
- parent/train/val/mapping SHA 与清单一致；
- epoch-0 logits 与 HL00/RM-LP 一致；
- 可训练参数组、LR、WD、有效 batch 与声明一致；
- 一步后 classifier 与 visual 参数均实际更新，loss/grad 有限，无 AMP overflow；
- 每组输出仅写入 `outputs/xjn/rematch750_f05_transfer/<experiment_id>/`；
- 最终回填 selected epoch、macro、micro、相对 HL00 差值、耗时、checkpoint/config
  SHA；未出预测 CSV/ZIP 前不得称为可提交候选。

本文件在 NPU 回填前只声明 `ready_for_npu`。
