# REMATCH750 AdamW Decay-Filter 2×2 — 预注册与 NPU 交接

## Material Passport

- artifact_type: code experiment plan
- protocol: `REMATCH750_DECAY_FILTER_2X2`
- owner: xjn
- created: 2026-09-24
- status: ready for NPU; no training result; no submission package; no platform score
- baseline: same-revision `RM_HL00_F05_CONTROL`
- parent: `RM_LP`, SHA-256 `d5cb8f5265754fd900d3efde23e24fefbcf616c747f2cab13e4dc2201fdd689b`

## 1. 研究问题

当前 Aegis 的 AdamW 参数组把同一 scope 内的全部可训练张量赋予同一个非零
weight decay。因此 classifier bias、视觉 LayerNorm affine、class token 等一维参数
与权重矩阵一起衰减。常见的矩阵-only decay 约定尚未被检验，已有
`REMATCH750_WD_RELAX_2X2` 只比较整个 head/visual scope 的衰减系数为 `1e-4`
还是 `0`，不能回答“一维参数是否应豁免”。

本轮保持两个 scope 的衰减系数都为 `1e-4`，仅改变**衰减过滤规则**：

- `all`：该 scope 内所有可训练张量应用配置的 AdamW decay；
- `matrix_only`：`ndim >= 2` 应用配置的 decay，`ndim < 2` 的 bias、norm、token
  向量使用同 LR 但 `weight_decay=0`。

## 2. 严格 2×2 因子设计

| trial | head filter | visual filter | 作用 |
|---|---|---|---|
| HL00 | all | all | 同版本控制；复用 `RM_HL00_F05_CONTROL` |
| DF01 | matrix_only | all | 单独检验 head 一维参数豁免 |
| DF02 | all | matrix_only | 单独检验 visual 一维参数豁免 |
| DF03 | matrix_only | matrix_only | 检验两端共同豁免及交互 |

因子效应按同版本 HL00 计算：head 主效应、visual 主效应，以及 DF03 是否超过两项
单独收益的加和。若 HL00 尚未产出，必须先跑 HL00；不得拿历史 F05 数字替代本次
代码版本控制。

## 3. 固定项与血缘

- 同一 RM-LP parent、train-dev/val-dev split、class mapping、seed42；
- 320px、`weak_rrc_flip`；
- microbatch 256 × accumulation 4 = effective batch 1024；
- 16 epochs、head LR 4e-4、backbone LR 1.2e-5；
- head/backbone weight decay 数值均固定为 1e-4；
- CE warmup 2 → GCE q=0.5、feature anchor 2.0；
- AMP、cosine、raw macro 选模与 raw micro 破平局；
- 无 TTA、EMA、融合、跨 checkpoint 平均或测试分布适配。

血缘哈希见 `configs/rematch750_decay_filter/manifest.json`；生成器会把三份配置和
各自 SHA 固定下来。

## 4. 与既有搜索去重

- `REMATCH750_WD_RELAX_2X2` 改的是 scope 的 decay **系数**；本轮系数始终为
  `1e-4`，改的是 scope 内的**张量选择规则**。
- `REMATCH750_HEAD_L2SP` 改 classifier 参数化并扫描残差头 WD；本轮保持普通线性头。
- `REMATCH750_F05_EVIDENCE_TRANSFER` 改 anchor、loss、LR、warmup 或训练长度；本轮
  均不触碰。
- V5 的分辨率、增强、SAM、teacher、adapter 等轴均不在本轮变化范围内。

## 5. 实现门禁

- 默认 `all` 必须保留历史的 `head`、`visual` 两组及原 decay 行为；
- `matrix_only` 下每个 trainable parameter 恰好进入一个 optimizer group；
- `ndim >= 2` 的参数进入原名组并使用 `1e-4`，`ndim < 2` 进入
  `*_no_decay` 并使用 `0`；
- 一次零梯度 AdamW step 中矩阵参数发生纯 decay 更新，一维参数逐位不变；
- 非法 filter 值在配置层和模型层均 fail closed；
- tracked config 保持 `npu:UNASSIGNED`，仅运行时副本绑定逻辑设备；
- parent/train/val/mapping/config SHA 必须与 manifest 一致。

## 6. 晋级、复核与止损

所有差值相对同版本 HL00：

- 晋级：raw macro 至少 `+0.30pp`，且 raw micro 不低于 HL00 `−0.10pp`；
- 止损：macro 或 micro 低于 HL00 `2.00pp`，关闭对应因子；
- seed42 达线后才复核 seed 3407、2026，至少 2/3 seed 同方向才保留；
- 只有独立过门的过滤规则才允许与其他机制组合；
- 平台包及是否占用提交名额由队长决定，本运行器不上传平台。

## 7. NPU 执行

必须从包含本方案的最新 `main` 执行；不要在未合并本提交的旧 V5 worktree 直接跑：

```bash
git pull --ff-only origin main
export PYTHONPATH=reproducibility/aegis_f1

python3 scripts/build_rematch750_decay_filter.py --check
/workspace/noise-npu-venv/bin/python scripts/run_rematch750_decay_filter.py \
  --device npu:0 --trials DF01 DF02 DF03

ASCEND_RT_VISIBLE_DEVICES=<physical_card> \
/workspace/noise-npu-venv/bin/python scripts/run_rematch750_decay_filter.py \
  --device npu:0 --trials DF01 --execute
```

三点互相独立，可分配到三张空闲 NPU；HL00 是共同依赖，可与它们并行跑，但统一
比较必须等同版本 HL00 完成。

## 8. 结果回填要求

每点回填 selected epoch、raw macro、raw micro、相对 HL00 差值、完整耗时、配置与
checkpoint SHA，并核验实际 optimizer group 名称、参数数目、LR、WD。未完成同版本
HL00 对照和 checkpoint 重载复评前，不得称为有效提升；未生成并校验 37,444 行
预测 CSV/ZIP 前，不得称为可提交候选。

本文件在 NPU 回填前只声明 `ready_for_npu`。
