# REMATCH750 weight-decay relaxation：NPU 预注册与执行交接

日期：2026-09-23

方案分支：`xjn/rematch750-wd-relax`

状态：实现完成，等待 NPU 执行

## 1. 研究问题

在当前本地最优单模型 `RM_V3_B1024_E16_LR4` 上，AdamW 对视觉 backbone 与分类头统一使用 `weight_decay=1e-4`。该值从 RM-FT 配方沿用，尚未在复赛 750 类、batch 1024、16 epoch 的 NPU 最优配方上做过参数组级消融。

本方案检验：**当前配方是否因参数级正则过强而欠拟合；如果是，收益主要来自放松 backbone、head，还是二者交互。**

## 2. 为什么现在测这个轴

证据来自当前阶段，而非泛泛扫参：

1. `RM_V3_B1024_E16_LR4` 从 epoch 10 到 16 仍由 macro 72.1079% 上升到 72.3472%，最终 best=last，没有明显后程过拟合回落。
2. 已完成的 MixUp A 轴在固定特征锚点 2.0 下呈单调负向：α=0 / 0.2 / 0.4 的 val micro 为 72.8696% / 72.5000% / 71.9288%，同时训练准确率随额外正则增强而下降。它不能证明 weight decay 必然有害，但使“降低另一种独立正则”成为有方向依据的测试。
3. 已登记的 V4 82 点覆盖增强、鲁棒损失、质量修复、特征锚定、部分解冻、长尾、分辨率、SAM 与冻结 head；未包含 head/backbone weight decay 扫描。
4. 优化器接线已核实：`trainer.py` 将 `head_weight_decay` 与 `backbone_weight_decay` 分别写入两个 AdamW 参数组，`NpuFusedAdamW(groups)` 直接消费这两个值。因此本实验不是“配置有字段但运行不生效”。

## 3. 2×2 因子设计

除实验 ID、输出目录和以下两个 weight-decay 字段外，四组与 V3 胜者完全一致：同一 RM-LP parent、split、seed 42、batch 1024、16 epochs、head LR 4e-4、backbone LR 1.2e-5、CE warmup 2 → GCE q=0.5、feature anchor 2.0、增强、调度器、AMP、选模和评估间隔。

| 点位 | backbone WD | head WD | 作用 |
|---|---:|---:|---|
| WD00 | 1e-4 | 1e-4 | 同 commit 配对控制 |
| WD01 | 0 | 1e-4 | 只放松 backbone |
| WD02 | 1e-4 | 0 | 只放松 head |
| WD03 | 0 | 0 | 同时放松，估计交互 |

配置：`configs/rematch750_wd_relax/{WD00,WD01,WD02,WD03}.yaml`。

本轮先测端点而不直接铺满 1e-5/3e-5/1e-4 网格。端点能用 4 次训练识别方向和参数组归属；只有出现正信号才在对应组上做对数区间细化，避免无依据扩大扫描。

## 4. 固定血缘

| 资产 | SHA-256 / 值 |
|---|---|
| parent | `RM_LP` |
| parent checkpoint | `d5cb8f5265754fd900d3efde23e24fefbcf616c747f2cab13e4dc2201fdd689b` |
| train CSV | `615d510481e9df610371e84cbdeeb0058cc9a176989786a8ccf4eb30235b92c6` |
| val CSV | `d91106df365b62f9bf56eff51474c222925301546642fa427f6778258cb833ab` |
| seed | 42 |
| V3 reference macro / micro | 0.7234723568 / 0.7341398001 |

执行前仍须由 NPU 机器复核 parent、train CSV、val CSV 的真实 SHA；任一不符立即停止。

## 5. 指标与判据

主指标为 `raw_macro`，副指标为 `raw_micro`；固定训练尾部 75 类 macro 只作机制诊断，不替代总体判据。

以同次执行的 WD00 为主要对照，历史 V3 只作血缘锚点：

- `raw_macro` 至少 +0.20pp，且 `raw_micro` 不低于 WD00 超过 0.10pp：记为第一阶段正信号；
- `raw_macro` 至少 +0.30pp，且 micro 同向：达到内部投资回报门，可在获胜参数组追加 `1e-5` 与 `3e-5` 两个邻近点；
- 四组差异均在 ±0.10pp：结论为 weight decay 在当前尺度不可辨识，关闭本轴；
- 取消 decay 导致 macro ≤−0.50pp 或出现持续后程回落：不细化该方向；
- NaN、非有限梯度、无有效 optimizer step、血缘或配置不一致：运行无效，先修门禁，不解释指标。

因子效应按四点共同解释，不只挑最高点：

```text
backbone main effect = mean(WD01, WD03) - mean(WD00, WD02)
head main effect     = mean(WD02, WD03) - mean(WD00, WD01)
interaction          = WD03 - WD01 - WD02 + WD00
```

不使用测试集选择 weight decay，不自动生成或上传平台包。只有本地达到门槛并经审核后，才对一个单 checkpoint 胜者执行确定性推理。

## 6. NPU 执行

在独立 worktree 拉取本分支；原始数据共享只读，输出固定到 `outputs/xjn/rematch750_wd_relax/`，不会覆盖 V3/V4 或队友目录。

```bash
git fetch origin
git worktree add /workspace/noise-worktrees/xjn-wd-relax xjn/rematch750-wd-relax
cd /workspace/noise-worktrees/xjn-wd-relax

source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=<当时空闲的物理卡号>
export OMP_NUM_THREADS=8
export PYTHONPATH=reproducibility/aegis_f1

# 先只物化并打印命令，不训练；检查 runtime YAML。
/workspace/noise-npu-venv/bin/python scripts/run_rematch750_wd_relax.py \
  --device npu:0 --trials WD00 WD01 WD02 WD03

# 审核无误后执行。每个点独立写盘；任一点失败时 subprocess fail-closed，
# 不自动跳过故障继续解释后续结果。
/workspace/noise-npu-venv/bin/python scripts/run_rematch750_wd_relax.py \
  --device npu:0 --trials WD00 WD01 WD02 WD03 --execute
```

跟踪配置保留 `npu:UNASSIGNED`；执行脚本要求显式设置 `ASCEND_RT_VISIBLE_DEVICES`，再把当前进程内逻辑设备绑定为 `npu:0`。物化后的绝对路径 YAML 保存在输出目录 `_runtime_configs/`，作为运行证据。

## 7. 首轮验收

WD00 第一个 optimizer step 后必须核对：

- parent / train / val SHA 与 §4 一致；
- batch size 1024，计划每轮 batch 数与 V3 一致；
- 参数组恰为 head + backbone，两组 LR 分别 4e-4 / 1.2e-5；
- 两组 WD 与当前点位表一致；
- 有效 optimizer step 增长、无 AMP overflow、loss/grad 有限；
- 输出路径只位于 `outputs/xjn/rematch750_wd_relax/`。

通过后串行跑完四点。训练结果、checkpoint SHA、逐 epoch 曲线和因子效应由 NPU 执行方回填；在此之前本文件只声明“ready for NPU”，不声明实验有效或有收益。
