# REMATCH750 head L2-SP：NPU 预注册与执行交接

日期：2026-09-24

方案分支：`xjn/rematch750-head-l2sp`

状态：实现完成，等待 NPU 执行

## 1. 研究问题

当前平台领先单模型 `RM_V4_F05` 使用 320px、普通线性分类头和 AdamW。它相对
224px V3 在本地 macro/micro 提升约 2.08/2.03pp，平台相对 F03 提升约
1.06pp，证明高分辨率细节能迁移；下一步不应回到旧 224px 基线做主搜索。

普通 AdamW 对整个分类头权重做 decay，目标点是零。本方案改为 L2-SP 语义：把
RM-LP 的分类头固定为 `W_parent`，只训练零初始化残差 `Delta W`，有效分类头为
`W_parent + Delta W`；AdamW 只衰减 `Delta W`，因此正则中心是父分类头而不是零。
视觉 backbone 仍按 F05 完整微调。

假设：F05 的视觉表示更新有益，但普通分类头可能同步漂离 LP 的稳定决策边界；对
head 增量做 L2-SP 能保留迁移性，同时允许 750 类边界适应高分辨率特征。

## 2. 不重叠说明

V4/V5 已覆盖固定/更高分辨率、SAM/GSAM、feature anchor、RRC 范围、attention-local、
质量修复、cached linear/cosine head 与多类组合，但未启用训练态
`classifier_mode=anchored_residual`。普通 WD 方案衰减完整 head 到零；本方案衰减
相对 RM-LP head 的残差，正则中心不同，不是 WD01–WD03 的重复。

## 3. 粗筛设计

四组均从同一个 RM-LP checkpoint 独立初始化并训练 16 epochs；不是从 F05 checkpoint
续训，避免 parent lineage 越权。固定 320px、weak RRC+flip、effective batch 1024、
head/backbone LR 4e-4/1.2e-5、CE warmup 2→GCE q=0.5、feature anchor 2.0、seed 42。

| 点位 | classifier | head WD | 含义 |
|---|---|---:|---|
| HL00 | ordinary linear | 1e-4 | 同 revision 的 F05 等价控制 |
| HL01 | anchored residual | 1e-4 | 与旧 WD 相同强度，但中心改为 RM-LP head |
| HL02 | anchored residual | 1e-3 | 10× head-increment anchor |
| HL03 | anchored residual | 1e-2 | 100× head-increment anchor |

`classifier_residual_scale=1.0` 固定，保证残差的有效学习率与 head LR 不被额外平方缩放。
四组均开启 epoch-0 evaluation；HL01–HL03 在训练前必须与 RM-LP logits 逐元素一致。

## 4. 判据

主比较必须使用同次 HL00，不把历史 F05 的跨 commit 数字当严格控制：

- `raw_macro` 相对 HL00 ≥+0.30pp，且 `raw_micro` 不低于 HL00 超过 0.10pp：成为候选；
- `raw_macro` +0.20~0.30pp：仅记为弱信号，不占平台名额；
- 四点均在 ±0.10pp：关闭本轴，不细扫；
- 任何点 ≤−0.50pp、非有限梯度、无 optimizer update 或 epoch-0 不同：判失败/关闭；
- 首轮若达到 +0.30pp，再用 seed 3407 与 2026 复验；三 seed 平均 macro 仍
  ≥+0.30pp 且至少 2/3 同向，才交给队长决定是否使用平台名额。

不自动生成或上传平台包。训练完成后仍应执行单 checkpoint 确定性推理、37,444 行
提交检查和 SHA 登记，作为可提交产物；实际上传权归队长。

## 5. NPU 执行

```bash
git fetch origin
git worktree add /workspace/noise-worktrees/xjn-head-l2sp xjn/rematch750-head-l2sp
cd /workspace/noise-worktrees/xjn-head-l2sp

source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=<空闲物理卡号>
export OMP_NUM_THREADS=8
export PYTHONPATH=reproducibility/aegis_f1

# 先物化并人工核对，不训练。
/workspace/noise-npu-venv/bin/python scripts/run_rematch750_head_l2sp.py \
  --device npu:0 --trials HL00 HL01 HL02 HL03

# 审核后串行执行；若有多张空闲卡，可把 trials 拆开，每卡仍使用逻辑 npu:0。
/workspace/noise-npu-venv/bin/python scripts/run_rematch750_head_l2sp.py \
  --device npu:0 --trials HL00 HL01 HL02 HL03 --execute
```

## 6. 首步门禁

- parent/train/val/class mapping SHA 与 manifest 一致；
- HL00 的模型、损失、优化器、320px 输入和有效 batch 与 F05 配方一致；
- HL01–HL03 epoch-0 logits 与 RM-LP parent 完全一致；
- frozen base classifier `weight/bias` 不在 optimizer，且一步后逐位不变；
- `residual_weight/residual_bias` 在 optimizer head group，一步后至少一个发生变化；
- head group WD 分别为 1e-4/1e-3/1e-2，visual WD 固定 1e-4；
- effective batch=1024（microbatch 256 × accumulation 4），更新数连续；
- 无 AMP overflow、loss/grad 有限，输出只写入 `outputs/xjn/rematch750_head_l2sp/`。

本文件在 NPU 结果回填前只声明 `ready_for_npu`，不声明方法有效或优于 F05。
