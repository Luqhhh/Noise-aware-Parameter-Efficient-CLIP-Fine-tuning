# PRELIM75_V2_20260912 执行记录

用户授权：按顺序执行。源基线83c03dd，main，远端fece41b。H0/H1 → V0/V1顺序执行；C只在真实平台反馈满足方案门槛时启动。不push、不操作平台账号。

## 初始实现与检查

实现完整P十视图global/local base/residual分片缓存；P epoch3实际软目标与权重；H0/H1共享头重拟合；V0/V1完整双Adapter联合短微调；真实候选attention诊断；固定单checkpoint、无校准推理及提交校验；顺序队列与每段冻结源码。

H checkpoint将表示训练scope登记为 `frozen`，严格加载的plain visual参数、O3/PTA及共享linear结构不变；这是实际冻结状态，不将backbone LR guard关闭。V继续登记full_finetune许可掩码。

首批8个手算/几何/有效batch检查通过，随后增加冻结scope严格重载和工程止损检查，专项共10 passed。首次CPU Aegis回归448 passed/1 skipped（12.21秒），根回归424 passed/1 skipped（51.89秒）。CPU回归设置CUDA_VISIBLE_DEVICES为空以隔离运行缓存；真实官方训练图的CUDA初始检查另行完成：局部原生logits与base/residual重构最大差0，argmax相同，global重构通过。V1首步还将单独核验local loss对visual proj和两个Adapter的梯度，不能仅由总loss的视觉梯度代替。

## 实际命令

以下从仓库根目录运行，缓存命令已启动，其余由缓存完成后的队列顺序执行。候选目录已有即停止，不自动覆盖/恢复。

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -u -m aegis_clip.cli.cache_prelim75_features --config configs/prelim75_v2.yaml
python3 scripts/run_prelim75_queue.py --config configs/prelim75_v2.yaml --segment head --execute
python3 scripts/run_prelim75_queue.py --config configs/prelim75_v2.yaml --segment visual --execute
```

队列默认不带 `--execute` 时只审计。执行时保存配置、main/ref/历史状态和冻结源码；每个候选完成固定训练→真实10视图诊断→未触发工程止损则真实测试推理→24967行CSV/ZIP检查和包内外一致性→候选报告。GPU时间、显存峰值和各阶段退出码来自真实运行，线上分数留空。

## 当前截面

运行目录 `outputs/prelim75_v2_20260912/`。完整训练缓存生成中；尚无已训练的新候选、提交包或平台成绩。初赛75%目标未达到，不把实现或本地重叠指标写成达成。后续在本记录追加各段实际末轮结果、哈希与提交路径。
