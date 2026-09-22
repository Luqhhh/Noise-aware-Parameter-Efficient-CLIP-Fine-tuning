# REMATCH750_V3：精度优先的耗时权衡

状态：预注册，未有新训练结果。用户明确要求探索精度/耗时配置，并选择macro下降≤0.1pp。

参考为V2同后端NPU batch32八轮：macro0.7190471887588501、micro0.7298387289047241、纯训练2670.459578s、完整CLI2870.871812s。精度可接受下界raw_macro=0.7180471887588501（71.80471887588501%）。一次达界只表示本次独立验证满足约束，不代表统计等价。

## 有界比较

先完整运行 `configs/rematch750_v3_b64.yaml`、`configs/rematch750_v3_b128.yaml`。只有当两者均未达精度约束时，最多补一次B64_LR2：batch64、head LR2e-4、visual LR6e-6，其余不变。该点是实测假设，不把LR线性缩放当作保证。最多三次八轮训练，不重跑B32/B1024、不扫描workers/prefetch，不恢复LoRA或关闭路线。

共同设置：910B2单卡、当前空闲设备才启动；workers16、prefetch2、pinned、fused AdamW、foreach norm、OMP4、既有taskset144–191（此机器device0记录）。同NPU LP SHA `d5cb8f5265754fd900d3efde23e24fefbcf616c747f2cab13e4dc2201fdd689b`，133815/14880同内容组划分；8 epochs/cosine horizon8、原head1e-4/visual3e-6（仅条件点LR2例外）、CE2轮/GCE6轮q=.5、anchor2.0、shuffle、无LR warmup。每20步及轮末记录成功更新，2/4/6/8验证，raw_macro主选模、同值raw_micro。

B64每轮2091步、八轮16728步；B128每轮1046步、八轮8368步。完整跑满八轮，不能只用CE段或重建optimizer续训冒充。各自独立目录，共享训练/验证数据、特征和初始化只读，Python依赖不修改。

## 选择与交付

执行、optimizer有限性、成功更新、checkpoint及逐类重载审计先通过；在macro满足阈值的配置中选**完整CLI耗时最短**者，另列纯训练秒数、macro/micro差、固定尾部75类及其余675类。若新点均失败则保留已验证B32。不用验证时间的短窗口估算代替完整耗时。

用户的效率筛选与+0.20pp平台精度晋级门分开：满足≤0.1pp精度损失的最快新配置可生成工程交付CSV/ZIP并独立校验，标记为效率候选，不自动声称平台更优、不自动上传。若无新点满足约束，保留并复核原FT交付包。报告包含所有失败点，不只报告胜者。

## 隔离与重放

本机worktree `/home/lux1/noise-worktrees/rematch750_v3_tradeoff`，分支 `codex/rematch750_v3_tradeoff`；远端独立源码目录 `/workspace/noise-worktrees/rematch750_v3_tradeoff`。原始数据/特征通过只读使用的路径共享，init路径显式指向同哈希NPU LP；输出 `outputs/codex/rematch750_v3_tradeoff/{RM_V3_B64,RM_V3_B128}/seed42/`。

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH=reproducibility/aegis_f1
taskset -c 144-191 /workspace/noise-npu-venv/bin/python -u -m aegis_clip.cli.rematch train --config configs/rematch750_v3_b64.yaml
# 第一组完成并确认卡空闲后，替换为 configs/rematch750_v3_b128.yaml
```

训练不串接平台上传。阶段结束推送方案分支，再在main集成目录以自动模式 `git pull --rebase --autostash origin main` 同步、合并、重新校验、推送；不执行手动stash pop。
