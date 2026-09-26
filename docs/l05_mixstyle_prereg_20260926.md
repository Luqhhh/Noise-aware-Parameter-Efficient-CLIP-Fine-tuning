# L05 Patch-stem MixStyle 配对续训（预注册，运行中）

分支 `codex/l05_mixstyle`；实验 `L05_MIXSTYLE_20260926`。2026-09-26 已同步 origin/main，
检查全部分支、V4 82 份配置、V5 64 份配置、focus 和近期关闭实验：没有 MixStyle 方向。
不是对三位置裁剪、非线性头、已关闭 ColorJitter/RandomErasing 的参数扫描。

## 依据与边界

依据 [MixStyle 作者仓库](https://github.com/KaiyangZhou/mixstyle-release) 与
[官方公式实现](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/modeling/ops/mixstyle.py)：
训练时混合不同样本的早期特征均值/标准差，尝试减少对图像风格的依赖。
**对 CLIP ViT patch stem 的迁移是本轮假设，不是论文已证明的本任务收益**。
只混合本阶段 train_dev 同 batch 的统计量，不借用测试域、不改变标签。

## 固定配方

两臂独立从冻结 L05 `35d17c0c…2397` 初始化，MS00 是普通续训、MS01 只新增 MixStyle。
保留 L05 的 global+attention-local 监督、GCE q=0.5、feature anchor=2.0 和384px弱 RRC+Flip；
local 从续训 epoch1 开启，无 CE warmup。各 2 epoch cosine，backbone LR=3e-6、head LR=1e-4、
WD 沿用1e-4，effective batch1024=4×256，seed42，串行执行。epoch0 参与 center raw macro
选模；每 epoch 验证。相比现役 macro 或 micro 降 ≥2pp 时该臂在该 epoch 后停止。

MS01 hook 位置是 `visual.conv1` 输出（N×768×12×12）。每次训练 forward 以 p=.5 激活，
对空间维计算每通道 mean/std，统计量 detach，lambda~Beta(.1,.1)，在 batch 内循环移位配对
（保证非自身）。混合后的统计量作用到标准化特征；标签和空间排列不变。
该 hook 对 global/local 训练 forward 都生效，eval 恒等。无可训练新参数，移除 hook 的
原生 Aegis 单模型可直接推理；不形成多模型/多头融合。
MixStyle 使用独立、由 seed+训练调用计数确定的 NumPy RNG，避免污染训练器随机增强/采样
RNG，保证配对顺序。调用计数与 checkpoint SHA 绑定保存，恢复时验证并恢复。

共享训练器使用只读固定 `focus/f05-four-lines@f050ecb` 源码快照，因为最新 main 尚未整合
该分支的 384px attention-local 续训与绑定修复。快照仅放本方案输出目录；运行前对每个源码
文件核对原 Git blob，不修改他人 worktree、不复制其拟合资产。数据/parent 仍是同阶段只读。
调用其已存在的 same_split_continue 血缘门，保留 dataset_manifest，不绕过父模型验证。

解码固定 Flip/mean_probabilities/T=1.4，候选自身 val 拟合 prior strength=.60。
MS01 必须同时超过现役 L05 与配对 MS00 解码 macro ≥.30pp，且 micro 不下降；MS00 若单独
超过现役同门槛可作为续训候选，不归因于 MixStyle。未过门关闭，不自动派生扫描。
过门后独立重载、条件交叉校准和生成单 checkpoint CSV/ZIP，平台上传仍由用户执行。
无效运行/实现 smoke 不进结果表。正式结果尚未产生。

## 规则合规性

- Backbone: CLIP ViT-B/32
- Pretrained weights: OpenAI official
- External data: No
- Test data used for training/adaptation: No
- Cross-stage data/checkpoint reuse: No
- Multi-model ensemble: No
- Manual cleaning required: No
- Fully reproducible: Yes
- Rule risk: None；训练增强只读本阶段 train_dev，推理无统计拟合或权重更新

## 复现入口

```bash
cd /home/lux1/noise-worktrees/l05_mixstyle
mkdir -p outputs/codex/l05_mixstyle/framework
git archive f050ecb reproducibility/aegis_f1 scripts/cache_validation_tta_logits.py scripts/evaluate_l05_candidate.py | tar -x -C outputs/codex/l05_mixstyle/framework
python3 -m pytest tests/test_l05_mixstyle.py -q
python3 scripts/run_l05_mixstyle.py --config configs/l05_mixstyle/fixed.json --phase prepare
python3 scripts/run_l05_mixstyle.py --config configs/l05_mixstyle/fixed.json --phase train --arm MS01 --smoke
python3 scripts/run_l05_mixstyle.py --config configs/l05_mixstyle/fixed.json --phase queue
```

`queue` 按 MS00→MS01 串行执行训练、缓存该模型自己的 val 双分支、固定解码比较。正式两臂
每次优化更新都记录进度（预启动 configs 中的 log_every=100 已留档，正式改为1，仅日志频率）。
各轮完整 checkpoint、优化器、RNG 和 MixStyle 调用状态保存在独立 run_dir，运行目录存在则
拒绝覆盖。故障后只允许从同臂 checkpoint 显式 `--phase train --arm MSxx --resume <last.pt>`，
无 metadata 或 SHA 不符拒绝恢复；不能仅因观察超时重启正在运行的进程。


## 实现审计进展（非方法结果）

源码快照共 484 个 Git blob 验证通过。首次 `MS01_SMOKE` 首步梯度审计通过，但 sidecar
发现 MixStyle `calls=0/applied=0`：Aegis 在 full_finetune 时仍让冻结 conv1 保持 eval。
这次 smoke **无效、不计为方法运行**，原日志/权重/sidecar 和 invalid_reason.json 保留。
修复为以外层 AegisCLIP 的 training 标志门控 hook，并新增“冻结 stem 为 eval、外层模型为
train”的回归测试；目前 5 项测试通过。`MS01_SMOKE_V2` 在新的独立目录重新跑。
正式 MS01 结束时若 calls 或 applied 为0则硬拒，不能静默成为控制臂。
