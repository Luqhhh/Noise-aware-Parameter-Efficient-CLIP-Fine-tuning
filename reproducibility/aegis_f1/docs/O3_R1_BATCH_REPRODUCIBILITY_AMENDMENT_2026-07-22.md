# O3-R1：批布局复现修订协议

日期：2026-07-22

状态：**预注册，尚未执行**

## 1. 原 O3 的关闭事实

原 O3 在任何参数更新前执行参考复现审计并失败，程序按预注册规则退出：

| 审计 | 最大绝对 logit 差 | argmax 一致率 |
|---|---:|---:|
| F1 center | `0.041015625` | `99.9127567%` |
| F1 + M1 | `8.243944168` | `99.6413350%` |

因此原 O3 不得被描述为已训练或门控失败；它是**训练前实现/复现审计失败**。原 validation cache 保留不覆盖，SHA-256 为 `a85faa9e4392918c107bcfc8cc91f70b40086923ef7732cb1c575db5a1acfbd6`。

## 2. 根因证据

旧 F1 center/M1 参考由 `cache_validation_logits.py` 生成，其默认 batch size 为 `128`；F1 checkpoint 的 `evaluation.batch_size` 与 `evaluation.inference_batch_size` 也均为 `128`。已获得平台 `63.3276` 的 F1+M1 测试推理继承同一数值条件。

原 O3 协议却显式以 `batch_size=64` 生成 validation cache。父 checkpoint、validation CSV、路径顺序、裁剪大小、top-patches、融合公式均完全一致，因此 batch 布局是唯一已确认的执行条件差异。

逐层审计结果：

- 旧 center 与旧 M1 内部 global logits 逐位一致；
- 旧 M1 的已存融合 logits 与用其 global/local logits 重新执行概率均值的最大差仅 `3.8147e-6`，融合公式自洽；
- batch-64 全局分支所有样本的逐样本最大差均不超过 `0.0411`，但改变 9 个低间隔预测；
- batch-64 局部分支因 AMP 微扰进入离散 attention top-5，75 个样本的最大 logit 差超过 `1.0`，94 个局部预测改变；
- 最终融合有 `37 / 10,316` 个预测改变；相对正式 batch-128 M1，clean-core micro `-0.06760pp`、trusted macro `-0.04884pp`、raw micro `-0.04847pp`。

解释：矩阵乘法的半精度结果可随 batch 布局产生微小数值差异；连续 logits 变化本身很小，但 top-5 patch 选择在并列/近并列 attention 处是不连续的，少量样本会换用不同局部中心。该现象不代表模型权重、图像顺序或融合公式不一致。

## 3. 唯一允许的 O3-R1 修订

只修正 validation 的数值复现条件，不改变科学假设、模型、训练数据、训练 cache、Adapter、损失、优化器、选择指标、融合权重或晋级门槛：

1. 原 O3 validation cache 保留，不覆盖；
2. 新建 `validation_bs128.pt`，固定 `batch_size=128`；
3. 缓存必须写入 batch size、AMP、设备与 PyTorch/CUDA 元数据；
4. 新缓存首先与旧 F1 center 和 F1+M1 参考做原有严格审计；
5. center 与 M1 的 prediction agreement 必须都为 `1.0`；最大差必须分别为 `0.0` 和不超过纯融合重算误差 `4e-6`；
6. 任一复现门槛失败，O3-R1 立即关闭，不再次更换 batch、AMP、裁剪或容差；
7. 只有复现门槛通过，才可使用原 O3 固定配置进行一次 CPU Adapter 训练；
8. 最终测试推理固定 `batch_size=128`，与已获平台分数的 F1+M1 数值条件一致。

## 4. 固定重建命令

启动前必须确认团队没有 GPU 进程；命令只能写入独立 Aegis 输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/x28639/projects/AegisCLIP-F6-A2LoRA \
  /home/x28639/projects/AegisCLIP-Noise-Robust/.venv/bin/python \
  -m aegis_clip.cli.cache_local_adapter_features \
  --checkpoint /home/x28639/projects/AegisCLIP-Noise-Robust/outputs/F1_VISUAL_LORA_CLEAN_CORE/seed42/checkpoints/best.pt \
  --split-csv /home/x28639/projects/AegisCLIP-Noise-Robust/artifacts/stages/preliminary/seed42/val.csv \
  --output /home/x28639/projects/AegisCLIP-F6-A2LoRA/outputs/O3_R1_F1_LOCAL_ONLY_FEATURE_ADAPTER/seed42/cache/validation_bs128.pt \
  --batch-size 128 --num-workers 4 --crop-size 160 --top-patches 5
```

## 5. 性能门控保持不变

- clean-core micro 相对正式 F1+M1 至少 `+0.20pp`；
- trusted macro 不下降；
- raw micro 不下降超过 `0.10pp`；
- 局部特征平均 cosine drift 不超过 `1%`；
- prediction empty classes 为 0；
- 全局 F1 分支与父 checkpoint 保持不变；
- 失败即不生成测试提交，不扫描 bottleneck、scale、dropout、LR、损失、crop、top-k 或融合权重。

## 6. 决策边界

O3-R1 不是根据性能结果调参，因为原 O3 从未进入训练，且本修订发生在任何 Adapter 指标产生之前。它只恢复既有 F1+M1 平台基线的已冻结 batch 数值条件。O3-R1 的运行仍需在报告本次审计失败后得到明确继续授权。
