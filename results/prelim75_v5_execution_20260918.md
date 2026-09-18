# PRELIM75 v5 执行记录（2026-09-18）

## 状态

`PRELIM75_V5_20260917` 的实现、正式 GPU smoke、F0/F1 各 3 epoch 固定训练、重叠诊断、测试推理、提交校验和真实平台回填均已完成。F0 为 69.0752%，F1 为 69.1393%；保留 F1 为当前无校准胜者并关闭 v5。没有自动上传、push、F2 或推理参数扫描。

权重父模型固定为 v4 C0（平台 68.4544%）：

- checkpoint：`outputs/prelim75_v4_20260916/C0/candidate.pt`
- SHA-256：`48d4f4ccec8758f93b126e059790107981c2359a94963d5d0147d49f05613f8a`

## 实际执行与中断

固定队列命令：

```bash
PYTHONPATH=reproducibility/aegis_f1 python3 -u scripts/run_prelim75_v5_queue.py \
  --config configs/prelim75_v5.yaml --execute
```

用户第一次暂停发生在 F0 epoch 10 的 83,232/103,218 行；第二次按指令在 epoch 11 完成后暂停，信号生效前 epoch 12 已执行一个 32 图 batch。两次均未生成候选 checkpoint，半程权重没有被恢复、评估或提交；对应目录为：

- `outputs/prelim75_v5_20260917_paused_20260917T160449/`
- `outputs/prelim75_v5_20260917_paused_20260917T175712/`

重新执行后 F0 完整结束。F1 首步融合梯度审计在任何 optimizer 更新前因 `weighted["fusion"]` 键名错误失败。提交 `2a9680084d68743cd13c5f044f51f3f4bf4e3073` 将其修复为 `weighted["fusion_gce"]`；损失公式、blend、温度、数据和随机序列均未改变。失败现场保存在 `outputs/prelim75_v5_20260917/F1_failed_keyerror/`，修复登记在 `f1_retry_registration.json`。

修复后验证：v5 18 passed；prelim75 49 passed；完整 `reproducibility/aegis_f1/tests` 489 passed、1 skipped。F1 从同一 C0 独立重启。F0/F1 首批身份 SHA-256 均为 `31915ba8a0af2b8517d2eebf65dc214fdfa72b2be4506dfcccc47aa007cf6d91`，global/local/separate/fusion GCE 与 anchor 公共分量逐项一致，blend 分别为 0.0/0.5。

## 训练与诊断

| 候选 | 固定末轮 loss | 重叠 raw micro | 重叠 clean-core micro | 相对 C0 raw/clean-core |
|---|---:|---:|---:|---:|
| F0 | 0.14837972 | 83.491665% | 96.098757% | +0.261730pp / +0.354654pp |
| F1 | 0.14630883 | 83.559519% | 96.166962% | +0.329584pp / +0.422859pp |

两组均为 103,218 行/epoch、3,226 update/epoch、总 9,678 update，batch 32、FP32、sample epoch 10/11/12。F1 的融合项单独梯度审计对 global/local logits、visual projection、共享 head、O3、PTA 均为有限非零。冻结参数、双 Adapter 严格重载及最终推理辅助资产隔离检查通过。

本地验证与训练重叠，只作工程诊断。F1 相对 F0 的 raw/clean-core 约 +0.0679pp/+0.0682pp，不能据此认定平台更好，也不用于选择 epoch。两组均未触发相对 C0 下降 2pp 的止损规则。

## 提交产物

| 项目 | F0 | F1 |
|---|---|---|
| checkpoint SHA-256 | `afd21e73442441fbb3b885c1081b34bde00306c910c3623c50e841ad3c7079f9` | `44e64a9528653f1d76db6a12b56a22c3ad23851665e8cef45d06f7ece9f52e1b` |
| CSV SHA-256 | `073bf727b61e00f1cfe4f10277440b9ffe5e1af8a5e3704216b694db76088488` | `eea66dec1c79c8c8a9392dfc225f24772933f02b1bfbd686052fd89cec3f5a50` |
| ZIP SHA-256 | `6bbe24731f2a83a5e4ba401e94fe8035944eb57d2bad04d23c5365c775a14a3a` | `0863b93c785042692ec928107429d679815b50d2b1971b4a984cc1bceb9a705d` |
| 平台成绩 | 69.0752% | 69.1393% |

路径：

- `outputs/prelim75_v5_20260917/F0/submission/submission.zip`
- `outputs/prelim75_v5_20260917/F1/submission/submission.zip`

两个 ZIP 均只含 `pred_results.csv`，覆盖 24,967 个测试文件且无重复，标签为 0000–0499 四位编号，包内外 CSV 字节一致；独立复检均通过。两份预测相差 299 行。最终协议为单 checkpoint、原生 224、四尺度 attention local、Flip、T=1.5、无 prior；训练框、监督和教师资产均不是最终推理依赖。

## 平台回填与停止决定

用户按 F0、F1 顺序回传真实平台 Top-1：69.0752%、69.1393%。精确正确数和上传时间未提供，保持 null。

- F0 比 C0 68.4544% 高 0.6208pp。
- F1 比 C0 高 0.6849pp，比同条件 F0 高 0.0641pp，因此保留 F1 为当前无校准胜者。
- F1 只比 `max(C0,F0)` 高 0.0641pp，未达到 0.30pp 投入回报门槛；这是真实小幅收益，不授权自动追加 F2。
- F1 仍比历史 prior0.90 最高 70.352866% 低 1.213566pp，距 75% 低 5.8607pp；历史绝对最高和 75% 目标均未刷新。

v5 按固定规则结束，不自动追加续训、prior、温度或融合比例扫描。F0/F1 包和 C0 回退包均保留。
