# V3_FINAL_DELIVERY_20261001

按用户“继续”完成原v3协议，2026-10-01 **01:26:03 CST**生成并校验提交包。
本段只独立复核已有结果、归档与同步，不启动新训练、推理或搜索，不上传平台。
主协议为V3_SUPERVISION_FT_20260929；恢复入口见
[checkpoint恢复记录](v3_resume_checkpoint_20260930.md)。

## 配方与产物

384、original/v1_supervision各2轮/2786更新，logical batch96、micro batch2、
workers2、官方CLIP初始化、同一冻结draws/增强/Mixup/LR、EMA .9995。
末轮EMA单checkpoint，384 center crop，无bias/TTA，使用原train_dev v1监督，非full。
对照臂从已完成epoch02补齐导出，未重训；两臂初始参数SHA和2786条完整轨迹相同。
恢复controller从09-30 23:02:45运行到10-01 01:26:03，约2小时23分18秒；
最终测试推理277.24秒，服务正常结束，本机训练显存释放。

冻结prepared目录：
`/home/lux1/noise/worktrees/v3_after_v1_20260930/outputs/codex/v3_after_v1_20260930/prepared_exif_recovery`。

- plan SHA：`615fc646d20cf5d7b02f64e3adcc4d659bd607036302b895d08901140fa9e224`。
- 两臂初始模型SHA：`1095bab4f55b749f6b37aa2feda3d0cd712b756c1e535f1a99ff5f209a4beba0`。
- 配对trace SHA：`d9b075be90cc97769da41eee2c89ff798a5540688d8e46039567e3c7263d1afc`。
- 提交checkpoint：prepared目录`runs/train/v1_supervision/selected.pt`，
  SHA `ed61d72f65ee4d9415d45fdfd68833b26cc6e34bd2ac9caef366e34776f63882`。
- CSV：prepared目录`submission/pred_results.csv`，
  SHA `a60acd2640458825c37449f63f7b92e1d8a619b9170b9e455e32a6e35bacf7f1`。
- ZIP：prepared目录`submission/submission.zip`，
  SHA `bd078e6df85084adf997e63103f2a03a876923a2c55827359672502b39011899`。

## 本地结果与判断

固定14880张val_dev、750类，训练未包含val_dev。完整末轮轨迹：

| 臂/权重 | macro | micro |
| --- | ---: | ---: |
| original raw | 59.6075% | 60.5914% |
| original EMA（对照） | 53.0370% | 54.0995% |
| v1_supervision raw | 55.8867% | 56.8414% |
| v1_supervision EMA（冻结交付） | 48.8111% | 49.8925% |

EMA相对对照macro **−4.2259pp**、micro **−4.2070pp**。
修正424、退化1050、净−626张；改判3848张。
尾部1003张/75类macro **−3.8563pp**，修正15、退化59、净−44；
other13877张/675类macro−4.2669pp、净−582；few_support只有1张/1类，不能外推。
完整[对照历史](../results/v3_final_delivery_20261001/original_history.json)、
[监督历史](../results/v3_final_delivery_20261001/supervision_history.json)、
[配对报告](../results/v3_final_delivery_20261001/comparison.json)、
[逐张配对](../results/v3_final_delivery_20261001/paired.csv)及
[逐类报告](../results/v3_final_delivery_20261001/per_class.csv)。

结果`no_support_for_ladder`：本固定两轮对照未产生全量或尾部正信号，
不自动扩大分辨率阶梯、追加训练或改选raw。含噪原标签指标不是干净真值；
本报告没有干净标签或平台收益证据，不能把局部结果当作平台变化。
v1原12轮SWA的六视图/bias协议与本实验不同，不作直接等条件泛化比较。
v3平台分未知；当前最高用户报告分仍为已绑定桌面full v1 SWA **70.98600576861446%**。

## 独立CPU复核与复现命令

新方案目录`/home/lux1/noise/worktrees/v3_final_delivery_20261001`，
分支`codex/v3_final_delivery_20261001`，从最新origin/main建立并同步已发布的v1工程记录。
`scripts/verify_v3_final_delivery.py`沿用原计划绑定的冻结PYTHONPATH。
CPU独立检查两臂endpoint、2786条连续轨迹与初始模型一致性、checkpoint/预测摘要，
从全量逐张预测重算所有分组指标并与report/history一致；
核对37444项StageContext测试绑定，ZIP唯一成员与外部CSV逐字节相等，
再次执行正式9项提交校验。真实完整数据复核通过，不为纯归档新增镜像测试。

```bash
env PYTHONPATH=/home/lux1/noise/worktrees/v3_after_v1_20260930/reproducibility/aegis_f1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python3 scripts/verify_v3_final_delivery.py \
  --plan /home/lux1/noise/worktrees/v3_after_v1_20260930/outputs/codex/v3_after_v1_20260930/prepared_exif_recovery/plan.json \
  --controller-status /home/lux1/noise/worktrees/v3_resume_checkpoint_20260930/outputs/codex/v3_resume_checkpoint_20260930/status.json \
  --output /home/lux1/noise/worktrees/v3_final_delivery_20261001/results/v3_final_delivery_20261001
```

复核archive须为新目录，避免覆盖历史。正式校验确切命令在
[独立验证](../results/v3_final_delivery_20261001/final_validation.json)的`checker_command`，
结果见[校验日志](../results/v3_final_delivery_20261001/submission_check.log)、
[最终controller](../results/v3_final_delivery_20261001/controller_final.json)和
[包绑定](../results/v3_final_delivery_20261001/delivery_report.json)。

现役full v1 SWA桌面包为`/mnt/c/Users/lqh22/Desktop/v1_full_swa_submission.zip`，
既有校验见[full最终验证](../results/v1_full_swa_20260930/final_validation.json)；
本次新增可提交v3包路径和校验已交付，不替换full桌面包。

## Git收尾

提交并推送方案分支；main集成使用自动模式
`git pull --rebase --autostash origin main`，合并、核对文档与JSON并推送。
改动为CPU复核脚本、结果归档、本页、恢复页状态说明及当前执行入口。
确切方案/集成commit和推送状态见本次最终回复，本检查点暂停。
