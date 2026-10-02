# V2_SIXVIEW_PAIR_HANDOFF_20261002

本段补齐现有产物的Windows交付，支持2026-10-03的平台配对判断；没有新训练、推理或预测。
当前现役仍是768 full + 固定bias，用户报告74.41512658903963%，80%目标尚差5.5849pp。

## 发现与范围

上一段v2 DEV诊断是已交付的有效进展，但没有给v2与现役768做平台排名。
本次fetch核对main为`eea47a0`，全部分支和近期历史没有六视图raw的Windows补交付；
本机无运行中的用户service/timer或CUDA计算进程，没有在途任务可等待。
远端已关机，不连接、探测或恢复。

Windows目录已有六视图bias，但`submission/submission.zip`是原四视图raw。
现成六视图raw已通过9项检查和全量独立复算，尚只在Linux方案产物中。
本段将它原样复制到新的`submission_raw_sixview/`，保留原四视图包。
这使后续raw/bias配对仅改变固定均衡校正，避免把视图变化也算作bias效果。

## 两个确切文件

共同根目录：`C:\Users\lqh22\Downloads\v2_continuation_20261001`。

| 用途 | 根目录内相对路径 | ZIP SHA256 |
|---|---|---|
| 用户指定的六视图bias候选 | `submission_bias_sixview\submission.zip` | `7164b26d9662a8c0e78c9c0636393efe0927739cb84df8dc88ed33e35f47659a` |
| 同六视图无bias对照 | `submission_raw_sixview\submission.zip` | `4228a78a390e34744e42fb9a8ef861c6807d37b4e28dedc61bc3984c98353c6c` |

同一full第5轮raw checkpoint SHA `3b9fcce3a4a2a113f2eb2ad3a2028c7f224fd1b11f1bdaadd6641ee1ccc85138`，
448/512/576各原图与水平翻转、概率均值取log、相同37,444张；bias固定200次、强度1。
同缓存两包改变3,077条预测，这不是净修正或平台增益。
原四视图`submission/submission.zip` ZIP SHA为`e7152143952f7853c3cf6617bd320675fa8339aa24f4ff841d1cd627e462124e`，
不作为这次纯bias配对的对照。

## 执行与核验

独立分支`codex/v2_sixview_pair_handoff_20261002`、目录
`/home/lux1/noise/worktrees/v2_sixview_pair_handoff_20261002`。只新增复制/核验脚本和交付回执，
并更新当前入口、六视图原交付及DEV诊断中的配对路径说明。
源推理配置继续是[原固定六视图配置](../configs/v2_sixview_bias_20261002.json)。

```bash
cd /home/lux1/noise/worktrees/v2_sixview_pair_handoff_20261002
python3 results/v2_sixview_pair_handoff_20261002/copy_raw_pair.py
# 后续只核验，不写文件：
python3 results/v2_sixview_pair_handoff_20261002/copy_raw_pair.py --verify-only
python3 scripts/check_submission.py \
  --test_dir /home/lux1/noise/test \
  --class-mapping /home/lux1/noise/artifacts/stages/repechage/20260921/class_to_idx.json \
  --csv /mnt/c/Users/lqh22/Downloads/v2_continuation_20261001/submission_raw_sixview/pred_results.csv \
  --zip /mnt/c/Users/lqh22/Downloads/v2_continuation_20261001/submission_raw_sixview/submission.zip
```

同样对`submission_bias_sixview/`执行正式提交校验。
复制仅接受已入库原报告及独立复算一致的文件，现存同名不同内容则拒绝覆盖。
回执记录实际两包SHA、同checkpoint/视图/缓存身份及逐文件大小；不重新拟合bias或生成预测。
原推理及独立float64复算见[原交付](v2_sixview_bias_20261002.md)。
实测已完成8份raw相关文件复制，两个Windows包分别通过37,444行、9项正式检查；
CSV与ZIP内字节一致，逐文件SHA及同六视图身份一致，逐行重计差异3,077张。
这是已有产物的可逆复制，验证覆盖实际目标文件，不添加模拟复制单测。
见[复制回执](../results/v2_sixview_pair_handoff_20261002/pair_receipt.json)、
[实际检查记录](../results/v2_sixview_pair_handoff_20261002/validation.json)、
[raw检查日志](../results/v2_sixview_pair_handoff_20261002/raw_check.log)及
[bias检查日志](../results/v2_sixview_pair_handoff_20261002/bias_check.log)。

## 下一步判定

六视图bias已于2026-10-03回填 **74.6688387992735%**（当时最高），提交文件SHA与本目录产物一致；
本段的六视图raw随后回填 **72.96229035359471%**，配对收益+1.70654844567879pp / +639张，
raw比v1 768 raw 71.5388%高+1.42345903215468pp / +533张，但修正后仅领先95张。
差值只能给总体净变化，平台没有逐图真值时不能拆成修正/退化。
不据这次配对扫描bias强度，不因GPU空闲、DEV小涨或缺少现役独立DEV分自动启动新训练。

现阶段最高包是本次bias包`submission_bias_sixview\submission.zip`（74.6688387992735%）；
此前最高为`C:\Users\lqh22\Desktop\noise\v1_768_full_test_bias_submission.zip`，
SHA `0f83a2458199d524d3c70f19744fc7043497a531ba968d740ab6f15ebfaf4fca`，
校验见[平台包复核](../results/v1_768_bias_platform_20261002/validation.json)；
待测包排序见[平台反馈记录](v2_sixview_platform_20261003.md)。
本段完成推送/集成后停在交付检查点，目标保持未达成。
