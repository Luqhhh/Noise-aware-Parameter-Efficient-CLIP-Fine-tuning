# V2_PREPROJECTION768_PAIR_20261002

用户提出“v2+768”，本段将其按现役768方案的**投影前768维CLS**解释。
这是一项新的v2单模型结构配对，使用本机CUDA；不会连接已关机服务器。
基线`c4bd237`，方案分支`codex/v2_preprojection768_pair_20261002`，独立同名worktree。
已核对全部本地/远端分支和近期main：现有768头/LoRA实验使用v1，未发现v2投影前结构的重叠实现或运行。

## 可证伪问题和固定投入

v2现有路径为`CLS768 @ P(768,512) -> Linear(512,750)`。
其有效分类权重秩至多512；直接768头允许学习投影子空间外的分类方向。
v1冻结特征768曾比512净修正247张，但v2的投影已参与全参数训练，不能沿用该收益结论。
v2 s3归档尚有3,352张DEV错误，五个既有模型共同错误2,411张，构成本次待检验预算，
并不意味着这些错误均可恢复。用已冻结五模型共同错误作为主要机制分组，同时保留tail75等原分组。

共同父为原s3第4轮EMA，`05f4c1a…6425f`，只用133,815张train_dev；14,880张val_dev不入训。
control保留768→512投影和512头，candidate去掉投影、接768头；全部视觉塔均可更新。
候选初始化为`W768 = W512 @ P.T`，bias保持，其他视觉参数原样严格加载。
该代数变换本身不代表提分。两臂共用BF16视觉塔与FP32投影/分类头读出，先核对全val的
零更新logits最大差≤0.0002、top1完全相等。与历史原生BF16头的差异只记录，不放宽或重开
已经关闭的teacher重放协议；本段是明确的新本机配对基线。

每臂固定512次成功逻辑更新、batch80/micro16、576训练，seed20260926。
沿用父的增强、sqrt-inverse频次采样、原标签、LS0.15、Mixup/Cutmix。
两臂共享冻结40,960次抽样、逐图增强种子和逐批Mixup/Cutmix种子，并比较训练输入/目标摘要。
AdamW backbone4e-6/head1e-4、WD0.15、clip1、10% warmup后cosine到0.02倍。
只评固定末步raw，不按验证分挑轮，不做EMA/强度/seed搜索。
改变了头参数化及相应优化几何，结果不能全归因为“多256维信息”。

先各8步实测成本后丢弃探针权重，正式两臂都从共同父重新开始。
训练、完整验证、双臂六视图推理及出包合计上限7,200秒，预测成本须保留1.25倍余量。
成本不通过即不开始正式训练；数值/资源异常保存失败事实，不隐式重启。
推进复核门：候选对control净≥75，对共同父净≥75，共同错误组对control净≥25，
macro不降，修正/退化≥1.25；未过则关闭此固定短续训，不否定全部v2+768方法。
通过只支持后续完整训练复核，不能声称平台提分。

## 交付与命令

工作目录`/home/lux1/noise/worktrees/v2_preprojection768_pair_20261002`。

```bash
PYTHONPATH=reproducibility/aegis_f1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python3 -u scripts/train_v2_preprojection_pair.py \
  --config configs/v2_preprojection768_pair_20261002.json \
  --output outputs/codex/v2_preprojection768_pair_20261002/pair
```

正式训练后两臂各导出单checkpoint、448/512/576各中心及flip的等权概率平均，
raw及固定200步/强度1均衡bias包，逐包37,444行和9项校验。
独立复算逐图指标/配对/门、训练输入同一性、冷加载及CSV/ZIP与预测一致性。
状态与实测结果将在结束后补入本文件；此预注册不是训练完成报告。

现役包仍为`C:\Users\lqh22\Desktop\noise\v1_768_full_test_bias_submission.zip`，
平台74.41512658903963%；9项及字节核验见
[现役验证](../results/v1_768_bias_platform_20261002/validation.json)。
此前v2六视图配对已交付到Downloads，见[配对回执](v2_sixview_pair_handoff_20261002.md)，平台待测。
