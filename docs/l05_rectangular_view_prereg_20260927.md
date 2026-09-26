# L05 全图矩形视图（预注册）

实验 `L05_RECTANGULAR_VIEW_20260927`，分支 `codex/l05_rectangular_view`，base main `9f403f3`。
开始前fetch并检查全部本地/远端分支及近期main结果。三位置裁剪仍是方形窗口，已关闭；
V5 V03/V07 letterbox是带填充的训练/推理方案、仍属blocked_implementation。
本轮是独立的无填充矩形推理，不接管/重复其训练，不派生三裁剪的融合权重或视图数量扫描。

## 固定假设与机制

保持冻结L05单checkpoint及分类头，检验保留全部图像上下文是否比center crop更有效。
每张图缩放系数 `min(384/min(W,H), 576/max(W,H))`；两边分别四舍五入到32倍数，
最短不低于32、最长不超过576。整个RGB图像bicubic缩放到该尺寸，无裁剪、无填充。
舍入会引入少量纵横比变化，极细长图像的32px下界可能产生较大变形；不依据val更改规则。

将父模型12×12空间位置编码以bicubic、align_corners=False、antialias=False插值到新网格，
CLS位置编码保留；原12×12直接返回原张量。参考
[timm官方位置插值实现](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed.py)
的二维网格思路，参数选择固定如上；不是对timm默认选项的逐项复现。
检查主线模型后确认Aegis `_encode_visual`已支持矩形网格；正式推理直接复用该原生路径。
另写短的独立OpenAI CLIP forward仅作审计参照，位置张量局部计算，不改模型参数。
这不是在测试图上拟合参数/统计量，也不创建第二个checkpoint。

固定两视图：全图矩形原图及其水平Flip，概率均值、T1.4、候选自身val拟合uniform prior0.60。
不与center logits融合、不扫分辨率/温度/先验。只读14,880张val；训练和测试图不用于本轮拟合
或筛选（parent训练资产按已有stage绑定验证）。测试只在候选通过后做确定性推理。
从文件哈希和严格像素解码检查开始，禁止坏图零填充。按尺寸分桶batch32，结果恢复原val顺序。
原预注册曾声明编码器/分类头float32、无autocast；该声明与冻结参照缓存的实际生成协议不符。
在任何方法结果产生前已更正为CUDA AMP float16，输出缓存存float32；审计阈值不变，详见文末。

## 判据与审计

先在32张原生384方形验证输入上核对独立审计forward与原生forward逐位相等，
再与冻结参照中心/Flip logits核对max_abs≤1e-4且top1全部一致；不过则禁止正式验证。
每种矩形网格首个batch再核对原生/独立forward max_abs≤1e-5且top1完全一致。
固定解码macro较L05≥.30pp且micro不下降为初筛门；通过后按内容组固定3折条件交叉校准，
macro≥.20pp且micro不下降才晋级。条件交叉校准不是独立模型选模验证。
未过门关闭，不派生几何/融合扫描。完整val结果以独立NumPy float64解码再核对；
最终参数逐位未变、源checkpoint哈希未变。推理不改变训练骨干。

晋级后再做独立图像复放和单checkpoint CSV/ZIP、9/9校验，平台上传由用户执行。
失败交付既有L05_T14_P060保底CSV/ZIP并复核9/9，记录路径和哈希。
无新平台成绩前最佳仍为66.94797564362783%，70分目标未达。

```bash
python3 -m pytest tests/test_l05_rectangular_view.py -q
python3 scripts/run_l05_rectangular_view.py --config configs/l05_rectangular_view/fixed.json
```

首个失败目录`outputs/codex/l05_rectangular_view/`保留；修正后输出到新目录
`outputs/codex/l05_rectangular_view_v2/`，拒绝覆盖完成结果；只用空闲GPU，
运行期间每30分钟监控。无外部数据、无跨阶段资产、无集成、无测试时训练。


## 数值协议修正（正式方法验证前）

首个进程240969已终止，停在32张原生方形重放检查，未进入矩形完整val验证，没有方法成绩。
独立检查缓存生成脚本：`use_amp = config.train.amp and device==cuda`；参照缓存本身没有
记录autocast字段。实测纯float32的原图/Flip logits与参照最大差0.02314949/0.02764320，
而AMP两路最大差均为0、32/32 top1一致；独立审计forward与原生forward最大差也为0。

错误在本轮预注册对参照精度的理解，不归因于几何机制。现明确修正为与参照一致的AMP，
保留1e-4原生重放阈值、1e-5矩形路径阈值、模型/几何/温度/先验/晋级门全部原样。
首轮源码559b859、日志、implementation.json、precision_diagnostic.json与invalid_reason.json
均保留，重跑使用独立v2目录。诊断只验证数值一致性，不对AMP/float32做方法选优。
完整证据见[精度审计](../results/l05_rectangular_precision_audit_20260927.json)。
