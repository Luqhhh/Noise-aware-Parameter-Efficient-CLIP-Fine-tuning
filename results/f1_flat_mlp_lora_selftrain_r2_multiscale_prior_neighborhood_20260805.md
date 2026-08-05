# R2 多尺度 Prior 邻域候选（2026-08-05）

## 目的

当前平台最佳采用 balanced-prior strength 0.85。为避免重复执行约 4.5 分钟的三尺度
GPU 推理，新增 `infer_prior_aligned_submission`：从 `infer --dump-logits` 保存的
prior 前 fused logits 离线生成一个完整可审计提交包。

生成器严格核对 dump 必需字段、有限值、`[24967,500]` 形状、官方测试文件顺序、
类别映射、checkpoint 与源 logits SHA，并要求显式确认 balanced test prior 风险；
最终复用 `create_submission` 生成 CSV、ZIP 与 lineage manifest。

定向测试 `8 passed`，全量测试 `264 passed, 8 warnings`。

## 固定源资产

- checkpoint：R2 epoch 3
- checkpoint SHA-256：
  `67efab2bf954139b59df074ccf00c0113cbc6ff96163d6e8d66ffbe553b910a4`
- fused logits：144/160/176、top5、local0.4、flip0.5、temperature1.5
- fused logits SHA-256：
  `9bb1e6542781e2a0def12eb1c080bd475bac2267ac2ba02010fa391bd452b8c5`

## 候选

| prior strength | 改变 vs 0.85 | CSV SHA-256 | ZIP SHA-256 | 状态 |
|---:|---:|---|---|---|
| 0.80 | 280 | `c653d1bb9cd01473a51cc2f33f29a8a2cf2dd1da95ab7290eaf92f4788aad692` | `6d90f0afc6fbf974a586bc1432ecf19ca76327b54facf3a8a61b07c16be4dd0d` | audited, held |
| 0.90 | 309 | `98faa9559160acf48ee8cc80a6cb5d056080836e4b00f5bae45d872180985c5b` | `161073e8939bb6bd1cfce4aafbb138f57a3d5b50ec681cf571a95f7f2964da40` | audited, held |

0.80 与 0.90 互相改变 584 个预测。两包均为 24,967 张、500 类、ZIP 仅含根目录
`pred_results.csv`。历史平台证据仍支持 0.85，因此当前桌面继续保留正在待测的
144/160 双尺度包；本表候选只在收到其平台结果后按证据选择。
