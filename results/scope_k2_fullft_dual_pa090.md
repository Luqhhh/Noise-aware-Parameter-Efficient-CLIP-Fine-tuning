# SCOPE-K2 / FULLFT_DUAL balanced-prior 0.90 实验报告

## 结论

SCOPE-K2 **不晋级**。在冻结父模型之上的 conditional 5×3 grouped nested OOF 中，SCOPE、matched PACE、no-topology、gate-only 与 margin-only 均被预注册的 beta/precision/Wilson 规则冻结为 no-switch，因此都与 Parent 相同：raw `8257/10316 = 80.040713%`，clean-core `6755/7331 = 92.142955%`。SCOPE 相对 Parent 的 10,000-draw exact-group bootstrap 为 point `0.000000pp`、95% CI `[0.000000, 0.000000]pp`。

失败终态严格回退到归档的 FULLFT_DUAL + balanced-prior 0.90 submission；未创建 SCOPE test parent cache、test evidence cache、deployment 或 submission。

> 限制：本报告只能称为 **conditional grouped nested OOF given the frozen parent**。冻结的 FULLFT_DUAL final parent 使用了 `validation_overlap_with_training: true`，因此这里的 grouped nested OOF 只隔离 beta、threshold、duplicate groups、消融和晋级决策，不能把父模型本身解释为 honest held-out OOF。

## 冻结身份与资产

- 实验基线：`main` / `830de947e9c07738ddb69b9a5d274f0c2a6269e3`，开始时 `HEAD...origin/main = 0/0`。
- 平台父结果：`17565/24967 = 70.352866%`。
- parent checkpoint SHA-256：`f72b0104257f49d2667fe335553a861dd1dea947753feebdc7301b8890b48765`。
- trust bundle SHA-256：`6868041cc7b995a3e8e557ae925d1d25160acf23af09202f46911ce92125b30f`。
- exact duplicate group artifact SHA-256：`41e2668e0fa5e10291051c14a9c75fc96096a69ae7d36d84d0e774270a99bb87`。
- fold artifact：`reproducibility/aegis_f1/protocol_artifacts/scope_k2_fullft_dual_pa090/nested_folds.pt`。
  - file SHA-256：`7d85c94296df713c2844f0fdb2eba21897d32f01bd5383d49122305d5078d87e`。
  - semantic SHA-256：`9160b7b0b01bdfae6efffda6ca838aa7771438c63eb7f7e0f2f8626046e3fe7b`。
  - 10,316 行全覆盖；outer folds `0..4`；每个 outer-train inner folds `0..2`；第二次调用只验证同一 artifact，file/semantic SHA 均不变。

## Cache 与复现审计

| Artifact | run1 file SHA-256 | run2 file SHA-256 | 跨运行语义 SHA-256 |
|---|---|---|---|
| Parent | `d02e8542baf6cd5cc6d8d7f1783bf1ac2fc0293ee58b71ef5fb05e5682423040` | `c142ae712e985f1157a1a5d211119d503b0e80b8af900c6b32a12c83ae4fbe6b` | `f95fb1524144cbe0273402cbea98c6bcfc58312633e833118a25d0aa37ca37cf` |
| Evidence | `e75c663a333327f45887ada4d96e622b09812f0efc1d9d65adad477543e837fb` | `b5e3b1a246dc43aa5200c048477c2161a340981e43a4941230409e6d2e21b25d` | `13366bc68d7c34327db24d7e96b6eee5488492ab4294e66f0ae8c88b25563dc5` |

Evidence 的 instance semantic SHA 分别为 `7ba0e0e3...73c6` 和 `14d3ca46...3a`，差异只来自它们严格记录的 parent cache **文件实例 SHA**；两份 parent 的语义完全相同。去除这一个文件实例绑定、仍保留 `parent_semantic_sha256` 与全部 evidence/gate 值后的跨运行语义 SHA 完全一致。逐字段比较确认 row、candidate、scores、margin、prior、constituent、boxes、corrupt、SCOPE/PACE/no-topology evidence 与 gates 均一致。

审计全部通过：

- 49 个 row-major patch nodes、84 条固定四邻接 edges；
- base/dual classifier-space max absolute error 均为 `0.0`；
- canonical reverse bitwise antisymmetry 为 `true`；独立反向重算 max absolute error 为 `0.0`；
- 全部 classifier direction norm 有效；Pass A/B row/path/candidate/crop/corrupt binding 通过；
- parent prior 迭代数 `50`，corrupt 行数 `0`；
- formal parent run1/run2 与 evidence run1/run2 均独立完成。

## 六方法同折结果

| 方法 | Raw correct | Raw accuracy | Δ Parent | Clean correct | Clean accuracy | Clean Δ | switches | corrections | regressions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Parent | 8257/10316 | 80.040713% | 0.000000pp | 6755/7331 | 92.142955% | 0.000000pp | 0 | 0 | 0 |
| margin-only | 8257/10316 | 80.040713% | 0.000000pp | 6755/7331 | 92.142955% | 0.000000pp | 0 | 0 | 0 |
| matched PACE-K2 | 8257/10316 | 80.040713% | 0.000000pp | 6755/7331 | 92.142955% | 0.000000pp | 0 | 0 | 0 |
| SCOPE gate-only | 8257/10316 | 80.040713% | 0.000000pp | 6755/7331 | 92.142955% | 0.000000pp | 0 | 0 | 0 |
| SCOPE no-topology | 8257/10316 | 80.040713% | 0.000000pp | 6755/7331 | 92.142955% | 0.000000pp | 0 | 0 | 0 |
| SCOPE-K2 full | 8257/10316 | 80.040713% | 0.000000pp | 6755/7331 | 92.142955% | 0.000000pp | 0 | 0 | 0 |

Parent 每折 correct/rows：fold0 `1657/2064`、fold1 `1665/2063`、fold2 `1666/2063`、fold3 `1643/2063`、fold4 `1626/2063`。SCOPE 五折逐一相同，paired delta 均为 `0`。

所有 learned evidence 方法的 outer-refit beta 在五折均为 `0.0`；没有 fit failure。SCOPE outer-train eligible counts 分别为 `1001, 1037, 1037, 1024, 1017`，故 no-switch 不是空 gate 导致。五份 inner OOF threshold 均为 `no_switch / no_qualified_candidate`：在冻结的 accuracy-changing precision `>=0.60` 与 Wilson lower `>=0.50` 下没有合格切点。

SCOPE macro accuracy `0.8002754393`，McNemar exact p-value `1.0`，runner-up oracle availability `428/10316`。这些诊断不参与阈值选择。

## 八项晋级门

| Gate | 要求 | 实际 | 通过 |
|---|---|---|---|
| Raw delta | `>= +0.20pp` | `+0.000000pp` | 否 |
| Clean-core delta | `>= +0.20pp` | `+0.000000pp` | 否 |
| Net correct | `>= 21` | `0` | 否 |
| Outer stability | `>=4/5` nonnegative | `5/5` | 是 |
| Cluster bootstrap | lower `>0` | `0.0` | 否 |
| Raw strict ablation | SCOPE `>` PACE 和 no-topology | `8257 = 8257 = 8257` | 否 |
| Clean strict ablation | SCOPE `>` PACE 和 no-topology | `6755 = 6755 = 6755` | 否 |
| Audits | 全部通过 | 全部通过 | 是 |

AND 结论：`promoted=false`。因此按预注册规则跳过 final deployment refit、test Pass A/B 与 SCOPE submission。

## 终态 submission

`terminal_submission=parent_fallback`。直接复用归档父字节，没有重生成 CSV 或重打 ZIP：

- CSV：`outputs/delivery/fullft_dual_pa0.9/pred_results.csv`，SHA-256 `790fabcfb57ada355bfdb2f732da5ea1e16d3c505cdbf77c558bccfe7112b16d`。
- ZIP：`outputs/delivery/fullft_dual_pa0.9/submission.zip`，SHA-256 `3d684c07027d905c3edf88e4ce88c3ef9f32a01a6304385600d3d0ced7af5251`。
- Manifest：`outputs/delivery/fullft_dual_pa0.9/manifest.json`，SHA-256 `67f42552e0c39f25f314a716349f4c9984d9980c1eedee9da781cee36da45570`。
- 根九项 checker：**PASS**；24,967 行、覆盖/去重/四位标签/范围/ZIP 单文件全部通过。

rejected-decision 冒烟测试也确认：`infer_scope_submission` 在读取 test cache/checkpoint 和创建 output directory 之前即拒绝，未产生任何 SCOPE test artifact。

## 正式命令

所有命令从 `reproducibility/aegis_f1` 运行，使用项目解释器 `.venv/bin/python`：

```bash
.venv/bin/python -m aegis_clip.cli.cache_scope_parent --config configs/scope_k2_fullft_dual_pa090.yaml --checkpoint artifacts/external/scope_k2/fullft_dual_pa090/best.pt --split validation --output outputs/scope_k2/fullft_dual_pa090/cache/validation_parent_run1.pt --batch-size 128 --num-workers 4 --device cuda
.venv/bin/python -m aegis_clip.cli.cache_scope_evidence --config configs/scope_k2_fullft_dual_pa090.yaml --checkpoint artifacts/external/scope_k2/fullft_dual_pa090/best.pt --parent-cache outputs/scope_k2/fullft_dual_pa090/cache/validation_parent_run1.pt --split validation --output outputs/scope_k2/fullft_dual_pa090/cache/validation_evidence_run1.pt --batch-size 128 --num-workers 4 --device cuda
.venv/bin/python -m aegis_clip.cli.prepare_scope_folds --config configs/scope_k2_fullft_dual_pa090.yaml --parent-cache outputs/scope_k2/fullft_dual_pa090/cache/validation_parent_run1.pt --group-artifact protocol_artifacts/pace_k2_r2_parttoken/content_groups.json --output protocol_artifacts/scope_k2_fullft_dual_pa090/nested_folds.pt
.venv/bin/python -m aegis_clip.cli.evaluate_scope_k2 --config configs/scope_k2_fullft_dual_pa090.yaml --parent-cache outputs/scope_k2/fullft_dual_pa090/cache/validation_parent_run1.pt --evidence-cache outputs/scope_k2/fullft_dual_pa090/cache/validation_evidence_run1.pt --replicate-parent-cache outputs/scope_k2/fullft_dual_pa090/cache/validation_parent_run2.pt --replicate-evidence-cache outputs/scope_k2/fullft_dual_pa090/cache/validation_evidence_run2.pt --fold-artifact protocol_artifacts/scope_k2_fullft_dual_pa090/nested_folds.pt --output-dir outputs/scope_k2/fullft_dual_pa090/evaluation/final
```

父回退 checker 从仓库根运行：

```bash
python3 scripts/check_submission.py --test_dir test --csv outputs/delivery/fullft_dual_pa0.9/pred_results.csv --zip outputs/delivery/fullft_dual_pa0.9/submission.zip
```

机器可读结果位于 `reproducibility/aegis_f1/outputs/scope_k2/fullft_dual_pa090/evaluation/final/`；其中 `decision.json` 明确记录失败门、fallback hashes、`terminal_submission=parent_fallback` 与 checker PASS。
