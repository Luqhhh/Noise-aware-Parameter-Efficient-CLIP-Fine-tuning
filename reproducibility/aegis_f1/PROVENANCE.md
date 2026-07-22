# Aegis 独立实验线 provenance

- Source repositories: `/home/x28639/projects/AegisCLIP-Noise-Robust` and `/home/x28639/projects/AegisCLIP-F6-A2LoRA`
- Original snapshot commit: `d542fc6` (`experiment: add noise-aware visual LoRA F1`)
- Incremental source commit: `0e06f0a` (`feat: add cross-fitted trajectory audit`)
- Team integration base: `7c8b966` (`origin/main` at integration start)
- First imported: 2026-07-19
- Incremental integration: 2026-07-22
- Merge mode: three-way additive A/M union under the existing isolated prefix. Team-side A2 STRICT and Phase 4 additions were preserved; only files added or modified by the independent Aegis line since `d542fc6` were imported.
- Portability: machine-specific data roots in committed configurations are kept relative to the team repository where required.

This directory remains intentionally isolated from the legacy team runner. The legacy `ROBUST_LORA` runner updates only the last block's attention output projection, whereas Aegis F1 updates Q/V/output weights in the final four visual blocks and uses a separate clean-core trust bundle plus feature anchoring. Treating the two runners as interchangeable would not reproduce the submitted model.

The integration includes source, configurations, tests, protocols and result metadata. It intentionally excludes datasets, feature caches, checkpoints, prediction CSV files and submission ZIPs.

## Confirmed independent platform results

| Experiment | Inference | Accuracy | Submission ZIP SHA-256 |
|---|---|---:|---|
| D1 | horizontal flip | 59.8500% | `e3ab2e85b37f9dfb34b35521c41410342395d9c7acf8df110606a6ee2689b5a0` |
| E20 | epoch-35/44 head soup + K7 + flip | 60.1794% | `37e524ef7aae81880b825595a26c90d272bb16753eac20222bf08349a5771a97` |
| E21 | full-train continuation + K7 + flip | 60.2195% | `52a4ed874a745c818790de09b1b287c9845c36f13c402523e46e9662ffc97b5c` |
| F1 | bare | 60.5159% | `6c81b7e38d5688cd67c36cb50868c2de507e0fc4fef3b69b9180c65f29f7a363` |
| F1 | horizontal-flip mean probability, T=0.5 | 61.1007% | `5773f52944af998ac349b7091386282484d8c7dcbc8af296461ae1978dd96657` |
| A2 + M1 | center + attention-local, 1:1 probability mean | 62.6747% | `b73eed1f826b37433962cce547cbfa6f15e57afd7d83b3c56557ce2ab399ecbd` |
| A2 + M3 | A2 Flip branch + M1 branch, 1:1 probability mean | 62.0259% | `8f757c6590e9d92ce7655e716d72eb36397d8f302e14c94f691b45e5e184ef4b` |
| F1 + M1 | center + attention-local, 1:1 probability mean | **63.3276%** | `eca9e7c6269c6a4a1cdb213228fa11e881a7ed9795df14da721d6799a1dab63c` |

D1 bare was reported only as lower than D1 Flip; its exact platform score is unavailable and is therefore not fabricated. F2 + M1, O1 + M1, N3 + M1 and N3 + M3 are audited packages awaiting platform evaluation, not confirmed scores.

The authoritative experiment/status index is [`../../docs/aegis_independent_experiments_2026-07-22.md`](../../docs/aegis_independent_experiments_2026-07-22.md); machine-readable results are in [`../../results/aegis_independent_platform_results.csv`](../../results/aegis_independent_platform_results.csv).
