# SCOPE-K2 conditional grouped nested OOF report

> Conditional grouped nested OOF given the frozen FULLFT_DUAL parent. The parent was trained with validation overlap; these folds isolate only beta, threshold, duplicate groups, ablations, and the promotion decision.

| Method | Raw correct | Raw accuracy | Delta (pp) | Clean correct | Clean accuracy | Clean delta (pp) | Switches | Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parent | 8257 / 10316 | 80.040713% | +0.000000 | 6755 / 7331 | 92.142955% | +0.000000 | 0 | +0 |
| margin_only | 8257 / 10316 | 80.040713% | +0.000000 | 6755 / 7331 | 92.142955% | +0.000000 | 0 | +0 |
| pace | 8257 / 10316 | 80.040713% | +0.000000 | 6755 / 7331 | 92.142955% | +0.000000 | 0 | +0 |
| gate_only | 8257 / 10316 | 80.040713% | +0.000000 | 6755 / 7331 | 92.142955% | +0.000000 | 0 | +0 |
| no_topology | 8257 / 10316 | 80.040713% | +0.000000 | 6755 / 7331 | 92.142955% | +0.000000 | 0 | +0 |
| scope | 8257 / 10316 | 80.040713% | +0.000000 | 6755 / 7331 | 92.142955% | +0.000000 | 0 | +0 |

## Paired cluster bootstrap

SCOPE minus Parent: point=+0.000000pp, 95% CI=[+0.000000, +0.000000]pp, draws=10000, seed=42.

## Promotion gates

- [ ] `raw_delta_at_least_0_20pp`
- [ ] `clean_delta_at_least_0_20pp`
- [ ] `net_correct_at_least_21`
- [x] `four_of_five_outer_nonnegative`
- [ ] `bootstrap_lower_strictly_positive`
- [ ] `strictly_better_raw_than_pace_and_no_topology`
- [ ] `strictly_better_clean_than_pace_and_no_topology`
- [x] `all_audits_passed`

Final decision: **REJECT / EXACT PARENT FALLBACK**.
