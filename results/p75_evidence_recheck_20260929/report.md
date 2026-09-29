# P75 existing-cache recheck

CPU cached-table statistics only. Original-label agreement is not label truth.

## Signal intersections
Mask order: neighbor >=12/20, centroid OOF, ridge OOF.
Hard: final center p_y <0.3 OR center prediction differs from original label.
| Mask | Rows | Hard | Classes | Structurally eligible | Eligible hard |
|---|---:|---:|---:|---:|---:|
| 000 | 53260 | 13898 | 750 | 50724 | 12072 |
| 001 | 14463 | 178 | 746 | 14142 | 73 |
| 010 | 4659 | 149 | 697 | 4495 | 81 |
| 011 | 29034 | 110 | 742 | 28609 | 45 |
| 100 | 451 | 2 | 155 | 451 | 2 |
| 101 | 1228 | 0 | 335 | 1227 | 0 |
| 110 | 385 | 1 | 149 | 382 | 0 |
| 111 | 30335 | 5 | 631 | 30258 | 3 |

## Signal-only / leave-one-out (diagnostic, not training admission)
| Signal | Rows | Hard | Classes | Eligible classes |
|---|---:|---:|---:|---:|
| neighbor | 32399 | 8 | 648 | 648 |
| centroid | 64413 | 265 | 748 | 748 |
| ridge | 75060 | 293 | 750 | 750 |
| all_three | 30335 | 5 | 631 | 630 |
| without_neighbor | 59369 | 115 | 745 | 744 |
| without_centroid | 31563 | 5 | 647 | 647 |
| without_ridge | 30720 | 6 | 631 | 630 |

## Validation relationships (mutually exclusive)
| Neighbor relationship | Rows | Errors | Prior unknown errors |
|---|---:|---:|---:|
| unique_original_and_model | 7728 | 0 | 0 |
| unique_original_not_model | 241 | 241 | 241 |
| unique_model_not_original | 842 | 842 | 811 |
| unique_neither | 3823 | 1390 | 1388 |
| tied | 2246 | 1001 | 1000 |
| no_votes | 0 | 0 | 0 |

Unique-original plurality below 12 votes, wrong L05, no own-group conflict: 215

## Confusion coverage
```json
{
  "all_directed_pairs": 2976,
  "P0_retained_undirected_pairs": 24,
  "P0_retained_errors": 207,
  "excluded_by_P0_pair_rule": 3267,
  "errors_on_pairs_with_no_reverse": 2779,
  "top_directed_pair_cumulative_errors": {
    "10": 69,
    "50": 233,
    "100": 376,
    "200": 576
  }
}
```

## Limits
- Original labels are noisy; agreement is not clean-label ground truth.
- OOF exists only for training. Validation uses fixed-decode L05 and existing neighbor votes.
- Final-snapshot hard means center p_y < 0.3 OR center top1 != original; it is not historical suppression.
- A vote mask tests >=12 votes; structural eligibility separately requires 20 neighbor groups and no own-group conflict.
- Signal-only/leave-one-out counts diagnose exclusions; they do not relax training admission.
- Class/error overlap and confusion are not causes or attainable improvement bounds.
- Missing support is not opposing evidence and cannot justify revoking class supervision.
- No new supervision table, candidate configuration, or training permission is produced.

Input SHA256: `6c89d11fb29e85c8f4c8270217892246efd81deed45858a310987b41a3e02dcc`
All summary count/hash checks passed.
