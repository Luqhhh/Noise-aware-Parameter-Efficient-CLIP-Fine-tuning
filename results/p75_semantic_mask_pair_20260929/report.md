# P75 semantic masking pair

Original-label accuracy; selected and remaining groups are content proxies, not clean truth.

| Epoch | Proxy | Rows | Control macro | Masked macro | Control micro | Masked micro | Fixed | Damaged | Net |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | all | 14880 | 72.941563% | 72.941270% | 73.696237% | 73.696237% | 24 | 24 | 0 |
| 4 | selected_proxy | 132 | 17.619048% | 10.000000% | 20.454545% | 14.393939% | 2 | 10 | -8 |
| 4 | remaining_proxy | 14748 | 73.330574% | 73.388627% | 74.172769% | 74.227014% | 22 | 14 | 8 |

Slice macro averages only classes present in that slice.

Content groups are automatic proxies, not clean labels.
Same used validation split, not an independent test.
Shared RM-LP parent head already saw original supervision.
No automatic full-training or platform promotion.
