# P75_TEXT_PAGE_MASK_V1 fixed detector check

Gate: `failed_empty_selection`. Training started: false.

| Group | Rows | Original-label errors | Mean sqrt(py) | Final logit L1 share |
|---|---:|---:|---:|---:|
| train_not_detected_proxy | 133815 | 13024 | 0.8699578194861086 | 1.0 |
| train_detected_text_proxy | 0 | 0 | None | 0.0 |
| val_not_detected_proxy | 14880 | 3474 | 0.7476279817602903 | 1.0 |
| val_detected_text_proxy | 0 | 0 | None | 0.0 |

No candidate outcome or platform gain was measured.

A zero selection invalidates this detector as an intervention entry; it does not disprove contamination or its possible training harm.

- Detected text proxy is not confirmed open-set truth.
- Not-detected complement is mixed and is not a clean/effective-target validation set.
- No semantic mapping and no replacement labels. Numerical classes are unchanged.
- Final center-view snapshots cannot reconstruct training history or parameter-gradient harm.
- 2*sqrt(py)*(1-py) is the per-example GCE logit-gradient L1 at q=.5, not parameter-gradient norm.
- Only a whole fixed detector may pass/fail review; no per-image manual whitelist/blacklist.
