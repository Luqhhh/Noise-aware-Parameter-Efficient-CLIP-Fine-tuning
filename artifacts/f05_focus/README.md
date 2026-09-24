# F05 Focus Artifacts

Large binary artifacts are intentionally not committed.  This directory is the
fixed location for:

- `hp_noise_manifest.csv` — N1 strict CL+kNN + duplicate-conflict sidecar;
- `clean070.csv` — frozen `clean_probability >= 0.70` subset shared by D1/D2;
- `prior_bias.pt` — validation-fitted uniform-prior additive bias;
- `f05_val_logits.pt` — F05 center validation logits;
- `f05_attention_cache.pt` — A0 native global / attention global / local logits.
