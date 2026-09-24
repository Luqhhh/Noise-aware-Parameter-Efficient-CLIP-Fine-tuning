# F05 Focus Configs

- `C0_cuda_control.yaml`, `N1_hp_noise_drop.yaml`: ordinary Aegis training configs.
- `A0_m1.yaml`, `P0_prior.yaml`, `D1_o3.yaml`, `D2_pta.yaml`: lightweight,
  machine-readable descriptors for the Round-0 probes and Round-2 adapters.
  They are not consumed by `aegis_clip.config.load_config`.
- `manifest.json`: the seven-unit contract, promotion matrix, artifacts and
  two-GPU queue.

Regenerate and verify with:

```bash
python3 scripts/build_f05_focus_bundle.py
python3 scripts/build_f05_focus_bundle.py --check
```

The protocol document is `docs/f05_focus_protocol_20260924.md`.
