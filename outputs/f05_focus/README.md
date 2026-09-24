# F05 Focus Outputs

Run directories are created by the focus runners and follow this layout:

```text
outputs/f05_focus/
  C0_F05_CUDA/seed42/
  N1_HP_DROP/seed42/
  A0_M1/
  P0_PRIOR/
  D1_O3/
  D2_PTA/
  D3_DUAL/
```

Large `*.pt` files are ignored by git; JSON/CSV/log evidence is retained.
The canonical result table is `results/f05_focus_summary.csv`.
