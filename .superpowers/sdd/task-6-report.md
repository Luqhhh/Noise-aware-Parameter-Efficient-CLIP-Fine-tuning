# Task 6: `tools/audit_experiment_pair.py` — Completed

## What was done

Created `/home/lux1/noise/tools/audit_experiment_pair.py`, a CLI tool wrapping `common.pair_protocol_audit.audit_experiment_pair`.

### CLI arguments
- `--reference-config`, `--candidate-config`, `--reference-ckpt`, `--candidate-ckpt`, `--output` (all required)
- `--allow-confounded-analysis` (flag)

### Exit codes
| Code | Condition |
|------|-----------|
| 0    | Paired audit passed (no warnings, paired_valid=true) |
| 2    | Warnings present, OR `--allow-confounded-analysis` with paired_valid=false |
| 3    | paired_valid=false and `--allow-confounded-analysis` not set |
| 4    | Missing input file, or audit raised an exception |

### Verification
- Import check passed: `python3 -c "from tools.audit_experiment_pair import main; print('OK')"`
- Files committed on branch `main` (commit 8fec432)
