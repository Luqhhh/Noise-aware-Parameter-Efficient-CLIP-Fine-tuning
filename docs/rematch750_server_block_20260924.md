# REMATCH750 V5 / ET / DF — Server Block Status (2026-09-24)

## Summary

- Data cutoff: `2026-09-24T13:21:37Z` (last successful ledger fetch); last container contact `2026-09-24T13:33Z`.
- Blocking server: `vllm-lqh-86`, endpoint `36.140.239.86:32014`.
- Container SSH stopped returning a banner at 13:37 UTC; host port 22 still serves SSH, but our key is not authorized there.
- All experiments that were not already complete by the cutoff are marked `server_blocked`.
- No ET or DF training had started by the cutoff. All ET01–ET10 and DF01–DF03 are `server_blocked`.
- HL00 control completed before the outage: macro `0.7443073987960815`, micro `0.7544354796409607`, selected epoch 14.

## V5 completed full-FT results (28 total completed slots: 16 full-FT + 12 H cached-head)

| trial | macro | micro | epoch | Δmacro vs HL00 | Δmicro vs HL00 |
|---|---:|---:|---:|---:|---:|
| L05 | 0.752391 | 0.762836 | 16 | +0.808pp | +0.840pp |
| R02 | 0.751377 | 0.761559 | 14 | +0.707pp | +0.712pp |
| R04 | 0.750722 | 0.760618 | 16 | +0.642pp | +0.618pp |
| R03 | 0.750705 | 0.761358 | 16 | +0.640pp | +0.692pp |
| K09 | 0.750306 | 0.760215 | 16 | +0.600pp | +0.578pp |
| K02 | 0.747375 | 0.757594 | 14 | +0.307pp | +0.316pp |
| L01 | 0.747109 | 0.757258 | 16 | +0.280pp | +0.282pp |
| L02 | 0.746992 | 0.757124 | 16 | +0.268pp | +0.269pp |
| K03 | 0.746773 | 0.756855 | 14 | +0.247pp | +0.242pp |
| S08 | 0.746583 | 0.756653 | 16 | +0.228pp | +0.222pp |
| R01 | 0.745298 | 0.756183 | 14 | +0.099pp | +0.175pp |
| S01 | 0.744376 | 0.754368 | 16 | +0.007pp | -0.007pp |
| K07 | 0.744346 | 0.754368 | 16 | +0.004pp | -0.007pp |
| S02 | 0.743733 | 0.753562 | 16 | -0.057pp | -0.087pp |
| S09 | 0.740267 | 0.751143 | 16 | -0.404pp | -0.329pp |
| V02 | 0.739720 | 0.749866 | 14 | -0.459pp | -0.457pp |

H cached-head completed: H01–H12; H05 is the fastest probe top at macro `0.746600`.

## V5 server-blocked

All V5 trials not present in `results/rematch750_search_v5/server_block_status_20260924.json` completed list are marked `server_blocked`, including six lanes that were in-flight at the cutoff:
`K04, K08, S03, S05, S06, S07`.

`N01–N04` and `X01–X08` retain their scientific blocked status as well.

## ET / DF

- ET01–ET10: `server_blocked`; no training results.
- DF01–DF03: `server_blocked`; no training results.
- HL00 control: completed; binding in `configs/rematch750_f05_transfer/hl00_control_binding.json`.

