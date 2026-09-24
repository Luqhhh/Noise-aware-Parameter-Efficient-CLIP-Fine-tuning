# REMATCH750 Decay-Filter 2×2 — Server Block Status (2026-09-24)

- Server: `vllm-lqh-86` / `36.140.239.86:32014`
- Last container contact: `2026-09-24T13:33Z`
- Container SSH stopped returning a banner at `13:37Z`; host port 22 SSH is up but our key is not authorized.
- DF training had **not started** by the outage cutoff.
- All `DF01–DF03` are marked `server_blocked`.
- HL00 same-revision control completed before outage:
  - macro `0.7443073987960815`
  - micro `0.7544354796409607`
  - selected epoch `14`
  - selected report SHA-256 `08a996e61508a757670f7da223ade7c09cd2f264ad664d22c82fc7fabbf9dc85`

Machine-readable status:
`results/rematch750_decay_filter/server_block_status_20260924.json`
