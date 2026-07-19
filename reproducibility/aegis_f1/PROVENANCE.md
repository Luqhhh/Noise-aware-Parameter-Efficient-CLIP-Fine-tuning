# Aegis F1 provenance

- Source repository: `/home/x28639/projects/AegisCLIP-Noise-Robust`
- Source commit: `d542fc6` (`experiment: add noise-aware visual LoRA F1`)
- Imported: 2026-07-19
- Import mode: exact Git tree snapshot under an isolated prefix
- Portability change after import: F1 `train_root` and `test_root` were changed from machine-specific absolute paths to paths relative to the team repository root.

This directory is intentionally isolated from the legacy team runner. The team runner's `ROBUST_LORA` updates only the last block's attention output projection, whereas Aegis F1 updates Q/V/output weights in the final four blocks and uses a separate clean-core trust bundle plus feature anchoring. Treating the two runners as interchangeable would not reproduce the submitted model.

Platform results:

- Bare: 60.5159%, ZIP SHA-256 `6c81b7e38d5688cd67c36cb50868c2de507e0fc4fef3b69b9180c65f29f7a363`
- Horizontal-flip mean-probability TTA, T=0.5: 61.1007%, ZIP SHA-256 `5773f52944af998ac349b7091386282484d8c7dcbc8af296461ae1978dd96657`

