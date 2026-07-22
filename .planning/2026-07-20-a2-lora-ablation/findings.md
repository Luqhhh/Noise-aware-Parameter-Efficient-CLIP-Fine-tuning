# Findings — A2 LoRA Ablation

## User-provided requirements

- `A2_LORA_MIN`: parent A2 `best.pt`; rank 4; last visual block only (block 11); Q/V/output attention adapters; feature distillation enabled.
- `A2_LORA_FULL`: same parent; rank 8; Q/V/output adapters on visual blocks 8–11; feature distillation enabled.
- The purpose is to test whether the minimal adapter saturates the gain before paying for the AEGIS F1-sized capacity.
- The supplied decision threshold is `+0.30 pp` bare over A2 for the minimal configuration.

## Repository facts

- `common/peft.py` currently exposes only `last_block_lora` and wraps `attn.out_proj` on one block.
- `common/feature_distillation.py` provides frozen-parent cosine feature distillation, but the mainline training path is not automatically wired to it.
- The isolated AEGIS F1 runner documents Q/V/out adapters on blocks 8–11, GCE q=0.5, six epochs, and feature-distillation weight 2.0. Its data and parent must not be silently substituted for A2.
- Existing Phase 3 planning already contains a conditional last-block LoRA idea; this new spec is narrower and explicitly compares a minimal row against a four-block upper-bound row.

## Sync blocker

- `git pull --rebase --autostash origin main` failed with `GnuTLS recv error (-110)`.
- A retry using HTTP/1.1 and low-speed limits timed out after 142 seconds connecting to GitHub port 443.
- No training or source changes were made while the pull was blocked.


## Actual platform outcome (2026-07-22)

- A2_LORA_MIN bare: **61.1167%**.
- A2_LORA_MIN horizontal_flip TTA: **61.6574%**.
- A2_LORA_FULL horizontal_flip TTA: **62.1781%**.
- A2_LORA_FULL bare: **61.5733%**.

See docs/a2_lora_platform_results_2026-07-22.md for absolute submission paths, hashes, and decision notes.
