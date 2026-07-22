# Progress — A2 LoRA Ablation

## 2026-07-20

- Read the file-based planning instructions and restored existing project context.
- Inspected the supplied image requirements and existing A2, AEGIS F1, PEFT, LoRA, and feature-distillation assets.
- Attempted latest-main sync with Git automatic stash mode (`git pull --rebase --autostash origin main`); it failed on WSL/GitHub TLS.
- Retried with HTTP/1.1; port 443 connection timed out.
- Drafted `docs/superpowers/specs/2026-07-20-a2-lora-ablation-design.md`.
- Drafted this isolated plan under `.planning/2026-07-20-a2-lora-ablation/`.
- Training intentionally not started.


## Actual platform outcome (2026-07-22)

- A2_LORA_MIN bare: **61.1167%**.
- A2_LORA_MIN horizontal_flip TTA: **61.6574%**.
- A2_LORA_FULL horizontal_flip TTA: **62.1781%**.
- A2_LORA_FULL bare: **61.5733%**.

See docs/a2_lora_platform_results_2026-07-22.md for absolute submission paths, hashes, and decision notes.
