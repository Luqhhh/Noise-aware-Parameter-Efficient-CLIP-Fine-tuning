# A2_AEGIS_PARENT_SWAP Protocol Repair and Revalidation Implementation Plan

> **Goal:** Fix A2 checkpoint → AEGIS Visual LoRA parent-child split lineage protocol,
> establish epoch-0 legal parent baseline, resubmit only if strict local gate passes.

## Root Cause

The original `F1_VISUAL_LORA_CLEAN_CORE_A2_PARENT` used **different train/val splits** for:
- **Parent (A2 `NR_CL_KNN_DROP`)**: `outputs/data/d3_strict/seed42/{train,val}.csv` — 91,195 train / 10,322 val
- **Child (AEGIS LoRA)**: `artifacts/stages/preliminary/seed42/{train,val}.csv` — 92,902 train / 10,316 val

This means A2 parent training saw samples that later appeared in the AEGIS child val set,
artificially inflating the local evaluation to **79.22% raw_micro** (+8.5pp over the legal ~70% range).
The platform score (60.29% bare, 60.87% TTA) is a valid observation but the causal conclusion
that "stronger parent improves LoRA" is **invalid**.

## Architecture

- Preserve AEGIS independent runner — do not replace with main repo PEFT.
- Add fail-closed lineage audit that unifies `train/...` and `train_dedup/...` canonical paths.
- Strict rerun uses A2's exact `d3_strict/seed42` train/val split.
- After loading parent checkpoint, perform **epoch-0 evaluation** as the legal baseline.
- LoRA epochs must significantly exceed epoch 0 to promote.

## Tasks

See the main implementation plan in the agent conversation context.
Each task is executed via the AEGIS `.venv` (system Python 3.10 fallback if uv unavailable).
