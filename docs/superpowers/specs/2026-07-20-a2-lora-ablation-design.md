# A2 LoRA Configuration Ablation — Specification Plan

**Date:** 2026-07-20  
**Status:** Draft; do not run until parent/data/code gates pass  
**Scope:** Two controlled experiments requested in the P1-3 image

## 1. Motivation and hypothesis

AEGIS F1 showed that visual LoRA can improve a clean-supervision model, but its configuration is broad: rank 8, Q/V/output adapters, the final four Transformer blocks, clean-probability filtering, GCE, weak augmentation, and feature distillation. The A2 parent is described as having a cleaner training subset and a strong frozen linear-head starting point (61.21% TTA in the supplied brief). The question is therefore causal and narrow:

> Does a rank-4 adapter on only the final visual block already capture the gain, or is the rank-8/four-block AEGIS-style capacity necessary?

The experiment must change adapter capacity only. It must not mix the result with a new split, a new cleaning threshold, a new loss, or a new augmentation policy.

## 2. Parent and data contract

The parent for both children is the exact A2 `best.pt`, not a re-trained approximation. Before implementation, resolve and record:

- A2 experiment ID, source commit, and parent checkpoint SHA-256;
- A2 train/val CSV SHA-256 and class-mapping SHA-256;
- the A2 cleaned-subset/consensus manifest SHA-256 and its coverage/rejection counts;
- the A2 optimizer/loss/augmentation configuration;
- the parent checkpoint's best epoch and frozen-head bare/TTA scores.

The child train and validation files must be byte-identical to A2. A child validation image must not occur in the parent training set. A runtime audit must fail closed on missing/extra paths, label mismatch, duplicate keys, or manifest coverage below 100%.

## 3. Pre-registered experiment matrix

| Field | `A2_LORA_MIN` | `A2_LORA_FULL` |
|---|---|---|
| Parent | A2 `best.pt` | A2 `best.pt` |
| Backbone | official OpenAI CLIP ViT-B/32 | same |
| Classifier | A2 linear head initialized from parent | same parent head |
| LoRA rank | 4 | 8 |
| LoRA alpha | 8 (fixed alpha/r = 2) | 16 (fixed alpha/r = 2) |
| Target visual blocks | block 11 only (the last block; zero-based index 11) | blocks 8, 9, 10, 11 |
| Target attention weights | Q, V, and output projection | Q, V, and output projection |
| LoRA dropout | copy AEGIS F1 value, fixed for both (reference: 0.0 unless recovered A2 protocol says otherwise) | identical |
| Feature distillation | enabled against frozen A2 parent visual features | identical |
| Distillation definition | cosine distance after visual projection; fixed weight copied from AEGIS F1 (reference λ=2.0) | identical |
| Supervised loss | A2 loss/cleaning protocol, unchanged | unchanged |
| Augmentation | A2 augmentation, unchanged | unchanged |
| Split/seed | A2 split and seed, unchanged | unchanged |
| Epoch budget | AEGIS-style 6 epochs unless A2 protocol has a pre-registered alternative | identical |

The exact recovered A2 values override the reference values only if they are documented before either run. No post-hoc tuning is allowed between MIN and FULL.

## 4. Implementation readiness

The current mainline `common/peft.py` supports only `last_block_lora` with `attn.out_proj`. It does not yet express the requested arbitrary block list or Q/V projections, and the mainline training path does not automatically construct the frozen-parent feature-distillation term. Therefore the spec is not executable as-is.

Before training, implement or select one audited runner that provides:

1. `target_blocks: [11]` or `[8, 9, 10, 11]` with bounds checks;
2. Q/V/out wrappers compatible with CLIP's `MultiheadAttention` fast path;
3. freeze audit proving only classifier and LoRA parameters are trainable;
4. parent checkpoint loading before adapter attachment, with epoch-0 equivalence check;
5. frozen parent feature extraction and finite cosine distillation loss;
6. checkpoint state-dict round-trip for every adapter and parent-head tensor;
7. a resolved-config snapshot that records actual adapter parameter count and target names.

The isolated AEGIS F1 runner already demonstrates the Q/V/out, four-block, trust-filter, and distillation concepts, but it must not silently substitute its own AEGIS data or parent. Either adapt it to the exact A2 artifacts or extend the mainline runner with an explicit `visual_lora` mode and tests.

## 5. Validation and selection protocol

### 5.1 Pre-training gates

- Parent checkpoint loads strictly and class mapping matches A2.
- Child epoch-0 logits match the parent on a fixed audit batch within a documented tolerance.
- Train/val/clean-manifest coverage is 100%; no external data or class-name enrichment is used.
- Trainable parameter names and counts match the intended matrix; all other backbone tensors remain bit-identical during the smoke test.
- Distillation target features are produced by a frozen copy of the A2 parent, not by the current student.

### 5.2 Metrics

For each best checkpoint, report:

- raw validation micro/macro/bottom-10 accuracy;
- clean-subset or A2-defined trusted metrics, if part of the parent protocol;
- feature drift from the A2 parent and horizontal-flip agreement;
- prediction-change rate versus the frozen A2 parent;
- best epoch, train/val loss, trainable parameter count, and all artifact hashes;
- bare test submission score and horizontal-flip TTA score separately.

Bare inference is the primary compliance result. TTA uses the same checkpoint with an additional horizontal-flip forward pass and is explicitly marked as a rules-interpretation risk.

### 5.3 Decision gate

Use the bare result as the primary decision:

1. If MIN is at least `+0.30 pp` over A2 bare and FULL is no more than `+0.10 pp` above MIN, choose MIN.
2. If MIN misses `+0.30 pp` but FULL reaches it, choose FULL and record that rank-4/last-block capacity was insufficient.
3. If neither reaches the threshold, close the LoRA branch; do not expand the search space.
4. TTA may be reported and submitted for comparison, but cannot rescue a failed bare gate.

## 6. Reproducibility and outputs

Use isolated output roots:

```text
outputs/a2_lora_min/seed42/
outputs/a2_lora_full/seed42/
```

Each root must contain the resolved config, protocol/lineage audit, train log, best/last checkpoints, re-evaluation metrics, per-class metrics, prediction records, and separate `submissions/` and `submissions_tta/` directories. Run `scripts/check_submission.py` against both ZIPs. Do not commit `.pt` checkpoints under the current repository policy; commit lightweight configs, audits, metrics, and submission artifacts only if explicitly requested later.

## 7. Non-goals

- No rank/alpha/dropout/block grid search beyond these two rows.
- No change to A2 cleaning, consensus threshold, loss, or augmentation.
- No ensemble, soup, voting, external data, or test-time training.
- No platform-based post-hoc checkpoint selection.


## Actual platform outcome (2026-07-22)

- A2_LORA_MIN bare: **61.1167%**.
- A2_LORA_MIN horizontal_flip TTA: **61.6574%**.
- A2_LORA_FULL horizontal_flip TTA: **62.1781%**.
- A2_LORA_FULL bare: **61.5733%**.

See docs/a2_lora_platform_results_2026-07-22.md for absolute submission paths, hashes, and decision notes.
