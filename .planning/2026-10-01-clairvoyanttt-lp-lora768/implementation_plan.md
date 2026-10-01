# LP_LORA768_20261001 implementation plan

Spec: docs/team_exploration_20261001/machine_a.md; fixed protocol: configs/team_exploration_20261001/machine_a.json. Execution is inline; subagents explicitly prohibited by the task spec. User requests full authorized completion; no repeated generic approval.

- [ ] inputs.py: verify ZIP against reviewed whitelist, unpacked files, original artifact bindings, local dataset and official SHA, every ordered training/validation path and label. Preserve original sidecars; record new location map. Validate frozen confusion groups and tail set against supplied parent predictions. No historical assets.
- [ ] model and replay: reuse v1 classifier with pre_projection; restore all 98 adapter/head parameters before freezing control LoRA. Full val zero-update FP32 replay must match 14,880 original argmax predictions exactly. Cold load exported full adapter/head state also reproduces outputs. Tests use actual handed-over parent and official images.
- [ ] paired runtime: dedicated sampling generator, dataset per-sample epoch RNG, independent mixup generator and isolated model RNG; deterministic batch/population/Mixup hashes compared across arms. Reuse logical_update AMP retry and weighted_mixup_loss semantics. Keep full LoRA in all exports even control frozen weights. AdamW and OneCycle values copied unchanged; epoch 2-4 EMA arithmetic export fixed.
- [ ] CUDA probe: first128 inference, full holdout, five actual logical updates per arm, checkpoint IO and single model load; no wall-clock cutoff. Probe discarded. Verify actual control frozen visual vs actual candidate adapter updates, reliability mass equivalence.
- [ ] four epochs per arm from untouched original parent. Full center metrics each epoch; fixed -2pp stop. Finish cold-load raw/EMA/average validation and paired report with frozen groups, correction/regression counts and ratios. Never select best epoch.
- [ ] six-view fixed no-bias decode for each average checkpoint, 37,444 rows, CSV/ZIP, 9 checks and SHA. Independent recomputation from saved per-image data and package bytes.
- [ ] required regression suite, self-review (spec forbids subagents), report exact commands and outputs; immediate verified branch commit/push; automatic main sync, merge, reverify and push; pause checkpoint.

Review focus: frozen control LoRA omitted by generic trainable-state helpers; incorrect .proj in 768 forward; portable bindings confused with source identity; sample RNG drifting when gradients consume random numbers; AMP skips changing OneCycle/EMA update counts.
