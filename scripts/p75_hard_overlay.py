#!/usr/bin/env python3
"""Apply a fail-closed Hard Support adapter to a pinned, private L05 source copy."""
from __future__ import annotations

from pathlib import Path


def once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f'Expected one Hard Support insertion point, found {count}: {old[:70]!r}')
    return source.replace(old, new, 1)


def transform(source: str) -> str:
    source = once(source,
        '    val_dataset = _build_dataset(\n',
        '    from p75_hard_support import load_support, restore_original_label_loss, local_admission\n'
        '    hard_support = load_support(config, train_dataset.paths, train_dataset.labels)\n'
        '    if hard_support is not None:\n'
        '        hard_support = hard_support.to(device)\n'
        '    val_dataset = _build_dataset(\n')
    source = once(source,
        '            batch_suspicious = (\n',
        '            batch_support = hard_support[batch_indices] if hard_support is not None else None\n'
        '            batch_suspicious = (\n')
    source = once(source,
        '                if cyclic_enabled:\n                    per_sample, cyclic_delta = smoothstep_damped_loss(\n',
        '                per_sample = restore_original_label_loss(\n'
        '                    per_sample, training_logits, mixed_targets, batch_support)\n'
        '                if cyclic_enabled:\n                    per_sample, cyclic_delta = smoothstep_damped_loss(\n')
    source = once(source,
        '                    global_confidence = F.softmax(\n',
        '                    local_per_sample = restore_original_label_loss(\n'
        '                        local_per_sample, local_training_logits, targets, batch_support)\n'
        '                    global_confidence = F.softmax(\n')
    source = once(source,
        '                    fallback_mask = global_confidence < attention_local_confidence_gate\n'
        '                    local_per_sample = _attention_local_fallback_per_sample(\n'
        '                        local_per_sample,\n'
        '                        per_sample,\n'
        '                        global_confidence,\n'
        '                        attention_local_confidence_gate,\n'
        '                    )\n',
        '                    admitted = local_admission(\n'
        '                        global_confidence, attention_local_confidence_gate,\n'
        '                        batch_support, local_inputs)\n'
        '                    fallback_mask = ~admitted\n'
        '                    local_per_sample = torch.where(admitted, local_per_sample, per_sample)\n')
    return source


def apply(path: Path) -> None:
    source = path.read_text()
    updated = transform(source)
    compile(updated, str(path), 'exec')
    path.write_text(updated)
