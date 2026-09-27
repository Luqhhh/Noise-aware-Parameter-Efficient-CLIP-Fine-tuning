#!/usr/bin/env python3
"""Apply strict local-loss SAM wiring to a private pinned L05 trainer copy."""
from __future__ import annotations

from pathlib import Path

from p75_hard_overlay import once


def transform(source: str) -> str:
    source = once(source,
        '                if attention_local_enabled and epoch >= attention_local_start_epoch:\n',
        '                patch_attention = None\n'
        '                if attention_local_enabled and epoch >= attention_local_start_epoch:\n')
    source = once(source,
        '                            ("attention_local_training", attention_local_enabled),\n',
        '')
    source = once(source,
        '                            "forward_inputs": forward_inputs.detach(),\n',
        '                            "forward_inputs": forward_inputs.detach().cpu(),\n'
        '                            "local_attention": (None if patch_attention is None\n'
        '                                else patch_attention.detach().cpu()),\n'
        '                            "local_crop_size": int(attention_local_config.get("crop_size", 160)),\n'
        '                            "local_top_patches": int(attention_local_config.get("top_patches", 5)),\n'
        '                            "local_weight": float(attention_local_config.get("local_supervision_weight", .5)),\n'
        '                            "local_gate": float(attention_local_confidence_gate),\n')
    for name in ('mixed_reference', 'mixed_targets', 'mixed_weights'):
        source = once(source,
            f'                            "{name}": {name}.detach(),\n',
            f'                            "{name}": {name}.detach().cpu(),\n')
    source = once(source,
        '                            sam_update = _sam_accumulated_optimizer_step(\n',
        '                            from p75_sam import full_l05_sam_step\n'
        '                            sam_update = full_l05_sam_step(\n')
    return source


def apply(path: Path) -> None:
    updated = transform(path.read_text())
    compile(updated, str(path), 'exec')
    path.write_text(updated)
