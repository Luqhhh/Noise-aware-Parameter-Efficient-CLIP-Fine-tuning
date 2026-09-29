"""Insertion points in pinned L05 source. Does not modify sampler, anchor or decoder."""
from p75_hard_overlay import once


def transform(source):
    source=once(source,'    val_dataset = _build_dataset(\n',
        '    from p75_text_page_runtime import TextPageState\n'
        '    text_page_state = TextPageState(config, train_dataset.paths, train_dataset.labels, device)\n'
        '    val_dataset = _build_dataset(\n')
    source=once(source,'    for epoch in range(start_epoch, epochs + 1):\n',
        '    for epoch in range(start_epoch, min(epochs, 6) + 1):\n')
    source=once(source,'                if cyclic_enabled:\n                    per_sample, cyclic_delta = smoothstep_damped_loss(\n',
        '                per_sample = text_page_state.global_loss(\n'
        '                    per_sample, training_logits, mixed_targets, batch_indices, totals)\n'
        '                if cyclic_enabled:\n                    per_sample, cyclic_delta = smoothstep_damped_loss(\n')
    # Mask local before gate/fallback; global fallback is already masked too.
    source=once(source,'                    global_confidence = F.softmax(\n',
        '                    local_per_sample = text_page_state.local_loss(local_per_sample)\n'
        '                    global_confidence = F.softmax(\n')
    source=once(source,'        train_metrics = {\n',
        '        atomic_json_dump(dict(epoch=epoch, enabled=text_page_state.enabled,\n'
        '            loss_phase="CE" if epoch <= 2 else "GCE",\n'
        '            **{k:v for k,v in totals.items() if k.startswith("text_page_")}),\n'
        '            log_dir / f"text_page_epoch_{epoch}.json")\n'
        '        train_metrics = {\n')
    compile(source,'text_page_trainer','exec')
    return source
