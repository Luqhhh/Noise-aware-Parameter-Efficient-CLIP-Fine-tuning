"""Narrow integration into the frozen L05 trainer; no inference modifications."""
from p75_hard_overlay import once


def transform(source):
    source=once(source,'    val_dataset = _build_dataset(\n',
        '    from p75_supervision_runtime import EvidenceState, ConfusionSampler\n'
        '    evidence_state = EvidenceState(config, train_dataset.paths, train_dataset.labels)\n'
        '    val_dataset = _build_dataset(\n')
    source=once(source,'    def make_train_loader():\n',
        '    if evidence_state.route == "R2":\n'
        '        train_sampler = ConfusionSampler(evidence_state.rows, evidence_state.edges, seed)\n'
        '    def make_train_loader():\n')
    source=once(source,'    for epoch in range(start_epoch, epochs + 1):\n',
        '    import os\n'
        '    evidence_stop_epoch = int(os.environ.get("P75_STOP_AFTER_EPOCH", epochs))\n'
        '    if evidence_stop_epoch < start_epoch or evidence_stop_epoch > epochs:\n'
        '        raise ValueError("Invalid evidence training epoch boundary")\n'
        '    for epoch in range(start_epoch, evidence_stop_epoch + 1):\n'
        '        evidence_state.begin_epoch(model, epoch)\n')
    source=once(source,'                if cyclic_enabled:\n                    per_sample, cyclic_delta = smoothstep_damped_loss(\n',
        '                per_sample = evidence_state.global_loss(model, arguments,\n'
        '                    training_logits, encoded, batch_indices, per_sample, totals)\n'
        '                if cyclic_enabled:\n                    per_sample, cyclic_delta = smoothstep_damped_loss(\n')
    source=once(source,
        '                    fallback_mask = global_confidence < attention_local_confidence_gate\n'
        '                    local_per_sample = _attention_local_fallback_per_sample(\n'
        '                        local_per_sample,\n                        per_sample,\n'
        '                        global_confidence,\n                        attention_local_confidence_gate,\n'
        '                    )\n',
        '                    local_per_sample, admitted = evidence_state.local_loss(\n'
        '                        local_training_logits, local_per_sample, global_confidence,\n'
        '                        attention_local_confidence_gate, local_inputs, per_sample)\n'
        '                    fallback_mask = ~admitted\n')
    source=once(source,'                fine_active_value = logits.new_zeros(())\n',
        '                loss = loss + evidence_state.auxiliary_loss\n'
        '                fine_active_value = logits.new_zeros(())\n')
    source=once(source,'        train_metrics = {\n',
        '        atomic_json_dump(dict(epoch=epoch, route=evidence_state.route,\n'
        '            **{k:v for k,v in totals.items() if k.startswith("evidence_")}),\n'
        '            log_dir / f"evidence_epoch_{epoch}.json")\n'
        '        train_metrics = {\n')
    compile(source,'p75_supervision_trainer','exec')
    return source
