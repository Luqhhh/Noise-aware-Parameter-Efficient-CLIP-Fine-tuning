"""Private frozen trainer extension; shared source and old E4 archive stay untouched."""
from p75_hard_overlay import once
from p75_text_page_overlay import transform as masking


def transform(source):
    source=masking(source)
    source=once(source,'    for epoch in range(start_epoch, min(epochs, 6) + 1):\n',
        '    import os, hashlib, time\n'
        '    pipeline_stop = int(os.environ["P75_PIPELINE_STOP"])\n'
        '    pipeline_deadline = float(os.environ["P75_PIPELINE_DEADLINE"])\n'
        '    if pipeline_stop not in (5, 6) or start_epoch != pipeline_stop:\n'
        '        raise ValueError("Extension must resume E4 to E5 or E5 to E6")\n'
        '    for epoch in range(start_epoch, pipeline_stop + 1):\n')
    source=once(source,'        for batch_index, batch in enumerate(train_loader):\n',
        '        pipeline_order = hashlib.sha256()\n'
        '        for batch_index, batch in enumerate(train_loader):\n'
        '            if time.time() >= pipeline_deadline:\n'
        '                # Partial epoch state is diagnostic only: never advertised as resumable.\n'
        '                save_checkpoint(checkpoint_dir / "incomplete.pt", model=model, optimizer=optimizer,\n'
        '                    scheduler=scheduler, scaler=scaler, epoch=epoch-1, global_step=global_step,\n'
        '                    best_selector=best_selector, config=config,\n'
        '                    metrics={"incomplete_epoch": epoch, "batch_index": batch_index, "resumable": False},\n'
        '                    adaptive_cap_state=None, data_generator_state=generator.get_state())\n'
        '                raise TimeoutError("New extension budget expired; partial state is not an epoch boundary")\n'
        '            pipeline_order.update(batch["index"].numpy().astype("<i8").tobytes())\n')
    source=once(source,'        train_metrics = {\n',
        '        atomic_json_dump(dict(epoch=epoch, order_sha256=pipeline_order.hexdigest(),\n'
        '            rows=int(totals["examples"]), successful_updates=successful_updates,\n'
        '            attempted_updates=global_step, lr=[g["lr"] for g in optimizer.param_groups]),\n'
        '            log_dir / f"pair_order_{epoch}.json")\n'
        '        train_metrics = {\n')
    source=once(source,'                    local_per_sample = text_page_state.local_loss(local_per_sample)\n',
        '                    for group, mask in (("selected", text_page_state.batch_selected), ("remaining", ~text_page_state.batch_selected)):\n'
        '                        key = "text_page_local_" + group + "_rows"\n'
        '                        totals[key] = totals.get(key, 0) + int(mask.sum())\n'
        '                    local_per_sample = text_page_state.local_loss(local_per_sample)\n')
    # Save the full endpoint without running either center inference or the trainer's final-best inference.
    # The orchestrator evaluates each E6 once through the existing paired TTA cache + prior routine.
    marker='        interval = int(evaluation_config.get("interval_epochs", 1))\n'
    source=once(source,marker,
        '        atomic_json_dump(train_metrics, log_dir / f"train_epoch_{epoch}.json")\n'
        '        endpoint = dict(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,\n'
        '            epoch=epoch, global_step=global_step, best_selector=best_selector, config=config,\n'
        '            metrics=train_metrics, adaptive_cap_state=None, data_generator_state=generator.get_state())\n'
        '        save_checkpoint(checkpoint_dir / "last.pt", **endpoint)\n'
        '        save_checkpoint(checkpoint_dir / f"epoch_{epoch}.pt", **endpoint)\n'
        '        return checkpoint_dir / "last.pt"\n'+marker)
    compile(source,'pipeline_private_trainer','exec')
    return source
