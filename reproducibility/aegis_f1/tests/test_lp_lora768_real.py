"""Actual handed-over checkpoint and images; opt-in to local real-asset checks."""
import importlib
import importlib.util
import os
from pathlib import Path
import numpy as np
import pytest
import torch

LOCATIONS = os.environ.get("LP_LORA768_LOCATIONS")
pytestmark = pytest.mark.skipif(not LOCATIONS, reason="Requires exact handover and official local images")


def implementation():
    assert importlib.util.find_spec("lp_lora768") is not None, "Independent 768 continuation entry point is missing"
    return importlib.import_module("lp_lora768.inputs")


def test_real_parent_zero_update_first128():
    api = implementation()
    context = api.Context(Path(LOCATIONS))
    model = api.initial_model(context, "lora_and_head")
    assert len(model.adapted_modules) == 48
    assert tuple(model.head.weight.shape) == (len(context.classes), 768)
    assert model.visual.proj is None
    predictions, _ = api.center_predictions(context, model, context.val[:128], batch_size=8)
    np.testing.assert_array_equal(predictions, context.frozen["unbalanced768"][:128])
    assert context.inputs_report["positive_weight_rows"] == 119074
    assert context.inputs_report["all_ordered_labels_and_paths_match"]


def test_frozen_control_export_restores_all_lora():
    api = implementation()
    context = api.Context(Path(LOCATIONS))
    control = api.initial_model(context, "head_only")
    complete = api.adapter_head_state(control)
    assert len([k for k in complete if "lora_" in k]) == 96
    assert all(not p.requires_grad for p in control.visual.parameters())
    restored = api.initial_model(context, "lora_and_head")
    api.restore_adapter_head(restored, complete)
    for key, value in api.adapter_head_state(restored).items():
        torch.testing.assert_close(value, complete[key], rtol=0, atol=0)
    assert context.parent["selected_policy"] == "frozen_visual_head_last20"


def test_real_paired_rng_reliability_and_updates():
    api = implementation()
    assert importlib.util.find_spec("lp_lora768.paired") is not None, "Independent paired RNG/update runtime is missing"
    paired = importlib.import_module("lp_lora768.paired")
    context = api.Context(Path(LOCATIONS))
    batches = []
    for perturbation in (0, 997):
        torch.rand(perturbation)
        loader = paired.training_loader(context, epoch=1)
        batches.append(next(iter(loader)))
    torch.testing.assert_close(batches[0][0], batches[1][0], rtol=0, atol=0)
    torch.testing.assert_close(batches[0][1], batches[1][1], rtol=0, atol=0)
    permutation, lam = paired.mix_plan(context, 1, 1, len(batches[0][1]))
    original = context.targets["weights"][batches[0][1]]
    mixed = lam*original+(1-lam)*original[permutation]
    torch.testing.assert_close(mixed.sum(), original.sum(), rtol=1e-6, atol=1e-6)
    states = {}
    for arm in context.protocol["arms"]:
        model = api.initial_model(context, arm)
        before = api.adapter_head_state(model)
        optimizer = paired.optimizer_for(model, context)
        model.train()
        paired.update_once(context, model, optimizer, *batches[0], epoch=1, batch=1)
        after = api.adapter_head_state(model)
        visual_changed = any(not torch.equal(before[k], after[k]) for k in before if "lora_" in k)
        assert visual_changed == (arm == "lora_and_head")
        assert not torch.equal(before["head.weight"], after["head.weight"])
        assert len(after) == len(context.parent["selected_state"])
        states[arm] = after
        del model, optimizer
        torch.cuda.empty_cache()
    assert len(states["head_only"]) == len(states["lora_and_head"])


def test_real_control_cold_checkpoint_keeps_frozen_lora(tmp_path):
    api = implementation()
    assert importlib.util.find_spec("lp_lora768.runtime") is not None, "Cold-checkpoint runtime is missing"
    runtime = importlib.import_module("lp_lora768.runtime")
    context = api.Context(Path(LOCATIONS))
    model = api.initial_model(context, "head_only")
    state = api.adapter_head_state(model)
    checkpoint = tmp_path / "probe.pt"
    runtime.export_checkpoint(context, "head_only", state, checkpoint, epoch=0, policy="probe", updates=0)
    cold, payload = runtime.cold_load(context, "head_only", checkpoint)
    assert payload["complete"] is False
    assert set(payload["selected_state"]) == set(context.parent["selected_state"])
    for key, value in api.adapter_head_state(cold).items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    predictions, _ = api.center_predictions(context, cold, context.val[:128], batch_size=8)
    np.testing.assert_array_equal(predictions, context.frozen["unbalanced768"][:128])


def test_real_recovery_preserves_optimizer_scheduler_scaler_and_ema(tmp_path):
    import importlib.util
    assert importlib.util.find_spec('lp_lora768.recovery') is not None, 'Durable real-state recovery implementation required'
    context=implementation().Context(Path(LOCATIONS))
    from lp_lora768.recovery import save_recovery, load_recovery, restore_batch_prefix
    from lp_lora768.paired import training_loader, optimizer_for, scheduler_for, update_once
    from lp_lora768.inputs import initial_model, adapter_head_state
    from aegis_clip.v1_strategy import WeightAverage
    batches=list(__import__('itertools').islice(training_loader(context,1),2))
    resumed_batch=next(iter(training_loader(context,1,start_batch=1)))
    assert torch.equal(resumed_batch[0],batches[1][0]) and torch.equal(resumed_batch[1],batches[1][1])
    for arm in context.protocol['arms']:
        output=tmp_path/arm;output.mkdir()
        model=initial_model(context,arm).train();optimizer=optimizer_for(model,context)
        scheduler=scheduler_for(optimizer,3722);scaler=torch.amp.GradScaler('cuda')
        live=lambda:{n:p.detach() for n,p in model.named_parameters() if n.startswith('head.') or 'lora_' in n}
        ema=WeightAverage(live(),.999)
        loss,_=update_once(context,model,optimizer,*batches[0],epoch=1,batch=1,scaler=scaler)
        scheduler.step();ema.update(live())
        average=WeightAverage(ema.state);average.update(ema.state)
        import json
        from lp_lora768.paired import batch_identity
        first_record=json.dumps(dict(epoch=1,batch=1,**batch_identity(context,*batches[0],1,1)))+'\n'
        (output/'paired_batches.jsonl').write_text(first_record)
        save_recovery(context,arm,output,model,optimizer,scheduler,scaler,ema,average,
            epoch=1,batch=1,updates=1,history=[],losses=[loss],elapsed_seconds=0.)
        with (output/'paired_batches.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=1,batch=2,**batch_identity(context,*batches[1],1,2)))+'\n')
        expected_loss,_=update_once(context,model,optimizer,*batches[1],epoch=1,batch=2,scaler=scaler)
        scheduler.step();ema.update(live())
        expected=adapter_head_state(model);expected_ema={k:v.cpu().clone() for k,v in ema.state.items()}
        expected_scale=scaler.state_dict();expected_scheduler=scheduler.state_dict()
        del model,optimizer,scheduler,scaler,ema;torch.cuda.empty_cache()
        cold=initial_model(context,arm).train();opt=optimizer_for(cold,context)
        sched=scheduler_for(opt,3722);scale=torch.amp.GradScaler('cuda')
        recovered=load_recovery(context,arm,output,cold,opt,sched,scale)
        assert restore_batch_prefix(output,recovered)==1
        assert (output/'paired_batches.jsonl').read_text()==first_record
        assert len(list((output/'recovery').glob('discarded_batches_*.jsonl')))==1
        assert recovered['updates']==1 and recovered['batch']==1 and recovered['epoch']==1
        assert recovered['average'].count==1 and recovered['ema'].count==1
        actual_loss,_=update_once(context,cold,opt,*batches[1],epoch=1,batch=2,scaler=scale)
        sched.step();recovered['ema'].update({n:p.detach() for n,p in cold.named_parameters() if n.startswith('head.') or 'lora_' in n})
        assert actual_loss==expected_loss
        assert scale.state_dict()==expected_scale and sched.state_dict()==expected_scheduler
        assert all(torch.equal(v,adapter_head_state(cold)[k]) for k,v in expected.items())
        assert all(torch.equal(v,recovered['ema'].state[k].cpu()) for k,v in expected_ema.items())
        del cold,opt,sched,scale,recovered;torch.cuda.empty_cache()