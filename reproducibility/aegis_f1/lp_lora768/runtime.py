"""Fixed local two-arm four-epoch runtime; never select a best checkpoint."""
from __future__ import annotations
import contextlib
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.v1_pipeline import Images, image_transform, load_artifact, save_artifact, read_rows
from aegis_clip.v1_strategy import WeightAverage
from v1_continuation.runtime import require_idle_cuda
from .inputs import ROOT, Context, initial_model, adapter_head_state, restore_adapter_head, center_predictions, read_json
from .paired import training_loader, optimizer_for, scheduler_for, update_once, batch_identity, check_real_mixup_loss
from .metrics import metrics, grouped_pair


def export_checkpoint(context, arm, state, path, *, epoch, policy, updates):
    if set(state) != set(context.parent["selected_state"]):
        raise ValueError("Export omitted original adapters or head")
    expected = 4*math.ceil(len(context.active)/32)
    if epoch == 4 and updates != expected:
        raise ValueError("Completed checkpoint requires every fixed logical update")
    save_artifact(path, dict(selected_state={k: v.detach().cpu().clone() for k, v in state.items()},
        classes=context.classes, model_config=context.parent["model_config"], arm=arm, epoch=epoch,
        selected_policy=policy, optimizer_updates=updates, complete=epoch == 4, bias=None,
        original_parent_binding=context.parent["binding"], fixed_protocol=context.protocol), context.binding)


def cold_load(context, arm, path):
    payload = load_artifact(path, context.binding)
    if payload["arm"] != arm or payload["classes"] != context.classes:
        raise ValueError("Cold load arm or class mismatch")
    model = initial_model(context, arm)
    restore_adapter_head(model, payload["selected_state"])
    return model.eval(), payload


@contextlib.contextmanager
def using_complete_state(model, state):
    original = adapter_head_state(model)
    mode = model.training
    restore_adapter_head(model, state)
    model.eval()
    try:
        yield model
    finally:
        restore_adapter_head(model, original)
        model.train(mode)


def visual_hash(model, *, frozen_only=False):
    digest = hashlib.sha256()
    for name, p in model.visual.named_parameters():
        if frozen_only and p.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def probe(context, output):
    output = Path(output)/"probe"
    output.mkdir(exist_ok=False)
    report = {}
    identities = None
    for arm in context.protocol["arms"]:
        model = initial_model(context, arm)
        _, initial_cost = center_predictions(context, model, context.val[:128], output=output/f"{arm}_initial128.npz")
        before = adapter_head_state(model)
        frozen_before = visual_hash(model, frozen_only=True)
        optimizer = optimizer_for(model, context)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        loader = training_loader(context, 1)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        current_ids, retries = [], []
        samples = None
        for batch, (images, indices) in enumerate(itertools.islice(loader, 5), 1):
            samples = images[:8].to(context.device)
            if batch == 1: loss_equivalence = check_real_mixup_loss(context, model, images, indices, 1, batch)
            current_ids.append(batch_identity(context, images, indices, 1, batch))
            loss, numeric = update_once(context, model, optimizer, images, indices, epoch=1, batch=batch, scaler=scaler)
            retries.append(dict(batch=batch, loss=loss, **numeric))
        torch.cuda.synchronize()
        update_seconds = time.monotonic()-started
        peak = torch.cuda.max_memory_allocated()
        state = adapter_head_state(model)
        changed = [k for k in state if not torch.equal(state[k], before[k])]
        if visual_hash(model, frozen_only=True) != frozen_before:
            raise ValueError("A frozen visual parameter changed")
        if arm == "head_only" and any("lora_" in k for k in changed):
            raise ValueError("Control LoRA changed")
        if arm == "lora_and_head" and not any("lora_" in k for k in changed):
            raise ValueError("Candidate LoRA never updated")
        if not any(k.startswith("head.") for k in changed):
            raise ValueError("Head never updated")
        if identities is None:
            identities = current_ids
        elif current_ids != identities:
            raise ValueError("Probe population/augmentation/Mixup/reliability differs")
        io_start = time.monotonic()
        checkpoint = output/f"{arm}.pt"
        export_checkpoint(context, arm, state, checkpoint, epoch=0, policy="probe_discarded", updates=5)
        model.eval()
        with torch.no_grad():
            expected = model(samples).cpu()
        del model, optimizer
        torch.cuda.empty_cache()
        cold, payload = cold_load(context, arm, checkpoint)
        with torch.no_grad():
            actual = cold(samples).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        io_seconds = time.monotonic()-io_start
        report[arm] = dict(initial128_cuda_cost=initial_cost, logical_updates=5, logical_batch=32, micro_batch=8, updates_seconds=update_seconds,
            cuda_peak_bytes=peak, checkpoint_and_cold_load_seconds=io_seconds, changed_parameters=changed,
            frozen_visual_unchanged=True, complete_export=True, cold_logits_exact=True,
            estimated_four_epoch_update_seconds=update_seconds/5*4*math.ceil(len(context.active)/32),
            batches=current_ids, numeric_updates=retries, real_mixup_loss_equivalence=loss_equivalence)
        del cold
        torch.cuda.empty_cache()
    holdout_cost = read_json(output.parent/"parent_replay.json")["cost"]
    report["full_holdout_measured_cost"] = holdout_cost
    report["estimated_two_six_view_submission_seconds"] = holdout_cost["seconds"]/holdout_cost["rows"]*37444*6*2
    report.update(status="passed", discarded_for_training=True, paired_batches_exact=True, wall_clock_cutoff=None)
    atomic_json_dump(report, output/"report.json")
    print(json.dumps(dict(phase="probe_passed", costs={arm:{k:v for k,v in report[arm].items() if k in ("updates_seconds","cuda_peak_bytes","estimated_four_epoch_update_seconds")} for arm in context.protocol["arms"]})), flush=True)
    return report


def fit_arm(context, arm, root, baseline, resume=False):
    from .recovery import save_recovery,load_recovery,restore_batch_prefix
    output=root/arm
    if resume and (output/'training_report.json').exists():
        result=read_json(output/'training_report.json')
        if result['status']!='four_epochs_complete' or result['optimizer_updates']!=14888:
            raise ValueError('Incomplete training report cannot be reused')
        load_artifact(output/'ema_swa_2_4.pt',context.binding)
        return result
    existed=output.exists()
    output.mkdir(exist_ok=resume)
    model=initial_model(context,arm)
    frozen_before=visual_hash(model,frozen_only=True)
    initial=adapter_head_state(model)
    optimizer=optimizer_for(model,context)
    steps=math.ceil(len(context.active)/32)
    scheduler=scheduler_for(optimizer,steps)
    scaler=torch.amp.GradScaler('cuda',enabled=True)
    live=lambda:{n:p.detach() for n,p in model.named_parameters() if n.startswith('head.') or 'lora_' in n}
    ema=WeightAverage(live(),.999);average=None
    labels=context.frozen['labels']
    history,updates=[],0
    started=time.monotonic();last_print=started;elapsed_base=0.
    cursor_epoch,cursor_batch,cursor_losses=1,0,[]
    actual_resume=False
    if resume and existed and (output/'recovery/latest.json').exists():
        state=load_recovery(context,arm,output,model,optimizer,scheduler,scaler)
        cursor_epoch,cursor_batch=state['epoch'],state['batch']
        updates,history,cursor_losses=state['updates'],state['history'],state['losses']
        ema,average=state['ema'],state['average'];elapsed_base=state['elapsed_seconds']
        if updates!=(cursor_epoch-1)*steps+cursor_batch:
            raise ValueError('Recovery cursor does not match successful optimizer updates')
        discarded=restore_batch_prefix(output,state)
        actual_resume=True
        with (output/'recovery/events.jsonl').open('a') as f:
            f.write(json.dumps(dict(updates=updates,epoch=cursor_epoch,batch=cursor_batch,discarded_attempted_batches=discarded))+'\n')
    elif resume and existed and any(output.iterdir()):
        raise ValueError('No complete recovery state; use a fresh original-parent run')
    else:
        save_recovery(context,arm,output,model,optimizer,scheduler,scaler,ema,average,
            epoch=1,batch=0,updates=0,history=[],losses=[],elapsed_seconds=0.)
    with (output/'paired_batches.jsonl').open('a' if actual_resume else 'x') as records:
        for epoch in range(cursor_epoch,5):
            offset=cursor_batch if epoch==cursor_epoch else 0
            loader=training_loader(context,epoch,start_batch=offset)
            model.train()
            losses=list(cursor_losses) if epoch==cursor_epoch else []
            for batch,(images,indices) in enumerate(loader,offset+1):
                identity=batch_identity(context,images,indices,epoch,batch)
                records.write(json.dumps(dict(epoch=epoch,batch=batch,**identity))+'\n')
                loss,numeric=update_once(context,model,optimizer,images,indices,epoch=epoch,batch=batch,scaler=scaler)
                losses.append(loss);scheduler.step();ema.update(live());updates+=1
                if numeric['amp_retries']:
                    with (output/'amp_retries.jsonl').open('a') as f:
                        f.write(json.dumps(dict(epoch=epoch,batch=batch,**numeric))+'\n')
                if updates%1000==0:
                    records.flush()
                    save_recovery(context,arm,output,model,optimizer,scheduler,scaler,ema,average,
                        epoch=epoch,batch=batch,updates=updates,history=history,losses=losses,
                        elapsed_seconds=elapsed_base+time.monotonic()-started)
                now=time.monotonic()
                if batch==offset+1 or batch==steps or now-last_print>=30:
                    status=dict(phase='training',arm=arm,epoch=epoch,epochs=4,batch=batch,batches=steps,
                        updates=updates,loss=loss,seconds=elapsed_base+now-started,cuda_peak_bytes=torch.cuda.max_memory_allocated())
                    print(json.dumps(status),flush=True);atomic_json_dump(status,root/'progress.json');last_print=now
            records.flush()
            if len(losses)!=steps:raise ValueError('Recovery omitted fixed logical updates')
            if visual_hash(model,frozen_only=True)!=frozen_before:raise ValueError('Frozen visual weights changed')
            if epoch>=2:
                if average is None:average=WeightAverage(ema.state)
                average.update(ema.state)
            with using_complete_state(model,ema.state):
                prediction,cost=center_predictions(context,model,context.val,output=output/f'val_epoch{epoch}_ema.npz')
            score=metrics(labels,prediction,len(context.classes))
            history.append(dict(epoch=epoch,updates=updates,loss=float(np.mean(losses)),ema=score,validation_cost=cost))
            atomic_json_dump(history,output/'history.json')
            export_checkpoint(context,arm,ema.state,output/f'ema_epoch{epoch}.pt',epoch=epoch,policy=f'ema_epoch{epoch}',updates=updates)
            if score['micro']-float(np.mean(baseline==labels))<-.02:raise ValueError('Frozen -2pp quality stop triggered')
            save_recovery(context,arm,output,model,optimizer,scheduler,scaler,ema,average,
                epoch=epoch+1,batch=0,updates=updates,history=history,losses=[],
                elapsed_seconds=elapsed_base+time.monotonic()-started)
    if updates!=4*steps or average.count!=3:raise ValueError('Four-epoch/EMA2-4 trajectory incomplete')
    state=adapter_head_state(model)
    if arm=='head_only' and any(not torch.equal(initial[k],state[k]) for k in initial if 'lora_' in k):
        raise ValueError('Formal control LoRA changed')
    for policy,values in [('last_raw',state),('last_ema',ema.state),('ema_swa_2_4',average.state)]:
        export_checkpoint(context,arm,values,output/f'{policy}.pt',epoch=4,policy=policy,updates=updates)
    del model,optimizer,ema,average
    torch.cuda.empty_cache();evaluations={}
    for policy in ('last_raw','last_ema','ema_swa_2_4'):
        cold,payload=cold_load(context,arm,output/f'{policy}.pt')
        pred,cost=center_predictions(context,cold,context.val,output=output/f'val_{policy}.npz')
        evaluations[policy]=dict(metrics=metrics(labels,pred,len(context.classes)),paired_vs_local_parent=grouped_pair(context,labels,baseline,pred),cost=cost)
        del cold;torch.cuda.empty_cache()
    result=dict(status='four_epochs_complete',arm=arm,epochs=4,optimizer_updates=updates,history=history,
        evaluations=evaluations,seconds=elapsed_base+time.monotonic()-started,primary='ema_swa_2_4',
        complete_state_recovery=actual_resume,recovery_every_updates=1000,platform_score=None)
    atomic_json_dump(result,output/'training_report.json')
    return result


@torch.no_grad()
def infer_arm(context, arm, root):
    from aegis_clip.submission import create_submission
    checkpoint=root/arm/"ema_swa_2_4.pt"
    model,payload=cold_load(context,arm,checkpoint)
    if not payload["complete"] or payload["selected_policy"]!="ema_swa_2_4" or payload["bias"] is not None:
        raise ValueError("Inference requires complete fixed average and no bias")
    rows=read_rows(context.base/"test_manifest.csv")
    names=[Path(r["image_path"]).name for r in rows]
    if len(rows)!=37444 or sorted(names)!=sorted(p.name for p in context.test_root.iterdir() if p.is_file()):
        raise ValueError("Test population changed")
    logits=torch.zeros((len(rows),len(context.classes)))
    started=time.monotonic()
    for view,scale in enumerate((448,512,576),1):
        loader=DataLoader(Images(context.test_root,rows,image_transform(448,scale=scale)),batch_size=8,num_workers=2,shuffle=False)
        last_print=time.monotonic()
        for batch,(images,indices) in enumerate(loader,1):
            images=images.to(context.device)
            values=model(images)+model(images.flip(3))
            if not torch.isfinite(values).all():raise ValueError("Nonfinite test logits")
            logits[indices]+=values.float().cpu()
            now=time.monotonic()
            if batch==1 or batch==len(loader) or now-last_print>=30:
                status=dict(phase="six_view_inference",arm=arm,scale=scale,scale_index=view,batch=batch,batches=len(loader),seconds=now-started)
                print(json.dumps(status),flush=True);atomic_json_dump(status,root/"progress.json");last_print=now
    out=root/arm/"submission"
    prediction=logits.argmax(1).tolist()
    torch.save(dict(logits=logits,names=names,classes=context.classes,checkpoint_sha256=sha256_file(checkpoint)),root/arm/"test_logits.pt")
    create_submission([(n,context.classes[p]) for n,p in zip(names,prediction)],names,out,checkpoint,
        inference_mode="fixed_448_512_576_flip_sum_no_bias",tta_risk_acknowledged=True,valid_labels=set(context.classes),space_after_comma=True,
        extra_manifest=dict(binding=context.binding,decoder=context.protocol["test_decoder"],policy="ema_swa_2_4",platform_score=None))
    checked=subprocess.run([sys.executable,str(ROOT/"scripts/check_submission.py"),"--test_dir",str(context.test_root),
        "--class-mapping",str(context.base/"class_to_idx.json"),"--csv",str(out/"pred_results.csv"),"--zip",str(out/"submission.zip")],capture_output=True,text=True)
    (out/"submission_check.log").write_text(checked.stdout+checked.stderr)
    checked.check_returncode()
    result=dict(status="package_verified",rows=len(rows),seconds=time.monotonic()-started,checkpoint=str(checkpoint),
                checkpoint_sha256=sha256_file(checkpoint),csv=str(out/"pred_results.csv"),zip=str(out/"submission.zip"),
                csv_sha256=sha256_file(out/"pred_results.csv"),zip_sha256=sha256_file(out/"submission.zip"),checks=9,platform_score=None)
    atomic_json_dump(result,out/"report.json")
    del model;torch.cuda.empty_cache()
    return result


def complete(context, root, allow_local_parent_baseline=False, resume=False):
    root=Path(root)
    require_idle_cuda()
    replay=read_json(root/"parent_replay.json")
    if not replay["exact_predictions_match"] and not allow_local_parent_baseline:
        raise ValueError("Strict source replay failed; explicit local-baseline protocol ruling required")
    if (root/"final_report.json").exists():raise FileExistsError("Completed run cannot be overwritten")
    if sha256_file(root/"parent_center.npz") != replay["parent_prediction_sha256"]:
        raise ValueError("Recorded local parent predictions changed")
    baseline=np.load(root/"parent_center.npz")["predictions"]
    if resume:
        if read_json(root/'current_inputs_verified.json')['binding']!=context.binding:
            raise ValueError('Resume implementation/input binding changed')
    else:
        atomic_json_dump(context.inputs_report,root/'current_inputs_verified.json')
    if not replay["exact_predictions_match"]:
        atomic_json_dump(dict(source_gate_passed=False,source_mismatch_count=replay["mismatch_count"],
            ruling="User 2026-10-01 prioritized actual local execution after being told about strict source replay differences; both arms share the recorded local parent baseline.",
            cross_machine_bit_exact_claim=False,train_recipe_changed=False),root/"local_baseline_ruling.json")
    if resume and (root/'probe/report.json').exists():
        probe_report=read_json(root/'probe/report.json')
        if probe_report['status']!='passed':raise ValueError('Resume requires the verified cost probe')
    else:probe_report=probe(context,root)
    reports={arm:fit_arm(context,arm,root,baseline,resume=resume) for arm in context.protocol['arms']}
    first=(root/"head_only/paired_batches.jsonl").read_bytes()
    second=(root/"lora_and_head/paired_batches.jsonl").read_bytes()
    if first!=second:raise ValueError("Formal paired batches differ")
    labels=context.frozen["labels"]
    predictions={arm:np.load(root/arm/"val_ema_swa_2_4.npz")["predictions"] for arm in context.protocol["arms"]}
    vs_control=grouped_pair(context,labels,predictions["head_only"],predictions["lora_and_head"])
    vs_parent=grouped_pair(context,labels,baseline,predictions["lora_and_head"])
    ratio=lambda p:p["regressions"]==0 or p["corrections"]>=1.25*p["regressions"]
    gate=(vs_control["all"]["net_correct"]>=75 and vs_parent["all"]["net_correct"]>=75
          and vs_control["target"]["net_correct"]>=25 and ratio(vs_control["all"]) and ratio(vs_parent["all"])
          and int(((baseline!=labels)&np.isin(labels,context.groups["target_classes"])).sum())>=75)
    result=dict(experiment_id=context.protocol["experiment_id"],status="training_complete_awaiting_packages",reports=reports,
        parent_replay=replay,baseline="local_actual_parent",source_replay_exact=replay["exact_predictions_match"],
        candidate_vs_control=vs_control,candidate_vs_parent=vs_parent,paired_batch_sha256=hashlib.sha256(first).hexdigest(),
        paired_batches_exact=True,numeric_review_gate=gate,decision="supports_review" if gate else "closed_fixed_recipe",
        frozen_groups=context.groups,platform_score=None,automatic_full_training=False)
    if resume and (root/'paired_validation.csv').exists():
        saved=list(csv.DictReader((root/'paired_validation.csv').open()))
        if len(saved)!=len(labels) or any(int(row[arm])!=predictions[arm][i] for i,row in enumerate(saved) for arm in context.protocol['arms']):
            raise ValueError('Existing paired CSV differs from fixed primary predictions')
    else:
        with (root/"paired_validation.csv").open("x",newline="") as handle:
            writer=csv.writer(handle);writer.writerow(["image_path","label","source_parent","local_parent","head_only","lora_and_head"])
            writer.writerows((r["image_path"],int(y),int(src),int(p),int(a),int(b)) for r,y,src,p,a,b in zip(context.val,labels,context.frozen["unbalanced768"],baseline,predictions["head_only"],predictions["lora_and_head"]))
    atomic_json_dump(result,root/"paired_report.json")
    result['packages']={}
    for arm in context.protocol['arms']:
        report_path=root/arm/'submission/report.json'
        if resume and report_path.exists():
            package=read_json(report_path)
            if package['status']!='package_verified' or package['checks']!=9 or any(sha256_file(Path(package[key]))!=package[key+'_sha256'] for key in ('csv','zip','checkpoint')):
                raise ValueError('Existing verified submission package changed')
            result['packages'][arm]=package
        else:result['packages'][arm]=infer_arm(context,arm,root)
    result["status"]="completed_verified_delivery"
    atomic_json_dump(result,root/"final_report.json")
    return result



def replay_parent(context, root):
    """Reproduce the zero-update full parent gate before running either arm."""
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    if (root/'parent_replay.json').exists() or (root/'parent_center.npz').exists():
        raise FileExistsError('Use a fresh run directory for parent replay')
    require_idle_cuda()
    atomic_json_dump(context.inputs_report,root/'inputs_verified.json')
    torch.cuda.reset_peak_memory_stats()
    model=initial_model(context,'head_only')
    predictions,cost=center_predictions(context,model,context.val,output=root/'parent_center.npz')
    expected=context.frozen['unbalanced768']
    mismatches=np.flatnonzero(predictions!=expected)
    report=dict(status='passed' if len(mismatches)==0 else 'failed',
        exact_predictions_match=len(mismatches)==0,mismatch_count=len(mismatches),
        mismatches=[dict(image_path=context.val[i]['image_path'],expected=int(expected[i]),actual=int(predictions[i])) for i in mismatches],
        **metrics(context.frozen['labels'],predictions,len(context.classes)),cost=cost,
        cuda_peak_bytes=torch.cuda.max_memory_allocated(),
        parent_prediction_sha256=sha256_file(root/'parent_center.npz'),training_started=False)
    atomic_json_dump(report,root/'parent_replay.json')
    del model
    torch.cuda.empty_cache()
    return report