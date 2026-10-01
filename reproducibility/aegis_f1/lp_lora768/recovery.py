"""Durable full training state; recovery never substitutes EMA for raw weights."""
import hashlib
from pathlib import Path
import uuid
import torch
from aegis_clip.runtime import atomic_json_dump,sha256_file
from aegis_clip.v1_pipeline import save_artifact,load_artifact
from aegis_clip.v1_strategy import WeightAverage
from .inputs import adapter_head_state,restore_adapter_head,read_json


def save_recovery(context,arm,output,model,optimizer,scheduler,scaler,ema,average,*,epoch,batch,updates,history,losses,elapsed_seconds):
    output=Path(output);directory=output/'recovery';directory.mkdir(exist_ok=True)
    path=directory/f'step_{updates:06d}_{uuid.uuid4().hex[:8]}.pt'
    records=output/'paired_batches.jsonl'
    record_hash=sha256_file(records) if records.exists() else hashlib.sha256(b'').hexdigest()
    payload=dict(arm=arm,selected_state=adapter_head_state(model),optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),ema=ema.payload(),
        average=None if average is None else average.payload(),epoch=epoch,batch=batch,updates=updates,
        history=history,losses=losses,elapsed_seconds=elapsed_seconds,paired_prefix_sha256=record_hash,
        purpose='resume_raw_optimizer_trajectory_not_a_submission',complete=False)
    save_artifact(path,payload,context.binding)
    atomic_json_dump(dict(path=path.name,sha256=sha256_file(path),updates=updates),directory/'latest.json')
    return path


def load_recovery(context,arm,output,model,optimizer,scheduler,scaler):
    output=Path(output);directory=output/'recovery';pointer=read_json(directory/'latest.json')
    path=directory/pointer['path']
    if path.parent!=directory or sha256_file(path)!=pointer['sha256']:raise ValueError('Recovery checkpoint identity changed')
    payload=load_artifact(path,context.binding)
    if payload['arm']!=arm or payload['complete'] or payload['updates']!=pointer['updates']:
        raise ValueError('Wrong arm or state used for recovery')
    restore_adapter_head(model,payload['selected_state'])
    optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
    scaler.load_state_dict(payload['scaler'])
    def average(item):
        if item is None:return None
        result=WeightAverage.restore(item)
        result.state={k:v.to(context.device) for k,v in result.state.items()}
        return result
    payload['ema']=average(payload['ema']);payload['average']=average(payload['average'])
    return payload


def restore_batch_prefix(output,recovered):
    """Preserve failed tail then restore the proven successful-update prefix."""
    output=Path(output);path=output/'paired_batches.jsonl'
    data=path.read_bytes() if path.exists() else b''
    prefix=b''.join(data.splitlines(keepends=True)[:recovered['updates']])
    if hashlib.sha256(prefix).hexdigest()!=recovered['paired_prefix_sha256']:
        raise ValueError('Successful paired-batch prefix does not match recovery state')
    if path.exists():path.rename(output/'recovery'/f'discarded_batches_{uuid.uuid4().hex[:8]}.jsonl')
    path.write_bytes(prefix)
    return len(data.splitlines())-recovered['updates']