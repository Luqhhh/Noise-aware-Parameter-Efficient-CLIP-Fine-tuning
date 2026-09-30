"""Fixed V2 raw window: checksum, epoch/trajectory identity, and missing snapshot refusal."""
import copy
from pathlib import Path

import pytest
import torch

from v2.runtime import save_checkpoint
from v2.swa import checked_snapshot, export


def examples(tmp_path,epoch=3):
    binding=dict(stage='full_576',plan_sha256='one-plan',parent={'weights':'raw','sha256':'parent'},
                 complete=False,completed_epochs=epoch)
    cfg=dict(train=dict(epochs=5))
    state=dict(weight=torch.tensor([1.,3.]))
    last=dict(binding=dict(copy.deepcopy(binding),complete=True,completed_epochs=5),config=cfg,epoch=4,
              global_step=10,num_classes=750,image_size=576,model=state,metrics={'chosen':'raw'})
    plan=dict(stages={'full_576':dict(steps_per_epoch=2)})
    payload=dict(binding=binding,config=cfg,epoch=epoch-1,global_step=epoch*2,num_classes=750,
                 image_size=576,model=state,weight_source='raw')
    path=tmp_path/f'epoch_{epoch:02d}_raw.pt';save_checkpoint(path,payload)
    return path,plan,last,payload


def test_fixed_raw_snapshot_validates(tmp_path):
    path,plan,last,_=examples(tmp_path)
    assert torch.equal(checked_snapshot(path,plan,last,3)['weight'],last['model']['weight'])


@pytest.mark.parametrize('change',['ema','stage','parent','step','nan','shape'])
def test_snapshot_rejects_mixed_lineage_and_wrong_source(tmp_path,change):
    path,plan,last,payload=examples(tmp_path)
    if change=='ema':payload['weight_source']='ema'
    if change=='stage':payload['binding']['stage']='s3_576'
    if change=='parent':payload['binding']['parent']['sha256']='other'
    if change=='step':payload['global_step']-=1
    if change=='nan':payload['model']={'weight':torch.tensor([float('nan'),3.])}
    if change=='shape':payload['model']={'weight':torch.zeros(3)}
    save_checkpoint(path,payload)
    with pytest.raises(ValueError):
        checked_snapshot(path,plan,last,3)


def test_export_missing_snapshots_refuses_without_creating_output(tmp_path,monkeypatch):
    import v2.swa as module
    monkeypatch.setattr(module,'verify_prepared',lambda path:{})
    output=tmp_path/'export'
    with pytest.raises(ValueError,match='Missing RAW'):
        export(tmp_path/'prepared/plan.json',output)
    assert not output.exists()


def test_export_averages_only_three_raw_snapshots(tmp_path,monkeypatch):
    import v2.swa as module
    import v2.runtime as runtime
    source=tmp_path/'prepared/runs/full_576';source.mkdir(parents=True)
    for epoch in (3,4,5):
        path,plan,last,payload=examples(source,epoch)
        payload['model']={'weight':torch.tensor([float(epoch),float(2*epoch)])}
        save_checkpoint(path,payload)
    save_checkpoint(source/'last.pt',last)
    plan_path=tmp_path/'prepared/plan.json';plan_path.write_text('{}')
    monkeypatch.setattr(module,'verify_prepared',lambda path:plan)
    monkeypatch.setattr(runtime,'read_checkpoint',lambda *a:last)
    path=export(plan_path,tmp_path/'export')
    payload=torch.load(path,weights_only=False)
    torch.testing.assert_close(payload['model']['weight'],torch.tensor([4.,8.]))
    assert payload['average_epochs']==[3,4,5] and payload['weight_source']=='raw'
    with pytest.raises(ValueError,match='overwrite'):
        export(plan_path,tmp_path/'export')
