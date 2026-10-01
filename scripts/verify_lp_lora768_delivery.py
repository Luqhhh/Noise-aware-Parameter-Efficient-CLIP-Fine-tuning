"""Independently audit the complete local machine-A delivery, without training code."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import torch


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def scores(y,p,n=750):
    count=np.bincount(y,minlength=n)
    good=np.bincount(y[y==p],minlength=n)
    supported=count>0
    return dict(rows=len(y),correct=int(good.sum()),micro=float((y==p).mean()) if len(y) else None,
                macro=float((good[supported]/count[supported]).mean()) if supported.any() else None,
                covered_classes=int(supported.sum()))


def compare(y,a,b):
    fixed=int(((a!=y)&(b==y)).sum());broken=int(((a==y)&(b!=y)).sum())
    return dict(before=scores(y,a),after=scores(y,b),changed=int((a!=b).sum()),corrections=fixed,
        regressions=broken,net_correct=fixed-broken,
        corrections_to_regressions=fixed/broken if broken else ('infinite' if fixed else None))


def groups(y,a,b,g):
    target=np.isin(y,g['target_classes']);tail=np.isin(y,g['tail_classes'])
    masks=dict(all=np.ones(len(y),bool),target=target,tail75=tail,non_target=~target,
        other=~(target|tail),target_tail_overlap=target&tail)
    result={k:compare(y[m],a[m],b[m]) for k,m in masks.items()}
    result['per_target_pair']={'-'.join(map(str,p['classes'])):compare(y[m],a[m],b[m])
        for p in g['top_pairs'] for m in [np.isin(y,p['classes'])]}
    return result


def audit(root,assets,repo):
    final=read(root/'final_report.json')
    assert final['status']=='completed_verified_delivery'
    assert final['paired_batches_exact'] and not final['automatic_full_training']
    original=torch.load(assets/'assets/a_parent/selected.pt',map_location='cpu',weights_only=False)
    targets=torch.load(assets/'assets/common/targets.pt',map_location='cpu',weights_only=False)
    source=np.load(assets/'assets/common/validation_predictions.npz')
    parent=np.load(root/'parent_center.npz')
    y=source['labels'];assert len(y)==14880
    np.testing.assert_array_equal(parent['labels'],y)
    np.testing.assert_array_equal(parent['image_paths'],source['image_paths'])
    frozen=read(repo/'results/three_machine_exploration_20261001/a_target_groups.json')
    assert int(np.isin(y,frozen['target_classes']).sum())==417
    active=set(torch.where(targets['weights']>0)[0].tolist());assert len(active)==119074
    exported={};predictions={};package_results={}
    arms=['head_only','lora_and_head']
    batch_hashes=[digest(root/arm/'paired_batches.jsonl') for arm in arms]
    assert batch_hashes[0]==batch_hashes[1]==final['paired_batch_sha256']
    for arm in arms:
        seen={e:[] for e in range(1,5)};counts={e:0 for e in range(1,5)}
        with (root/arm/'paired_batches.jsonl').open() as f:
            for line in f:
                item=json.loads(line);epoch=item['epoch'];counts[epoch]+=1
                assert item['batch']==counts[epoch]
                seen[epoch].extend(item['indices'])
                assert sorted(item['permutation'])==list(range(len(item['indices'])))
                assert 0<=item['lam']<=1 and item['reliability_mass']>0
                assert len(item['sha256'])==64
        assert all(counts[e]==3722 and len(seen[e])==119074 and set(seen[e])==active for e in seen)
        history=read(root/arm/'history.json');assert [x['epoch'] for x in history]==[1,2,3,4]
        assert [x['updates'] for x in history]==[3722,7444,11166,14888]
        exports={p:torch.load(root/arm/f'{p}.pt',map_location='cpu',weights_only=False)
                 for p in ['last_raw','last_ema','ema_swa_2_4']}
        classes=original['classes'];assert len(classes)==750
        for policy,item in exports.items():
            assert item['complete'] and item['epoch']==4 and item['optimizer_updates']==14888
            assert item['bias'] is None and item['classes']==classes
            state=item['selected_state'];assert state.keys()==original['selected_state'].keys() and len(state)==98
            assert all(torch.isfinite(v).all() for v in state.values())
            if arm=='head_only':
                assert all(torch.equal(v,original['selected_state'][k]) for k,v in state.items() if 'lora_' in k)
        if arm=='lora_and_head':
            assert any(not torch.equal(v,original['selected_state'][k]) for k,v in exports['last_raw']['selected_state'].items() if 'lora_' in k)
        average=None
        for epoch in (2,3,4):
            item=torch.load(root/arm/f'ema_epoch{epoch}.pt',map_location='cpu',weights_only=False)
            state=item['selected_state']
            if average is None:average={k:v.clone() for k,v in state.items()}
            else:
                for k,v in state.items():average[k].lerp_(v,1/(epoch-1))
        for k,v in average.items():torch.testing.assert_close(v,exports['ema_swa_2_4']['selected_state'][k],rtol=1e-6,atol=1e-7)
        average_delta=max(float((v-exports['ema_swa_2_4']['selected_state'][k]).abs().max()) for k,v in average.items())
        for policy in exports:
            val=np.load(root/arm/f'val_{policy}.npz')
            np.testing.assert_array_equal(val['labels'],y)
            np.testing.assert_array_equal(val['image_paths'],source['image_paths'])
            np.testing.assert_array_equal(val['predictions'],val['logits'].argmax(1))
            assert scores(y,val['predictions'])==final['reports'][arm]['evaluations'][policy]['metrics']
        predictions[arm]=np.load(root/arm/'val_ema_swa_2_4.npz')['predictions']
        out=root/arm/'submission';package=read(out/'report.json')
        assert package['checks']==9 and package['rows']==37444
        assert digest(out/'pred_results.csv')==package['csv_sha256']
        assert digest(out/'submission.zip')==package['zip_sha256']
        assert digest(root/arm/'ema_swa_2_4.pt')==package['checkpoint_sha256']
        raw=(out/'pred_results.csv').read_bytes()
        with zipfile.ZipFile(out/'submission.zip') as archive:
            assert archive.namelist()==['pred_results.csv'] and archive.read('pred_results.csv')==raw
        parsed=list(csv.reader(raw.decode('utf-8-sig').splitlines()))
        assert len(parsed)==37444 and len({r[0] for r in parsed})==37444
        logits=torch.load(root/arm/'test_logits.pt',map_location='cpu',weights_only=False)
        assert logits['checkpoint_sha256']==package['checkpoint_sha256'] and logits['classes']==classes
        assert torch.isfinite(logits['logits']).all()
        assert [r[0] for r in parsed]==logits['names']
        assert [r[1] for r in parsed]==[' '+classes[i] for i in logits['logits'].argmax(1).tolist()]
        assert all(len(r[1])==5 and r[1][0]==' ' and r[1][1:].isdigit() for r in parsed)
        package_results[arm]=package
        exported[arm]=dict(updates=14888,epochs=4,complete_state=True,ema_average_verified=True,ema_average_max_abs_delta=average_delta,
            frozen_control_lora_exact=arm=='head_only',batch_population_exact=True)
    control=groups(y,predictions['head_only'],predictions['lora_and_head'],frozen)
    vs_parent=groups(y,parent['predictions'],predictions['lora_and_head'],frozen)
    assert control==final['candidate_vs_control'] and vs_parent==final['candidate_vs_parent']
    rows=list(csv.DictReader((root/'paired_validation.csv').open()))
    assert len(rows)==14880
    for i,r in enumerate(rows):
        assert r['image_path']==source['image_paths'][i] and int(r['label'])==y[i]
        assert int(r['source_parent'])==source['unbalanced768'][i] and int(r['local_parent'])==parent['predictions'][i]
        assert all(int(r[arm])==predictions[arm][i] for arm in arms)
    ratio=lambda x:x['regressions']==0 or x['corrections']>=1.25*x['regressions']
    budget=int(((parent['predictions']!=y)&np.isin(y,frozen['target_classes'])).sum())
    gate=control['all']['net_correct']>=75 and vs_parent['all']['net_correct']>=75 and control['target']['net_correct']>=25 and ratio(control['all']) and ratio(vs_parent['all']) and budget>=75
    assert gate==final['numeric_review_gate']
    report=dict(status='independently_verified',source_replay_exact=final['source_replay_exact'],
        source_parent=scores(y,source['unbalanced768']),local_parent=scores(y,parent['predictions']),
        arms=exported,packages=package_results,candidate_vs_control=control,candidate_vs_local_parent=vs_parent,
        candidate_vs_source_parent=groups(y,source['unbalanced768'],predictions['lora_and_head'],frozen),
        paired_batch_sha256=batch_hashes[0],decision=final['decision'],platform_score=None)
    (root/'independent_delivery_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--assets',required=True,type=Path)
    args=parser.parse_args()
    result=audit(args.output,args.assets,Path(__file__).resolve().parents[1])
    print(json.dumps(dict(status=result['status'],decision=result['decision'])))