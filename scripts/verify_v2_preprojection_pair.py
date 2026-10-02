"""Independent CPU reconstruction of a completed V2 512/768 pair or failed gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from verify_v1_768_full_delivery import numpy_uniform_bias


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def metrics(y,p,mask,classes):
    subset=y[mask]; pred=p[mask]
    counts=np.bincount(subset,minlength=classes)
    correct=np.bincount(subset[subset==pred],minlength=classes)
    present=counts>0
    return dict(rows=len(subset),correct=int(correct.sum()),errors=int((subset!=pred).sum()),
                classes=int(present.sum()),micro=float((subset==pred).mean()) if len(subset) else None,
                macro=float(np.mean(correct[present]/counts[present])) if len(subset) else None)


def comparison(y,a,b,mask):
    y,a,b=y[mask],a[mask],b[mask]
    corrected=int(((a!=y)&(b==y)).sum()); regressed=int(((a==y)&(b!=y)).sum())
    return dict(rows=len(y),corrections=corrected,regressions=regressed,net=corrected-regressed,
                changed=int((a!=b).sum()),both_wrong=int(((a!=y)&(b!=y)).sum()))


def verify(directory,metadata_only=False):
    start=time.monotonic(); out=Path(directory).resolve(); binding=read(out/'binding.json')
    sources=binding['sources']
    for path,expected in sources.items():
        assert digest(path)==expected,path
    cfg_path=next(p for p in sources if Path(p).name.startswith('v2_preprojection768_pair_20261002') and Path(p).suffix=='.json')
    cfg=read(cfg_path); stage=Path(cfg['stage_root'])
    if 'total_experiment_budget_seconds' in cfg:
        assert cfg['budget_seconds']+cfg['consumed_seconds_before_resource_revision']==cfg['total_experiment_budget_seconds']==10800
    for name in ('groups','schedule'):
        assert digest(out/f'{name}.npz')==binding[f'{name}_sha256']
    frozen=arrays(out/'schedule.npz'); group=arrays(out/'groups.npz')
    train,val=rows(stage/'train_dev.csv'),rows(stage/'val_dev.csv')
    y=np.array([int(r['label']) for r in val]); paths=np.array([r['image_path'] for r in val])
    np.testing.assert_array_equal(group['labels'],y); np.testing.assert_array_equal(group['image_paths'],paths)
    assert not {r['content_group'] for r in train}&{r['content_group'] for r in val}
    train_y=np.array([int(r['label']) for r in train]); counts=np.bincount(train_y,minlength=750)
    weights=torch.tensor(1/np.sqrt(counts[train_y]),dtype=torch.double)
    expected=torch.multinomial(weights,512*80,replacement=True,generator=torch.Generator().manual_seed(cfg['seed'])).numpy()
    np.testing.assert_array_equal(frozen['indices'],expected)
    rng=np.random.default_rng(cfg['seed'])
    np.testing.assert_array_equal(frozen['image_seeds'],rng.integers(0,2**31,512*80,dtype=np.int64))
    np.testing.assert_array_equal(frozen['mix_seeds'],rng.integers(0,2**31,512,dtype=np.int64))
    previous=read(cfg['error_report'] if Path(cfg['error_report']).is_absolute() else Path(cfg_path).parents[1]/cfg['error_report'])
    old=rows(previous['aligned_csv'])
    common=np.array([all(int(r[k])!=int(r['label']) for k in
        ('s3_ema','lr512_swa','head768','balanced768','wft448_ema')) for r in old])
    np.testing.assert_array_equal(common,group['common_five_errors']); assert common.sum()==2411
    parent=arrays(out/'parent.npz')
    assert parent['logits'].shape==(14880,750) and np.isfinite(parent['logits']).all()
    np.testing.assert_array_equal(parent['image_paths'],paths);np.testing.assert_array_equal(parent['labels'],y)
    np.testing.assert_array_equal(parent['predictions'],parent['logits'].argmax(1))
    initial=read(out/'initial_equivalence.json')
    assert initial['max_abs_logit_difference']==float(np.abs(parent['logits']-parent['folded_logits']).max())
    assert initial['top1_changes']==int((parent['predictions']!=parent['folded_logits'].argmax(1)).sum())
    assert initial['native_local_bf16_vs_archive_changes']==int((parent['native_bf16_logits'].argmax(1)!=group['original_s3']).sum())
    assert initial['fp32_readout_vs_original_archive_changes']==int((parent['predictions']!=group['original_s3']).sum())
    equivalence_passed=initial['max_abs_logit_difference']<=cfg['initial_max_abs_logit_difference'] and initial['top1_changes']==0
    if not (out/'report.json').exists():
        failure=read(out/'failure.json')
        assert failure['status']=='failed_closed' and failure['automatic_restart'] is False
        if 'cost gate failed' in failure['error']:
            assert equivalence_passed
            cost=read(out/'cost.json')
            assert cost['passed'] is False
            assert cost['reserved_seconds']+cost['elapsed_seconds']>=cost['budget_seconds']
            assert not (out/'control'/'selected.pt').exists() and not (out/'candidate'/'selected.pt').exists()
        else:
            assert failure['error']=='Initial function-equivalence check failed' and not equivalence_passed
        return dict(status='failed_gate_independently_verified',failure=failure['error'],initial_equivalence=initial,
                    formal_training_started=False,source_count=len(sources),elapsed_seconds=time.monotonic()-start)
    report=read(out/'report.json'); assert equivalence_passed
    for path,expected in report['artifacts'].items():
        assert digest(path)==expected,path
    if metadata_only:
        return dict(status='metadata_verified',source_count=len(sources),artifact_count=len(report['artifacts']),
                    report_sha256=digest(out/'report.json'),elapsed_seconds=time.monotonic()-start)
    predictions={'parent':parent['predictions']}; metric={}; pairs={}; histories={}; package_results={}
    groups={k:v for k,v in group.items() if k not in ('labels','image_paths','original_s3')}
    for arm,dim in (('control',512),('candidate',768)):
        d=out/arm; z=arrays(d/'validation.npz'); cold=arrays(d/'cold_validation64.npz')
        assert z['logits'].shape==(14880,750) and np.isfinite(z['logits']).all()
        for k,expected in [('labels',y),('image_paths',paths),('predictions',z['logits'].argmax(1))]:
            np.testing.assert_array_equal(z[k],expected)
        np.testing.assert_array_equal(cold['logits'],z['logits'][:64])
        predictions[arm]=z['predictions']; histories[arm]=read(d/'training.json')
        h=histories[arm]
        assert h['updates']==h['scheduler_last_epoch']==512 and len(h['input_trace'])==len(h['losses'])==512
        assert all(s==512 for s in h['optimizer_parameter_steps'])
        checkpoint=torch.load(d/'selected.pt',map_location='cpu',weights_only=False,mmap=True)
        assert checkpoint['feature_dim']==dim and checkpoint['updates']==512 and checkpoint['partition']=='train_dev'
        assert checkpoint['model']['head.weight'].shape==(750,dim)
        assert ('visual.proj' in checkpoint['model'])==(dim==512)
        assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
        del checkpoint
        data=arrays(d/'test_predictions.npz'); p=data['probabilities']
        assert p.shape==(37444,750) and np.isfinite(p).all() and (p>=0).all()
        np.testing.assert_allclose(p.sum(1),1,atol=2e-6,rtol=0)
        official=sorted(Path(r['image_path']).name for r in rows(stage/'test_manifest.csv'))
        assert data['names'].tolist()==official
        score=np.log(np.maximum(p.astype(np.float64),1e-30))
        independent=numpy_uniform_bias(score,200)
        np.testing.assert_allclose(data['bias'],independent,atol=2e-5,rtol=0)
        raw=p.argmax(1); corrected=(score+independent).argmax(1)
        np.testing.assert_array_equal(raw,data['raw']);np.testing.assert_array_equal(corrected,data['calibrated'])
        for name,pred in (('submission_raw',raw),('submission_bias',corrected)):
            encoded=''.join(f'{n}, {int(c):04d}\n' for n,c in zip(official,pred)).encode()
            assert (d/name/'pred_results.csv').read_bytes()==encoded
            with zipfile.ZipFile(d/name/'submission.zip') as archive:
                assert archive.namelist()==['pred_results.csv'] and archive.read('pred_results.csv')==encoded
            assert 'All checks passed' in (d/name/'check.log').read_text()
        package_results[arm]=dict(rows=37444,bias_max_abs_error=float(np.abs(independent-data['bias']).max()),
                                  full_independent_decisions_equal=True)
    assert histories['control']['input_trace']==histories['candidate']['input_trace']
    for a,p in predictions.items():
        metric[a]={g:metrics(y,p,m,750) for g,m in groups.items()}
    for name,a,b in [('candidate_vs_control','candidate','control'),('candidate_vs_parent','candidate','parent'),
                      ('control_vs_parent','control','parent')]:
        pairs[name]={g:comparison(y,predictions[b],predictions[a],m) for g,m in groups.items()}
    assert metric==report['metrics'] and pairs==report['comparisons']
    g=cfg['review_gate']; c=pairs['candidate_vs_control']['all']
    passed=(c['net']>=g['net_vs_control_min'] and pairs['candidate_vs_parent']['all']['net']>=g['net_vs_parent_min'] and
        pairs['candidate_vs_control']['common_five_errors']['net']>=g['common_error_net_vs_control_min'] and
        metric['candidate']['all']['macro']-metric['control']['all']['macro']>=g['macro_delta_vs_control_min'] and
        (c['regressions']==0 or c['corrections']/c['regressions']>=g['corrections_to_regressions_min']))
    assert report['review_gate_passed']==passed
    assert report['decision']==('supports_full_training_review' if passed else 'close_fixed_short_continuation')
    return dict(status='passed',report_sha256=digest(out/'report.json'),initial_equivalence=initial,
        full_validation_rows=14880,pair_count=len(pairs)*len(groups),metric_count=len(metric)*len(groups),
        schedule_draws=512*80,paired_training_batches=512,source_count=len(sources),artifact_count=len(report['artifacts']),
        packages=package_results,review_gate_passed=passed,decision=report['decision'],
        elapsed_seconds=time.monotonic()-start,verifier_sha256=digest(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--metadata-only',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    result=verify(args.directory,args.metadata_only)
    with Path(args.output).open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
