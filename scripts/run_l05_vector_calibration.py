#!/usr/bin/env python3
"""One fixed conditional cross-fit of frozen L05 class-wise affine calibration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from run_l05_nonlinear_head import preflight, write_json, sha256_file, fixed_decode
from aegis_clip.tta import fuse_paired_logits
from aegis_clip.prior_alignment import fit_prior_bias, apply_prior_bias
from audit_l05_mixstyle_decode import numpy_decode
from l05_vector_calibration_support import group_folds, fit, apply, metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    assert (cfg['temperature'],cfg['prior_strength'],cfg['folds'],cfg['seed'],cfg['regularization'])==(1.4,.6,3,42,1.)
    torch.set_num_threads(2)
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'implementation.json').exists():raise FileExistsError('Preserve existing execution')
    rows,reference,identity=preflight(cfg)
    assert identity['reference_cache']==cfg['reference_cache_sha256']
    write_json(out/'implementation.json',{'command':os.sys.argv,'pid':os.getpid(),'config':cfg,'identity':identity,
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'script_sha256':sha256_file(__file__),'support_sha256':sha256_file(Path(__file__).with_name('l05_vector_calibration_support.py'))})
    original,flip=reference['original_logits'],reference['flip_logits'];labels=reference['labels'].numpy();classes=original.shape[1]
    frozen,bias,prediction,_=fixed_decode(original,flip,reference['labels'],cfg)
    independent,_,_=numpy_decode(original.numpy().astype(np.float64),flip.numpy().astype(np.float64))
    assert np.array_equal(independent,prediction.numpy())
    assert abs(frozen['macro']-.7582651238982523)<1e-9 and abs(frozen['micro']-.7665322580645161)<1e-9
    fused=fuse_paired_logits(original,flip,mode='mean_probabilities',temperature=1.4)
    ids=group_folds([r['content_group'] for r in rows['val']],cfg['folds'],cfg['seed'])
    base_prediction=np.full(len(labels),-1);candidate_prediction=np.full(len(labels),-1)
    folds=[];arrays={'fold_id':ids,'labels':labels};started=time.monotonic()
    for fold in range(3):
        hold=ids==fold;train=~hold
        assert not ({r['content_group'] for r,t in zip(rows['val'],train) if t}&{r['content_group'] for r,h in zip(rows['val'],hold) if h})
        fold_bias,prior_report=fit_prior_bias(fused[train])
        base=apply_prior_bias(fused,fold_bias,strength=.6).numpy()
        scale,offset,report=fit(base[train],labels[train],regularization=1.)
        base_prediction[hold]=base[hold].argmax(1)
        corrected=apply(base[hold],scale,offset)
        candidate_prediction[hold]=corrected.argmax(1)
        # Apply audit with independent torch centering/multiplication.
        t=torch.tensor(base[hold],dtype=torch.float64)
        check=((t-t.mean(1,keepdim=True))*torch.from_numpy(scale)+torch.from_numpy(offset)).argmax(1).numpy()
        assert np.array_equal(check,candidate_prediction[hold])
        arrays.update({f'prior_{fold}':fold_bias.numpy(),f'scale_{fold}':scale,f'offset_{fold}':offset})
        folds.append({'fold':fold,'fit_samples':int(train.sum()),'held_out_samples':int(hold.sum()),'fit':report,'prior_fit':prior_report})
    assert np.all(base_prediction>=0) and np.all(candidate_prediction>=0)
    baseline=metrics(base_prediction,labels,classes);candidate=metrics(candidate_prediction,labels,classes)
    delta={k:100*(candidate[k]-baseline[k]) for k in candidate}
    frozen_delta={k:100*(candidate[k]-frozen[k]) for k in candidate}
    passed=delta['macro']>=.3 and delta['micro']>=0 and frozen_delta['macro']>=.3 and frozen_delta['micro']>=0
    arrays.update(baseline_prediction=base_prediction,candidate_prediction=candidate_prediction)
    np.savez_compressed(out/'crossfit.npz',**arrays)
    result={'experiment_id':cfg['experiment_id'],'identity':identity,'config':cfg,'frozen_baseline':frozen,
        'crossfit_baseline':baseline,'crossfit_candidate':candidate,'delta_pp':delta,'delta_vs_frozen_pp':frozen_delta,
        'folds':folds,'promotion_pass':passed,'test_used':False,'labels_used_for_calibration':True,
        'independent_heldout_predictions_identical':len(labels),'baseline_numpy_replay_identical':len(labels),
        'corrections':int(((candidate_prediction==labels)&(base_prediction!=labels)).sum()),
        'regressions':int(((candidate_prediction!=labels)&(base_prediction==labels)).sum()),
        'crossfit_seconds':time.monotonic()-started,'crossfit_sha256':sha256_file(out/'crossfit.npz'),
        'limitation':'Conditional calibration cross-fit; parent checkpoint and base recipe selected using whole validation; not unbiased full-pipeline OOF'}
    if passed:
        full=apply_prior_bias(fused,bias,strength=.6).numpy()
        scale,offset,report=fit(full,labels,regularization=1.)
        np.savez_compressed(out/'frozen_calibration.npz',prior=bias.numpy(),scale=scale,offset=offset)
        result['full_calibration']={'fit':report,'sha256':sha256_file(out/'frozen_calibration.npz')}
    write_json(out/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
