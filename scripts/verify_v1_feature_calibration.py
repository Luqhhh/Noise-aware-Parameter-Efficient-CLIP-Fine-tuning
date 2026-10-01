"""Independent NumPy replay of feature-path and conditional-bias diagnostics."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from aegis_clip.runtime import atomic_json_dump, sha256_file

SOURCE=Path('/home/lux1/noise')
STAGE=SOURCE/'artifacts/stages/repechage/20260921'


def counts(prediction,labels,classes):
    total=np.bincount(labels,minlength=classes)
    correct=np.bincount(labels[prediction==labels],minlength=classes)
    return dict(samples=len(labels),correct=int(correct.sum()),micro=float(correct.sum()/len(labels)),
        macro=float((correct[total>0]/total[total>0]).mean()),
        per_class_correct=correct.tolist(),per_class_samples=total.tolist())


def compare(recorded,actual):
    for k in actual:
        if k in ('micro','macro'):
            assert abs(recorded[k]-actual[k])<1e-10,(k,recorded[k],actual[k])
        else:assert recorded[k]==actual[k],k


def numpy_bias(logits,labels):
    # Independent float64 implementation of the fixed 200 updates.
    x=logits.astype(np.float64)
    c=x.shape[1]
    n=np.bincount(labels,minlength=c)
    assert np.all(n>0)
    w=1/n[labels];w/=w.sum()
    b=np.zeros(c)
    for _ in range(200):
        p=x+b;p-=p.max(1,keepdims=True);np.exp(p,out=p);p/=p.sum(1,keepdims=True)
        mass=(p*w[:,None]).sum(0)
        b-=np.log(np.maximum(mass,1e-8)*c);b-=b.mean()
    return b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--features',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    root=args.root
    ca=root/'calibration_r1'
    arrays=np.load(ca/'predictions.npz');report=json.loads((ca/'report.json').read_text())
    labels=arrays['labels'];folds=arrays['folds'];score=folds>=0;classes=arrays['logits'].shape[1]
    rows=list(csv.DictReader((STAGE/'val_dev.csv').open()))
    assert np.array_equal(labels,[int(r['label']) for r in rows])
    compare(report['raw_heldout_population'],counts(arrays['raw'][score],labels[score],classes))
    compare(report['crossfit'],counts(arrays['crossfit'][score],labels[score],classes))
    bias_errors=[]
    for f in (0,1):
        fit=folds!=f;hold=folds==f
        assert not {r['content_group'] for r,m in zip(rows,fit) if m}&{r['content_group'] for r,m in zip(rows,hold) if m}
        recomputed=numpy_bias(arrays['logits'][fit],labels[fit])
        bias_errors.append(float(np.abs(recomputed-arrays['biases'][f]).max()))
        assert bias_errors[-1]<1e-4
        prediction=(arrays['logits'][hold].astype(np.float64)+recomputed).argmax(1)
        assert np.array_equal(prediction,arrays['crossfit'][hold])
    verification=dict(calibration_predictions_replayed=int(score.sum()),bias_float64_max_errors=bias_errors,
        calibration_content_groups_disjoint=True,calibration_sha256=sha256_file(ca/'predictions.npz'))
    if args.features:
        fe=root/'features';a=np.load(fe/'predictions.npz');r=json.loads((fe/'report.json').read_text())
        population=list(csv.DictReader((STAGE/'full_train.csv').open()))
        ix={row['image_path']:i for i,row in enumerate(population)}
        val_ix=[ix[row['image_path']] for row in rows]
        feature_artifact=torch.load(fe/'features768.pt',map_location='cpu',weights_only=True)
        assert feature_artifact['image_paths']==[row['image_path'] for row in population]
        cached=torch.load(STAGE/'features/features.pt',map_location='cpu',weights_only=True)
        for d,tensor in [(512,cached),(768,feature_artifact['preprojection'])]:
            state=torch.load(fe/f'head{d}.pt',map_location='cpu',weights_only=True)
            x=tensor[val_ix].numpy();x=x/np.linalg.norm(x,axis=1,keepdims=True)
            w=state['weight'].numpy();w=w/np.linalg.norm(w,axis=1,keepdims=True)
            prediction=(x@w.T).argmax(1)
            assert np.array_equal(prediction,a[f'head{d}'])
            compare(r['metrics'][str(d)]['head'],counts(prediction,a['labels'],classes))
            compare(r['metrics'][str(d)]['knn'],counts(a[f'knn{d}'],a['labels'],classes))
        # CPU recheck the native 768->512 projection against the entire audited cache.
        z=feature_artifact['preprojection'];p=feature_artifact['projection']
        projected=torch.nn.functional.normalize(z@p,dim=1)
        error=float((projected-cached).abs().max())
        assert error<2e-4
        verification.update(feature_head_predictions_replayed=2*len(labels),
            full_projection_samples=len(z),full_projection_cpu_max_error=error,
            feature_predictions_sha256=sha256_file(fe/'predictions.npz'),
            feature_tensor_sha256=sha256_file(fe/'features768.pt'))
    atomic_json_dump(verification,root/('independent_verification.json' if args.features else 'calibration_verification.json'))
    print(json.dumps(verification),flush=True)


if __name__=='__main__':main()
