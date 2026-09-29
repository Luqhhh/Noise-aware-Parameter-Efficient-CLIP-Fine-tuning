#!/usr/bin/env python3
"""Join fixed pixel evidence to existing final snapshots. Never decide new labels."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw
from p75_text_page_detector import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--p0-summary',type=Path,required=True)
    p.add_argument('--image-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists(): p.error('Output already exists')
    scan=json.loads((args.scan/'scan_summary.json').read_text())
    ref=json.loads(args.p0_summary.read_text())
    if any(v != ref['identity']['split_hashes'][k] for k,v in scan['splits'].items()):
        raise ValueError('Scan uses a different data stage/split')
    if not scan.get('complete') or sha(args.scan/'pixels.jsonl')!=scan['pixels_sha256']:
        raise ValueError('Incomplete or modified pixel scan')
    if sha(args.evidence)!=ref['archived_artifacts'][args.evidence.name]:
        raise ValueError('P0 evidence identity mismatch')
    with (args.scan/'pixels.jsonl').open() as f: rows=[json.loads(s) for s in f]
    by_path={r['image_path']:r for r in rows}
    if len(rows)!=len(by_path): raise ValueError('Duplicate paths')
    evidence_paths=set()
    values={(s,b):[] for s in ('train','val') for b in (False,True)}
    summaries={(s,b):Counter() for s in ('train','val') for b in (False,True)}
    classes=[Counter() for _ in range(ref['identity']['num_classes'])]
    joined=[]
    with gzip.open(args.evidence,'rt') as f:
        for r in csv.DictReader(f):
            q=by_path[r['image_path']]
            if r['image_path'] in evidence_paths or q['split']!=r['split'] or q['label']!=int(r['original_label']) or q['content_group']!=r['content_group']:
                raise ValueError('Pixel/P0 row alignment mismatch')
            evidence_paths.add(r['image_path'])
            key=(r['split'],q['selected']); py=float(r['center_label_probability'])
            pred=int(r['l05_fixed_decode_prediction'] if r['split']=='val' else r['l05_prediction'])
            s=summaries[key]; s['rows']+=1; s['original_label_errors']+=int(pred!=q['label'])
            s['low_py_001']+=int(py<.01); s['low_py_03']+=int(py<.3); s['high_py_07']+=int(py>=.7)
            s['sqrt_py_sum']+=py**.5
            s['logit_gradient_l1_sum']+=2*(py**.5)*(1-py)
            values[key].append(py)
            if q['split']=='train':
                classes[q['label']]['train_rows']+=1
                classes[q['label']]['selected']+=int(q['selected'])
            if q['selected']:
                joined.append(dict(q,center_py=py,gce_relative_scale=py**.5,
                                   final_logit_gradient_l1=2*(py**.5)*(1-py),prediction=pred,
                                   historical_sample_trajectory='missing'))
    if evidence_paths != set(by_path): raise ValueError('Incomplete P0 join')
    expected={'train':ref['identity']['training_samples'],'val':ref['identity']['validation_samples']}
    for s,n in expected.items():
        if sum(summaries[s,b]['rows'] for b in (False,True))!=n: raise ValueError('Count mismatch')
    if sum(summaries['val',b]['original_label_errors'] for b in (False,True))!=ref['baseline_errors']:
        raise ValueError('Fixed-decode baseline errors mismatch')
    report={}
    for (s,b),counter in summaries.items():
        out=dict(counter)
        v=values[s,b]; count=len(v)
        out['rows']=count
        out['py_quantiles']=dict(zip(['min','p25','median','p75','max'],np.quantile(v,[0,.25,.5,.75,1]).tolist())) if count else None
        out['mean_sqrt_py']=out.get('sqrt_py_sum',0)/count if count else None
        total=summaries[s,False]['logit_gradient_l1_sum']+summaries[s,True]['logit_gradient_l1_sum']
        out['final_logit_l1_share']=out.get('logit_gradient_l1_sum',0)/total if total else None
        report[s+('_detected_text_proxy' if b else '_not_detected_proxy')]=out
    selection=sorted([r for r in joined if r['split']=='train'],key=lambda r:r['image_path'])
    rng=random.Random(scan['rule']['review_seed'])
    review=rng.sample(selection,min(scan['rule']['review_limit'],len(selection)))
    rng.shuffle(review)
    for i,row in enumerate(review,1): row['review_id']=f'T{i:02d}'
    result=dict(experiment=scan['rule']['id'],scan_sha256=sha(args.scan/'scan_summary.json'),
        evidence_sha256=sha(args.evidence),p0_summary_sha256=sha(args.p0_summary),groups=report,
        selected_train_classes=sum(c['selected']>0 for c in classes),
        emptied_classes=[i for i,c in enumerate(classes) if c['train_rows'] and c['selected']==c['train_rows']],
        review_ids=[r['review_id'] for r in review],
        training_started=False,detector_gate='pending_visual_false_positive_check' if selection else 'failed_empty_selection',
        limitations=['Detected text proxy is not confirmed open-set truth.',
            'Not-detected complement is mixed and is not a clean/effective-target validation set.',
            'No semantic mapping and no replacement labels. Numerical classes are unchanged.',
            'Final center-view snapshots cannot reconstruct training history or parameter-gradient harm.',
            '2*sqrt(py)*(1-py) is the per-example GCE logit-gradient L1 at q=.5, not parameter-gradient norm.',
            'Only a whole fixed detector may pass/fail review; no per-image manual whitelist/blacklist.'])
    if result['emptied_classes']:
        result['detector_gate']='failed_class_coverage'
    result['pixel_rejection_reasons']=dict(Counter(r['reason'] for r in rows if 'reason' in r))
    args.out.mkdir(parents=True)
    (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.out/'selected_snapshot.json').write_text(json.dumps(joined,indent=2)+'\n')
    (args.out/'review_manifest.json').write_text(json.dumps(review,indent=2)+'\n')
    with (args.out/'class_coverage.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['class','train_rows','selected','remaining'],lineterminator='\n');writer.writeheader()
        for i,c in enumerate(classes): writer.writerow(dict({'class':i},train_rows=c['train_rows'],selected=c['selected'],remaining=c['train_rows']-c['selected']))
    # Verify previews against the official preprocessing, when hits exist.
    if review:
        from clip.clip import _transform
        preprocess = _transform(384)
    # Content-only previews; model/label information stays outside these sheets.
    # At most 16 automatically sampled hits, not a repeat of the 200-case audit.
    for offset in range(0,len(review),4):
        canvas=Image.new('RGB',(1536,820),'#ddd'); draw=ImageDraw.Draw(canvas)
        for col,row in enumerate(review[offset:offset+4]):
            path=args.image_root/row['image_path']
            if sha(path)!=row['file_sha256']: raise ValueError('Review image changed')
            with Image.open(path) as image:
                original=image.convert('RGB'); preview=original.copy();preview.thumbnail((380,380))
                canvas.paste(preview,(col*384+(384-preview.width)//2,28+(380-preview.height)//2))
                w,h=original.size
                new=(384,int(384*h/w)) if w<h else (int(384*w/h),384)
                resized=original.resize(new,Image.Resampling.BICUBIC)
                left=round((new[0]-384)/2); top=round((new[1]-384)/2)
                center=resized.crop((left,top,left+384,top+384))
                expected=preprocess.transforms[1](preprocess.transforms[0](original))
                if not np.array_equal(np.asarray(center),np.asarray(expected)):
                    raise ValueError('Displayed center crop differs from official transform')
                canvas.paste(center,(col*384,430))
                draw.text((col*384+8,5),row['review_id']+' original / center384',fill='black')
        canvas.save(args.out/f'review_{offset//4+1:02d}.jpg',quality=95)
    lines=['# P75_TEXT_PAGE_MASK_V1 fixed detector check', '',
        f"Gate: `{result['detector_gate']}`. Training started: false.", '',
        '| Group | Rows | Original-label errors | Mean sqrt(py) | Final logit L1 share |',
        '|---|---:|---:|---:|---:|']
    for name, group in report.items():
        lines.append(f"| {name} | {group['rows']} | {group.get('original_label_errors',0)} | {group['mean_sqrt_py']} | {group['final_logit_l1_share']} |")
    lines += ['', 'No candidate outcome or platform gain was measured.', '',
        'A zero selection invalidates this detector as an intervention entry; it does not disprove contamination or its possible training harm.', '',
        *['- '+s for s in result['limitations']]]
    (args.out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
