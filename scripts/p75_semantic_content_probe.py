#!/usr/bin/env python3
"""Training-only semantic content development. No species relabeling or mask export."""
import argparse,csv,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
import clip


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sheets(rows,root,out,prefix):
    for offset in range(0,len(rows),4):
        canvas=Image.new('RGB',(1536,580),'#ddd');d=ImageDraw.Draw(canvas)
        for j,r in enumerate(rows[offset:offset+4]):
            with Image.open(root/r['image_path']) as im:
                im=im.convert('RGB'); preview=im.copy();preview.thumbnail((380,350))
                canvas.paste(preview,(j*384+(384-preview.width)//2,25+(350-preview.height)//2))
                from clip.clip import _transform
                prep=_transform(224); crop=prep.transforms[1](prep.transforms[0](im))
                canvas.paste(crop,(j*384+80,355))
                d.text((j*384+8,5),r['review_id']+' original / cached center224',fill='black')
        canvas.save(out/f'{prefix}_{offset//4+1:02d}.jpg',quality=95)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--descriptions',type=Path,required=True);p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Output exists')
    spec=json.loads(a.descriptions.read_text()); pre=json.loads(a.preflight.read_text())
    if pre['status']!='cache_compatibility_passed' or spec['authorization']['source']!='user_message':raise ValueError('Preconditions missing')
    if sha(a.checkpoint)!=pre['official_checkpoint_sha256'] or sha(a.stage/'features/features.pt')!=pre['feature_tensor_sha256']:raise ValueError('Assets changed')
    torch.set_num_threads(4)
    model,_=clip.load(str(a.checkpoint),device='cpu',jit=False);model.eval().float().requires_grad_(False)
    texts=[s for g in ['B','A','N'] for s in spec[g]]
    with torch.no_grad():text=F.normalize(model.encode_text(clip.tokenize(texts)).float(),dim=1)
    paths=json.loads((a.stage/'features/image_paths.json').read_text()); lookup={s:i for i,s in enumerate(paths)}
    bank=torch.load(a.stage/'features/features.pt',map_location='cpu',weights_only=True)
    with (a.stage/'train_dev.csv').open() as f:rows=list(csv.DictReader(f))
    bank=bank[[lookup[r['image_path'].removeprefix('train/')] for r in rows]]
    scores=F.normalize(bank,dim=1)@text.T
    b=len(spec['B']);aa=len(spec['A']);sb=scores[:,:b].max(1).values;sa=scores[:,b:b+aa].max(1).values
    sn,n=scores[:,b+aa:].max(1);margin=sn-torch.maximum(sb,sa)
    for i,r in enumerate(rows):
        r.update(sB=float(sb[i]),sA=float(sa[i]),sN=float(sn[i]),margin=float(margin[i]),non_target_type=spec['N'][int(n[i])])
    # Development samples: fixed top-3 per N subtype, plus top-1 per protected
    # description with a distinct content group. No validation outcomes used.
    selected=[];groups=set()
    for phrase in spec['N']:
        pool=sorted([r for r in rows if r['non_target_type']==phrase],key=lambda r:(-r['margin'],r['image_path']))
        taken=0
        for r in pool:
            if r['content_group'] in groups:continue
            selected.append(dict(r,development_role='N_subtype_top'));groups.add(r['content_group']);taken+=1
            if taken==3:break
    for col in range(b,b+aa):
        for i in torch.argsort(scores[:,col],descending=True).tolist():
            r=rows[i]
            if r['content_group'] not in groups:
                selected.append(dict(r,development_role='protection_top'));groups.add(r['content_group']);break
    random.Random(spec['seed']).shuffle(selected)
    for i,r in enumerate(selected,1):r['review_id']=f'D{i:02d}'
    a.out.mkdir(parents=True)
    with (a.out/'training_scores.csv').open('w',newline='') as f:
        keys=['image_path','label','content_group','sB','sA','sN','margin','non_target_type']
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
    (a.out/'development_cases.json').write_text(json.dumps(selected,indent=2)+'\n')
    torch.save(text,a.out/'text_features.pt')
    report=dict(stage='development_training_only',descriptions_sha256=sha(a.descriptions),preflight_sha256=sha(a.preflight),text_sha256=sha(a.out/'text_features.pt'),scores_sha256=sha(a.out/'training_scores.csv'),training_rows=len(rows),review_rows=len(selected),margin_quantiles=np.quantile(margin.numpy(),[0,.5,.9,.95,.99,1]).tolist(),positive_margin_rows=int((margin>0).sum()),validation_scored=False,mask_created=False)
    (a.out/'development_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    sheets(selected,Path(json.loads((a.stage/'dataset_manifest.json').read_text())['train_root']).parent,a.out,'development')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
