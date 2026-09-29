#!/usr/bin/env python3
"""Bounded, development-only image audit. Never emits training labels or candidates."""
import argparse
import csv
import gzip
import hashlib
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BUDGET = {'original_plurality':16, 'model_plurality':40, 'third_plurality':64,
          'tied':40, 'control_agreement':40}
EXPECTED = {'original_plurality':241, 'model_plurality':842, 'third_plurality':1390,
            'tied':1001, 'control_agreement':11406}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def write_json(path, data):
    with Path(path).open('x') as f: json.dump(data,f,ensure_ascii=False,indent=2); f.write('\n')


def classify(row):
    y,p=int(row['original_label']),int(row['l05_fixed_decode_prediction'])
    votes={int(k):int(v) for k,v in json.loads(row['neighbor_votes']).items()}
    leaders=sorted(k for k,v in votes.items() if v==max(votes.values())) if votes else []
    if y==p: return 'control_agreement',leaders
    if len(leaders)!=1: return 'tied',leaders
    return ('original_plurality' if leaders[0]==y else 'model_plurality' if leaders[0]==p else 'third_plurality'),leaders


def sample(rows, seed):
    pools=defaultdict(list)
    for row in rows:
        if row['split']=='val': pools[classify(row)[0]].append(row)
    if {k:len(v) for k,v in pools.items()} != EXPECTED:
        raise ValueError('Cached population differs from preregistered five strata')
    chosen=[]
    for group,n in BUDGET.items():
        candidates=sorted(pools[group],key=lambda r:r['image_path'])
        chosen += random.Random(f'{seed}:stratum:{group}').sample(candidates,n)
    random.Random(f'{seed}:blind-order').shuffle(chosen)
    return chosen


def reference_sample(training, labels, excluded_groups, seed):
    output={}
    for label in sorted(labels):
        groups=defaultdict(list)
        for r in training[label]:
            if r['content_group'] not in excluded_groups: groups[r['content_group']].append(r)
        rng=random.Random(f'{seed}:reference:{label}')
        selected=rng.sample(sorted(groups),min(3,len(groups)))
        output[label]=[rng.choice(sorted(groups[g],key=lambda r:r['image_path'])) for g in selected]
    return output


def prepare(args):
    if args.out.exists(): raise ValueError('Output exists; refusing overwrite')
    summary=json.loads(args.summary.read_text())
    if sha(args.input)!=summary['archived_artifacts'][args.input.name]: raise ValueError('Evidence hash mismatch')
    splitrows={}
    for split in ('train','val'):
        path=args.assets/f'{split}_dev.csv'
        if sha(path)!=summary['identity']['split_hashes'][path.name]: raise ValueError('Split hash mismatch')
        with path.open() as f: splitrows[split]={r['image_path']:r for r in csv.DictReader(f)}
    mapping=args.assets/'class_to_idx.json'
    if sha(mapping)!=summary['identity']['split_hashes'][mapping.name]: raise ValueError('Mapping hash mismatch')
    with gzip.open(args.input,'rt') as f: rows=list(csv.DictReader(f))
    training=defaultdict(list)
    for r in rows:
        split=r['split']; actual=splitrows[split][r['image_path']]
        if int(actual['label'])!=int(r['original_label']) or actual['content_group']!=r['content_group']:
            raise ValueError('Evidence/split mismatch')
        if split=='train': training[int(r['original_label'])].append(r)
    chosen=sample(rows,args.seed)
    excluded={r['content_group'] for r in chosen}
    labels={int(r[k]) for r in chosen for k in ('original_label','l05_fixed_decode_prediction')}
    # Only unique plurality alternatives; tied classes are not arbitrarily broken.
    for r in chosen:
        leaders=classify(r)[1]
        if len(leaders)==1: labels.update(leaders)
    refs=reference_sample(training,labels,excluded,args.seed)
    config=json.loads(args.l05_config.read_text())
    if config['model']['input_resolution']!=384: raise ValueError('Unexpected L05 resolution')
    from PIL import Image,ImageDraw,ImageFile,ImageOps
    from clip.clip import _transform
    import numpy as np
    import torch
    torch.set_num_threads(1)
    ImageFile.LOAD_TRUNCATED_IMAGES=False
    preprocess=_transform(384)
    args.out.mkdir(parents=True)
    for d in ('images','blind','reveal','references'): (args.out/d).mkdir()
    loaded={}
    def load(row,split):
        rel=row['image_path']; path=(args.data_root/rel).resolve()
        if args.data_root.resolve() not in path.parents or not rel.startswith('train/'):
            raise ValueError('Only current official training pool images allowed')
        expected=splitrows[split][rel]['file_sha256']
        if sha(path)!=expected: raise ValueError('Image byte mismatch')
        with Image.open(path) as im: rgb=im.convert('RGB')
        loaded[rel]=expected
        return rgb
    def tile(im,size):
        canvas=Image.new('RGB',size,'#f1f1f1'); contained=ImageOps.contain(im,size)
        canvas.paste(contained,((size[0]-contained.width)//2,(size[1]-contained.height)//2))
        return canvas
    cases=[]
    for i,r in enumerate(chosen,1):
        cid=f'V{i:03d}'; original=load(r,'val'); resized=preprocess.transforms[0](original)
        crop=preprocess.transforms[1](resized)
        # Verify the displayed crop produces the exact normalized model input.
        reconstructed=preprocess.transforms[4](preprocess.transforms[3](crop))
        if not torch.equal(reconstructed,preprocess(original)): raise ValueError('Displayed input differs from L05 transform')
        crop.save(args.out/'images'/f'{cid}_center.png')
        # Original preview keeps complete field of view; local link points to full-resolution original.
        tile(original,(768,768)).save(args.out/'images'/f'{cid}_original.jpg',quality=95)
        group,leaders=classify(r)
        relevant=list(dict.fromkeys([int(r['original_label']),int(r['l05_fixed_decode_prediction'])]+(leaders if len(leaders)==1 else [])))
        cases.append(dict(case_id=cid,image_path=r['image_path'],content_group=r['content_group'],
            stratum=group,population=EXPECTED[group],sample_size=BUDGET[group],
            inclusion_probability=BUDGET[group]/EXPECTED[group],weight=EXPECTED[group]/BUDGET[group],
            original_label=int(r['original_label']),l05_fixed_decode_prediction=int(r['l05_fixed_decode_prediction']),
            unique_neighbor_label=leaders[0] if len(leaders)==1 else None,reference_classes=relevant,
            original_size=list(original.size),resized_size=list(resized.size),
            crop_box=[round((resized.width-384)/2),round((resized.height-384)/2),384,384],
            center_rgb_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
            retained_area_fraction=384*384/(resized.width*resized.height)))
    for label,rr in refs.items():
        for n,r in enumerate(rr):
            tile(load(r,'train'),(256,256)).save(args.out/'references'/f'{label:04d}_{n}.jpg',quality=95)
    # Contact sheets contain only IDs/original/center. Labels and strata are absent.
    for start in range(0,len(cases),10):
        canvas=Image.new('RGB',(1024,5*282),'white'); draw=ImageDraw.Draw(canvas)
        for j,c in enumerate(cases[start:start+10]):
            x=(j%2)*512;y=(j//2)*282;cid=c['case_id']
            draw.text((x+5,y+4),f'{cid} original | center384',fill='black')
            for k,name in enumerate(('original.jpg','center.png')):
                with Image.open(args.out/'images'/f'{cid}_{name}') as im: canvas.paste(tile(im,(256,256)),(x+k*256,y+24))
        canvas.save(args.out/'blind'/f'page_{start//10+1:02d}.jpg',quality=95)
    # Five cases per reveal sheet; 3 random independent-group references per class.
    for start in range(0,len(cases),5):
        canvas=Image.new('RGB',(1152,5*164),'white');draw=ImageDraw.Draw(canvas)
        for j,c in enumerate(cases[start:start+5]):
            y=j*164;cid=c['case_id']
            draw.text((4,y+2),f"{cid} original={c['original_label']:04d} L05={c['l05_fixed_decode_prediction']:04d} neighbor={c['unique_neighbor_label']} ; reference folder labels are unverified",fill='black')
            for k,label in enumerate(c['reference_classes']):
                for n,_ in enumerate(refs[label]):
                    x=(k*3+n)*128
                    draw.text((x,y+17),f'{label:04d} random {n+1}',fill='black')
                    with Image.open(args.out/'references'/f'{label:04d}_{n}.jpg') as im: canvas.paste(tile(im,(128,128)),(x,y+32))
        canvas.save(args.out/'reveal'/f'page_{start//5+1:02d}.jpg',quality=95)
    reference_manifest={str(k):[dict(image_path=r['image_path'],content_group=r['content_group'],file_sha256=loaded[r['image_path']]) for r in v] for k,v in refs.items()}
    write_json(args.out/'cases.json',cases)
    write_json(args.out/'reference_manifest.json',reference_manifest)
    protocol=dict(experiment='P75_VISUAL_AUDIT_200',seed=args.seed,budget=BUDGET,populations=EXPECTED,
        review_kind='assistant_visual_development_analysis_not_human_or_expert_adjudication',
        evidence_sha256=sha(args.input),summary_sha256=sha(args.summary),config_sha256=sha(args.l05_config),
        script_sha256=sha(Path(__file__)),classes_sha256=sha(mapping),
        data_root=str(args.data_root),assets=str(args.assets),l05_config=str(args.l05_config),
        case_count=len(cases),reference_images=sum(map(len,refs.values())),reference_classes=len(refs),
        seed_policy='stratum-specific seeded uniform sample; independent seeded shuffle; three uniformly sampled content groups per reference class',
        display='RGB; native OpenAI CLIP _transform(384), bicubic short-side resize and center crop; exact tensor equivalence checked for all queries; no EXIF rotation',
        decode_failures=0,confidence_displayed=False,model_loaded=False,training_executed=False,
        outputs_for_training=False,official_semantic_mapping='not found; numeric class IDs only',
        loaded_image_hashes=loaded)
    write_json(args.out/'protocol.json',protocol)
    parts=['<!doctype html><meta charset="utf-8"><title>P75 local development audit</title>',
           '<style>body{font:16px sans-serif;max-width:1200px;margin:auto}article{border-bottom:1px solid #aaa;padding:1em}img.query{max-width:47%;height:384px;object-fit:contain}img.ref{width:160px}summary{cursor:pointer}</style>',
           '<h1>Development-only visual audit</h1><p>First record visible facts from original/center, then expand labels. No confidences. Folder labels are unverified. No training export.</p>']
    for c in cases:
        cid=c['case_id']; raw=(args.data_root/c['image_path']).as_uri()
        parts += [f'<article><h2>{cid}</h2><a href="{html.escape(raw)}">Full original</a><br><img class="query" src="images/{cid}_original.jpg"><img class="query" src="images/{cid}_center.png">',
                  f'<details><summary>Reveal after recording visible facts</summary><p>Original {c["original_label"]:04d}; L05 {c["l05_fixed_decode_prediction"]:04d}; unique neighbor {c["unique_neighbor_label"]}</p>']
        for label in c['reference_classes']:
            parts += [f'<p>Class folder {label:04d}: uniform random independent groups; labels unverified</p>']
            for n,rr in enumerate(refs[label]):
                raw=(args.data_root/rr['image_path']).as_uri()
                parts += [f'<a href="{html.escape(raw)}"><img class="ref" src="references/{label:04d}_{n}.jpg"></a>']
        parts += ['</details></article>']
    (args.out/'index.html').write_text('\n'.join(parts))
    print(json.dumps({k:protocol[k] for k in ('case_count','reference_images','reference_classes','decode_failures')}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--l05-config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260929)
    prepare(p.parse_args())


if __name__=='__main__': main()
