#!/usr/bin/env python3
"""CPU-only cache compatibility check. No text encoding, content scores or weights."""
import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import random
import time
import torch
import torch.nn.functional as F
from PIL import Image
import clip
from clip.clip import _MODELS


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();require(not a.out.exists(),'Output exists')
    start=time.monotonic();torch.set_num_threads(4)
    base=a.stage/'features';meta=json.loads((base/'manifest.json').read_text())
    binding=meta['rematch_binding'];dataset=json.loads((a.stage/'dataset_manifest.json').read_text())
    require(binding['stage']=='repechage' and binding['data_version']=='20260921','Wrong stage')
    require(sha(a.stage/'dataset_manifest.json')==binding['dataset_manifest_sha256'],'Dataset manifest changed')
    require(sha(a.stage/'class_to_idx.json')==binding['class_mapping_sha256'],'Class mapping changed')
    for name,h in dataset['files'].items(): require(sha(a.stage/name)==h,'Dataset asset changed: '+name)
    require(sha(a.checkpoint)==binding['official_checkpoint_sha256']==_MODELS['ViT-B/32'].split('/')[-2],'Not the bound official checkpoint')
    require(sha(base/'features.pt')==meta['tensor_sha256'],'Feature tensor changed')
    require(sha(base/'image_paths.json')==meta['paths_file_sha256'],'Feature path index changed')
    require(meta['backbone']=='ViT-B/32' and meta['pretrained']=='openai' and meta['normalized'] is True,'Cache model metadata mismatch')
    require(meta['external_data'] is False and meta['test_data_used'] is False,'Invalid cache data scope')
    require(binding['encoder_precision']=='float32' and binding['feature_precision']=='float32' and binding['autocast'] is False,'Cache precision mismatch')
    paths=json.loads((base/'image_paths.json').read_text())
    with (a.stage/'full_train.csv').open() as f: full=list(csv.DictReader(f))
    require(paths==[r['image_path'].removeprefix('train/') for r in full],'Cache order differs from full training pool')
    require(len(paths)==len(set(paths))==dataset['train_samples'],'Cache count/uniqueness mismatch')
    bank=torch.load(base/'features.pt',map_location='cpu',weights_only=True)
    require(bank.dtype==torch.float32 and tuple(bank.shape)==(len(paths),512),'Cache tensor shape/dtype mismatch')
    require(bool(torch.isfinite(bank).all()),'Nonfinite cache')
    norms=bank.norm(dim=1);require(float((norms-1).abs().max())<1e-5,'Cache is not normalized')
    # Explicit local path: loading cannot download an alternative model.
    model,preprocess=clip.load(str(a.checkpoint),device='cpu',jit=False)
    model.eval().float().requires_grad_(False)
    require(model.visual.input_resolution==224,'Unexpected official input resolution')
    require(tuple(model.visual.proj.shape)==(768,512) and tuple(model.text_projection.shape)==(512,512),'Incompatible image/text projection dimensions')
    with (a.stage/'train_dev.csv').open() as f: train=sorted(csv.DictReader(f),key=lambda r:r['image_path'])
    chosen=random.Random(20260929).sample(train,4)
    index={p:i for i,p in enumerate(paths)};checks=[]
    for row in chosen:
        path=Path(dataset['train_root'])/row['image_path'].removeprefix('train/')
        require(sha(path)==row['file_sha256'],'Probe image changed')
        with Image.open(path) as image: x=preprocess(image.convert('RGB')).unsqueeze(0)
        with torch.no_grad():v=F.normalize(model.encode_image(x).float(),dim=1)[0]
        cached=bank[index[row['image_path'].removeprefix('train/')]]
        cosine=float(F.cosine_similarity(v[None],cached[None]));error=float((v-cached).abs().max())
        print(row['image_path'], 'cosine',cosine,'max_abs_error',error,flush=True)
        require(cosine>=.99999 and float((v-cached).norm())<=.0045,'Official image feature does not match cache')
        checks.append(dict(image_path=row['image_path'],file_sha256=row['file_sha256'],cosine=cosine,max_abs_error=error,l2_error=float((v-cached).norm())))
    source=Path(inspect.getfile(type(model)))
    report=dict(status='cache_compatibility_passed',rule_approval='user_confirmed_specific_use_20260929',text_encoder_executed=False,
        content_scores_created=False,training_weights_created=False,gpu_used=False,training_started=False,
        numeric_check=dict(min_cosine=.99999,max_l2_error=.0045,note='CPU/CUDA values are not bitwise identical. Initial coordinate 1e-4 and L2 .001 checks were unnecessarily stricter than cosine>=.99999; corrected L2 bound to sqrt(2*(1-.99999)), rounded upward to .0045. These tolerances check alignment, not detector accuracy; retain actual differences.'),feature_shape=list(bank.shape),feature_dtype=str(bank.dtype),norm_min=float(norms.min()),norm_max=float(norms.max()),
        visual_projection_shape=list(model.visual.proj.shape),text_projection_shape=list(model.text_projection.shape),
        official_checkpoint_sha256=sha(a.checkpoint),feature_tensor_sha256=meta['tensor_sha256'],
        feature_manifest_sha256=sha(base/'manifest.json'),dataset_manifest_sha256=sha(a.stage/'dataset_manifest.json'),
        cache_artifact_scope=meta.get('artifact_scope'),scope_note='Generic unverified field is retained; explicit current-stage binding and numerical probes checked here, not overwritten.',
        source_hashes={'script':sha(__file__),'installed_clip_model.py':sha(source)},probe_seed=20260929,probes=checks,
        limitation='Four image-only numerical probes support cache compatibility, not semantic detector quality or rule permission. Cached 224px center crops omit full-image content.',
        elapsed_seconds=time.monotonic()-start)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
