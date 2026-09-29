#!/usr/bin/env python3
"""Verify the released semantic proxy table and reuse of the original mask adapter."""
import argparse,gzip,hashlib,json,tempfile
from pathlib import Path
import torch
from p75_text_page_runtime import TextPageState


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--image-root',type=Path);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];m=json.loads((a.archive/'manifest.json').read_text())
 for name,h in m['files'].items():
  if sha(a.archive/name)!=h:raise ValueError('Changed artifact '+name)
 for name,h in m['sources'].items():
  if sha(root/name)!=h:raise ValueError('Changed source '+name)
 rule=json.loads((root/'configs/p75_semantic_content_probe_20260929/frozen_rule.json').read_text());summary=json.loads((a.archive/'frozen_summary.json').read_text())
 with gzip.open(a.archive/'frozen_selection.jsonl.gz','rt') as f:rows=[json.loads(s) for s in f]
 if len({r['image_path'] for r in rows})!=len(rows):raise ValueError('Repeated table paths')
 for r in rows:
  if abs(r['margin']-(r['sN']-max(r['sB'],r['sA'])))>1e-7 or r['selected']!=(r['margin']>=rule['minimum_margin']):raise ValueError('Nonautomatic selection')
 train=[r for r in rows if r['split']=='train']
 for split in ['train','val']:
  for selected,name in [(True,'selected'),(False,'remaining')]:
   if sum(r['split']==split and r['selected']==selected for r in rows)!=summary['groups'][split+'_'+name]['rows']:raise ValueError('Group mismatch')
 dev=json.loads((a.archive/'development_development_cases.json').read_text());cases=json.loads((a.archive/'acceptance_cases.json').read_text())
 if {r['content_group'] for r in dev}&{r['content_group'] for r in cases}:raise ValueError('Development leakage into acceptance')
 if len({r['content_group'] for r in cases})!=len(cases):raise ValueError('Repeated acceptance groups')
 if a.image_root:
  for r in json.loads((a.archive/'review_image_identity.json').read_text()):
   if sha(a.image_root/r['image_path'])!=r['sha256']:raise ValueError('Reviewed image changed')
 with tempfile.TemporaryDirectory(prefix='p75_semantic_mask_verify_') as temporary:
  table=Path(temporary)/'selection.jsonl'
  with gzip.open(a.archive/'frozen_selection.jsonl.gz','rb') as f:table.write_bytes(f.read())
  if sha(table)!=summary['selection_sha256']:raise ValueError('Table byte hash mismatch')
  cfg={'loss':{'text_page_mask':{'enabled':True,'selection':str(table),'sha256':sha(table)},'mixup_probability':0},'trust':{'enabled':False}}
  state=TextPageState(cfg,[r['image_path'].removeprefix('train/') for r in train],[r['label'] for r in train],torch.device('cpu'))
  if int(state.selected.sum())!=summary['groups']['train_selected']['rows']:raise ValueError('Runtime differs')
 print(json.dumps(dict(status='passed',table_rows=len(rows),train_selected=int(state.selected.sum()),validation_selected=summary['groups']['val_selected']['rows'],acceptance_cases=len(cases),development_overlap=0,review_image_bytes_verified=bool(a.image_root),existing_mask_adapter_loaded=True,training_started=False),indent=2))

if __name__=='__main__':main()
