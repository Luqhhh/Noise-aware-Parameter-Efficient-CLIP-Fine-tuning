#!/usr/bin/env python3
"""Apply an accepted frozen content rule; only automatic scores determine weights."""
import argparse,csv,gzip,json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from p75_semantic_content_probe import sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['stage','development','acceptance','rule','evidence','p0-summary','out']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():raise ValueError('Output exists')
 rule=json.loads(a.rule.read_text());accept=json.loads((a.acceptance/'acceptance_report.json').read_text());protocol=json.loads((a.acceptance/'protocol.json').read_text());dev=json.loads((a.development/'development_summary.json').read_text());meta=json.loads((a.stage/'features/manifest.json').read_text());p0=json.loads(a.p0_summary.read_text())
 if accept['status']!='passed_limited_content_acceptance' or protocol['rule_sha256']!=sha(a.rule) or protocol['descriptions_sha256']!=rule['descriptions_sha256']:raise ValueError('Acceptance/binding missing')
 for name,h in p0['identity']['split_hashes'].items():
  if (a.stage/name).is_file() and sha(a.stage/name)!=h:raise ValueError('Stage identity changed')
 if sha(a.evidence)!=p0['archived_artifacts'][a.evidence.name] or sha(a.stage/'features/features.pt')!=meta['tensor_sha256'] or sha(a.development/'text_features.pt')!=dev['text_sha256']:raise ValueError('Bound cache changed')
 torch.set_num_threads(4)
 text=torch.load(a.development/'text_features.pt',map_location='cpu',weights_only=True)
 bank=torch.load(a.stage/'features/features.pt',map_location='cpu',weights_only=True)
 paths=json.loads((a.stage/'features/image_paths.json').read_text());lookup={v:i for i,v in enumerate(paths)}
 spec=json.loads((Path(__file__).resolve().parents[1]/'configs/p75_semantic_content_probe_20260929/development.json').read_text())
 if sha(Path(__file__).resolve().parents[1]/'configs/p75_semantic_content_probe_20260929/development.json')!=rule['descriptions_sha256']:raise ValueError('Text descriptions changed')
 sim=F.normalize(bank,dim=1)@F.normalize(text,dim=1).T
 b,aa=len(spec['B']),len(spec['A']);sb=sim[:,:b].max(1).values;sa=sim[:,b:b+aa].max(1).values;sn,n=sim[:,b+aa:].max(1);margin=sn-torch.maximum(sb,sa)
 evidence={}
 with gzip.open(a.evidence,'rt') as f:
  for r in csv.DictReader(f):evidence[r['image_path']]=r
 rows=[];stats={};classes=[Counter() for _ in range(p0['identity']['num_classes'])];probabilities={};types=Counter();groups=set()
 for split in ['train','val']:
  with (a.stage/(split+'_dev.csv')).open() as f:source=list(csv.DictReader(f))
  for r in source:
   idx=lookup[r['image_path'].removeprefix('train/')];selected=bool(margin[idx]>=rule['minimum_margin']); y=int(r['label']); e=evidence[r['image_path']]
   if e['split']!=split or int(e['original_label'])!=y or e['content_group']!=r['content_group']:raise ValueError('P0 alignment mismatch')
   pred=int(e['l05_fixed_decode_prediction'] if split=='val' else e['l05_prediction']);py=float(e['center_label_probability']);key=split+('_selected' if selected else '_remaining')
   c=stats.setdefault(key,Counter());c['rows']+=1;c['original_label_errors']+=int(pred!=y);c['sqrt_py_sum']+=py**.5;c['logit_gradient_l1_sum']+=2*(py**.5)*(1-py);c['py_lt_001']+=int(py<.01);c['py_ge_07']+=int(py>=.7);probabilities.setdefault(key,[]).append(py)
   if split=='train':
    classes[y]['raw']+=1;classes[y]['selected']+=int(selected)
    if selected:types[spec['N'][int(n[idx])]]+=1;groups.add(r['content_group'])
   rows.append(dict(split=split,image_path=r['image_path'],label=y,content_group=r['content_group'],file_sha256=r['file_sha256'],selected=selected,sB=float(sb[idx]),sA=float(sa[idx]),sN=float(sn[idx]),margin=float(margin[idx]),non_target_type=spec['N'][int(n[idx])]))
 assert sum(c['rows'] for k,c in stats.items() if k.startswith('val_'))==p0['identity']['validation_samples']
 assert sum(c['original_label_errors'] for k,c in stats.items() if k.startswith('val_'))==p0['baseline_errors']
 if any(c['raw'] and c['raw']==c['selected'] for c in classes):raise ValueError('Rule emptied a training class')
 # Recomputing the matrix must reproduce the already accepted training membership.
 with (a.development/'training_scores.csv').open() as f:old={r['image_path']:float(r['margin'])>=rule['minimum_margin'] for r in csv.DictReader(f)}
 assert all(r['selected']==old[r['image_path']] for r in rows if r['split']=='train')
 for key,c in stats.items():
  c['mean_sqrt_py']=c['sqrt_py_sum']/c['rows'];c['py_quantiles']=np.quantile(probabilities[key],[0,.25,.5,.75,1]).tolist()
  split=key.split('_')[0];total=sum(v['logit_gradient_l1_sum'] for k,v in stats.items() if k.startswith(split+'_'));c['final_logit_l1_share']=c['logit_gradient_l1_sum']/total
 a.out.mkdir(parents=True)
 with (a.out/'selection.jsonl').open('w') as f:
  for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
 with (a.out/'class_coverage.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['class','raw','selected','remaining'],lineterminator='\n');w.writeheader()
  for i,c in enumerate(classes):w.writerow(dict({'class':i},raw=c['raw'],selected=c['selected'],remaining=c['raw']-c['selected']))
 report=dict(status='accepted_frozen_proxy_table_not_training_result',rule_sha256=sha(a.rule),descriptions_sha256=rule['descriptions_sha256'],acceptance_sha256=sha(a.acceptance/'acceptance_report.json'),selection_sha256=sha(a.out/'selection.jsonl'),feature_manifest_sha256=sha(a.stage/'features/manifest.json'),text_sha256=dev['text_sha256'],evidence_sha256=sha(a.evidence),groups=stats,selected_training_classes=sum(c['selected']>0 for c in classes),selected_training_content_groups=len(groups),emptied_training_classes=[],min_remaining_per_class=min(c['raw']-c['selected'] for c in classes),training_selected_types=dict(types),training_started=False,manual_case_flags_used_for_selection=False,limitations=['Proxy not confirmed open-set labels; unselected set remains mixed.','Final center snapshot only; no historical training gradient or harmful parameter-gradient claim.','Logit-gradient L1 is 2*sqrt(py)*(1-py), not parameter-gradient norm.','Limited training-only visual acceptance does not prove full-population precision or platform gain.','Training control/masked pair has not been executed.'])
 (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
