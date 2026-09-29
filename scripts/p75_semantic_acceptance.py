#!/usr/bin/env python3
"""Select independent training content groups for whole-rule acceptance, not labels."""
import argparse,csv,json,random
from pathlib import Path
from p75_semantic_content_probe import sha,sheets

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--development',type=Path,required=True);p.add_argument('--rule',type=Path,required=True);p.add_argument('--image-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise ValueError('Output exists')
 rule=json.loads(a.rule.read_text());summary=json.loads((a.development/'development_summary.json').read_text())
 if rule['descriptions_sha256']!=summary['descriptions_sha256'] or sha(a.development/'training_scores.csv')!=summary['scores_sha256']:raise ValueError('Development changed')
 with (a.development/'training_scores.csv').open() as f:rows=list(csv.DictReader(f))
 for r in rows:
  for k in ['sB','sA','sN','margin']:r[k]=float(r[k])
 threshold=rule['minimum_margin'];settings=rule['acceptance'];rng=random.Random(settings['seed'])
 excluded={r['content_group'] for r in json.loads((a.development/'development_cases.json').read_text())}
 used=set(excluded); chosen=[]
 def take(pool,n,role):
  added=0
  for r in pool:
   if r['content_group'] in used:continue
   chosen.append(dict(r,acceptance_role=role,selected=r['margin']>=threshold));used.add(r['content_group']);added+=1
   if added==n:break
  if added!=n:raise ValueError('Insufficient distinct groups for '+role)
 selected=sorted([r for r in rows if r['margin']>=threshold],key=lambda r:r['image_path']);rng.shuffle(selected)
 take(selected,settings['selected_random'],'selected_random')
 take(sorted(selected,key=lambda r:(r['margin'],r['image_path'])),settings['selected_boundary'],'selected_boundary')
 # Independent likely-protected mixed/drawing/specimen cases with stronger A than B,
 # ranked by closeness to selection. Inspection may also find non-target misses.
 protected=sorted([r for r in rows if r['margin']<threshold and r['sA']>r['sB'] and r['sA']>=.27],key=lambda r:(-r['margin'],r['image_path']))
 take(protected,settings['protected_challenges'],'protected_challenge')
 rng.shuffle(chosen)
 for i,r in enumerate(chosen,1):r['review_id']=f'A{i:02d}'
 a.out.mkdir(parents=True)
 (a.out/'cases.json').write_text(json.dumps(chosen,indent=2)+'\n')
 (a.out/'protocol.json').write_text(json.dumps(dict(rule_sha256=sha(a.rule),descriptions_sha256=rule['descriptions_sha256'],development_cases_sha256=sha(a.development/'development_cases.json'),training_selected_count=len(selected),acceptance_rows=len(chosen),development_group_overlap=len(excluded&{r['content_group'] for r in chosen}),validation_used=False,manual_training_labels=False),indent=2)+'\n')
 sheets(chosen,a.image_root,a.out,'acceptance')
 print((a.out/'protocol.json').read_text())
if __name__=='__main__':main()
