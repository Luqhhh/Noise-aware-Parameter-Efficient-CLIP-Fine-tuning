#!/usr/bin/env python3
"""Verify archived pair results against all rows and original local artifacts."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from run_p75_semantic_pair import sha
from p75_semantic_pair_report import paired


def verify(archive):
    manifest=json.loads((archive/'artifact_manifest.json').read_text())
    for name,expected in manifest['archived_files'].items():
        if sha(archive/name)!=expected: raise ValueError(f'Archived artifact changed: {name}')
    run=json.loads((archive/'run_manifest.json').read_text())
    for path,expected in run['files'].items():
        if sha(path)!=expected: raise ValueError(f'Runtime/source/input changed: {path}')
    report=json.loads((archive/'report.json').read_text())
    execution=json.loads((archive/'execution_status.json').read_text())
    used=sum(e['seconds'] for e in execution['history'])
    if abs(used-execution['used_seconds'])>1e-6 or used>run['budget_seconds'] or execution['current'] is not None:
        raise ValueError('Budget/state reconciliation mismatch')
    if str(execution['matched_epoch']) not in report['epochs']:
        raise ValueError('Delivered epoch lacks paired report')
    if execution.get('closure_source_sha256'):
        root=Path(__file__).resolve().parents[1]
        if sha(root/'scripts/close_p75_semantic_pair_e4.py')!=execution['closure_source_sha256']:
            raise ValueError('Closure implementation changed')
        recovered=[e for e in execution['history'] if e.get('recovered_after_supervisor_exit')]
        if len(recovered)!=1 or recovered[0]['returncode'] is not None:
            raise ValueError('Unknown child exit code must remain unknown')
        if execution['matched_epoch']!=4 or report['primary_complete']:
            raise ValueError('Four-epoch closure misreported as primary completion')
    for epoch,value in report['epochs'].items():
        with gzip.open(archive/f'paired_epoch_{epoch}.csv.gz','rt') as f:
            rows=list(csv.DictReader(f))
        y,a,b,l05=[np.array([int(r[k]) for r in rows]) for k in ('original_label','control','masked','l05')]
        mask=np.array([r['selected_proxy']=='True' for r in rows])
        if len(rows)!=14880 or mask.sum()!=132: raise ValueError('Validation coverage changed')
        for group,selected in [('all',np.ones(len(rows),dtype=bool)),('selected_proxy',mask),('remaining_proxy',~mask)]:
            if paired(y,a,b,selected)!=value['same_epoch_pair'][group]: raise ValueError('Pair metrics mismatch')
            for arm,pred in [('control',a),('masked',b)]:
                if paired(y,l05,pred,selected)!=value['versus_l05'][arm][group]: raise ValueError('L05 comparison mismatch')
        for arm,binding in value['binding'].items():
            for kind in ('checkpoint','cache'):
                if sha(binding[kind])!=binding[kind+'_sha256']: raise ValueError(f'{arm} {kind} changed')
        for order in value['training_audit']['orders']:
            if order['rows']!=133815 or order['successful_updates']!=order['epoch']*131:
                raise ValueError('Incomplete training epoch')
    delivery=manifest['delivery']
    for name in ('pred_results.csv','submission.zip'):
        if sha(Path(delivery['directory'])/name)!=delivery['sha256'][name]: raise ValueError('Submission changed')
    config=json.loads(Path(delivery['config']).read_text())
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/'scripts/check_submission.py'),
        '--test_dir',config['data']['test_root'],'--class-mapping',config['data']['class_mapping'],
        '--csv',str(Path(delivery['directory'])/'pred_results.csv'),
        '--zip',str(Path(delivery['directory'])/'submission.zip')],check=True)
    print(json.dumps(dict(verified=True,epochs=list(report['epochs']),primary_complete=report['primary_complete'],
        matched_rows=14880,submission_rows=37444),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    verify(p.parse_args().archive)
