"""Run the approved fixed head/visual candidates sequentially; never upload.

Default is read-only audit. An explicit --execute starts new segments once.
The real feature cache must already be complete. No resume/overwrite paths.
Platform feedback is required before any combination C.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'reproducibility/aegis_f1'))
from aegis_clip.prelim75 import load_plan
from aegis_clip.runtime import atomic_json_dump, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT/'configs/prelim75_v2.yaml'))
    parser.add_argument('--segment', choices=['head', 'visual'], required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = load_plan(args.config)
    output = Path(plan['output'])
    candidates = ['H0', 'H1'] if args.segment == 'head' else ['V0', 'V1']
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip() != 'main':
        raise ValueError('Approved experiment work remains on main')
    if not (output/'cache/manifest.json').exists():
        raise ValueError('Finish and verify the real composite cache first')
    cache = json.loads((output/'cache/manifest.json').read_text())
    if cache['status'] != 'complete' or cache['parent_sha256'] != plan['parent_sha256']:
        raise ValueError('Incomplete/mismatched cache')
    if args.segment == 'visual':
        for candidate in ('H0', 'H1'):
            diag = output/candidate/'diagnostic/result.json'
            if not diag.exists():
                raise ValueError('Complete H0/H1 before visual continuation')
            state = json.loads(diag.read_text())
            if not state.get('engineering_stop') and not (output/candidate/'candidate_report.json').exists():
                raise ValueError('Finish each eligible head submission first')
    for name in candidates:
        if (output/name).exists():
            raise FileExistsError(f'Existing candidate directory: {name}; do not overwrite/resume')
    registration = {
        'plan': plan, 'segment':args.segment, 'candidates':candidates,
        'user_authorization':'按顺序执行', 'execute':args.execute,
        'cache_manifest_sha256':sha256_file(output/'cache/manifest.json'),
        'status':'audit_passed', 'new_platform_upload':False,
        'platform_feedback_required_for_C':True,
    }
    print(json.dumps(registration,ensure_ascii=False,indent=2),flush=True)
    if not args.execute:
        return
    subprocess.run(['git','fetch','origin'],cwd=ROOT,check=True)
    registration['git_status'] = subprocess.check_output(['git','status','--short','--branch'],cwd=ROOT,text=True)
    registration['branch_refs'] = subprocess.check_output(['git','for-each-ref','--format=%(refname:short) %(objectname) %(subject)','refs/heads','refs/remotes'],cwd=ROOT,text=True)
    registration['recent_history'] = subprocess.check_output(['git','log','-8','--oneline'],cwd=ROOT,text=True)
    registration['origin_main'] = subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip()
    if subprocess.run(['git','merge-base','--is-ancestor','origin/main','HEAD'],cwd=ROOT).returncode:
        raise RuntimeError('Origin/main changed: synchronize before starting the new segment')
    snapshot = output/f'training_source_{args.segment}'
    snapshot.mkdir(exist_ok=False)
    package = ROOT/'reproducibility/aegis_f1/aegis_clip'
    for src in package.rglob('*.py'):
        dst = snapshot/'aegis_clip'/src.relative_to(package)
        dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    shutil.copy2(__file__,snapshot/'run_prelim75_queue.py')
    registration['training_source_sha256'] = {str(p.relative_to(snapshot)):sha256_file(p) for p in snapshot.rglob('*.py')}
    atomic_json_dump(registration,output/f'queue_registration_{args.segment}.json')
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(snapshot)
    status_path = output/f'queue_status_{args.segment}.json'
    history = []
    def run(module, name, log_name):
        command = [sys.executable,'-u','-m',module,'--config',plan['_config_path'],'--candidate',name]
        begin = time.monotonic()
        atomic_json_dump({'status':'running','candidate':name,'module':module,'completed_steps':history},status_path)
        with (output/log_name).open('w') as log:
            result = subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        history.append({'command':command,'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-begin,'log':log_name})
        if result.returncode:
            atomic_json_dump({'status':'failed','candidate':name,'module':module,'completed_steps':history},status_path)
            raise RuntimeError(f'Queue stopped: {module} {name}; inspect {log_name}')
    if not (output/'P_diagnostic/result.json').exists():
        run('aegis_clip.cli.evaluate_prelim75','P','P_diagnostic.log')
    for name in candidates:
        run('aegis_clip.cli.train_shared_head_refit' if args.segment=='head' else 'aegis_clip.cli.train_global_local_joint',name,f'{name}_training.log')
        run('aegis_clip.cli.evaluate_prelim75',name,f'{name}_diagnostic.log')
        diagnostic = json.loads((output/name/'diagnostic/result.json').read_text())
        if diagnostic['engineering_stop']:
            atomic_json_dump({'status':'engineering_stopped_no_submission','diagnostic':diagnostic,'online_accuracy':None},output/name/'status.json')
        else:
            run('aegis_clip.cli.infer_prelim75',name,f'{name}_delivery.log')
        atomic_json_dump({'status':'running','candidate_complete':name,'completed_steps':history},status_path)
    atomic_json_dump({'status':'complete_pending_platform_feedback','candidates':candidates,'completed_steps':history,
                      'combination_C_not_started':True},status_path)
    print(f'{args.segment}: complete; no automatic upload or combination',flush=True)

if __name__ == '__main__':
    main()
