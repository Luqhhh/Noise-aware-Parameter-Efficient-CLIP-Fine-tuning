"""Thin, audit-first node runner over existing CLIs.

This first version executes synthetic recipes only. Formal recipes remain
blocked until transitive scope auditing is implemented; an approved boolean
alone cannot unlock formal training. Node-boundary reuse is supported, not
mid-step or mid-epoch resume.
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from aegis_clip.runtime import atomic_json_dump, sha256_file

MODULES = {
    'aegis_clip.cli.train', 'aegis_clip.cli.cache_features',
    'aegis_clip.cli.prepare_final_train', 'aegis_clip.cli.infer',
}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _path(value: str, base: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base/path).resolve()


def audit_recipe(manifest_path: str | Path, repository_root: str | Path) -> dict:
    source = Path(manifest_path).resolve()
    manifest = json.loads(source.read_text())
    repo = Path(repository_root).resolve()
    base = source.parent
    code_root = repo / 'reproducibility/aegis_f1/aegis_clip'
    code_sha256 = _digest({str(p.relative_to(code_root)):sha256_file(p) for p in sorted(code_root.rglob('*.py'))})
    errors = []
    if manifest.get('schema_version') != 1: errors.append('schema_version must be 1')
    protocol = manifest.get('protocol', {})
    if protocol.get('approved') is not True or not protocol.get('decision_source'):
        errors.append('pending_user_decision: protocol and decision source required')
    budget = protocol.get('budget', {})
    nodes = manifest.get('nodes', [])
    if not nodes: errors.append('recipe needs actual nodes')
    if not isinstance(budget.get('max_nodes'), int) or budget.get('max_nodes',0)<len(nodes):
        errors.append('registered node budget missing or exceeded')
    if not isinstance(budget.get('max_wall_seconds'), (int,float)) or budget.get('max_wall_seconds',0)<=0:
        errors.append('positive wall-clock budget required')
    if manifest.get('fit_scope') != 'synthetic_dryrun':
        errors.append('formal execution blocked: transitive scope audit not implemented')
    if not manifest.get('dataset_id') or manifest.get('stage') not in {'preliminary','repechage','semifinal'}:
        errors.append('explicit stage and dataset_id required')
    output_root = _path(manifest['output_root'], base)
    state_path = output_root / 'reproduction_run.json'
    previous = json.loads(state_path.read_text()) if state_path.is_file() else {}
    seen = set(); produced = set(); normalized = []
    producers = {}; ancestors = {}
    for node in nodes:
        identifier = node.get('node_id')
        if not isinstance(identifier,str) or not identifier or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in identifier) or identifier in seen:
            raise ValueError('unique simple node_id required')
        if not set(node.get('depends_on',[])) <= seen: errors.append(f'{identifier}: dependencies must precede node')
        dependencies = set(node.get('depends_on', []))
        ancestors[identifier] = dependencies | set().union(*(ancestors.get(x, set()) for x in dependencies))
        seen.add(identifier)
        argv=node.get('argv',[])
        if len(argv)<3 or argv[0] not in {'python','python3','{python}'} or argv[1]!='-m' or argv[2] not in MODULES:
            errors.append(f'{identifier}: only registered existing Python CLIs are supported')
        if len(argv) >= 3 and node.get('operation') != argv[2]:
            errors.append(f'{identifier}: operation does not match actual CLI')
        flags = [x.split('=', 1)[0] for x in argv if isinstance(x, str) and x.startswith('--')]
        if len(flags) != len(set(flags)):
            errors.append(f'{identifier}: duplicate CLI option can shadow bound arguments')
        if any(x in argv for x in ['--overwrite','--resume','--init-checkpoint']):
            errors.append(f'{identifier}: overwrite and implicit resume/init overrides forbidden')
        if not all(isinstance(v,str) for v in argv):raise ValueError('argv must contain strings')
        if any(x.startswith(('--overwrite=', '--resume=', '--init-checkpoint=')) for x in argv):
            errors.append(f'{identifier}: alternate override syntax forbidden')
        cwd=_path(node.get('cwd','.'),repo)
        if not cwd.is_relative_to(repo) or not cwd.is_dir():errors.append(f'{identifier}: cwd must be within repository')
        inputs=[]
        for artifact in node.get('input_artifacts',[]):
            path=_path(artifact['path'],base)
            if artifact.get('stage') != manifest.get('stage') or artifact.get('scope') != manifest.get('fit_scope'):
                errors.append(f'{identifier}: input stage/scope mismatch')
            if path in produced:
                if producers[path] not in ancestors[identifier]:
                    errors.append(f'{identifier}: producer missing from dependency ancestry')
                inputs.append({'path':str(path),'producer_output':True})
                continue
            if not path.is_file():errors.append(f'{identifier}: missing input {path}')
            elif not artifact.get('sha256') or sha256_file(path)!=artifact['sha256']:
                errors.append(f'{identifier}: input hash mismatch {path}')
            inputs.append({'path':str(path),'sha256':artifact.get('sha256')})
        outputs=[_path(v,base) for v in node.get('output_artifacts',[])]
        if not outputs:errors.append(f'{identifier}: output artifacts required')
        for path in outputs:
            if not path.is_relative_to(output_root) or path==output_root or path in produced:
                errors.append(f'{identifier}: overlapping or uncontained output {path}')
            if path in [Path(x['path']) for x in inputs]:errors.append(f'{identifier}: output aliases input')
        config_path=node.get('config_path')
        owned_output_dir = None
        if config_path:
            config_path=_path(config_path,base)
            if not config_path.is_file() or sha256_file(config_path)!=node.get('config_sha256'):
                errors.append(f'{identifier}: config hash mismatch')
            if '--config' not in argv or _path(argv[argv.index('--config')+1],cwd)!=config_path:
                errors.append(f'{identifier}: argv config must match bound config')
        if len(argv)>2 and argv[2]=='aegis_clip.cli.train':
            if not config_path: errors.append(f'{identifier}: actual training config required')
            elif config_path.is_file():
                from aegis_clip.config import load_config
                config=load_config(str(config_path))
                if config['project']['stage']!=manifest.get('stage'):errors.append(f'{identifier}: config stage mismatch')
                declared={Path(x['path']) for x in inputs}
                paths=[config['data'][k] for k in ('train_csv','val_csv','class_mapping')]
                paths += [v for k,v in config['features'].items() if k in {'tensor_path','paths_path','manifest_path'} and v]
                init=config['train'].get('init_checkpoint')
                if init:paths.append(init)
                if config.get('trust',{}).get('enabled'):paths.append(config['trust']['bundle_path'])
                if config['train'].get('require_lineage_for_init_checkpoint'):
                    paths += [config['lineage'][k] for k in ('parent_train_csv', 'parent_val_csv')]
                if any(Path(p).resolve() not in declared for p in paths):errors.append(f'{identifier}: config has undeclared input assets')
                train_root=Path(config['output']['root']).resolve()/config['project']['experiment_id']/f"seed{config['project'].get('seed',42)}"
                owned_output_dir = str(train_root)
                if not train_root.is_relative_to(output_root):errors.append(f'{identifier}: training output outside run root')
                if not any(p.is_relative_to(train_root/'checkpoints') and p.suffix=='.pt' for p in outputs):
                    errors.append(f'{identifier}: training must declare an actual checkpoint output')
        if len(argv)>2 and argv[2]=='aegis_clip.cli.cache_features' and not config_path:
            errors.append(f'{identifier}: feature producer config binding required')
        if len(argv)>2 and argv[2]=='aegis_clip.cli.infer':
            declared = {Path(x['path']) for x in inputs}
            if '--checkpoint' not in argv:
                errors.append(f'{identifier}: explicit bound checkpoint required')
            for flag in ('--checkpoint', '--prior-config'):
                if flag in argv:
                    index = argv.index(flag) + 1
                    if index >= len(argv) or _path(argv[index], cwd) not in declared:
                        errors.append(f'{identifier}: undeclared {flag} input')
            if any(x.startswith('--prior-alignment-strength') for x in argv):
                errors.append(f'{identifier}: test-batch prior fitting is not a reproduction default')
        if len(argv)>2 and argv[2] in {'aegis_clip.cli.cache_features','aegis_clip.cli.infer'}:
            if '--output-dir' not in argv:
                errors.append(f'{identifier}: explicit owned output directory required')
            else:
                destination=_path(argv[argv.index('--output-dir')+1],cwd)
                owned_output_dir=str(destination)
                if not destination.is_relative_to(output_root) or not all(p.is_relative_to(destination) for p in outputs):
                    errors.append(f'{identifier}: CLI output directory outside declared ownership')
        if len(argv)>2 and argv[2]=='aegis_clip.cli.prepare_final_train':
            for flag in ('--train-csv','--val-csv','--output-csv'):
                if flag not in argv:errors.append(f'{identifier}: {flag} required')
                else:
                    actual=_path(argv[argv.index(flag)+1],cwd)
                    allowed=outputs if flag=='--output-csv' else [Path(x['path']) for x in inputs]
                    if actual not in allowed:errors.append(f'{identifier}: undeclared {flag} path')
        normalized.append({**node,'argv':[sys.executable]+argv[1:],'cwd':str(cwd),
                           'producer_code_sha256':code_sha256,
                           'config_path':str(config_path) if config_path else None,
                           'owned_output_dir':owned_output_dir,
                           'input_artifacts':inputs,'output_artifacts':[str(p) for p in outputs]})
        old = previous.get('nodes', {}).get(identifier, {})
        completed = old.get('status') == 'checks_passed' and previous.get('manifest_sha256') == sha256_file(source)
        if completed:
            bound_paths = [x['path'] for x in inputs] + ([str(config_path)] if config_path else [])
            actual = {p:sha256_file(p) for p in bound_paths if Path(p).is_file()}
            if len(actual) != len(set(bound_paths)) or old.get('fingerprint') != _digest({'node':normalized[-1],'inputs':actual}):
                errors.append(f'{identifier}: completed node input/config/code binding changed')
            if any(not Path(p).is_file() or sha256_file(p) != old.get('outputs',{}).get(p) for p in normalized[-1]['output_artifacts']):
                errors.append(f'{identifier}: completed node output changed')
        elif any(p.exists() for p in outputs) or (owned_output_dir and Path(owned_output_dir).exists()):
            errors.append(f'{identifier}: output conflict with unverified or incomplete run')
        producers.update({path: identifier for path in outputs})
        produced.update(outputs)
    return {'schema_version':1,'manifest_sha256':sha256_file(source),'output_root':str(output_root),
            'fit_scope':manifest.get('fit_scope'),'budget':budget,'nodes':normalized,'errors':errors,
            'status':'blocked' if errors else 'checks_passed','resume_granularity':'completed_node_only'}


def run_recipe(manifest_path: str | Path, repository_root: str | Path, *, execute=False, resume=False) -> dict:
    audit=audit_recipe(manifest_path,repository_root)
    if not execute:return audit
    if audit['errors']:raise ValueError('; '.join(audit['errors']))
    root=Path(audit['output_root']);root.mkdir(parents=True,exist_ok=True)
    lock=root/'.reproduce.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    state_path=root/'reproduction_run.json'
    try:
        if state_path.exists():
            if not resume:raise FileExistsError('run already exists; request verified node resume')
            state=json.loads(state_path.read_text())
            if state['manifest_sha256']!=audit['manifest_sha256']:raise ValueError('manifest changed; new run required')
        else:state={'manifest_sha256':audit['manifest_sha256'],'nodes':{},'status':'running','elapsed_seconds':0.}
        start=time.monotonic();previous_elapsed=state.get('elapsed_seconds',0.)
        for node in audit['nodes']:
            identity=node['node_id']; before={x['path']:sha256_file(x['path']) for x in node['input_artifacts']}
            if node.get('config_path'):before[node['config_path']]=sha256_file(node['config_path'])
            fingerprint=_digest({'node':node,'inputs':before})
            old=state['nodes'].get(identity)
            if old:
                if old['status']!='checks_passed':raise ValueError('incomplete node needs a new run; no implicit partial resume')
                if old['fingerprint']!=fingerprint or any(not Path(p).is_file() or sha256_file(p)!=h for p,h in old['outputs'].items()):
                    raise ValueError('completed node input/config/output changed; refuse reuse')
                continue
            if any(Path(p).exists() for p in node['output_artifacts']):raise FileExistsError('preexisting node output; refuse overwrite')
            if node.get('owned_output_dir') and Path(node['owned_output_dir']).exists():
                raise FileExistsError('owned output directory already exists; refuse partial output reuse')
            remaining=audit['budget']['max_wall_seconds']-previous_elapsed-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('registered budget exhausted')
            state['nodes'][identity]={'status':'running','fingerprint':fingerprint,'argv':node['argv'],'cwd':node['cwd'],'inputs':before}
            atomic_json_dump(state,state_path)
            env=os.environ.copy();env['PYTHONPATH']=node['cwd']
            with (root/f'{identity}.log').open('x') as log:
                process=subprocess.run(node['argv'],cwd=node['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining,check=False)
            if process.returncode:raise RuntimeError(f'{identity} exited {process.returncode}')
            if any(sha256_file(p)!=h for p,h in before.items()):raise RuntimeError('inputs changed during node execution')
            outputs={p:sha256_file(p) for p in node['output_artifacts']}
            state['nodes'][identity].update(status='checks_passed',outputs=outputs,exit_code=0)
            state['elapsed_seconds']=previous_elapsed+time.monotonic()-start
            atomic_json_dump(state,state_path)
        state['status']='checks_passed';atomic_json_dump(state,state_path);return state
    except BaseException as error:
        if 'state' in locals():
            state['status']='failed';state['error']=f'{type(error).__name__}: {error}'
            for entry in state['nodes'].values():
                if entry['status']=='running':entry['status']='failed'
            atomic_json_dump(state,state_path)
        raise
    finally:lock.unlink()
