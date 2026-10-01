"""Run the two authorized follow-ups after verified 768 FULL integration."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from run_serial_continuations import now, read, write, sha, git, gpu_pids

ROUTES=['512_control','crt768_dev']


def environment(root):
    return dict(os.environ,PYTHONPATH=str(root/'reproducibility/aegis_f1'),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')


def validate_job(job):
    root=Path(job['worktree'])
    if git(root,'branch','--show-current')!=job['branch'] or git(root,'rev-parse','HEAD')!=job['prepared_commit']:
        raise ValueError('Prepared job branch/commit changed')
    if (sha(job['config'])!=job['config_sha256'] or sha(job['prepared'])!=job['prepared_sha256']
            or sha(root/'scripts/run_v1_post768_job.py')!=job['worker_sha256']):
        raise ValueError('Prepared protocol/source changed')
    command=[sys.executable,'scripts/run_v1_post768_job.py','--config',job['config'],'--preflight']
    fresh=json.loads(subprocess.check_output(command,cwd=root,env=environment(root),text=True))
    if fresh['binding']!=read(job['prepared'])['binding']:
        raise ValueError('Parent/data/target binding changed after preflight')
    if any(fresh[key]!=job[key] for key in ('experiment_id','route','output')):
        raise ValueError('Job identity/output differs from the prepared config')
    if Path(job['output']).exists():raise FileExistsError('Job output exists; automatic retry is disabled')
    return root


def archive(job,main):
    root=Path(job['worktree']);output=Path(job['output']);report=read(output/'report.json')
    if report['status']!='completed_verified_delivery' or read(output/'status.json')['status']!='completed':
        raise ValueError('Job did not deliver verified packages')
    git(root,'fetch','origin');git(root,'merge','origin/main','-m','Sync main before post-768 delivery record')
    result=root/'results'/job['name'];write(result/'delivery_report.json',report)
    if job['route']=='512_control':
        write(result/'test_bias_report.json',read(output/'test_bias_report.json'))
    else:
        for arm in ('unbalanced','balanced'):
            write(result/(arm+'_test_bias_report.json'),read(output/arm/'test_bias_report.json'))
    doc=root/'docs'/(job['name']+'.md')
    text=doc.read_text()+'\n## 已验证交付\n\n'
    text+=f"{now()} 完成，用时{report['elapsed_seconds']/60:.2f}分钟。原始标签诊断不证明平台改善；平台分待回填。\n\n"
    packages=report['packages'] if job['route']=='512_control' else {
        arm+'/'+kind:package for arm,data in report['packages'].items() for kind,package in data['packages'].items()}
    if job['route']=='crt768_dev':
        for arm,m in report['metrics'].items():
            p=report['parent_to_head'][arm]
            text+=f"- {arm}: macro/micro {m['macro']*100:.4f}% / {m['micro']*100:.4f}%；相对原512中心视图修正{p['corrections']}、退化{p['regressions']}、净{p['net_correct']}；tail75 macro {m['tail75_macro']*100:.4f}%。\n"
        p=report['unbalanced_to_balanced']
        text+=f"\n均衡相对同768普通头修正{p['corrections']}、退化{p['regressions']}、净{p['net_correct']}；尾部净{p['tail75']['net_correct']}。不自动追加full训练。\n\n"
    for name,package in packages.items():
        text+=f"- {name}: `{package['zip']}`；SHA `{package['zip_sha256']}`；37,444行，9项校验通过，独立NumPy预测与CSV/ZIP一致。\n"
    text+=f"\n[完整指标与血缘](../results/{job['name']}/delivery_report.json)。\n"
    doc.write_text(text)
    current=root/'docs/current_execution_plan.md';text=current.read_text()
    marker='## 已授权后续串行队列：V1_POST768_SERIAL_20261001'
    section=f"## 已交付后续检查点：{job['experiment_id']}\n\n固定流程完成且CSV/ZIP校验通过；平台分未知，现役未替换。见[{job['experiment_id']}]({job['name']}.md)。\n\n"
    if marker not in text:raise ValueError('Current execution queue marker changed')
    current.write_text(text.replace(marker,section+marker,1))
    git(root,'add',str(result.relative_to(root)),str(doc.relative_to(root)),'docs/current_execution_plan.md')
    git(root,'diff','--cached','--check')
    git(root,'commit','-m',f"Deliver verified {job['experiment_id']} post-768 packages")
    proposal=git(root,'rev-parse','HEAD');git(root,'push','-u','origin',job['branch'])
    if git(main,'branch','--show-current')!='main':raise ValueError('Integration root changed branch')
    git(main,'pull','--rebase','--autostash','origin','main')
    git(main,'merge','--no-ff',job['branch'],'-m',f"Integrate verified {job['experiment_id']} delivery")
    subprocess.run([sys.executable,'-m','pytest','tests/test_v1_post768_jobs.py',
        'reproducibility/aegis_f1/tests/test_v1_preprojection_test_bias.py','-q'],cwd=main,env=environment(main),check=True)
    git(main,'diff','--check')
    if read(main/'results'/job['name']/'delivery_report.json')!=report:raise ValueError('Integrated results differ')
    git(main,'push','origin','main')
    return dict(experiment_id=job['experiment_id'],status='completed_integrated',proposal_commit=proposal,
        main_commit=git(main,'rev-parse','HEAD'),branch_pushed=True,main_pushed=True,report=str(result/'delivery_report.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True,type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required')
    cfg=read(args.queue);output=Path(cfg['controller_output']);output.mkdir(parents=True,exist_ok=True)
    status_path=output/'status.json'
    if status_path.exists():raise FileExistsError('Fresh controller required; no retry')
    if [job['route'] for job in cfg['jobs']]!=ROUTES or [job['max_minutes'] for job in cfg['jobs']]!=[60,90]:
        raise ValueError('Only the authorized fixed pair and budgets are supported')
    state=dict(experiment_id='V1_POST768_SERIAL_20261001',status='waiting_dependency',runner_pid=os.getpid(),
        started_at=now(),jobs=[],automatic_retry=False,automatic_full_training=False,platform_upload=False,
        dependency=cfg['dependency_completion'],queue_sha256=sha(args.queue))
    write(status_path,state);child=None
    try:
        for job in cfg['jobs']:validate_job(job)
        while True:
            completed=read(cfg['dependency_completion']);training=read(cfg['dependency_training'])
            if completed['status']=='failed' or training['status']=='failed':
                raise RuntimeError('Required 768 FULL training/delivery/integration failed; no follow-up started')
            if completed['status']=='completed':break
            if training['status'] not in ('completed','failed') and not Path(f"/proc/{training['runner_pid']}").exists():
                raise RuntimeError('Required training runner disappeared')
            state.update(status='waiting_dependency',heartbeat_at=now(),dependency_stage=completed.get('stage'),
                dependency_training_progress=training.get('training_progress'))
            write(status_path,state);time.sleep(30)
        for job in cfg['jobs']:
            root=validate_job(job)
            idle_since=None
            while True:
                utilization=int(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu',
                    '--format=csv,noheader,nounits'],text=True).strip())
                if not gpu_pids() and utilization<=5:
                    if idle_since is None:idle_since=time.monotonic()
                    if time.monotonic()-idle_since>=30:break
                else:idle_since=None
                state.update(status='waiting_local_resource',current_experiment=job['experiment_id'],heartbeat_at=now())
                write(status_path,state);time.sleep(15)
            command=[sys.executable,'-u','scripts/run_v1_post768_job.py','--config',job['config'],
                '--prepared',job['prepared'],'--execute']
            started=time.monotonic()
            with (output/(job['name']+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=root,env=environment(root),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:
                    elapsed=time.monotonic()-started
                    if elapsed>job['max_minutes']*60:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        status=Path(job['output'])/'status.json'
                        if status.exists():
                            failed=read(status);failed.update(status='failed',error='Fixed queue wall-clock budget exceeded')
                            write(status,failed)
                        raise TimeoutError(f"{job['experiment_id']} exceeded its fixed wall-clock budget; partial outputs retained")
                    state.update(status='running',current_experiment=job['experiment_id'],child_pid=child.pid,
                        elapsed_seconds=elapsed,heartbeat_at=now(),command=shlex.join(command))
                    write(status_path,state);time.sleep(15)
                if child.returncode:raise RuntimeError(f"{job['experiment_id']} failed with code {child.returncode}; no retry")
            child=None
            state.update(status='integrating',heartbeat_at=now());write(status_path,state)
            state['jobs'].append(archive(job,Path(cfg['integration_root'])))
            write(status_path,state)
        state.update(status='completed',completed_at=now(),current_experiment=None)
        write(status_path,state)
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=10)
        state.update(status='failed',error=str(error),failed_at=now());write(status_path,state);raise


if __name__=='__main__':main()
