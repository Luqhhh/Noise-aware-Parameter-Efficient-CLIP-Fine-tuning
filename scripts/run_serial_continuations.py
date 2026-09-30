"""User-authorized local serial DEV runs and evidence-gated FULL runs, through delivery."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
MAIN=Path('/home/lux1/noise')


def now():return datetime.now(timezone.utc).isoformat()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root,*args):
    return subprocess.check_output(['git',*args],cwd=root,text=True,stderr=subprocess.STDOUT).strip()


def full_support(report):
    """Conservative automatic funding gate, not a candidate validity/local-score veto.

    Both predetermined exports must support whole-set and tail improvements;
    few-support classes must not regress. Raw and EMA trajectory must agree.
    Mixed evidence retains both DEV artifacts for review and does not spend FULL compute.
    """
    reasons=[]
    if report.get('status')!='development_complete' or report.get('epochs')!=4:
        return dict(eligible=False,status='evidence_requires_review',reasons=['incomplete DEV'],assessment='')
    for policy in ('last_ema','ema_swa_2_4'):
        paired=report['paired'][policy]['slices']
        full,tail,few=paired['all'],paired['tail'],paired['few_support']
        if full['net']<=0 or full['delta_macro_pp']<=0:reasons.append(policy+': whole-set evidence mixed')
        if not tail['rows'] or tail['net']<0 or tail['delta_macro_pp']<0:reasons.append(policy+': tail not supported')
        if few['rows'] and (few['net']<0 or few['delta_macro_pp']<0):reasons.append(policy+': few-support regressed')
    history=report['trajectory']
    if len(history)!=4:reasons.append('incomplete trajectory')
    else:
        raw=history[-1]['raw']['slices']['all']
        if raw['net']<=0 or raw['delta_macro_pp']<=0:reasons.append('raw/EMA direction not jointly supported')
        if history[-1]['ema']['slices']['all']['net']<history[1]['ema']['slices']['all']['net']:
            reasons.append('late EMA lost gains seen at epoch2')
    eligible=not reasons
    summary={policy:{name:report['paired'][policy]['slices'][name] for name in
                     ('all','tail','head','few_support')} for policy in ('last_ema','ema_swa_2_4')}
    assessment=('Both fixed DEV endpoints improve whole-set net and macro; tail and few-support groups do not regress; '
                'final raw agrees with EMA and late EMA retains epoch2 net gains. Four-epoch FULL uses its own v1 full '
                'parent/targets and no overlapping validation. Platform migration and clean-label improvement remain unknown. '
                'This is a bounded funding decision, not a promise of platform gain.') if eligible else ''
    return dict(eligible=eligible,status='supports_fixed_full' if eligible else 'evidence_requires_review',
                reasons=reasons,evidence=summary,assessment=assessment,candidate_rejected=False,
                platform_gain_known=False)


def gpu_pids():
    result=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    lines=[x.strip() for x in result.splitlines() if x.strip()]
    if any(not x.isdigit() for x in lines):raise RuntimeError('Cannot establish local CUDA occupancy')
    return lines


def stage_commands(job):
    cp=job['output']+'/run/training/selected.pt'
    common=[sys.executable,'-u','-m','v1_continuation.runtime']
    return [('train',common+['train','--plan',job['plan'],'--output',job['output']+'/run','--execute']),
            ('infer',common+['infer','--plan',job['plan'],'--run-root',job['output']+'/run',
                            '--checkpoint',cp,'--output',job['output']+'/submission','--execute'])]


def verify_delivery(job):
    root=Path(job['output']);run=root/'run';package=root/'submission'
    report=read(run/'report.json');delivery=read(package/'report.json')
    if report['status'] not in ('development_complete','full_training_complete') or report['epochs']!=4:
        raise ValueError('Fixed training incomplete')
    for path,expected in report['artifacts'].items():
        if sha(path)!=expected:raise ValueError('Training/evaluation evidence changed: '+path)
    if delivery['status']!='package_ready' or delivery['rows']!=37444:raise ValueError('Wrong package coverage')
    if sha(package/'pred_results.csv')!=delivery['csv_sha256'] or sha(package/'submission.zip')!=delivery['zip_sha256']:
        raise ValueError('Package checksum mismatch')
    with zipfile.ZipFile(package/'submission.zip') as archive:
        if archive.namelist()!=['pred_results.csv'] or archive.read('pred_results.csv')!=(package/'pred_results.csv').read_bytes():
            raise ValueError('ZIP/CSV mismatch')
    if 'All checks passed' not in (package/'submission_check.log').read_text():raise ValueError('Submission checker failed')
    if report['partition']=='train_dev':
        import numpy as np
        sys.path.insert(0,str(Path(job['worktree'])/'reproducibility/aegis_f1'))
        from v1_continuation.runtime import comparison
        plan=read(job['plan'])
        with np.load(run/'baseline.npz',allow_pickle=False) as base:
            labels=base['labels'];pred=base['predictions'];paths=base['image_paths']
        for policy,file in [('last_ema','val_epoch04_ema.npz'),('ema_swa_2_4','val_epoch04_ema_swa_2_4.npz')]:
            with np.load(run/file,allow_pickle=False) as data:
                if not np.array_equal(labels,data['labels']) or not np.array_equal(paths,data['image_paths']):
                    raise ValueError('Validation row alignment differs')
                measured=comparison(plan,labels,pred,data['predictions'])
            if measured!=report['paired'][policy]:raise ValueError('Independent paired recount differs')
    return dict(status='completed_delivered',experiment_id=job['experiment_id'],source_report=str(run/'report.json'),
        source_report_sha256=sha(run/'report.json'),plan=job['plan'],plan_sha256=sha(job['plan']),
        checkpoint=delivery['checkpoint'],checkpoint_sha256=delivery['checkpoint_sha256'],
        csv=str(package/'pred_results.csv'),zip=str(package/'submission.zip'),
        csv_sha256=delivery['csv_sha256'],zip_sha256=delivery['zip_sha256'],rows=37444,
        submission_checks_passed=9,zip_csv_identical=True,paired_independently_recounted=report['partition']=='train_dev',
        scores_are_original_label_diagnostics=report['partition']=='train_dev',platform_score=None)


def archive_and_integrate(job,verified,support,queue_state):
    root=Path(job['worktree']);name=job['name']
    # Synchronize the already-pushed scheme by merge before editing shared status.
    git(root,'fetch','origin')
    git(root,'merge','origin/main','-m','Sync current main before the validated checkpoint record')
    result=root/'results'/name;result.mkdir(parents=True,exist_ok=False)
    write(result/'final_validation.json',verified)
    write(result/'support.json',support)
    # Keep complete machine-readable evidence, without committing model tensors or test predictions.
    (result/'report.json').write_text(json.dumps(read(verified['source_report']),ensure_ascii=False,separators=(',',':'))+'\n')
    report=read(verified['source_report']);lines=[]
    if report['partition']=='train_dev':
        for policy in ('last_ema','ema_swa_2_4'):
            group=report['paired'][policy]['slices']['all']
            lines.append(f"- {policy}: macro/micro {group['candidate']['macro']*100:.4f}% / {group['candidate']['micro']*100:.4f}%; "
                         f"修正{group['corrections']}、退化{group['regressions']}、净{group['net']}。")
    else:lines.append('- 全量4轮完成，无独立val分；没有利用重叠val选模、校准或测试分布拟合。')
    commands='\n'.join(' '.join(argv) for _,argv in stage_commands(job))
    doc=root/'docs'/(name+'.md')
    doc.write_text(f"# {job['experiment_id']} 本机固定4轮执行\n\n状态 completed_delivered，{now()} 独立核验。\n\n"+
        '\n'.join(lines)+f"\n\n训练配置：`{job['config']}`，冻结plan SHA `{verified['plan_sha256']}`。\n"
        f"分支 `{job['branch']}`，启动源码commit `{job['code_commit']}`。\n"
        f"配方、轨迹与分组：[完整报告](../results/{name}/report.json)；"
        f"[独立校验](../results/{name}/final_validation.json)；[FULL支持判断](../results/{name}/support.json)。\n\n"
        f"CSV：`{verified['csv']}`；ZIP：`{verified['zip']}`。37,444行、9项通过、ZIP内外CSV字节相等。\n"
        "DEV统计相对原标签，非干净真值；新平台分未知，现役最高仍引用full v1 SWA用户回填70.9860%。\n\n"
        f"```bash\n{commands}\n```\n\n用户授权串行执行；本实验在已验证产物检查点封存并推送，下一独立路线按既定队列执行。\n")
    current=root/'docs/current_execution_plan.md'
    text=current.read_text();marker='## 最新授权本机串行执行：V1_CONTINUATIONS_SERIAL_20261001'
    section=f"## 已交付串行检查点：{job['experiment_id']}\n\n"+'\n'.join(lines)+f"\n\n37,444行CSV/ZIP通过9项校验；"
    section+=f"FULL证据状态 `{support['status']}`，未声称平台提升。见[{job['experiment_id']}]({name}.md)。\n\n"
    if marker in text:text=text.replace(marker,section+marker,1)
    else:text=text.replace('\n', '\n\n'+section,1)
    current.write_text(text)
    git(root,'add',str(result.relative_to(root)),str(doc.relative_to(root)),'docs/current_execution_plan.md')
    git(root,'diff','--cached','--check')
    git(root,'commit','-m',f"Deliver verified {job['experiment_id']} fixed continuation and package")
    commit=git(root,'rev-parse','HEAD');git(root,'push','-u','origin',job['branch'])
    if git(MAIN,'branch','--show-current')!='main':raise RuntimeError('Integration directory is not on main')
    git(MAIN,'pull','--rebase','--autostash','origin','main')
    git(MAIN,'merge','--no-ff',job['branch'],'-m',f"Integrate verified {job['experiment_id']} delivery")
    git(MAIN,'diff','--check')
    if read(MAIN/'results'/name/'final_validation.json')!=verified:raise ValueError('Integrated record differs')
    git(MAIN,'push','origin','main')
    return dict(branch=job['branch'],commit=commit,branch_pushed=True,main_commit=git(MAIN,'rev-parse','HEAD'),main_pushed=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue',required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if not args.execute:p.error('Serial execution requires --execute')
    cfg=read(args.queue);output=Path(cfg['controller_output']);output.mkdir(parents=True,exist_ok=True)
    status_path=output/'status.json'
    if status_path.exists():raise FileExistsError('Fresh controller required; no automatic retry')
    if cfg['experiments']!=['WFT448_DEV','LR512_DEV','WFT448_FULL','LR512_FULL']:
        raise ValueError('Only the fixed serial queue is authorized')
    state=dict(status='starting',runner_pid=os.getpid(),started_at=now(),jobs=[],automatic_retry=False,
               test_usage='inference_only',platform_upload=False,current_stage=None)
    write(status_path,state);child=None
    try:
        for job in cfg['jobs']:
            job=dict(job);state['current_experiment']=job['experiment_id'];state['current_stage']='prepare'
            if job['partition']=='full_train':
                dev_report=read(job['development_report']);support=full_support(dev_report)
                if not support['eligible']:
                    state['jobs'].append(dict(experiment_id=job['experiment_id'],status='evidence_requires_review',
                                              reasons=support['reasons'],candidate_rejected=False))
                    write(status_path,state);continue
                git(MAIN,'fetch','origin')
                if Path(job['worktree']).exists():raise FileExistsError(job['worktree'])
                git(MAIN,'worktree','add','-b',job['branch'],job['worktree'],'origin/main')
                root=Path(job['worktree'])
                env=dict(os.environ,PYTHONPATH=str(root/'reproducibility/aegis_f1'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2')
                subprocess.run([sys.executable,'-m','v1_continuation.plan','prepare','--config',job['config'],
                    '--output',str(Path(job['plan']).parent),'--development-report',job['development_report'],
                    '--support-assessment',support['assessment']],cwd=root,env=env,check=True)
            root=Path(job['worktree']);job['code_commit']=git(root,'rev-parse','HEAD')
            env=dict(os.environ,PYTHONPATH=str(root/'reproducibility/aegis_f1'),CUDA_VISIBLE_DEVICES='0',
                     OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            job_output=Path(job['output']);job_output.mkdir(parents=True,exist_ok=True)
            write(job_output/'execution_protocol.json',dict(job=job,commands=stage_commands(job),user_authorized=True))
            for stage,argv in stage_commands(job):
                while gpu_pids():
                    state.update(status='waiting_local_resource',current_stage=stage,heartbeat_at=now())
                    write(status_path,state);time.sleep(30)
                log=job_output/(stage+'.log')
                with log.open('x') as handle:
                    child=subprocess.Popen(argv,cwd=root,env=env,stdout=handle,stderr=subprocess.STDOUT)
                    state.update(status='running',current_stage=stage,child_pid=child.pid,log=str(log),heartbeat_at=now())
                    write(status_path,state);print(json.dumps(state),flush=True)
                    while True:
                        try:code=child.wait(timeout=15);break
                        except subprocess.TimeoutExpired:
                            state['heartbeat_at']=now()
                            for name in ('training/progress.json','evaluation_progress.json','inference_progress.json'):
                                path=job_output/'run'/name
                                if path.exists():state[name.replace('/','_')]=read(path)
                            write(status_path,state)
                    if code!=0:raise RuntimeError(f"{job['experiment_id']} {stage} failed ({code}); see {log}")
                    child=None
            verified=verify_delivery(job)
            report=read(verified['source_report'])
            support=full_support(report) if job['partition']=='train_dev' else dict(status='fixed_full_complete',eligible=False)
            state.update(current_stage='checkpoint_git_sync',heartbeat_at=now());write(status_path,state)
            integration=archive_and_integrate(job,verified,support,state)
            state['jobs'].append(dict(experiment_id=job['experiment_id'],status='completed_delivered',
                                      verification=verified,support=support,integration=integration))
            state['child_pid']=None;write(status_path,state)
        state.update(status='completed',current_stage='delivered_or_evidence_review',completed_at=now())
        write(status_path,state);print(json.dumps(state),flush=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state.update(status='failed',error=str(error),failed_at=now(),automatic_retry=False)
        write(status_path,state);raise


if __name__=='__main__':main()
