#!/usr/bin/env python3
"""One explicitly authorized local-step timing probe; discarded weights, charged to A."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from p75_pipeline import verify, require
from run_p75_semantic_pair import write,sha,idle_gpu
from run_p75_pipeline_a import authorize


def run(root,authorization):
    verify(root); authorize(root,authorization,'A'); idle_gpu()
    out=root/'A/cost_probe'; require(not out.exists(),'No cost probe replay')
    out.mkdir(); spec=json.loads((root/'A/manifest.json').read_text())['arms']['control']
    config=json.loads(Path(spec['config']).read_text())
    config['project'].update(experiment_id='P75_E6_COST_ONLY',trial_id='P75_E6_COST_ONLY')
    config['output']['root']=str(out/'runs')
    config['train'].update(max_steps=534,log_every_steps=2)
    write(out/'config.json',config)
    source=Path(spec['resume']); resume=out/'resume_E4.pt'; shutil.copyfile(source,resume)
    binding=json.loads(source.with_suffix('.binding.json').read_text())
    binding.update(training_config_sha256=sha(out/'config.json'),purpose='10-update local throughput measurement only; never resume resulting weights')
    write(resume.with_suffix('.binding.json'),binding)
    env=os.environ.copy(); fw=root/'framework'
    env.update(PYTHONPATH=os.pathsep.join((str(fw),str(fw/'reproducibility/aegis_f1'))),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',
        P75_PIPELINE_STOP='5',P75_PIPELINE_DEADLINE=str(time.time()+810))
    argv=[sys.executable,'-m','aegis_clip.cli.train','--config',str(out/'config.json'),'--resume',str(resume)]
    state=dict(status='running',argv=argv,budget_source='new A 21600 seconds',weight_reuse_forbidden=True,
        config_sha256=sha(out/'config.json'),source_sha256=sha(__file__),started_unix=time.time())
    write(out/'status.json',state); started=time.monotonic()
    with (out/'stdout.log').open('x') as log:
        child=subprocess.Popen(argv,cwd=fw,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['pid']=child.pid; write(out/'status.json',state)
        try:
            code=child.wait(timeout=890)
        except BaseException:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
        finally:
            state.update(seconds=time.monotonic()-started,returncode=child.returncode,status='measured' if child.returncode==0 else 'incomplete')
            write(out/'status.json',state)
    require(code==0,'Local timing probe failed; time remains charged')
    progress=out/'runs/P75_E6_COST_ONLY/seed42/logs/progress.jsonl'
    rows=[json.loads(s) for s in progress.read_text().splitlines()]
    require(len(rows)>=3 and rows[-1]['global_step']>=532,'Insufficient real local updates')
    steps=[(b['cumulative_training_seconds']-a['cumulative_training_seconds'])/(b['global_step']-a['global_step']) for a,b in zip(rows,rows[1:])]
    state.update(seconds_per_local_update=max(steps)*1.2,measured_intervals=steps,margin_multiplier=1.2,
        progress_sha256=sha(progress),completed_probe_updates=534-524,final_weights_not_a_valid_E5=True)
    write(out/'status.json',state);print(json.dumps(state,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True)
    a=p.parse_args();run(a.out.resolve(),a.authorization)
