#!/usr/bin/env python3
"""Run the authorized bounded A segment and archive its terminal checkpoint."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from p75_pipeline import require,verify
from p75_pipeline_report import summarize
from run_p75_semantic_pair import write,sha
from run_p75_pipeline_a import run

ROOT=Path(__file__).resolve().parents[1]
BRANCH='codex/p75_mask_e6_execution_20260929'
ARCHIVE=ROOT/'results/p75_mask_e6_execution_20260929'


def git(*args,cwd=ROOT):
    return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()


def archive(root):
    state=json.loads((root/'A/status.json').read_text())
    require(state['status'] in ('local_result','incomplete'),'Cannot archive a live experiment')
    if state['status']=='local_result':
        rows=json.loads((root/'groups.json').read_text())
        with (root/'A/predictions.csv').open() as f: pred=list(csv.DictReader(f))
        require([r['image_path'] for r in pred]==[r['image_path'] for r in rows],'Final prediction order mismatch')
        got=summarize(rows,[int(r['label']) for r in pred],[int(r['control']) for r in pred],[int(r['masked']) for r in pred])
        require(got==json.loads((root/'A/report.json').read_text())['groups'],'Final pair report mismatch')
    for sub in ('A','B'):
        for file in (root/sub).glob('*'):
            if file.is_file() and file.suffix in ('.json','.csv'):
                target=ARCHIVE/sub/file.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
    for file in (root/'A/runs').glob('*/seed42/logs/*'):
        if file.is_file():
            target=ARCHIVE/'training_logs'/file.relative_to(root/'A/runs')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
    conclusion=json.loads((root/'A/conclusion.json').read_text())
    if state['status']=='local_result':
        metrics=json.loads((root/'A/report.json').read_text())['groups']
        text=(f"**2026-09-29 P75_MASK_E6_EXTENSION E6 配对已完成：**全量净修正{metrics['all']['net']}张，"
              f"未命中代理净修正{metrics['remaining_proxy']['net']}张；判断 `{conclusion['decision']}`。"
              '代理不是真值，无新平台成绩；B因缺少现有绑定flip特征未启动，不自动扩训或上传。')
    else:
        text='**2026-09-29 P75_MASK_E6_EXTENSION 本段未完成：**自有任务已停止，完整E6主判断未形成，不解释为机制无效；旧预算未重置，B未启动，无新平台成绩。'
    text+=' 详见[执行记录](docs/p75_mask_e6_execution_20260929.md)。'
    readme=ROOT/'README.md'; content=readme.read_text();start=content.index('**2026-09-29 P75_MASK_E6_EXTENSION 已授权')
    end=content.index('\n\n',start);readme.write_text(content[:start]+text+content[end:])
    current=ROOT/'docs/current_execution_plan.md'
    current.write_text(current.read_text().replace('A执行状态：`running`', 'A执行状态：`'+state['status']+'`',1))
    path=ROOT/'docs/p75_mask_e6_execution_20260929.md'
    with path.open('a') as f:
        f.write('\n## 终态\n\n'+text.replace('(docs/','(')+'\n\n')
        f.write('任务状态与成本：\n\n```json\n'+json.dumps({k:v for k,v in state.items() if k not in ('history','current')},indent=2,ensure_ascii=False)+'\n```\n')
    write(ARCHIVE/'completion.json',dict(status=state['status'],conclusion=conclusion,platform_uploaded=False,new_full_candidate=False))
    write(ARCHIVE/'archive_sha256.json',{str(p.relative_to(ARCHIVE)):sha(p) for p in ARCHIVE.rglob('*') if p.is_file() and p.name!='archive_sha256.json'})


def main(root,authorization,expected_head):
    require(git('rev-parse','HEAD')==expected_head and git('branch','--show-current')==BRANCH,'Launch source changed')
    try:
        run(root,authorization)
    except BaseException as exc:
        write(root/'service_error.json',dict(error=repr(exc)))
        if not (root/'A/status.json').exists(): raise
    try:
        require(git('rev-parse','HEAD')==expected_head,'Concurrent branch change; leave results for manual integration')
        require(not git('diff','--name-only') and not git('diff','--cached','--name-only'),'Concurrent tracked edits; do not overwrite')
        verify(root); archive(root)
        subprocess.check_call([sys.executable,'-m','pytest','tests/test_p75_pipeline.py','tests/test_p75_text_page_runtime.py','tests/test_p75_semantic_pair.py','-q'],cwd=ROOT)
        git('diff','--check')
        git('add','README.md','docs/current_execution_plan.md','docs/p75_mask_e6_execution_20260929.md',str(ARCHIVE.relative_to(ROOT)))
        git('commit','-m','Archive bounded P75 E6 extension terminal result')
        commit=git('rev-parse','HEAD'); git('push','-u','origin',BRANCH)
        integration=Path('/home/lux1/noise')
        require(git('branch','--show-current',cwd=integration)=='main','Integration checkout is not main')
        git('pull','--rebase','--autostash','origin','main',cwd=integration)
        git('merge','--no-ff',commit,'-m','Merge verified P75 E6 extension terminal result',cwd=integration)
        subprocess.check_call([sys.executable,'-m','pytest','tests/test_p75_pipeline.py','tests/test_p75_text_page_runtime.py','tests/test_p75_semantic_pair.py','-q'],cwd=integration)
        git('diff','--check','HEAD^','HEAD',cwd=integration); git('push','origin','main',cwd=integration)
        write(root/'delivery_status.json',dict(status='integrated',experiment_commit=commit,main_commit=git('rev-parse','HEAD',cwd=integration)))
    except BaseException as exc:
        write(root/'delivery_status.json',dict(status='needs_manual_review',error=repr(exc)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True);p.add_argument('--expected-head',required=True)
    a=p.parse_args();main(a.out.resolve(),a.authorization,a.expected_head)
