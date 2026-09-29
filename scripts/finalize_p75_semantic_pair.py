#!/usr/bin/env python3
"""Wait for this fixed campaign, verify, publish results and stop. No GPU work."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from run_p75_semantic_pair import ROOT, OUT, CONFIGS, run_dir, sha, write, verify
from p75_semantic_pair_report import report

ARCHIVE=ROOT/'results/p75_semantic_mask_pair_20260929'
BRANCH='codex/p75_semantic_mask_pair_20260929'
MAIN=Path('/home/lux1/noise-main-integration-l05')


def command(args,cwd=ROOT):
    subprocess.run(args,cwd=cwd,check=True)


def finish(expected_head):
    # Never overwrite concurrent edits or combine them into this experiment's commit.
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=expected_head or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Experiment worktree changed during training; leave artifacts for manual integration')
    verify()
    state=json.loads((OUT/'status.json').read_text())
    if state['status'] not in ('paired_training_validation_delivery_complete','epoch4_only_primary_incomplete'):
        raise RuntimeError(f"Campaign has no complete deliverable: {state['status']}")
    report(ARCHIVE)
    shutil.copy2(OUT/'manifest.json',ARCHIVE/'run_manifest.json')
    shutil.copy2(OUT/'status.json',ARCHIVE/'execution_status.json')
    if (OUT/'interrupted_status.json').exists():
        shutil.copy2(OUT/'interrupted_status.json',ARCHIVE/'interrupted_status.json')
    epoch=state['matched_epoch']
    package=OUT/f'deliveries/P75_SEMANTIC_MASKED_E{epoch}'
    for arm in ('control','masked'):
        target=ARCHIVE/'training_logs'/arm
        target.mkdir(parents=True)
        for source in (run_dir(arm)/'logs').iterdir():
            if source.suffix in ('.json','.jsonl','.csv'):
                shutil.copy2(source,target/source.name)
    shutil.copy2(package/'manifest.json',ARCHIVE/'submission_manifest.json')
    if state.get('closure_source_sha256'):
        if sha(ROOT/'scripts/close_p75_semantic_pair_e4.py')!=state['closure_source_sha256']:
            raise ValueError('Closure source changed')
    config=json.loads((CONFIGS/'masked.json').read_text())
    with (ARCHIVE/'submission_check.log').open('x') as f:
        subprocess.run([sys.executable,str(ROOT/'scripts/check_submission.py'),'--test_dir',config['data']['test_root'],
            '--class-mapping',config['data']['class_mapping'],'--csv',str(package/'pred_results.csv'),
            '--zip',str(package/'submission.zip')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    write(ARCHIVE/'artifact_manifest.json',dict(
        archived_files={str(p.relative_to(ARCHIVE)):sha(p) for p in ARCHIVE.rglob('*') if p.is_file()},
        delivery=dict(directory=str(package),config=str(CONFIGS/'masked.json'),
            sha256={name:sha(package/name) for name in ('pred_results.csv','submission.zip')},
            platform_submitted=False,diagnostic_only=True)))
    command([sys.executable,'scripts/verify_p75_semantic_pair.py','--archive',str(ARCHIVE)])
    command([sys.executable,'-m','pytest','tests/test_p75_text_page_runtime.py','tests/test_p75_semantic_pair.py','-q'])
    result=json.loads((ARCHIVE/'report.json').read_text())
    values=result['epochs'][str(epoch)]['same_epoch_pair']
    full,remaining,selected=[values[k] for k in ('all','remaining_proxy','selected_proxy')]
    completion='六轮配对完成' if epoch==6 else '四轮配对评估完成，六轮主判断未完成'
    conclusion=(f"其余代理净修正{remaining['net']}张；"+
        ('未观察到该组净改善，不扩大固定方案。' if remaining['net']<=0 else
         '该数字仍是含噪标签代理结果，不自动晋级完整训练或占用平台名额。'))
    if epoch!=6: conclusion='局部分支后的完整配对未完成，不能用四轮结果否定机制。'+conclusion
    statement=(f"**2026-09-29 语义屏蔽{completion}：**同RM-LP父权重，取消自动命中1,246张训练图的分类项；四轮时局部分支尚未启用。"
        f"第{epoch}轮全量macro/micro：control {full['control_macro']:.4%}/{full['control_micro']:.4%}，"
        f"masked {full['masked_macro']:.4%}/{full['masked_micro']:.4%}；修正{full['corrected']}、退化{full['regressed']}、"
        f"净{full['net']}张。{conclusion}单学生诊断CSV/ZIP共37,444行、9/9校验通过；无新平台成绩，SAM未操作。"
        "现役L05_T14_P060平台仍为 **66.94797564362783%**。见[配对记录](docs/p75_semantic_mask_pair_20260929.md)及[当前执行入口](docs/current_execution_plan.md)。")
    readme=ROOT/'README.md'
    parts=readme.read_text().split('\n\n')
    if not parts[1].startswith('**2026-09-29 语义内容阶段A完成：**'):
        raise ValueError('README state moved; merge manually')
    parts[1]=statement
    readme.write_text('\n\n'.join(parts))
    record=ROOT/'docs/p75_semantic_mask_pair_20260929.md'
    record.write_text(record.read_text()+f"\n## 已验证结果\n\n{completion}。预算计费累计{state['used_seconds']:.2f}秒；无预算追加。\n\n"+
        (ARCHIVE/'report.md').read_text()+f"\n{conclusion}\n\n"+
        ("本次训练调度器中断后，用户明确要求只做四轮评估与诊断出包。恢复时保留原状态，补记masked任务开始至恢复时刻的保守墙钟耗时（含空闲中断间隔）；原退出码未知，未伪造成功退出，未重置预算、未续训第5–6轮。恢复入口：`python3 scripts/close_p75_semantic_pair_e4.py`；先冻结同轮次配对报告再出包。\n\n" if state.get('closure_reason') else "")+
        f"诊断单学生包：`{package}/pred_results.csv`、`{package}/submission.zip`。9/9校验通过；未上传平台。"
        "逐张预测、分组训练期统计、顺序/更新数审计、执行命令与用时均归档。报告中的相对L05结果仅为现役参考，主对照为同轮次control。\n\n"
        "复核：`python3 scripts/verify_p75_semantic_pair.py --archive results/p75_semantic_mask_pair_20260929`。"
        "方案结果提交、main合并并重新校验推送后暂停，不自动扩训。\n")
    current=ROOT/'docs/current_execution_plan.md'
    text=current.read_text().replace('# 当前执行入口：P75语义内容阶段A验收与自动表完成（2026-09-29）',
        f'# 当前执行入口：P75语义屏蔽{completion}（2026-09-29）',1)
    start=text.index('## 最新语义内容入口：')
    end=text.index('## 已关闭的像素内容入口',start)
    text=text[:start]+f"## 最新语义内容入口：{completion}\n\n"+statement.replace('(docs/','(')+\
        "\n\n阶段A的描述、阈值及自动表未变；两臂样本顺序、实际更新与学习率已核对。代理不是干净真值，污染主导平台分差仍未被证实。配对在规定预算内收尾并暂停，不追加训练或检测器扫描。\n\n"+text[end:]
    old='本轮`P75_SEMANTIC_CONTENT_PROBE`完成阶段A验收和自动表，配对训练尚未执行，没有新训练、候选指标或提交包。'
    text=text.replace(old,f'本轮`P75_SEMANTIC_MASK_PAIR`{completion}；同轮次结果、单学生诊断包及提交校验见最新配对记录。')
    text=text.replace('当前误差与成本门禁。本轮P0快照已完成，未启动新训练，不改其他既有进程。',
        '当前误差与成本门禁。本轮语义分类屏蔽固定配对已按预算收尾，不改其他既有进程。')
    current.write_text(text)
    command(['git','diff','--check'])
    paths=[str(ARCHIVE.relative_to(ROOT)),'README.md','docs/current_execution_plan.md','docs/p75_semantic_mask_pair_20260929.md']
    command(['git','add',*paths])
    command(['git','commit','-m',f'Record bounded semantic masking pair at matched epoch {epoch}'])
    command(['git','push','-u','origin',BRANCH])
    branch_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','branch','--show-current'],cwd=MAIN,text=True).strip()!='main':
        raise RuntimeError('Integration worktree is not on main')
    print('Main integration uses automatic mode: pull --rebase --autostash; no manual stash pop.',flush=True)
    command(['git','pull','--rebase','--autostash','origin','main'],MAIN)
    command(['git','merge','--no-ff',BRANCH,'-m','Merge verified bounded semantic masking pair'],MAIN)
    command([sys.executable,'scripts/verify_p75_semantic_pair.py','--archive','results/p75_semantic_mask_pair_20260929'],MAIN)
    command([sys.executable,'-m','pytest','tests/test_p75_text_page_runtime.py','tests/test_p75_semantic_pair.py','-q'],MAIN)
    command(['git','diff','--check'],MAIN)
    command(['git','push','origin','main'],MAIN)
    merge_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=MAIN,text=True).strip()
    command(['git','pull','--rebase','--autostash','origin','main'],Path('/home/lux1/noise'))
    write(OUT/'delivery_status.json',dict(status='verified_pushed_merged_checkpoint_paused',
        branch=BRANCH,branch_commit=branch_commit,main_commit=merge_commit,package=str(package),
        matched_epoch=epoch,primary_complete=epoch==6,platform_submitted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-head',required=True)
    args=p.parse_args()
    try:
        while json.loads((OUT/'status.json').read_text())['status']=='running':
            state=json.loads((OUT/'status.json').read_text())
            if time.time()-state['started_unix']>8*3600:
                raise RuntimeError('Campaign status stale beyond its fixed budget; no automatic restart')
            time.sleep(30)
        finish(args.expected_head)
    except Exception as exc:
        write(OUT/'delivery_status.json',dict(status='needs_review_no_automatic_retry',error=repr(exc)))
        raise
