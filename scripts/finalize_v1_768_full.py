"""Finish one authorized running trajectory through verified Git integration.

This watches the existing local job; it never starts or retries training.
Every mutation follows successful delivery and independent verification.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from run_v1_training import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--integration-root', required=True, type=Path)
    parser.add_argument('--desktop-raw', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = args.config.resolve()
    recipe = yaml.safe_load(config.read_text())
    experiment = recipe['project']['experiment_id']
    branch = 'codex/v1_768_full_test_bias_20261001'
    if experiment != 'V1_768_FULL_TEST_BIAS_20261001':
        raise ValueError('This completion workflow is scoped to the authorized fixed experiment')
    if subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip() != branch:
        raise ValueError('The experiment worktree has changed branches')
    output = (config.parent / recipe['output']['root']).resolve()
    status_path = output/'completion_status.json'
    if status_path.exists():
        raise FileExistsError('Do not duplicate an existing completion workflow')
    state = dict(experiment_id=experiment,status='watching_existing_training',
        started_at=datetime.now(timezone.utc).isoformat(),training_retry=False)
    write_json(status_path,state)
    environment = dict(os.environ,PYTHONPATH=str(root/'reproducibility/aegis_f1'),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def run(stage,command,working=root):
        state.update(status='running',stage=stage,command=command)
        write_json(status_path,state)
        with (output/f'completion_{stage}.log').open('x') as log:
            process_environment = dict(environment,PYTHONPATH=str(working/'reproducibility/aegis_f1'))
            subprocess.run(command,cwd=working,env=process_environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        while True:
            status = json.loads((output/'status.json').read_text())
            if status['status'] == 'failed':
                raise RuntimeError('Training/delivery failed: '+status.get('error','unknown failure'))
            if status['status'] == 'completed':
                break
            if not Path(f'/proc/{status["runner_pid"]}').exists():
                raise RuntimeError('Training runner exited without a delivered or failed status')
            state.update(heartbeat_at=datetime.now(timezone.utc).isoformat(),
                training_stage=status.get('stage'),training_progress=status.get('training_progress'))
            write_json(status_path,state)
            time.sleep(30)
        results = root/'results/v1_768_full_test_bias_20261001'
        verification_path = results/'delivery_verification.json'
        run('independent_verification',[sys.executable,'-u','scripts/verify_v1_768_full_delivery.py',
            '--config',str(config),'--report',str(verification_path),'--desktop-raw',str(args.desktop_raw)])
        verification = json.loads(verification_path.read_text())
        bias_report = json.loads((output/'test_bias_report.json').read_text())
        targets = json.loads((output/'target_report.json').read_text())
        history = json.loads((output/'training/history.json').read_text())
        summary = dict(experiment_id=experiment,status='full_training_and_delivery_verified',
            training_elapsed_seconds=status['elapsed_seconds'],train_samples=148695,
            target_samples=targets['used'],classes=750,feature_dimension=768,epochs=12,
            selected_policy='swa_ema',swa_epochs=list(range(4,13)),
            checkpoint=status['checkpoint'],checkpoint_sha256=verification['checkpoint_sha256'],
            optimizer_steps=sum(row['optimizer_steps'] for row in history),
            first_epoch_loss=history[0]['loss'],last_epoch_loss=history[-1]['loss'],
            raw_to_bias_prediction_changes=verification['raw_to_bias_prediction_changes'],
            raw_argmax_min=bias_report['raw']['argmax_min'],raw_argmax_max=bias_report['raw']['argmax_max'],
            calibrated_argmax_min=bias_report['calibrated']['argmax_min'],
            calibrated_argmax_max=bias_report['calibrated']['argmax_max'],
            calibrated_soft_uniform_max_relative_error=bias_report['calibrated']['soft_uniform_max_relative_error'],
            packages=verification['packages'],desktop_zip=verification['desktop_zip'],
            desktop_raw_zip=verification['desktop_raw_zip'],
            independent_float64_bias_max_error=verification['independent_float64_bias_max_error'],
            independent_refit_prediction_matches=verification['independent_refit_prediction_matches'],
            official_permission_source='user-relayed official confirmation 2026-10-01',
            test_statistical_fitting=True,test_labels_used=False,model_parameter_updates_during_calibration=False,
            independent_accuracy=None,platform_score=None,incumbent_unchanged=True)
        for filename,value in [('delivery_summary.json',summary),('training_history.json',history),
                               ('test_bias_report.json',bias_report),('target_report.json',targets)]:
            write_json(results/filename,value)
        document = root/'docs/v1_768_full_test_bias_20261001.md'
        text = document.read_text()
        text += ('\n## 已验证交付检查点\n\n'
            f'固定12轮完整训练及两包交付完成，用时{status["elapsed_seconds"]/3600:.4f}小时；'
            f'训练使用{targets["used"]:,}张样本、750类，单checkpoint为EMA4–12 SWA。'
            'full_train无独立验证准确率，不将训练loss或预测分布改进当作平台提升。\n\n'
            f'独立重放两份全部37,444条预测和CSV/ZIP字节，两个包均重新通过提交校验。'
            f'NumPy FP64独立200次拟合bias最大误差{verification["independent_float64_bias_max_error"]:.8g}，'
            f'重拟合预测一致{verification["independent_refit_prediction_matches"]:,}/37,444。'
            f'bias改变{verification["raw_to_bias_prediction_changes"]:,}条预测；不代表准确率提升。\n\n'
            f'选中checkpoint SHA `{verification["checkpoint_sha256"]}`。\n\n')
        for directory in ['submission','submission_raw']:
            package = verification['packages'][directory]
            text += f'- {directory}: `{package["zip"]}`；ZIP SHA `{package["zip_sha256"]}`。\n'
        text += (f'\n桌面候选：`{verification["desktop_zip"]}`；无bias对照：`{verification["desktop_raw_zip"]}`。'
                 '\n\n现役70.98600576861446%包校验为未改变；新平台分待用户回填。结果见'
                 '`results/v1_768_full_test_bias_20261001/delivery_summary.json`与`delivery_verification.json`。'
                 '完成方案/主线推送后停在交付检查点，不派生新候选。\n')
        document.write_text(text)
        current = root/'docs/current_execution_plan.md'
        text = current.read_text().replace('## 新授权固定实验：V1_768_FULL_TEST_BIAS_20261001',
            '## 最新交付：V1_768_FULL_TEST_BIAS_20261001')
        text = text.replace('75项CPU回归及官方CPU/CUDA真实检查通过；北京时间12:08已启动，第1/12轮，去噪后136,631张、750类均有目标，尚无新包/平台分。',
            f'固定12轮完成，136,631张、750类均有目标；两份37,444行CSV/ZIP通过校验并独立重放。'
            f'bias改变{verification["raw_to_bias_prediction_changes"]:,}条预测；新平台分待回填，现役保持。')
        current.write_text(text)
        files = ['docs/current_execution_plan.md','docs/v1_768_full_test_bias_20261001.md']+[
            str((results/name).relative_to(root)) for name in (
                'delivery_verification.json','delivery_summary.json','training_history.json',
                'test_bias_report.json','target_report.json')]
        run('diff_check',['git','diff','--check'])
        run('stage',['git','add',*files])
        run('commit',['git','commit','-m','Deliver verified 768 full v1 and balanced-test bias packages'])
        state['proposal_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        run('push_proposal',['git','push','origin',branch])
        integration = args.integration_root.resolve()
        if subprocess.check_output(['git','branch','--show-current'],cwd=integration,text=True).strip() != 'main':
            raise ValueError('The designated main integration worktree has changed branches')
        run('pull_main',['git','pull','--rebase','--autostash','origin','main'],integration)
        run('merge_main',['git','merge','--no-ff',branch,'-m','Integrate verified 768 full training and paired bias delivery'],integration)
        checks = ['reproducibility/aegis_f1/tests/'+name for name in (
            'test_v1.py','test_v1_preprojection_test_bias.py','test_v1_export.py',
            'test_prior_alignment.py','test_calibration_binding.py')]
        run('main_tests',[sys.executable,'-m','pytest',*checks,'-q'],integration)
        run('main_diff_check',['git','diff','--check'],integration)
        run('push_main',['git','push','origin','main'],integration)
        state.update(status='completed',stage='verified_checkpoint',
            main_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=integration,text=True).strip(),
            summary=str(results/'delivery_summary.json'),verification=str(verification_path),
            completed_at=datetime.now(timezone.utc).isoformat())
        write_json(status_path,state)
    except BaseException as error:
        state.update(status='failed',error=str(error),failed_at=datetime.now(timezone.utc).isoformat())
        write_json(status_path,state)
        raise


if __name__=='__main__':
    main()
