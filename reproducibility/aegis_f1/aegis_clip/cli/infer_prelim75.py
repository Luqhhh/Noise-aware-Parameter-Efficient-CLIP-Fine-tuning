"""One-checkpoint, fixed-protocol submission with validation and no prior."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile
from aegis_clip.prelim75 import load_plan, repository_root
from aegis_clip.runtime import atomic_json_dump, sha256_file


def infer(plan, name):
    root = Path(plan['output'])/name
    diagnostic = json.loads((root/'diagnostic/result.json').read_text())
    if diagnostic['status'] != 'complete' or diagnostic.get('engineering_stop', True):
        raise ValueError('Candidate stopped by the fixed engineering budget')
    checkpoint = root/'candidate.pt'
    if sha256_file(checkpoint) != diagnostic['checkpoint_sha256']:
        raise ValueError('Candidate changed after real diagnostic')
    output = root/'submission'
    if output.exists():
        raise FileExistsError('Do not overwrite an existing submission')
    command = [sys.executable, '-u', '-m', 'aegis_clip.cli.infer',
               '--checkpoint', str(checkpoint), '--output-dir', str(output),
               '--local-view', 'attention_multiscale', '--local-crop-sizes', '112,128,144,160',
               '--local-scale-weights', '0.20,0.30,0.40,0.10', '--local-top-k', '5',
               '--local-weight', '0.40', '--local-temperature', '1.5',
               '--adapt-local-features', '--adapt-part-token-features',
               '--tta', 'horizontal_flip', '--tta-fusion', 'mean_probabilities',
               '--tta-temperature', '1.5', '--tta-view-weight', '0.50',
               '--acknowledge-local-view-risk', '--acknowledge-tta-risk', '--batch-size', '64']
    with (root/'inference.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    repo = repository_root()
    check_command = [sys.executable, str(repo/'scripts/check_submission.py'),
                     '--test_dir', plan['test_root'], '--num-classes', '500',
                     '--csv', str(output/'pred_results.csv'), '--zip', str(output/'submission.zip')]
    with (root/'submission_validation.log').open('w') as log:
        subprocess.run(check_command, check=True, stdout=log, stderr=subprocess.STDOUT)
    raw = (output/'pred_results.csv').read_bytes()
    with ZipFile(output/'submission.zip') as archive:
        if archive.namelist() != ['pred_results.csv'] or archive.read('pred_results.csv') != raw:
            raise ValueError('ZIP CSV byte identity failed')
    if len(raw.splitlines()) != 24967:
        raise ValueError('Official test row count mismatch')
    report = {'candidate': name, 'status': 'submission_ready_pending_platform',
              'checkpoint_sha256': sha256_file(checkpoint),
              'csv_sha256': sha256_file(output/'pred_results.csv'),
              'zip_sha256': sha256_file(output/'submission.zip'),
              'inference_manifest_sha256': sha256_file(output/'manifest.json'),
              'resolved_config_sha256': sha256_file(root/'resolved_config.json'),
              'inference_command': command, 'validation_command': check_command,
              'validation_exit_code': 0, 'rows': 24967, 'zip_internal_csv_byte_equal': True,
              'online_accuracy': None, 'actual_platform_upload_time': None,
              'diagnostic': diagnostic, 'single_checkpoint': True, 'prior_in_inference': False}
    atomic_json_dump(report, root/'candidate_report.json')
    atomic_json_dump(report, root/'status.json')
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--candidate', choices=['H0', 'H1', 'V0', 'V1', 'C'], required=True)
    args = parser.parse_args()
    infer(load_plan(args.config), args.candidate)

if __name__ == '__main__':
    main()
