"""Independently replay both packages and the fixed bias using NumPy on CPU."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.v1_pipeline import StageContext, load_artifact, load_recipe


def read_predictions(path):
    with path.open() as handle:
        return [(name, label.strip()) for name, label in csv.reader(handle)]


def numpy_uniform_bias(matrix, iterations):
    """Float64 proportional fitting, independent of the torch implementation."""
    bias = np.zeros(matrix.shape[1], dtype=np.float64)
    for iteration in range(iterations):
        mass = np.zeros(matrix.shape[1], dtype=np.float64)
        for start in range(0, len(matrix), 8192):
            z = matrix[start:start+8192].astype(np.float64) + bias
            z -= z.max(1, keepdims=True)
            probability = np.exp(z)
            probability /= probability.sum(1, keepdims=True)
            mass += probability.sum(0)
        bias += np.log(1/matrix.shape[1]) - np.log(np.maximum(mass/len(matrix), 1e-12))
        bias -= bias.mean()
        if (iteration+1) % 50 == 0:
            print(f'independent NumPy bias {iteration+1}/{iterations}', flush=True)
    return bias


def check_package(source_root, test_root, class_mapping, package):
    """The real checker logs through logging.StreamHandler, usually stderr."""
    checked = subprocess.run([sys.executable,str(source_root/'scripts/check_submission.py'),
        '--test_dir',str(test_root),'--class-mapping',str(class_mapping),
        '--csv',str(package/'pred_results.csv'),'--zip',str(package/'submission.zip')],
        capture_output=True,text=True,check=True)
    output = checked.stdout + checked.stderr
    if 'All checks passed' not in output:
        raise ValueError('Independent submission checker did not pass')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--desktop-raw', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.monotonic()
    context = StageContext(load_recipe(args.config))
    output = Path(context.config['output']['root'])
    status = json.loads((output/'status.json').read_text())
    if status['status'] != 'completed' or status['stage'] != 'delivered':
        raise ValueError('The full trajectory has not completed delivery')
    checkpoint = output/'training/selected.pt'
    selected = load_artifact(checkpoint, context.binding)
    targets = load_artifact(output/'targets.pt', context.binding)
    cache = load_artifact(output/'test_logits.pt', context.binding)
    fitted = load_artifact(output/'test_calibration.pt', context.binding)
    decoder = context.config['decode']
    checkpoint_hash = sha256_file(checkpoint)
    if (selected['epoch'] != 12 or selected['selected_policy'] != 'swa_ema'
            or selected['swa_epochs'] != list(range(4,13)) or selected['swa']['count'] != 9
            or len(selected['history']) != 12
            or tuple(selected['selected_state']['head.weight'].shape) != (750,768)
            or tuple(targets['teacher_state']['weight'].shape) != (750,768)
            or targets['image_paths'] != [row['image_path'] for row in context.full]
            or selected['targets_sha256'] != sha256_file(output/'targets.pt')):
        raise ValueError('Wrong training population, head dimension or fixed SWA window')
    for artifact in (cache,fitted):
        if artifact['checkpoint_sha256'] != checkpoint_hash or artifact['decoder'] != decoder:
            raise ValueError('Logits/bias use another model or decoder')
    if fitted['test_logits_sha256'] != sha256_file(output/'test_logits.pt'):
        raise ValueError('Fitted bias cache checksum changed')
    if (fitted['test_data_used'] is not True or fitted['backbone_or_head_updates'] is not False
            or fitted['official_permission_confirmed'] is not True
            or fitted['source'] != 'unlabelled_current_stage_test_logits'):
        raise ValueError('Bias provenance misstates test use or the official confirmation')
    matrix = cache['logits'].numpy()
    bias = fitted['bias'].numpy()
    if matrix.shape != (context.manifest['test_samples'],len(context.classes)):
        raise ValueError('Test matrix shape changed')
    raw = matrix.argmax(1)
    corrected = (matrix + np.float32(decoder['bias_strength'])*bias).argmax(1)
    package_reports = {}
    source_root = Path(context.config['source']['root'])
    for directory,prediction,usage in (
        ('submission_raw',raw,'inference_only'),
        ('submission',corrected,'unlabelled_test_marginal_bias_fitting'),
    ):
        package = output/directory
        expected = [(name,context.classes[index]) for name,index in zip(cache['names'],prediction)]
        observed = read_predictions(package/'pred_results.csv')
        if observed != expected:
            raise ValueError(f'{directory} does not reproduce all predictions from saved logits/bias')
        with zipfile.ZipFile(package/'submission.zip') as archive:
            if archive.namelist() != ['pred_results.csv'] or archive.read('pred_results.csv') != (package/'pred_results.csv').read_bytes():
                raise ValueError('ZIP members or CSV bytes differ')
        metadata = json.loads((package/'manifest.json').read_text())
        if metadata['checkpoint_sha256'] != checkpoint_hash or metadata['test_usage'] != usage:
            raise ValueError('Package lineage misstates the selected checkpoint or test usage')
        checker_output = check_package(source_root, context.test_root,
            context.reference['data']['class_mapping'], package)
        package_reports[directory] = dict(samples=len(observed),numpy_replay_matches=len(observed),
            csv=str(package/'pred_results.csv'),zip=str(package/'submission.zip'),
            csv_sha256=sha256_file(package/'pred_results.csv'),zip_sha256=sha256_file(package/'submission.zip'),
            test_usage=usage,checker_output=checker_output)
    independent = numpy_uniform_bias(matrix,decoder['bias_iterations'])
    error = float(np.max(np.abs(independent-bias)))
    if error > 1e-3:
        raise ValueError(f'Independent float64 bias differs by {error}')
    independent_prediction = (matrix.astype(np.float64)+independent).argmax(1)
    changed = np.flatnonzero(independent_prediction != corrected)
    if changed.size:
        scores = matrix[changed].astype(np.float64)+independent
        two = np.partition(scores,-2,axis=1)[:,-2:]
        if np.max(two[:,1]-two[:,0]) > 2*error+1e-5:
            raise ValueError('Independent bias changes predictions beyond its numerical error')
    desktop = Path(status['desktop_zip'])
    if sha256_file(desktop) != package_reports['submission']['zip_sha256']:
        raise ValueError('Delivered desktop ZIP differs')
    if args.desktop_raw:
        if not args.desktop_raw.exists():
            with (output/'submission_raw/submission.zip').open('rb') as source,args.desktop_raw.open('xb') as destination:
                shutil.copyfileobj(source,destination)
        if sha256_file(args.desktop_raw) != package_reports['submission_raw']['zip_sha256']:
            raise ValueError('Raw desktop ZIP differs')
    incumbent = source_root/'worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/submission.zip'
    if sha256_file(incumbent) != '1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e':
        raise ValueError('The incumbent submission has changed')
    atomic_json_dump(dict(experiment_id=context.config['project']['experiment_id'],status='verified',
        binding=context.binding,checkpoint_sha256=checkpoint_hash,packages=package_reports,
        independent_float64_bias_max_error=error,
        independent_refit_prediction_matches=int(np.sum(independent_prediction==corrected)),
        independent_refit_numerical_boundary_changes=int(changed.size),
        raw_to_bias_prediction_changes=int(np.sum(raw!=corrected)),
        teacher_dimension=768,student_head_dimension=768,full_train_rows=len(context.full),
        used_training_rows=targets['stats']['used'],epochs=12,swa_ema_epochs=list(range(4,13)),
        desktop_zip=str(desktop),desktop_raw_zip=str(args.desktop_raw) if args.desktop_raw else None,
        incumbent_zip=str(incumbent),incumbent_unchanged=True,
        platform_score=None,accuracy_gain_unknown=True,elapsed_seconds=time.monotonic()-started),Path(args.report))
    print(f'verified both {len(raw)}-row packages; independent bias max error {error:.8g}',flush=True)


if __name__=='__main__':
    main()
