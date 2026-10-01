"""Fixed CPU teacher/student recovery budget, with no fusion or training."""
from __future__ import annotations

import argparse
import csv
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from diagnose_v1_candidate_errors import require, sha, read_json, read_rows, load_aligned_npz, metrics, paired

ROOT = Path(__file__).resolve().parents[1]


def replay_head(features, state, chunk=2048):
    weight = state['weight'].numpy().astype(np.float64)
    require(np.isfinite(weight).all() and np.all(np.linalg.norm(weight, axis=1) > 0), 'Invalid teacher weights')
    weight /= np.linalg.norm(weight, axis=1, keepdims=True)
    scale = float(np.clip(np.exp(float(state['logit_scale'])), 1, 100))
    predictions, confidences, margins = [], [], []
    for start in range(0, len(features), chunk):
        x = np.array(features[start:start+chunk], dtype=np.float64, copy=True)
        require(x.shape[1] == weight.shape[1] and np.isfinite(x).all() and
                np.all(np.linalg.norm(x, axis=1) > 0), 'Invalid teacher features')
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        logits = scale * x @ weight.T
        top = logits.max(1)
        second = np.partition(logits, -2, axis=1)[:, -2]
        denominator = np.exp(logits-top[:, None]).sum(1)
        confidence = 1/denominator
        predictions.append(logits.argmax(1)); confidences.append(confidence)
        margins.append(confidence-np.exp(second-top)/denominator)
    return dict(prediction=np.concatenate(predictions), confidence=np.concatenate(confidences),
                margin=np.concatenate(margins))


def supported(scores, rule):
    return (scores['confidence'] >= rule['confidence_min']) & (scores['margin'] >= rule['probability_margin_min'])


def gate_pass(comparison, common_recoveries, gate):
    return bool(comparison['net'] >= gate['supported_net_min'] and
                comparison['corrections'] >= gate['supported_corrections_min'] and
                common_recoveries >= gate['common_error_supported_recoveries_min'] and
                comparison['corrections'] >= gate['corrections_to_regressions_min']*comparison['regressions'])


def inputs(config):
    cfg = read_json(config)
    require(cfg['teacher_dimensions'] == [512, 768] and cfg['primary_teacher_dimension'] == 768 and
            cfg['support_rule'] == dict(confidence_min=.7, probability_margin_min=.2) and
            all(cfg[k] is False for k in ['new_training','gpu_encoding','test_data_used',
                'fusion_predictions_created','automatic_training','parameter_search']), 'Fixed CPU protocol changed')
    sources = {}
    def check(path, expected=None):
        path = Path(path); digest = sha(path)
        require(expected is None or digest == expected, f'Source checksum changed: {path}')
        sources[str(path.resolve())] = digest
        return path
    check(config); check(__file__)
    stage = Path(cfg['stage_root'])
    manifest = read_json(check(stage/'dataset_manifest.json', cfg['dataset_manifest_sha256']))
    require(manifest['data_version'] == '20260921' and manifest['stage'] == 'repechage', 'Wrong data stage')
    for filename in ['class_to_idx.json','train_dev.csv','val_dev.csv','full_train.csv']:
        check(stage/filename, manifest['files'][filename])
    mapping = read_json(stage/'class_to_idx.json')
    classes = [name for name, index in sorted(mapping.items(), key=lambda item:item[1])]
    require(sorted(mapping.values()) == list(range(len(classes))) and len(classes) == manifest['num_classes'], 'Class order differs')
    train, val, full = [read_rows(stage/name) for name in ['train_dev.csv','val_dev.csv','full_train.csv']]
    paths, labels = np.array([r['image_path'] for r in val]), np.array([int(r['label']) for r in val])
    require(len(paths) == len(set(paths)) == manifest['val_dev_samples'] and
            len(train) == manifest['train_dev_samples'], 'Wrong DEV population')
    require(not {r['content_group'] for r in train} & {r['content_group'] for r in val}, 'Train/val content overlap')
    summary = read_json(check(ROOT/cfg['feature_summary']))
    assets = summary['artifact_sha256']
    def asset(filename):
        matched = [p for p in assets if Path(p).name == filename and Path(p).parent.name == 'features']
        require(len(matched) == 1, f'Missing teacher asset: {filename}')
        return check(matched[0], assets[matched[0]])
    feature_file = asset('features768.pt')
    teacher_root = feature_file.parent
    protocol = read_json(check(teacher_root/'protocol.json'))
    teacher_report = read_json(check(teacher_root/'report.json'))
    require(protocol['binding']['dataset_manifest_sha256'] == cfg['dataset_manifest_sha256'] and
            protocol['binding']['class_mapping_sha256'] == sha(stage/'class_to_idx.json') and
            protocol['binding']['data_version'] == '20260921' and
            teacher_report['test_data_used'] is False and teacher_report['candidate_training'] is False and
            teacher_report['train_samples'] == len(train) and teacher_report['val_samples'] == len(val) and
            teacher_report['epochs'] == 20,
            'Teacher scope changed')
    check(stage/'features/manifest.json', protocol['binding']['feature_manifest_sha256'])
    cache_manifest = read_json(stage/'features/manifest.json')
    cached512 = stage/'features/features.pt'
    check(cached512, cache_manifest['tensor_sha256'])
    check(stage/'features/image_paths.json', cache_manifest['paths_file_sha256'])
    require(read_json(stage/'features/image_paths.json') ==
            [Path(r['image_path']).relative_to('train').as_posix() for r in full], '512 feature row order changed')
    check(Path('/home/lux1/.cache/clip/ViT-B-32.pt'), protocol['binding']['official_checkpoint_sha256'])
    # The rule is inherited from the existing v1 recipe, not fitted on this val.
    import yaml
    source_recipe = Path('/home/lux1/noise/worktrees/v1_swa_trial_20260930/outputs/codex/v1_swa_trial_20260930/candidate/config.yaml')
    recipe = yaml.safe_load(check(source_recipe).read_text())
    require(recipe['denoise']['pseudo_threshold'] == .7 and recipe['denoise']['pseudo_margin'] == .2, 'Rule differs from v1')
    feature_artifact = torch.load(feature_file, map_location='cpu', weights_only=True)
    require(feature_artifact['image_paths'] == [r['image_path'] for r in full], 'Feature population order changed')
    full_index = {r['image_path']:i for i,r in enumerate(full)}
    indices = [full_index[p] for p in paths]
    predictions = np.load(asset('predictions.npz'), allow_pickle=False)
    require(np.array_equal(predictions['labels'], labels), 'Teacher labels differ')
    feature_arrays = {768:feature_artifact['preprojection'][indices].numpy(),
                     512:torch.load(cached512, map_location='cpu', weights_only=True)[indices].numpy()}
    heads = {d:torch.load(asset(f'head{d}.pt'),map_location='cpu',weights_only=True) for d in cfg['teacher_dimensions']}
    require(all(heads[d]['weight'].shape == (len(classes),d) for d in heads), 'Teacher class/dimension differs')
    student_report = read_json(check(ROOT/cfg['student_report']))
    plan = read_json(check(student_report['plan'],student_report['plan_sha256']))
    require(student_report['validation_is_independent'] is True and student_report['partition'] == 'train_dev' and
            plan['source_binding']['dataset_manifest_sha256'] == cfg['dataset_manifest_sha256'], 'Student validation is not independent/current')
    student_path = next(p for p in student_report['artifacts'] if Path(p).name == 'val_epoch04_ema_swa_2_4.npz')
    student = load_aligned_npz(check(student_path,student_report['artifacts'][student_path]),paths,labels)['predictions']
    errors = read_json(check(ROOT/cfg['candidate_error_report']))
    aligned = check(errors['aligned_csv'],errors['aligned_csv_sha256'])
    aligned_rows = read_rows(aligned)
    require(np.array_equal(paths,[r['image_path'] for r in aligned_rows]) and
            np.array_equal(labels,[int(r['label']) for r in aligned_rows]), 'Candidate rows differ')
    common_wrong = np.ones(len(val),bool)
    for name in errors['candidate_names']:
        common_wrong &= np.array([int(r[name]) for r in aligned_rows]) != labels
    require(int(common_wrong.sum()) == errors['candidate_error_overlap']['all']['all_candidates_wrong'], 'Common error count differs')
    require(np.array_equal(student,[int(r['lr512_swa']) for r in aligned_rows]), 'Student/candidate cache differs')
    incumbent = read_json(check(ROOT/cfg['incumbent_reference']))
    for kind in ['csv','zip']:
        check(incumbent['artifacts'][kind]['path'],incumbent['artifacts'][kind]['sha256'])
    with zipfile.ZipFile(incumbent['artifacts']['zip']['path']) as archive:
        require(archive.namelist() == ['pred_results.csv'] and archive.read('pred_results.csv') ==
                Path(incumbent['artifacts']['csv']['path']).read_bytes(), 'Incumbent ZIP/CSV differs')
    require(incumbent['submission_checks_passed'] == 9 and incumbent['submission_rows'] == manifest['test_samples'], 'Incumbent check record incomplete')
    check(ROOT/incumbent['submission_check_log'])
    counts = np.bincount([int(r['label']) for r in train],minlength=len(classes))
    tail = np.lexsort((np.arange(len(classes)),counts))[:max(1,len(classes)//10)]
    groups = dict(all=np.ones(len(val),bool), common_candidate_wrong=common_wrong,
        originally_small=np.array([min(int(r['width']),int(r['height'])) < 224 for r in val]),
        tail75=np.isin(labels,tail), other_student_errors=(student != labels)&~common_wrong)
    return cfg, labels, paths, classes, feature_arrays, heads, predictions, student, groups, sources, incumbent


def analyze(config, output):
    start = time.monotonic(); output = Path(output).resolve()
    require(not output.exists(), 'Preserve prior diagnostic output')
    cfg, labels, paths, classes, arrays, heads, archived, student, groups, sources, incumbent = inputs(config)
    scores = {d:replay_head(arrays[d],heads[d]) for d in cfg['teacher_dimensions']}
    reports, cached = {}, dict(labels=labels, image_paths=paths, student=student, **groups)
    for d, score in scores.items():
        require(np.array_equal(score['prediction'],archived[f'head{d}']), 'Native teacher predictions do not replay')
        support = supported(score,cfg['support_rule'])
        comparisons = {g:dict(teacher=metrics(labels,score['prediction'],len(classes),mask),
            student=metrics(labels,student,len(classes),mask), comparison=paired(labels,student,score['prediction'],mask),
            supported_rows=int((mask&support).sum()),
            supported_comparison=paired(labels,student,score['prediction'],mask&support)) for g,mask in groups.items()}
        reports[str(d)] = dict(native_predictions_replayed=len(labels),support_rows=int(support.sum()),
                              unsupported_rows=int((~support).sum()), groups=comparisons)
        cached.update({f'teacher{d}':score['prediction'],f'confidence{d}':score['confidence'],
                       f'margin{d}':score['margin'],f'supported{d}':support})
    primary = reports[str(cfg['primary_teacher_dimension'])]
    recoveries = primary['groups']['common_candidate_wrong']['supported_comparison']['corrections']
    passed = gate_pass(primary['groups']['all']['supported_comparison'],recoveries,cfg['feasibility_gate'])
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'scores.npz',**cached)
    report = dict(experiment_id=cfg['experiment_id'],status='completed_cpu_diagnostic',
        source_files=sources, config_sha256=sha(config),script_sha256=sha(__file__),
        score_file=str(output/'scores.npz'),score_sha256=sha(output/'scores.npz'),validation_rows=len(labels),classes=len(classes),
        teacher_reports=reports,primary_teacher_dimension=cfg['primary_teacher_dimension'],
        gate_passed=passed,decision='supports_transfer_review' if passed else 'close_fixed_confident_teacher_logit_transfer',
        no_fusion_predictions_created=True,training_started=False,new_candidate=False,test_data_used=False,
        native_decoder_comparison_is_causal=False,catastrophic_forgetting_proven=False,
        original_labels_noisy=True,platform_gain_known=False,automatic_training=False,
        seconds=time.monotonic()-start,incumbent_reference=incumbent['artifacts'],
        limitation='Cross-recipe teacher/student recovery budget on original noisy validation labels; no attainable gain, fusion candidate, forgetting causality or platform effect is established')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],decision=report['decision'],primary=primary,
                         seconds=report['seconds'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(2);analyze(args.config,args.output)
