"""One fixed content-distinct k16 diagnostic on task-trained 768 features."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml
from aegis_clip.runtime import atomic_json_dump
from diagnose_v1_candidate_errors import require, read_json, read_rows, sha, metrics, paired

ROOT = Path(__file__).resolve().parents[1]


def gallery_indices(train):
    by_group = defaultdict(list)
    for i, row in enumerate(train):
        by_group[row['content_group']].append(i)
    indices, conflict_rows, conflict_groups = [], 0, 0
    for members in by_group.values():
        if len({train[i]['label'] for i in members}) != 1:
            conflict_rows += len(members)
            conflict_groups += 1
            continue
        indices.append(min(members, key=lambda i: train[i]['image_path']))
    indices.sort(key=lambda i: train[i]['image_path'])
    return np.array(indices), dict(original_train_rows=len(train), original_content_groups=len(by_group),
        gallery_rows=len(indices), excluded_conflict_groups=conflict_groups, excluded_conflict_rows=conflict_rows,
        excluded_redundant_same_label_rows=len(train)-conflict_rows-len(indices))


def normalized64(features):
    x = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    require(np.isfinite(x).all() and (norms > 0).all(), 'Invalid feature vectors')
    return x/norms


def inspect(config):
    cfg = read_json(config)
    require(cfg['neighbors'] == 16 and cfg['consensus_votes_min'] == 12 and cfg['cosine_dtype'] == 'float64' and
        cfg['cosine_round_decimals'] == 9 and cfg['neighbor_tie_rule'] == 'gallery_image_path_ascending' and
        cfg['vote_tie_rule'] == 'class_index_ascending' and cfg['query_chunk'] == 128 and cfg['gallery_chunk'] == 8192 and
        cfg['cost_rows'] == cfg['independent_neighbor_rows'] == 64 and cfg['seed'] == 42 and
        all(cfg[k] is False for k in ['new_training','parameter_search','automatic_full_training','test_predictions_used','platform_upload']),
        'Only the frozen k16 diagnostic is supported')
    files = {}
    def checked(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        require(expected is None or digest == expected, f'Source changed: {path}')
        files[str(path)] = digest
        return path
    source = Path(cfg['source_root'])
    report = read_json(checked(ROOT/cfg['source_report']))
    require(report == read_json(checked(source/'report.json')) and report['status'] == 'completed_verified_delivery' and
        report['validation_independent'] is True and report['visual_parameter_updates'] is False, 'CRT source not delivered')
    binding = report['binding']
    stage = Path(cfg['stage_root'])
    manifest = read_json(checked(stage/'dataset_manifest.json', binding['dataset_manifest_sha256']))
    mapping = read_json(checked(stage/'class_to_idx.json', binding['class_mapping_sha256']))
    classes = [c for c, i in sorted(mapping.items(), key=lambda item: item[1])]
    require(manifest['stage'] == binding['stage'] == 'repechage' and
        manifest['data_version'] == binding['data_version'] == '20260921', 'Wrong current stage')
    train, val, full = [read_rows(checked(stage/name)) for name in ['train_dev.csv','val_dev.csv','full_train.csv']]
    require(not {r['content_group'] for r in train} & {r['content_group'] for r in val}, 'Train/val content overlap')
    parent_cfg = yaml.safe_load(checked(cfg['source_parent_config']).read_text())
    require(parent_cfg['source']['partition'] == 'train_dev', 'Parent trained on validation')
    official = checked('/home/lux1/.cache/clip/ViT-B-32.pt', binding['official_checkpoint_sha256'])
    cache_path = checked(source/'frozen_features768.pt', cfg['feature_cache_sha256'])
    side = read_json(checked(cache_path.with_suffix('.sha256.json')))
    require(side['sha256'] == cfg['feature_cache_sha256'] and side['binding'] == binding, 'Feature cache sidecar differs')
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    require(cache['binding'] == binding and cache['feature_path'] == 'ln_post_pre_projection' and
        cache['image_size'] == 448 and cache['visual_parameter_updates'] is False and
        cache['image_paths'] == [r['image_path'] for r in full] and len(set(cache['image_paths'])) == len(full) and
        cache['features'].shape == (len(full), 768), 'Invalid task-trained feature cache')
    closure = read_json(checked(ROOT/cfg['source_closure']))
    checked(source/'validation_predictions.npz', closure['validation_npz_sha256'])
    parent = checked(Path(cfg['source_parent_config']).parent/'selected.pt', binding['parent_checkpoint_sha256'])
    parent_side = read_json(checked(parent.with_suffix('.sha256.json')))
    require(parent_side['sha256'] == binding['parent_checkpoint_sha256'], 'Parent sidecar differs')
    parent_payload = torch.load(parent, map_location='cpu', weights_only=False)
    require(parent_payload['classes'] == classes, 'Parent class order differs')
    selected_path = checked(source/'unbalanced/selected.pt', report['packages']['unbalanced']['checkpoint_sha256'])
    selected_side = read_json(checked(selected_path.with_suffix('.sha256.json')))
    selected = torch.load(selected_path, map_location='cpu', weights_only=False)
    require(selected_side['sha256'] == report['packages']['unbalanced']['checkpoint_sha256'] and
        selected_side['binding'] == selected['binding'] and selected['classes'] == classes and
        selected['binding']['frozen_features_sha256'] == cfg['feature_cache_sha256'] and
        selected['binding']['arm'] == 'unbalanced', 'Head checkpoint provenance differs')
    full_index = {p: i for i, p in enumerate(cache['image_paths'])}
    train_cache = np.array([full_index[r['image_path']] for r in train])
    val_cache = np.array([full_index[r['image_path']] for r in val])
    require(not set(train_cache) & set(val_cache), 'Feature indices leak validation into gallery')
    labels = np.array([int(r['label']) for r in val]); paths = np.array([r['image_path'] for r in val])
    with np.load(source/'validation_predictions.npz', allow_pickle=False) as z:
        require(np.array_equal(z['labels'], labels) and np.array_equal(z['image_paths'], paths), 'CRT native rows differ')
        native = z['unbalanced768'].copy(); tail = z['tail'].copy()
        require(np.array_equal(cache['parent_center_val_logits'].argmax(1).numpy(), z['parent512']), 'Source parent cache differs')
    # Same FP32 replay used in the delivered CRT audit; no new head fitting.
    val_features = cache['features'][val_cache].numpy()
    weight = selected['selected_state']['head.weight'].numpy()
    x = val_features/np.maximum(np.linalg.norm(val_features, axis=1, keepdims=True), 1e-12)
    w = weight/np.maximum(np.linalg.norm(weight, axis=1, keepdims=True), 1e-12)
    require(np.array_equal((x@w.T).argmax(1), native), 'All native head predictions must replay')
    require(int((native == labels).sum()) == report['metrics']['unbalanced']['correct'], 'Native metric differs')
    reflection = read_json(checked(ROOT/cfg['source_reflection']))
    common_path = next(p for p in reflection['artifacts'] if Path(p).name == 'lr512_swa.npz')
    checked(common_path, reflection['artifacts'][common_path])
    with np.load(common_path, allow_pickle=False) as z:
        require(np.array_equal(z['labels'], labels) and np.array_equal(z['image_paths'], paths), 'Common-error alignment differs')
        common = z['common_candidate_wrong'].copy()
    require((native[common] != labels[common]).all(), 'Frozen common-error definition differs')
    gi, gallery_info = gallery_indices(train)
    gallery_labels = np.array([int(train[i]['label']) for i in gi])
    gallery_paths = np.array([train[i]['image_path'] for i in gi])
    gallery_info['classes'] = len(set(gallery_labels.tolist()))
    gallery_info['classes_absent'] = sorted(set(range(len(classes)))-set(gallery_labels.tolist()))
    groups = dict(all=np.ones(len(val), bool), common_candidate_wrong=common, tail75=np.isin(labels, tail),
        small=np.array([min(int(r['width']), int(r['height'])) < 224 for r in val]))
    incumbent = read_json(checked(ROOT/cfg['incumbent_reference']))
    for k in ['csv','zip']:
        checked(incumbent['artifacts'][k]['path'], incumbent['artifacts'][k]['sha256'])
    checked(ROOT/incumbent['submission_check_log'])
    require(incumbent['submission_checks_passed'] == 9, 'Incumbent check reference incomplete')
    return dict(cfg=cfg, sources=files, classes=classes, labels=labels, paths=paths, native=native, groups=groups,
        gallery_indices=gi, gallery_paths=gallery_paths, gallery_labels=gallery_labels, gallery_info=gallery_info,
        gallery=normalized64(cache['features'][train_cache[gi]].numpy()), query=normalized64(val_features), incumbent=incumbent)


def binding(config, data):
    return dict(config_sha256=sha(config), script_sha256=sha(__file__), source_files=data['sources'])


def prepare(config, output):
    data = inspect(config); out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out/'frozen.npz', labels=data['labels'], image_paths=data['paths'], native=data['native'],
        gallery_train_indices=data['gallery_indices'], gallery_image_paths=data['gallery_paths'],
        gallery_labels=data['gallery_labels'], **data['groups'])
    pre = dict(status='prepared_verified_cpu', experiment_id=data['cfg']['experiment_id'], binding=binding(config, data),
        frozen_sha256=sha(out/'frozen.npz'), gallery=data['gallery_info'], native_rows_replayed=len(data['labels']),
        native={g:metrics(data['labels'], data['native'], len(data['classes']), m) for g, m in data['groups'].items()},
        new_training=False, test_predictions_used=False, wall_clock_time_limit=None)
    atomic_json_dump(pre, out/'preflight.json'); print(json.dumps(pre, ensure_ascii=False), flush=True)


def stable_best(scores, indices, k):
    by_index = torch.argsort(indices, dim=1, stable=True)
    scores = scores.gather(1, by_index); indices = indices.gather(1, by_index)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
    return scores.gather(1, order), indices.gather(1, order)


@torch.no_grad()
def neighbors(query, gallery, cfg, output=None, phase='diagnostic'):
    gallery = torch.from_numpy(gallery).to('cuda'); nearest, keys = [], []
    start_time = time.monotonic(); last_report = start_time
    for start in range(0, len(query), cfg['query_chunk']):
        q = torch.from_numpy(query[start:start+cfg['query_chunk']]).to('cuda')
        best_s = best_i = None
        for left in range(0, len(gallery), cfg['gallery_chunk']):
            scores = torch.round((q@gallery[left:left+cfg['gallery_chunk']].T)*10**cfg['cosine_round_decimals']).long()
            index = torch.arange(left, left+scores.shape[1], device='cuda').expand(len(q), -1)
            # Each gallery block already has ascending paths, so score-stable sort suffices.
            order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :cfg['neighbors']]
            scores, index = scores.gather(1, order), index.gather(1, order)
            if best_s is None: best_s, best_i = scores, index
            else: best_s, best_i = stable_best(torch.cat([best_s, scores], 1), torch.cat([best_i, index], 1), cfg['neighbors'])
        nearest.append(best_i.cpu().numpy()); keys.append(best_s.cpu().numpy())
        now = time.monotonic()
        if output is not None and (start == 0 or now-last_report >= 30 or start+len(q) == len(query)):
            progress = dict(status='retrieving', phase=phase, rows_completed=start+len(q), rows=len(query), elapsed_seconds=now-start_time)
            atomic_json_dump(progress, output/'progress.json'); print(json.dumps(progress), flush=True); last_report=now
    return np.concatenate(nearest), np.concatenate(keys)


def votes(indices, gallery_labels, classes):
    counts = np.zeros((len(indices), classes), dtype=np.int64)
    for i, labels in enumerate(gallery_labels[indices]): counts[i] = np.bincount(labels, minlength=classes)
    prediction = counts.argmax(1)
    return prediction, counts.max(1), counts


def context(config, output):
    data = inspect(config); out = Path(output).resolve(); pre = read_json(out/'preflight.json')
    require(pre['binding'] == binding(config, data) and pre['frozen_sha256'] == sha(out/'frozen.npz'), 'Frozen inputs changed')
    return data, out, pre


def require_idle():
    import subprocess
    result = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True, capture_output=True, check=True)
    require(not result.stdout.strip(), 'Another CUDA compute task is running')


def cost(config, output):
    data, out, pre = context(config, output); require(not (out/'cost.json').exists(), 'No repeated cost probe')
    require_idle(); torch.cuda.reset_peak_memory_stats(); start=time.monotonic()
    idx, key = neighbors(data['query'][:data['cfg']['cost_rows']], data['gallery'], data['cfg'])
    require(idx.shape == (64,16) and len(np.unique(idx[0])) == 16, 'Real retrieval cost check failed')
    result = dict(status='passed_discarded_cost_outputs', query_rows=64, gallery_rows=len(data['gallery']),
        seconds=time.monotonic()-start, peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated(), dtype='float64',
        training_updates=0, wall_clock_time_limit=None)
    atomic_json_dump(result, out/'cost.json'); print(json.dumps(result), flush=True)


def run(config, output):
    data, out, pre = context(config, output); cfg=data['cfg']
    require(not (out/'status.json').exists() and read_json(out/'cost.json')['status']=='passed_discarded_cost_outputs', 'Fresh fixed run required')
    require_idle(); start=time.monotonic(); atomic_json_dump(dict(status='retrieving'), out/'status.json')
    try:
        idx, scores = neighbors(data['query'], data['gallery'], cfg, out)
        pred, strength, counts = votes(idx, data['gallery_labels'], len(data['classes']))
        supported = strength >= cfg['consensus_votes_min']; disagree = pred != data['native']
        masks = dict(data['groups'], supported=supported, unsupported=~supported,
            supported_disagreement=supported&disagree, common_wrong_supported=data['groups']['common_candidate_wrong']&supported)
        validation = {g:dict(native=metrics(data['labels'],data['native'],len(data['classes']),m),
            neighbors=metrics(data['labels'],pred,len(data['classes']),m),
            paired_to_native=paired(data['labels'],data['native'],pred,m)) for g,m in masks.items()}
        np.savez_compressed(out/'neighbors.npz', labels=data['labels'], image_paths=data['paths'], native=data['native'],
            neighbor_gallery_indices=idx, cosine_keys=scores, prediction=pred, max_votes=strength, class_votes=counts, **masks)
        g=cfg['mechanism_gate'];target=validation['supported_disagreement'];pair=target['paired_to_native']
        criteria=dict(target_errors=target['native']['errors']>=g['supported_disagreement_parent_errors_min'],
            corrections=pair['corrections']>=g['supported_disagreement_corrections_min'], net=pair['net']>=g['supported_disagreement_net_min'],
            correction_regression_ratio=pair['corrections']>=g['supported_corrections_to_regressions_min']*pair['regressions'],
            common_wrong_corrections=validation['common_wrong_supported']['paired_to_native']['corrections']>=g['common_wrong_supported_corrections_min'])
        passed=all(criteria.values())
        report=dict(status='completed_fixed_neighbor_diagnostic',experiment_id=cfg['experiment_id'],binding=pre['binding'],
            gallery=data['gallery_info'],source_native_rows_replayed=len(pred),validation=validation,gate_criteria=criteria,
            mechanism_gate_passed=passed,decision='supports_bounded_task_geometry_review' if passed else 'close_fixed_task_neighbor_entry',
            artifacts={str(out/'neighbors.npz'):sha(out/'neighbors.npz')},elapsed_seconds=time.monotonic()-start,
            training_updates=0,new_candidate=False,test_predictions_used=False,fused_predictions_generated=False,
            parameter_search=False,automatic_full_training=False,platform_gain_known=False,clean_labels_known=False,
            incumbent=data['incumbent']['artifacts'],wall_clock_time_limit=None)
        atomic_json_dump(report,out/'report.json'); atomic_json_dump(dict(status=report['status']),out/'status.json')
        print(json.dumps({k:v for k,v in report.items() if k not in ['binding','incumbent','artifacts']},ensure_ascii=False),flush=True)
    except Exception as error:
        atomic_json_dump(dict(status='failed',error=str(error),automatically_restarted=False),out/'status.json');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','cost','run'])
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.action!='prepare' and not args.execute:parser.error('CUDA requires --execute')
    torch.set_num_threads(2);dict(prepare=prepare,cost=cost,run=run)[args.action](args.config,args.output)
