"""Run C only after package-bound, user-reported platform improvements."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from aegis_clip.prelim75 import cache_features, load_plan, train_head
from aegis_clip.runtime import atomic_json_dump, sha256_file


def select_combination(scores, stopped=()):
    valid = {}
    for name in ('H0','H1','V0','V1'):
        row = scores.get(name)
        if not row:
            continue
        score = row.get('accuracy_percent')
        if row.get('source') != 'user_reported_platform' or not isinstance(score,(int,float)) or isinstance(score,bool) or not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError('Only real user-reported platform scores may trigger C')
        valid[name] = score
    if any(v >= 75 for v in valid.values()):
        return {'status':'target_reached_no_new_training'}
    required = set(('H0','H1','V0','V1')) - set(stopped)
    # A fully scored route below the gate rules out C immediately.
    # The other route's pending scores cannot change that decision.
    for route in (('H0', 'H1'), ('V0', 'V1')):
        active = set(route) & required
        if not active:
            return {'status':'closed_no_eligible_route_pair'}
        if active <= set(valid) and max(valid[name] for name in active) < 67.0681:
            return {'status':'closed_below_combination_gate'}
    if not required <= set(valid):
        return {'status':'pending_real_platform_feedback'}
    if not any(k in valid for k in ('H0','H1')) or not any(k in valid for k in ('V0','V1')):
        return {'status':'closed_no_eligible_route_pair'}
    h = max((k for k in ('H0','H1') if k in required), key=lambda k:valid[k])
    v = max((k for k in ('V0','V1') if k in required), key=lambda k:valid[k])
    if valid[h] < 67.0681 or valid[v] < 67.0681:
        return {'status':'closed_below_combination_gate'}
    return {'status':'eligible','head_winner':h,'visual_winner':v,
            'head_accuracy_percent':valid[h],'visual_accuracy_percent':valid[v]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    plan = load_plan(args.config); root = Path(plan['output'])
    score_file = root/'platform_scores.json'
    scores = json.loads(score_file.read_text()) if score_file.exists() else {}
    stopped = []
    for name in ('H0','H1','V0','V1'):
        diagnostic = root/name/'diagnostic/result.json'
        if diagnostic.exists() and json.loads(diagnostic.read_text()).get('engineering_stop') is True:
            stopped.append(name)
    selection = select_combination(scores, stopped)
    print(json.dumps(selection,indent=2),flush=True)
    if selection['status'] != 'eligible' or not args.execute:
        return
    # All four prescribed candidates must finish before this fifth candidate.
    for name in ('H0','H1','V0','V1'):
        diagnostic = json.loads((root/name/'diagnostic/result.json').read_text())
        if not diagnostic['engineering_stop'] and not (root/name/'candidate_report.json').exists():
            raise ValueError('Complete fixed H/V candidate delivery before C')
    for name in (selection['head_winner'],selection['visual_winner']):
        report = json.loads((root/name/'candidate_report.json').read_text())
        if scores[name].get('zip_sha256') != report['zip_sha256'] or sha256_file(root/name/'submission/submission.zip') != report['zip_sha256']:
            raise ValueError('Reported platform score must bind the actual submitted package')
    visual = selection['visual_winner']
    checkpoint = root/visual/'candidate.pt'
    report = json.loads((root/visual/'candidate_report.json').read_text())
    if sha256_file(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('Visual winner checkpoint changed')
    if (root/'C').exists() or (root/'C_cache').exists():
        raise FileExistsError('C budget already started; no resume/overwrite')
    atomic_json_dump({'selection':selection,'score_file_sha256':sha256_file(score_file),
                      'score_rows':scores,'single_model_new_features_new_head':True},root/'combination_registration.json')
    plan['_root_parent_sha256'] = plan['parent_sha256']
    plan['parent'] = str(checkpoint)
    plan['parent_sha256'] = report['checkpoint_sha256']
    plan['_feature_parent_experiment_id'] = f'PRELIM75_V2_{visual}'
    plan['_cache_directory'] = 'C_cache'
    plan['_head_variant'] = selection['head_winner']
    plan['_combination_approved'] = True
    cache_features(plan)
    train_head(plan,'C')

if __name__ == '__main__':
    main()
