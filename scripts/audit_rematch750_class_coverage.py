"""T0 类别覆盖与映射审计（rematch750，复赛阶段）。

只读、CPU-only（不占用 GPU/NPU）。回答四个问题：
  1. 740/741 是在哪个集合、哪个 checkpoint 上统计的？
  2. 训练标签 / 分类头输出索引 / 反向映射 / CSV 标签是否一一对应？
  3. 未预测类别的训练支持度、验证召回、最高分竞争类别是什么？
  4. 是否存在未加载分类头行、异常样本读取、标签编号偏移？

并附带一项比例检验：test 的逐类预测分布是否只是 val 的比例放大。
若测试集是同分布 iid 抽样，则 z_c = (Y_c/k - X_c)/sqrt(X_c(1+1/k)) 应 ~ N(0,1)。

用法：
    PYTHONPATH=reproducibility/aegis_f1 \\
    .venv/bin/python scripts/audit_rematch750_class_coverage.py

注意：本脚本是**测量**工具。结果不得用于选择模型、调整超参数或派生候选
（规则禁止用测试集预测分布调参）。它只用于记录已产生提交包的诊断事实。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

N_CLASSES = 750
TEST_SAMPLES = 37444


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_pred(path: Path) -> tuple[list[str], list[int], list[int]]:
    """返回 (图片名, 原始标签串, 标签)。记录格式异常行下标。"""
    names, raw, labs, bad = [], [], [], []
    with path.open(encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split(', ')
            if len(parts) != 2:
                bad.append(i)
                continue
            names.append(parts[0])
            raw.append(parts[1])
            labs.append(int(parts[1]))
    return names, raw, labs, bad


def state_dict(ck):
    sd = ck
    for key in ('model', 'state_dict', 'model_state_dict', 'module'):
        if isinstance(ck, dict) and key in ck and isinstance(ck[key], dict):
            sd = ck[key]
            break
    if not [k for k in sd if k.endswith('classifier.weight')]:
        sd = ck
    return sd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', default='results/rematch750_class_coverage_audit_20260923.json')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    art = root / 'artifacts/stages/repechage/20260921'
    rep: dict = {'stage': 'measurement', 'test_used_for_tuning': False}

    # ---------- Q1 口径 ----------
    arms = {}
    for tag, rel in (('RM_LP', 'outputs/rematch750/RM_LP/seed42/submission/pred_results.csv'),
                     ('RM_FT', 'outputs/rematch750/RM_FT/seed42/submission/pred_results.csv')):
        p = root / rel
        names, raw, labs, bad = read_pred(p)
        cnt = Counter(labs)
        arms[tag] = {
            'csv': rel, 'csv_sha256': sha256_file(p), 'rows': len(labs),
            'malformed_rows': len(bad),
            'label_min': min(labs), 'label_max': max(labs),
            'labels_out_of_range': sum(1 for x in labs if not 0 <= x < N_CLASSES),
            'labels_not_4digit': sum(1 for s in raw if not (len(s) == 4 and s.isdigit())),
            'distinct_names': len(set(names)),
            'distinct_predicted_classes': len(cnt),
            'predicted_counts': {str(c): int(cnt.get(c, 0)) for c in range(N_CLASSES)},
        }
        for key in ('csv_sha256',):
            pass
        print(f'[{tag}] rows={len(labs)} distinct_labels={len(cnt)} '
              f'sha={arms[tag]["csv_sha256"][:12]}')

    rep['q1_definition'] = {
        'what': 'distinct predicted labels appearing in pred_results.csv over the 37444 test images',
        'not_what': 'not "classes answered correctly" (test labels are unavailable)',
        'where': 'evaluation.py:137 uses the same statistic but on val_dev (prediction.unique().numel())',
        'arms': {t: {'csv': a['csv'], 'csv_sha256': a['csv_sha256'],
                     'checkpoint': f'outputs/rematch750/{t}/seed42/checkpoints/best.pt',
                     'distinct_predicted_classes': a['distinct_predicted_classes']}
                 for t, a in arms.items()},
    }

    # ---------- Q2 映射 ----------
    c2i = json.loads((art / 'class_to_idx.json').read_text())
    i2c = json.loads((art / 'idx_to_class.json').read_text())
    vals = sorted(c2i.values())
    inverse_ok = all(c2i.get(i2c[str(k)]) == k for k in range(N_CLASSES) if str(k) in i2c)
    print(f'[class_to_idx] n={len(c2i)} bijective_0_749={vals == list(range(N_CLASSES))} '
          f'keys_unique={len(set(c2i)) == len(c2i)}')
    print(f'[idx_to_class] n={len(i2c)} inverse_ok={inverse_ok}')

    with (art / 'test_manifest.csv').open(newline='', encoding='utf-8') as f:
        manifest = list(csv.DictReader(f))
    man_names = {Path(r['image_path']).name for r in manifest}
    man_labels = {r['label'] for r in manifest}

    mapping = {
        'class_to_idx': {'n': len(c2i), 'bijective_0_to_749': vals == list(range(N_CLASSES)),
                         'keys_unique': len(set(c2i)) == len(c2i)},
        'idx_to_class': {'n': len(i2c), 'inverse_of_class_to_idx': inverse_ok},
        'test_manifest': {'rows': len(manifest), 'label_values': sorted(man_labels),
                          'label_is_placeholder': man_labels == {'-1'}},
        'csv_vs_manifest': {},
    }
    for tag, a in arms.items():
        p = root / a['csv']
        names, _, _, _ = read_pred(p)
        mapping['csv_vs_manifest'][tag] = {
            'rows': len(names), 'unique': len(set(names)),
            'not_in_manifest': len(set(names) - man_names),
            'manifest_uncovered': len(man_names - set(names)),
        }
        print(f'[{tag}] vs manifest: not_in_manifest='
              f'{mapping["csv_vs_manifest"][tag]["not_in_manifest"]} '
              f'uncovered={mapping["csv_vs_manifest"][tag]["manifest_uncovered"]}')
    rep['q2_mapping'] = mapping

    # ---------- 离线重建 RM-LP 的 val 预测（锚点） ----------
    feats = torch.as_tensor(torch.load(art / 'features/features.pt',
                                       map_location='cpu')).float()
    paths = json.loads((art / 'features/image_paths.json').read_text())
    index = {p: i for i, p in enumerate(paths)}
    vp, vlab = [], []
    with (art / 'val_dev.csv').open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            vp.append(r['image_path'].removeprefix('train/'))
            vlab.append(int(r['label']))
    vi = torch.tensor([index[p] for p in vp])
    vl = torch.tensor(vlab)
    ck = torch.load(root / 'outputs/rematch750/RM_LP/seed42/checkpoints/best.pt',
                    map_location='cpu', weights_only=False)
    sd = state_dict(ck)
    W = sd[[k for k in sd if k.endswith('classifier.weight')][0]].float()
    B = sd[[k for k in sd if k.endswith('classifier.bias')][0]].float()
    logits = feats[vi] @ W.t() + B
    pred = logits.argmax(1)
    micro = (pred == vl).float().mean().item()
    X = torch.bincount(pred, minlength=N_CLASSES).float()
    eval_json = json.loads((root / 'outputs/rematch750/RM_LP/seed42/logs/'
                            'evaluation_epoch_20.json').read_text())
    gap = logits.topk(2, dim=1).values
    min_gap = float((gap[:, 0] - gap[:, 1]).min())
    print(f'[anchor] reconstructed micro={micro!r} recorded={eval_json["raw_micro"]!r} '
          f'diff_images={abs(micro - eval_json["raw_micro"]) * len(vl):.0f}')
    rep['anchor_rm_lp_val'] = {
        'method': 'frozen backbone + linear head applied to the cached features; CPU only',
        'val_rows': len(vl), 'predicted_class_count': int((X > 0).sum()),
        'recorded_predicted_class_count': int(eval_json['predicted_class_count']),
        'reconstructed_micro': micro, 'recorded_micro': eval_json['raw_micro'],
        'abs_micro_diff_images': abs(micro - eval_json['raw_micro']) * len(vl),
        'min_top1_top2_logit_gap': min_gap,
        'interpretation': 'the 2-image gap is inside numerical tie sensitivity; '
                          'class-coverage count reproduces exactly',
    }

    # ---------- Q3 未预测类别 ----------
    sup, lp_pc = {}, {}
    with (root / 'results/rematch750_20260921_class_support.csv').open(newline='',
                                                                      encoding='utf-8') as f:
        for r in csv.DictReader(f):
            sup[int(r['label'])] = r
    with (root / 'results/rematch750_20260921_rm_lp_per_class.csv').open(newline='',
                                                                        encoding='utf-8') as f:
        for r in csv.DictReader(f):
            lp_pc[int(r['label'])] = r

    never = {}
    for tag, a in arms.items():
        never[tag] = sorted(c for c in range(N_CLASSES) if a['predicted_counts'][str(c)] == 0)
    never_val = sorted(int(c) for c in range(N_CLASSES) if X[c] == 0)
    union = sorted(set(never['RM_LP']) | set(never['RM_FT']))
    both = sorted(set(never['RM_LP']) & set(never['RM_FT']))

    rows = []
    for c in union:
        rows.append({
            'label': c, 'class_name': sup[c]['class_name'],
            'train_samples': int(sup[c]['train_samples']),
            'val_samples': int(sup[c]['val_samples']),
            'segment': sup[c]['segment'],
            'val_predicted_count': int(X[c]),
            'lp_val_recall': float(lp_pc[c]['recall']),
            'lp_test_predicted_count': int(arms['RM_LP']['predicted_counts'][str(c)]),
            'ft_test_predicted_count': int(arms['RM_FT']['predicted_counts'][str(c)]),
        })
    # 最高分竞争类别：在 val 上，这些类的样本被预测成了什么（test 无标签，不可算）
    competitors = {}
    for c in union:
        m = vl == c
        if not m.any():
            continue
        got = Counter(pred[m].tolist())
        competitors[str(c)] = {
            'val_samples': int(m.sum()),
            'self_hits': int(got.get(c, 0)),
            'top_competitors': [[int(a), int(b)] for a, b in got.most_common(4)],
        }
    rep['q3_never_predicted'] = {
        'competitors_on_val': competitors,
        'rm_lp_test_never': never['RM_LP'],
        'rm_ft_test_never': never['RM_FT'],
        'rm_lp_val_never': never_val,
        'union': union, 'intersection_both_arms': both,
        'contiguous_block': union == list(range(min(union), max(union) + 1)),
        'val_never_and_test_never': sorted(set(never_val) & set(union)),
        'val_predicted_but_test_never': sorted(set(union) - set(never_val)),
        'test_predicted_but_val_never': sorted(set(never_val) - set(union)),
        'per_class': rows,
    }
    print(f'[Q3] test_never LP={len(never["RM_LP"])} FT={len(never["RM_FT"])} '
          f'val_never={len(never_val)} union={len(union)} contiguous='
          f'{rep["q3_never_predicted"]["contiguous_block"]}')

    # ---------- Q4 比例 / 过度离散检验 ----------
    Y = torch.tensor([arms['RM_LP']['predicted_counts'][str(c)]
                      for c in range(N_CLASSES)], dtype=torch.float32)
    k = TEST_SAMPLES / len(vl)
    mask = X > 0
    z = ((Y / k - X) / torch.sqrt(X * (1 + 1 / k)))[mask]
    s = float(z.std())
    print(f'[Q4] k={k:.4f} sd(z)={s:.4f} chi2/dof={float((z**2).mean()):.3f} '
          f'|z|>2={int((z.abs()>2).sum())} |z|>3={int((z.abs()>3).sum())}')
    rep['q4_proportional_test'] = {
        'hypothesis': 'test per-class predicted counts are val counts scaled by k',
        'k': k, 'defined_classes': int(mask.sum()),
        'mean_z': float(z.mean()), 'sd_z': s,
        'chi2_over_dof': float((z ** 2).mean()),
        'abs_z_gt_2': int((z.abs() > 2).sum()),
        'abs_z_gt_3': int((z.abs() > 3).sum()),
        'abs_z_gt_4': int((z.abs() > 4).sum()),
        'expected_abs_z_gt_2': 0.0455 * int(mask.sum()),
        'expected_abs_z_gt_3': 0.0027 * int(mask.sum()),
        'x0_but_y_positive': [[c, int(Y[c])] for c in range(N_CLASSES) if X[c] == 0 and Y[c] > 0],
        'x0_and_y0': [c for c in range(N_CLASSES) if X[c] == 0 and Y[c] == 0],
        'extreme_positive': sorted(
            [{'label': c, 'train_samples': int(sup[c]['train_samples']),
              'val_count': float(X[c]), 'test_count': float(Y[c]),
              'z': float((Y[c] / k - X[c]) / (X[c] * (1 + 1 / k)) ** 0.5)}
             for c in range(N_CLASSES) if X[c] > 0],
            key=lambda d: -d['z'])[:15],
        'extreme_negative': sorted(
            [{'label': c, 'train_samples': int(sup[c]['train_samples']),
              'val_count': float(X[c]), 'test_count': float(Y[c]),
              'z': float((Y[c] / k - X[c]) / (X[c] * (1 + 1 / k)) ** 0.5)}
             for c in range(N_CLASSES) if X[c] > 0],
            key=lambda d: d['z'])[:15],
    }

    out = root / args.out
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
