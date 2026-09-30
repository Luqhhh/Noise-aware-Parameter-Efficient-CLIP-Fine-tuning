"""Edge classification, evidence ordering, discrete class caps and diagnostics."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import statistics

from v2.plan import require

LEVEL_WEIGHT = {'weak': 1.0, 'medium': 0.5, 'strong': 0.2}
FIELDS = ['sample_id', 'image_path', 'label', 'content_group', 'nd_level', 'nd_weight',
          'raw_nd_level', 'raw_nd_weight', 'max_conflict_similarity', 'num_conflict_neighbors',
          'mutual_support', 'perceptual_support', 'reason']


def boolean(value):
    return value is True or str(value).lower() == 'true'


def classify(pair, rows, medium, strong):
    require(.94 <= medium <= strong <= .995, 'Invalid frozen CLIP thresholds')
    a, b = rows[int(pair['i'])], rows[int(pair['j'])]
    if a['label'] == b['label'] or a['content_group'] == b['content_group']:
        return 'weak'
    s = float(pair['cosine_similarity'])
    mutual, perceptual = boolean(pair['mutual_nn']), boolean(pair['perceptual_pass'])
    if s >= strong and mutual and perceptual:
        return 'strong'
    if s >= medium and (mutual or perceptual):
        return 'medium'
    return 'weak'


def manifest(rows, pairs, medium, strong, *, eligible=None, cap=.10):
    """Minimum edge weight; restore weakest samples to 1 until the cap holds.

    Only edges wholly within eligible can influence those rows. Output always
    covers full_train; held-out rows remain 1 for dev. Cap uses eligible counts.
    """
    require(cap == .10, 'First-round safety cap is frozen at 10%')
    eligible = set(range(len(rows))) if eligible is None else set(eligible)
    result = [dict(sample_id=r['image_path'], image_path=r['image_path'], label=r['label'],
                   content_group=r['content_group'], nd_level='weak', nd_weight=1.0,
                   raw_nd_level='weak', raw_nd_weight=1.0, max_conflict_similarity=0.0,
                   num_conflict_neighbors=0, mutual_support=False, perceptual_support=False,
                   reason='no_verified_non_exact_cross_label_conflict') for r in rows]
    best = {}
    neighbors = defaultdict(set)
    for pair in pairs:
        i, j = int(pair['i']), int(pair['j'])
        require(0 <= i < j < len(rows), 'Pairs must be unique canonical training indices')
        if not {i, j} <= eligible:
            continue
        level = classify(pair, rows, medium, strong)
        if level == 'weak':
            continue
        rank = (2 if level == 'strong' else 1, int(boolean(pair['perceptual_pass'])),
                int(boolean(pair['mutual_nn'])), float(pair['cosine_similarity']))
        for a, b in ((i, j), (j, i)):
            r = result[a]
            neighbors[a].add(b)
            if LEVEL_WEIGHT[level] < r['nd_weight']:
                r.update(nd_level=level, nd_weight=LEVEL_WEIGHT[level])
            best[a] = max(best.get(a, rank), rank)
            r['max_conflict_similarity'] = max(r['max_conflict_similarity'], float(pair['cosine_similarity']))
            r['mutual_support'] |= boolean(pair['mutual_nn'])
            r['perceptual_support'] |= boolean(pair['perceptual_pass'])
            r['reason'] = 'non_exact_cross_label_near_duplicate'
    classes = defaultdict(list)
    for i in sorted(eligible):
        classes[rows[i]['label']].append(i)
    for indices in classes.values():
        budget, used = cap * len(indices), 0.0
        candidates = sorted((i for i in indices if i in best), key=lambda i: (*(-v for v in best[i]), rows[i]['image_path']))
        for i in candidates:
            r = result[i]
            r.update(raw_nd_level=r['nd_level'], raw_nd_weight=r['nd_weight'], num_conflict_neighbors=len(neighbors[i]))
            cost = 1.0 - r['nd_weight']
            if used + cost <= budget + 1e-10:
                used += cost
            else:
                r.update(nd_level='weak', nd_weight=1.0, reason='class_safety_cap_restored')
    return result


def impact(rows, weights, eligible=None):
    eligible = set(range(len(rows))) if eligible is None else set(eligible)
    groups = defaultdict(list)
    for i in eligible:
        groups[rows[i]['label']].append(i)
    ordered = sorted(groups, key=lambda c: (len(groups[c]), c))
    tail = set(ordered[:max(1, math.ceil(len(ordered) * .10))])
    head = set(ordered[-max(1, math.ceil(len(ordered) * .10)):])
    result = []
    for c in sorted(groups):
        idx = groups[c]
        loss = sum(1 - weights[i]['nd_weight'] for i in idx)
        require(loss / len(idx) <= .10 + 1e-10, 'Class protection failed')
        result.append(dict(label=c, samples=len(idx), raw_affected=sum(weights[i]['raw_nd_weight'] < 1 for i in idx),
                           affected=sum(weights[i]['nd_weight'] < 1 for i in idx),
                           strong=sum(weights[i]['nd_level'] == 'strong' for i in idx),
                           medium=sum(weights[i]['nd_level'] == 'medium' for i in idx),
                           affected_rate=sum(weights[i]['nd_weight'] < 1 for i in idx) / len(idx),
                           equivalent_supervision_loss=loss, supervision_loss_rate=loss / len(idx),
                           stratum='tail' if c in tail else 'head' if c in head else 'middle'))
    return result


def sweep(rows, pairs):
    result = []
    for step in range(12):
        t = round(.94 + step * .005, 3)
        selected = [p for p in pairs if float(p['cosine_similarity']) >= t
                    and rows[int(p['i'])]['content_group'] != rows[int(p['j'])]['content_group']]
        conflict = [p for p in selected if rows[int(p['i'])]['label'] != rows[int(p['j'])]['label']]
        n = len(selected)
        result.append(dict(threshold=t, candidate_pairs=n, cross_label_pairs=len(conflict),
                           involved_images=len({int(p[k]) for p in selected for k in ('i', 'j')}),
                           cross_label_images=len({int(p[k]) for p in conflict for k in ('i', 'j')}),
                           cross_label_ratio=len(conflict) / n if n else None,
                           mutual_nn_ratio=sum(boolean(p['mutual_nn']) for p in selected) / n if n else None,
                           perceptual_confirm_ratio=sum(boolean(p['perceptual_pass']) for p in selected) / n if n else None))
    return result


def clusters(rows, pairs, medium, strong):
    """Connected components are descriptive only; never passed to manifest()."""
    parent = {}
    def find(i):
        parent.setdefault(i, i)
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != i:
            previous, parent[i] = parent[i], root
            i = previous
        return root
    for p in pairs:
        if classify(p, rows, medium, strong) != 'weak':
            a, b = find(int(p['i'])), find(int(p['j']))
            parent[b] = a
    counts = Counter(find(i) for i in list(parent))
    sizes = list(counts.values())
    return dict(clusters=len(sizes), median_cluster_size=statistics.median(sizes) if sizes else 0,
                largest_cluster=max(sizes, default=0))


def gates(rows, weights, trust, eligible=None):
    eligible = set(range(len(rows))) if eligible is None else set(eligible)
    raw = [i for i, r in enumerate(weights) if i in eligible and r['raw_nd_weight'] < 1]
    active = [i for i, r in enumerate(weights) if i in eligible and r['nd_weight'] < 1]
    covered = [i for i in raw if rows[i]['image_path'] in trust and trust[rows[i]['image_path']]['v1_low_trust'] is not None]
    low = sum(trust[rows[i]['image_path']]['v1_low_trust'] for i in covered)
    fraction = low / len(covered) if covered else None
    coverage = len(covered) / len(raw) if raw else 0.0
    bounds = [low / len(raw), (low + len(raw) - len(covered)) / len(raw)] if raw else [None, None]
    a = len(raw) >= math.ceil(.005 * len(eligible))
    b = len({rows[i]['label'] for i in raw}) >= 50
    # V1 has no val_dev trust. Treat missing rows conservatively as low trust
    # in the upper bound, rather than inventing values or blocking 90% coverage.
    c = bounds[1] is not None and bounds[1] <= .95 and len(covered) > low
    # Prevent a passed raw gate from hiding a cap that removes almost all treatment.
    active_ok = len(active) >= math.ceil(.005 * len(eligible)) and len({rows[i]['label'] for i in active}) >= 50
    return dict(gate_a=a, gate_b=b, gate_c=c, low_trust_given_nd=fraction,
                low_trust_given_nd_bounds=bounds, high_trust_nd_images=len(covered) - low,
                trust_coverage=coverage, raw_conflict_images=len(raw), weighted_images=len(active),
                affected_classes=len({rows[i]['label'] for i in active}), post_cap_scale_ok=active_ok,
                training_allowed=a and b and c and active_ok,
                decision='eligible_for_probe' if a and b and c and active_ok else 'close_or_missing_independent_evidence')
