"""Content-group split feasibility without selecting a fallback policy."""
from collections import Counter, defaultdict
import math


def diagnose_group_split(labels, groups, *, num_classes, val_ratio, folds=None):
    if len(labels) != len(groups) or not labels:
        raise ValueError('nonempty equally sized labels and groups required')
    if not 0 < val_ratio < 1 or num_classes < 1:
        raise ValueError('valid class count and validation ratio required')
    if any(isinstance(x, bool) or int(x) != x or not 0 <= x < num_classes for x in labels):
        raise ValueError('class index outside mapping')
    if folds is not None and (not isinstance(folds, int) or folds < 2):
        raise ValueError('folds must be an integer >=2')
    by_group = defaultdict(list)
    for label, group in zip(labels, groups):
        if not isinstance(group, str) or not group:
            raise ValueError('nonempty content-group identity required')
        by_group[group].append(int(label))
    raw = Counter(labels); memberships = [set() for _ in range(num_classes)]
    conflicts = Counter(); majority = Counter()
    for group, values in by_group.items():
        counts = Counter(values); majority[counts.most_common(1)[0][0]] += 1
        for label in counts:
            memberships[label].add(group)
            if len(counts) > 1: conflicts[label] += counts[label]
    rows = [dict(class_id=c, raw_samples=raw[c], unique_content_groups=len(memberships[c]),
                 majority_label_groups=majority[c], conflict_group_samples=conflicts[c])
            for c in range(num_classes)]
    bad = [c for c in range(num_classes) if majority[c] < 2]
    val_groups = math.ceil(len(by_group) * val_ratio)
    errors = []
    if bad: errors.append({'reason':'fewer_than_two_majority_groups','classes':bad})
    if min(val_groups, len(by_group)-val_groups) < num_classes:
        errors.append({'reason':'split_capacity_below_class_count','classes':list(range(num_classes))})
    if folds is not None:
        insufficient = [c for c in range(num_classes) if len(memberships[c]) < folds]
        if insufficient:errors.append({'reason':'too_few_independent_groups_for_folds','classes':insufficient})
    return dict(status='blocked' if errors else 'checks_passed', classes=rows, errors=errors,
                unique_content_groups=len(by_group), target_val_groups=val_groups,
                grouping='SHA256 byte content; near duplicates not guaranteed excluded',
                automatic_fallback=False)
