"""Read-only class support diagnostics for a fixed sample/group manifest."""
from collections import Counter, defaultdict


def summarize_support(raw_rows, fit_rows, validation_rows, groups, class_names):
    """Rows use canonical relative paths; groups must come from a bound source.

    No inferred head/tail definition or fabricated accuracy for unsupported classes.
    Content conflicts describe recorded labels, not true annotation errors.
    """
    count = len(class_names)
    if not count:
        raise ValueError('Empty class mapping')

    def validate(rows):
        result = {}
        for row in rows:
            path, label = row['image_path'], int(row['label'])
            if path in result:
                raise ValueError('Duplicate sample identity')
            if path not in groups or not 0 <= label < count:
                raise ValueError('Missing content group or invalid class')
            result[path] = label
        return result

    raw, fit, val = map(validate, (raw_rows, fit_rows, validation_rows))
    for subset in (fit, val):
        if any(path not in raw or raw[path] != label for path, label in subset.items()):
            raise ValueError('Subset differs from raw sample identities/labels')
    labels_by_group = defaultdict(set)
    for path, label in raw.items():
        labels_by_group[groups[path]].add(label)
    conflicts = {group for group, labels in labels_by_group.items() if len(labels) > 1}
    raw_counts, fit_counts, val_counts = map(Counter, (raw.values(), fit.values(), val.values()))
    unique = defaultdict(set)
    conflict_counts = Counter()
    for path, label in raw.items():
        unique[label].add(groups[path])
        if groups[path] in conflicts:
            conflict_counts[label] += 1
    fit_groups = {groups[path] for path in fit}
    overlap = Counter(label for path, label in val.items() if groups[path] in fit_groups)
    return [dict(class_id=c, class_name=str(name), raw_samples=raw_counts[c],
                 unique_content_groups=len(unique[c]), conflict_group_samples=conflict_counts[c],
                 fit_samples=fit_counts[c], validation_support=val_counts[c],
                 validation_groups_seen_by_current_fit_samples=overlap[c],
                 raw_correct=None, raw_accuracy=None, clean_core_support=None,
                 clean_core_accuracy=None, frequency_segment=None)
            for c, name in enumerate(class_names)]
