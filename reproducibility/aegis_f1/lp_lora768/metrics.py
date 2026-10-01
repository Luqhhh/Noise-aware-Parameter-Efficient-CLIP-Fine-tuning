"""Original-label paired diagnostics on the pre-frozen population and groups."""
import numpy as np


def metrics(labels, predictions, classes):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    count = np.bincount(labels, minlength=classes)
    correct = np.bincount(labels[labels == predictions], minlength=classes)
    supported = count > 0
    return dict(rows=len(labels), correct=int(correct.sum()), micro=float(np.mean(labels == predictions)) if len(labels) else None,
                macro=float(np.mean(correct[supported]/count[supported])) if supported.any() else None,
                covered_classes=int(supported.sum()))


def pair(labels, before, after, classes):
    labels, before, after = map(np.asarray, (labels, before, after))
    corrections = int(((before != labels) & (after == labels)).sum())
    regressions = int(((before == labels) & (after != labels)).sum())
    return dict(before=metrics(labels, before, classes), after=metrics(labels, after, classes),
                changed=int((before != after).sum()), corrections=corrections, regressions=regressions,
                net_correct=corrections-regressions,
                corrections_to_regressions=corrections/regressions if regressions else (None if not corrections else "infinite"))


def grouped_pair(context, labels, before, after):
    labels = np.asarray(labels)
    target = np.isin(labels, context.groups["target_classes"])
    tail = np.isin(labels, context.groups["tail_classes"])
    masks = dict(all=np.ones(len(labels), dtype=bool), target=target, tail75=tail,
                 non_target=~target, other=~(target | tail), target_tail_overlap=target & tail)
    result = {k: pair(labels[m], np.asarray(before)[m], np.asarray(after)[m], len(context.classes)) for k, m in masks.items()}
    result["per_target_pair"] = {"-".join(map(str, p["classes"])): pair(labels[m], np.asarray(before)[m], np.asarray(after)[m], len(context.classes))
        for p in context.groups["top_pairs"] for m in [np.isin(labels, p["classes"]) ]}
    return result
