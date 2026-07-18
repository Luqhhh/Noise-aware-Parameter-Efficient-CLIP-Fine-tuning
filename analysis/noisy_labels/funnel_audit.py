"""Funnel audit — automatic per-step survivor counts for all 3 selectors.

Outputs audit/task3_filter_funnel.json and audit/task3_filter_funnel.csv.
No TBD, no estimated, no manual — all counts are real selector output.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path("/home/lux1/noise")
AUDIT = REPO / "audit"


def load_data():
    quality = pd.read_csv(REPO / "outputs/phase/phase3/oof/sample_quality_with_kta.csv")
    logits_data = torch.load(
        REPO / "outputs/phase/phase3/oof/oof_logits.pt", map_location="cpu"
    )
    probs = F.softmax(logits_data["logits"].float(), dim=1)
    labels = torch.tensor(quality["original_label"].to_numpy(copy=True))
    return quality, probs, labels


def get_issues(quality, probs, labels):
    from analysis.noisy_labels.confident_joint import (
        build_confident_joint, estimate_class_thresholds, rank_label_issues,
    )
    num_classes = 500
    thresholds = estimate_class_thresholds(probs, labels, num_classes)
    cj = build_confident_joint(probs, labels, thresholds, num_classes)
    knn = quality["knn_agreement"].to_numpy() if "knn_agreement" in quality.columns else None
    flip = quality["flip_consistency"].to_numpy() if "flip_consistency" in quality.columns else None
    marg = quality["top1_margin"].to_numpy() if "top1_margin" in quality.columns else None
    return rank_label_issues(
        probs, labels, thresholds, cj,
        max_class_reject_rate=0.10, max_global_reject_rate=0.10,
        knn_agreement=knn, flip_consistency=flip, top1_margin=marg,
    )


def funnel_cl_classwise_drop(quality, probs, labels):
    """cl_classwise_drop: confident joint + classwise cap only."""
    issues = get_issues(quality, probs, labels)
    issue_indices = set(issues[issues["selected"]]["index"].values)
    n = len(quality)
    return [
        ("total", n),
        ("cj_issue_selected", len(issue_indices)),
        ("rejected_after_caps", len(issue_indices)),
    ]


def funnel_cl_knn_drop(quality, probs, labels):
    """cl_knn_drop: CJ + OOF/kNN consensus + margin + knn_agreement + caps."""
    NUM_CLASSES = 500
    issues = get_issues(quality, probs, labels)
    issue_indices = set(issues[issues["selected"]]["index"].values)
    n = len(quality)

    class_margin_q75 = {}
    for c in range(NUM_CLASSES):
        mask = quality["original_label"] == c
        if mask.sum() > 0:
            class_margin_q75[c] = float(quality.loc[mask, "top1_margin"].quantile(0.75))
        else:
            class_margin_q75[c] = 0.0

    EPS = 1e-6

    def count(cond_fn):
        return sum(1 for i, row in quality.iterrows() if cond_fn(i, row))

    cj = count(lambda i, r: i in issue_indices)
    oof = count(lambda i, r: i in issue_indices and int(r["oof_top1"]) != int(r["original_label"]))
    knn = count(lambda i, r: i in issue_indices and int(r["oof_top1"]) != int(r["original_label"])
                and int(r["knn_top1"]) != int(r["original_label"]))
    eq = count(lambda i, r: i in issue_indices and int(r["oof_top1"]) != int(r["original_label"])
               and int(r["knn_top1"]) != int(r["original_label"])
               and int(r["oof_top1"]) == int(r["knn_top1"]))
    mar = count(lambda i, r: i in issue_indices and int(r["oof_top1"]) != int(r["original_label"])
                and int(r["knn_top1"]) != int(r["original_label"])
                and int(r["oof_top1"]) == int(r["knn_top1"])
                and float(r["top1_margin"]) >= class_margin_q75.get(int(r["original_label"]), 0.0))
    kan = count(lambda i, r: i in issue_indices and int(r["oof_top1"]) != int(r["original_label"])
                and int(r["knn_top1"]) != int(r["original_label"])
                and int(r["oof_top1"]) == int(r["knn_top1"])
                and float(r["top1_margin"]) >= class_margin_q75.get(int(r["original_label"]), 0.0)
                and float(r.get("knn_agreement", 1.0)) <= 0.20 + EPS)
    dup = count(lambda i, r: i in issue_indices and int(r["oof_top1"]) != int(r["original_label"])
                and int(r["knn_top1"]) != int(r["original_label"])
                and int(r["oof_top1"]) == int(r["knn_top1"])
                and float(r["top1_margin"]) >= class_margin_q75.get(int(r["original_label"]), 0.0)
                and float(r.get("knn_agreement", 1.0)) <= 0.20 + EPS
                and not bool(r.get("duplicate_conflict_flag", False)))

    from analysis.noisy_labels.consensus import select_consensus_drop
    final = len(select_consensus_drop(quality, issues))

    return [
        ("total", n),
        ("cj_issue", cj),
        ("oof_ne_original", oof),
        ("knn_ne_original", knn),
        ("oof_eq_knn", eq),
        ("margin_ge_q75", mar),
        ("knn_agreement_le_020", kan),
        ("no_duplicate_conflict", dup),
        ("final_selected_after_caps", final),
    ]


def funnel_relabel_v2(quality, probs, labels, top_k=100):
    """consensus_relabel_v2: core hard + 3-of-5 aux + top-k."""
    issues = get_issues(quality, probs, labels)
    issue_indices = set(issues[issues["selected"]]["index"].values)
    n = len(quality)
    NUM_CLASSES = 500

    q90 = {}
    q75 = {}
    for c in range(NUM_CLASSES):
        mask = quality["original_label"] == c
        if mask.sum() > 0:
            q90[c] = float(quality.loc[mask, "p_top1"].quantile(0.90))
            q75[c] = float(quality.loc[mask, "top1_margin"].quantile(0.75))
        else:
            q90[c] = 0.90
            q75[c] = 0.50

    g90 = float(quality["p_top1"].quantile(0.90))
    g75 = float(quality["top1_margin"].quantile(0.75))

    # Core hard survivors
    core = []
    for i, row in quality.iterrows():
        if i not in issue_indices:
            continue
        oof_t1 = int(row.get("oof_top1", -1))
        knn_t1 = int(row.get("knn_top1", -1))
        orig = int(row["original_label"])
        if oof_t1 == orig or knn_t1 == orig or oof_t1 != knn_t1:
            continue
        if bool(row.get("duplicate_conflict_flag", False)):
            continue
        core.append(i)

    # Aux counts on core survivors
    prot_ok = sum(1 for i in core if int(quality.iloc[i].get("prototype_top1", -1)) == int(quality.iloc[i]["oof_top1"]))
    ptop_ok = sum(1 for i in core if float(quality.iloc[i]["p_top1"]) >= q90.get(int(quality.iloc[i]["original_label"]), g90))
    marg_ok = sum(1 for i in core if float(quality.iloc[i]["top1_margin"]) >= q75.get(int(quality.iloc[i]["original_label"]), g75))
    kta_ok = sum(1 for i in core if float(quality.iloc[i].get("knn_top1_agreement", 0.5)) >= 0.60)
    flip_ok = sum(1 for i in core if float(quality.iloc[i].get("flip_consistency", 0.0)) == 1.0)

    # 3 of 5
    aux3 = 0
    for i in core:
        r = quality.iloc[i]
        a = 0
        if int(r.get("prototype_top1", -1)) == int(r["oof_top1"]):
            a += 1
        if float(r["p_top1"]) >= q90.get(int(r["original_label"]), g90):
            a += 1
        if float(r["top1_margin"]) >= q75.get(int(r["original_label"]), g75):
            a += 1
        if float(r.get("knn_top1_agreement", 0.5)) >= 0.60:
            a += 1
        if float(r.get("flip_consistency", 0.0)) == 1.0:
            a += 1
        if a >= 3:
            aux3 += 1

    from analysis.noisy_labels.consensus import select_consensus_relabel_v2
    final = len(select_consensus_relabel_v2(quality, issues, top_k=top_k))

    return [
        ("total", n),
        ("cj_issue", len(issue_indices)),
        ("core_hard_conditions", len(core)),
        ("aux_prototype_eq_oof", prot_ok),
        ("aux_p_top1_ge_class_p90", ptop_ok),
        ("aux_margin_ge_class_q75", marg_ok),
        ("aux_knn_top1_agreement_ge_060", kta_ok),
        ("aux_flip_consistency_eq_1", flip_ok),
        ("aux_at_least_3_of_5", aux3),
        (f"final_selected_top{top_k}", final),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["all", "drop", "relabel"])
    args = parser.parse_args()

    AUDIT.mkdir(parents=True, exist_ok=True)
    quality, probs, labels = load_data()

    all_steps = []

    if args.mode in ("all", "drop"):
        print("=== cl_classwise_drop ===", flush=True)
        for name, cnt in funnel_cl_classwise_drop(quality, probs, labels):
            print(f"  {name}: {cnt}", flush=True)
            all_steps.append(("cl_classwise_drop", name, cnt))

        print("\n=== cl_knn_drop ===", flush=True)
        for name, cnt in funnel_cl_knn_drop(quality, probs, labels):
            print(f"  {name}: {cnt}", flush=True)
            all_steps.append(("cl_knn_drop", name, cnt))

    if args.mode in ("all", "relabel"):
        for tk in [100, 300]:
            print(f"\n=== consensus_relabel_v2_top{tk} ===", flush=True)
            for name, cnt in funnel_relabel_v2(quality, probs, labels, top_k=tk):
                print(f"  {name}: {cnt}", flush=True)
                all_steps.append((f"consensus_relabel_v2_top{tk}", name, cnt))

    # Write JSON
    json_out = {}
    for selector, name, cnt in all_steps:
        json_out.setdefault(selector, {})[name] = int(cnt)
    (AUDIT / "task3_filter_funnel.json").write_text(json.dumps(json_out, indent=2))

    # Write CSV
    with open(AUDIT / "task3_filter_funnel.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["selector", "step", "survivors"])
        for selector, name, cnt in all_steps:
            w.writerow([selector, name, int(cnt)])

    print(f"\nWritten audit/task3_filter_funnel.json + audit/task3_filter_funnel.csv")


if __name__ == "__main__":
    main()
