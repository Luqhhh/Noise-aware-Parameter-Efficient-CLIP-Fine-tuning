"""Compare two runs under the pre-registered OOF down-weighting criteria.

Reads the ``best_evaluation.json`` each run already writes, recomputes the
criteria from its per-class records, and prints the verdict against the gates
registered in ``docs/rematch750_strategy_oof_downweight_20260921.md``.

The criteria, and why they are these three:

* **Primary -- fixed training tail 10% macro.** The 75 classes with the fewest
  ``train_dev`` samples, ties broken by class id. Identical definition to
  ``docs/rematch750_local_comparison_20260922.md``. This is the primary because
  down-weighting acts on the tail by construction, and because overall macro on
  14,880 validation images separates the paired RM_FT/RM_LT runs by only
  0.03pp -- too little to read.
* **Secondary -- overall macro.** Must not fall more than 0.30pp below the
  baseline: trading a little overall accuracy for tail accuracy is allowed, but
  not a broad regression.
* **Stop-loss -- overall macro.** A 2pp local drop closes the round outright.

Two of the 75 tail classes cannot be judged on a single validation image, so
every tail number is reported both with and without them (``--exclude-class``,
default 183, which has 4 training and 1 validation sample). One validation
image flipping moves the tail macro by ~0.75pp, nearly 4x the 0.20pp promotion
gate, so the exclusion is what makes the primary criterion readable.

The tail is a macro over only 75 classes holding ~1,000 validation images, so
the gate needs a resolution check before it can be read as a pass. Bootstrapping
classes shows the *level* of the tail macro has an SE of ~3.3pp -- but both runs
score the same classes on the same images, so that spread is shared and cancels
in the difference. What the delta rests on is the spread of the *per-class
difference*, which this tool reports as a paired SE and a t. A delta that clears
0.20pp while sitting inside one paired SE is reported as INCONCLUSIVE rather
than PASS; that rule was added before any variant result existed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


TAIL_FRACTION = 0.10
PROMOTION_GATE_PP = 0.20
STOP_LOSS_PP = 2.00
OVERALL_TOLERANCE_PP = 0.30


def _load_per_class(path: Path) -> tuple[dict, list[dict]]:
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    per_class = evaluation.get("per_class")
    if not per_class:
        raise ValueError(f"{path} has no per_class records")
    required = {"label", "correct", "recall", "train_samples", "val_samples"}
    missing = required - set(per_class[0])
    if missing:
        raise ValueError(f"{path} per_class is missing fields: {sorted(missing)}")
    return evaluation, per_class


def _macro(records: list[dict]) -> float:
    return statistics.fmean(record["recall"] for record in records)


def _micro(records: list[dict]) -> float:
    total = sum(record["val_samples"] for record in records)
    if total == 0:
        raise ValueError("no validation samples across the per-class records")
    return sum(record["correct"] for record in records) / total


def _fixed_train_tail(records: list[dict], count: int) -> list[dict]:
    """Fewest train_dev samples first; ties by class id, as the shared doc does."""
    return sorted(records, key=lambda r: (r["train_samples"], r["label"]))[:count]


def _worst_recall(records: list[dict], count: int) -> list[dict]:
    return sorted(records, key=lambda r: (r["recall"], r["label"]))[:count]


def _summarise(path: Path, exclude: set[int], tail_count: int) -> dict:
    evaluation, per_class = _load_per_class(path)

    # Anchor the recomputation to the run's own reported metrics, so a change in
    # the per-class export cannot silently change what these criteria mean.
    overall_macro = _macro(per_class)
    overall_micro = _micro(per_class)
    reported_macro = evaluation.get("raw_macro")
    if reported_macro is not None and abs(overall_macro - reported_macro) > 1e-6:
        raise ValueError(
            f"{path}: recomputed macro {overall_macro} != reported raw_macro "
            f"{reported_macro}; the per-class records no longer define this metric"
        )

    tail = _fixed_train_tail(per_class, tail_count)
    kept = [record for record in tail if record["label"] not in exclude]
    excluded = [record for record in tail if record["label"] in exclude]

    return {
        "path": str(path),
        "selector": evaluation.get("selector_metric"),
        "overall_macro": overall_macro,
        "overall_micro": overall_micro,
        "reported_raw_macro": reported_macro,
        "fixed_train_tail_macro": _macro(tail),
        "fixed_train_tail_macro_excluding": _macro(kept) if kept else None,
        "fixed_train_tail_excluded_labels": sorted(r["label"] for r in excluded),
        "fixed_train_tail_support_range": (
            tail[0]["train_samples"],
            tail[-1]["train_samples"],
        ),
        "fixed_train_tail_val_samples": sum(r["val_samples"] for r in tail),
        "bottom_10pct_macro": _macro(_worst_recall(per_class, tail_count)),
        "tail_labels": sorted(record["label"] for record in tail),
        "recall_by_label": {record["label"]: record["recall"] for record in per_class},
    }


def _paired(control: dict, variant: dict, labels: list[int], unit: str) -> dict:
    """Statistics of the per-class difference, which is what the delta rests on.

    Both runs see the same classes and the same validation images, so the
    class-to-class spread in *level* is shared and cancels; what remains is the
    spread in the per-class *difference*. Resampling classes is therefore the
    right unit here, and it is far tighter than the unpaired spread of the
    metric itself.
    """
    differences = [
        variant["recall_by_label"][label] - control["recall_by_label"][label]
        for label in labels
    ]
    n = len(differences)
    mean = statistics.fmean(differences)
    if n < 2:
        return {"unit": unit, "n": n, "mean_pp": mean * 100, "se_pp": None, "t": None}
    se = statistics.stdev(differences) / math.sqrt(n)
    return {
        "unit": unit,
        "n": n,
        "mean_pp": mean * 100,
        "sd_pp": statistics.stdev(differences) * 100,
        "se_pp": se * 100,
        "t": (mean / se) if se > 0 else None,
    }


def _pp(value: float) -> str:
    return f"{value * 100:+.4f}pp"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="the paired control")
    parser.add_argument("--variant", required=True, help="the run under test")
    parser.add_argument(
        "--tail-count",
        type=int,
        default=None,
        help="tail size; default = 10%% of the class count, rounded down",
    )
    parser.add_argument(
        "--exclude-class",
        type=int,
        action="append",
        default=None,
        help="class id to drop from the tail (repeatable); default 183",
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    exclude = {183} if args.exclude_class is None else set(args.exclude_class)

    baseline = _load_per_class(Path(args.baseline))[1]
    tail_count = args.tail_count or int(len(baseline) * TAIL_FRACTION)
    if tail_count <= 0 or tail_count > len(baseline):
        raise ValueError(f"tail count {tail_count} is out of range for {len(baseline)} classes")

    control = _summarise(Path(args.baseline), exclude, tail_count)
    variant = _summarise(Path(args.variant), exclude, tail_count)

    if control["tail_labels"] != variant["tail_labels"]:
        raise ValueError(
            "the two runs disagree on the fixed training tail; they are not "
            "paired on the same split, so the comparison is not attributable"
        )

    tail_delta = variant["fixed_train_tail_macro"] - control["fixed_train_tail_macro"]
    tail_delta_excl = (
        variant["fixed_train_tail_macro_excluding"]
        - control["fixed_train_tail_macro_excluding"]
    )
    overall_delta = variant["overall_macro"] - control["overall_macro"]

    print(f"tail = {tail_count} classes, train_dev support "
          f"{control['fixed_train_tail_support_range'][0]}"
          f"..{control['fixed_train_tail_support_range'][1]}, "
          f"{control['fixed_train_tail_val_samples']} validation images")
    print(f"excluded from tail: {sorted(exclude & set(control['tail_labels']))}")
    print()
    header = f"{'metric':38s} {'control':>11s} {'variant':>11s} {'delta':>12s}"
    print(header)
    print("-" * len(header))
    rows = [
        ("overall macro", "overall_macro"),
        ("overall micro", "overall_micro"),
        ("fixed train tail macro", "fixed_train_tail_macro"),
        ("fixed train tail macro (excl.)", "fixed_train_tail_macro_excluding"),
        ("bottom-10% macro", "bottom_10pct_macro"),
    ]
    for label, key in rows:
        c, v = control[key], variant[key]
        print(f"{label:38s} {c * 100:10.4f}% {v * 100:10.4f}% {_pp(v - c):>12s}")
    print()

    tail_labels_kept = [label for label in control["tail_labels"] if label not in exclude]
    paired = [
        _paired(control, variant, control["tail_labels"], "fixed train tail macro (incl.)"),
        _paired(control, variant, tail_labels_kept, "fixed train tail macro (excl.)"),
        _paired(control, variant, sorted(control["recall_by_label"]), "overall macro"),
    ]
    print("resolution (paired over classes; both runs see the same split)")
    for stats in paired:
        if stats["se_pp"] is None:
            print(f"  {stats['unit']:34s} n={stats['n']:3d}  SE unavailable")
            continue
        t_text = f"{stats['t']:+.2f}" if stats["t"] is not None else "n/a (zero spread)"
        print(
            f"  {stats['unit']:34s} n={stats['n']:3d}  "
            f"delta={stats['mean_pp']:+.4f}pp  SE={stats['se_pp']:.4f}pp  "
            f"t={t_text}"
        )
    tail_se = paired[1]["se_pp"]
    if tail_se:
        print(
            f"  the +{PROMOTION_GATE_PP:.2f}pp gate is "
            f"{PROMOTION_GATE_PP / tail_se:.2f} SE of the paired tail difference"
        )
    print()

    above_gate = tail_delta_excl * 100 >= PROMOTION_GATE_PP
    resolvable = paired[1]["t"] is not None and paired[1]["t"] >= 1.0
    if not above_gate:
        primary = "FAIL"
    elif resolvable:
        primary = "PASS"
    else:
        primary = "INCONCLUSIVE (clears the gate but sits inside one SE)"

    print("verdict")
    print(f"  primary   fixed train tail macro (excl.) {_pp(tail_delta_excl)}"
          f"  vs gate +{PROMOTION_GATE_PP:.2f}pp  ->  {primary}")
    print(f"  primary   fixed train tail macro (incl.) {_pp(tail_delta)}")
    print(f"  secondary overall macro {_pp(overall_delta)}"
          f"  vs tolerance -{OVERALL_TOLERANCE_PP:.2f}pp  ->  "
          f"{'PASS' if overall_delta * 100 >= -OVERALL_TOLERANCE_PP else 'FAIL'}")
    print(f"  stop-loss overall macro {overall_delta * 100:.4f}pp"
          f"  vs -{STOP_LOSS_PP:.2f}pp  ->  "
          f"{'CLOSE' if overall_delta * 100 <= -STOP_LOSS_PP else 'ok'}")

    if args.json_out:
        payload = {
            "tail_count": tail_count,
            "excluded_classes": sorted(exclude),
            "control": control,
            "variant": variant,
            "deltas_pp": {
                "fixed_train_tail_macro": tail_delta * 100,
                "fixed_train_tail_macro_excluding": tail_delta_excl * 100,
                "overall_macro": overall_delta * 100,
                "overall_micro": (variant["overall_micro"] - control["overall_micro"]) * 100,
                "bottom_10pct_macro": (
                    variant["bottom_10pct_macro"] - control["bottom_10pct_macro"]
                ) * 100,
            },
            "paired": paired,
            "gates": {
                "promotion_pp": PROMOTION_GATE_PP,
                "stop_loss_pp": STOP_LOSS_PP,
                "overall_tolerance_pp": OVERALL_TOLERANCE_PP,
            },
            "verdict": {
                "primary": primary,
                "primary_above_gate": above_gate,
                "primary_resolvable": resolvable,
                "secondary_pass": overall_delta * 100 >= -OVERALL_TOLERANCE_PP,
                "stop_loss_triggered": overall_delta * 100 <= -STOP_LOSS_PP,
            },
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
