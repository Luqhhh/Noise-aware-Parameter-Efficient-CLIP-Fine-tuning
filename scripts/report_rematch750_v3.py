"""Compare audited V3 runs against the same-NPU batch-32 accuracy/time baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BASELINE = "RM_FT_NPU_TUNED"
REQUIRED = {"RM_V3_B64", "RM_V3_B128_E16", "RM_V3_B1024_E16_LR4"}
OPTIONAL = "RM_V3_B64_LR2"


def compare(baseline: dict, candidates: list[dict]) -> dict:
    by_id = {row["candidate"]: row for row in candidates}
    if len(by_id) != len(candidates) or set(by_id) - (REQUIRED | {OPTIONAL}):
        raise ValueError("V3 candidate IDs must be unique and pre-registered")
    if not REQUIRED <= set(by_id):
        raise ValueError(f"Missing complete V3 runs: {sorted(REQUIRED - set(by_id))}")
    if baseline["candidate"] != BASELINE or baseline["status"] != "passed":
        raise ValueError("The B32 baseline audit is missing or invalid")
    for row in candidates:
        if row["status"] != "passed":
            raise ValueError(f"Execution audit did not pass: {row['candidate']}")
        for key in ("init_checkpoint_sha256", "train_csv_sha256", "val_csv_sha256", "fixed_tail75_class_ids"):
            if row[key] != baseline[key]:
                raise ValueError(f"Source or fixed-tail mismatch: {row['candidate']} {key}")
        if row["train_cli_seconds"] <= 0 or row["training_seconds"] <= 0:
            raise ValueError(f"Missing full-run timing: {row['candidate']}")
    floor = baseline["raw_macro"] - 0.001
    if OPTIONAL in by_id and any(by_id[k]["raw_macro"] >= floor for k in ("RM_V3_B64", "RM_V3_B128_E16")):
        raise ValueError("Conditional B64_LR2 was only authorized if B64 and B128 both miss the floor")
    rows = []
    for row in [baseline, *candidates]:
        rows.append({
            "candidate": row["candidate"],
            "selected_epoch": row["selected_epoch"],
            "raw_macro": row["raw_macro"],
            "raw_micro": row["raw_micro"],
            "macro_delta_pp_vs_b32": 100 * (row["raw_macro"] - baseline["raw_macro"]),
            "fixed_tail75_macro": row["fixed_tail75_macro"],
            "rest675_macro": row["rest675_macro"],
            "successful_optimizer_updates": row["successful_optimizer_updates"],
            "training_seconds": row["training_seconds"],
            "train_cli_seconds": row["train_cli_seconds"],
            "speedup_vs_b32": baseline["train_cli_seconds"] / row["train_cli_seconds"],
            "meets_accuracy_floor": row["raw_macro"] >= floor,
        })
    eligible = [row for row in rows if row["meets_accuracy_floor"]]
    winner = min(eligible, key=lambda row: row["train_cli_seconds"])
    return {
        "accuracy_floor": floor,
        "baseline": BASELINE,
        "winner": winner["candidate"],
        "decision": "retained_b32" if winner["candidate"] == BASELINE else "new_efficiency_candidate",
        "rows": rows,
        "platform_score": None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = compare(json.loads(args.baseline.read_text()), [json.loads(path.read_text()) for path in args.candidate])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
