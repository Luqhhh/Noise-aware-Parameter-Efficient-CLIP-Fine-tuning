#!/usr/bin/env python3
"""Delete V4 best checkpoints when a point has no significant local gain.

The V4 plan defines a significant local candidate as relative to frozen V3:

    macro delta >= +1.0 pp  AND  micro delta >= +1.0 pp

Small-positive or negative points keep their small JSON/CSV/log evidence and
resolved config, but their multi-hundred-MB ``best.pt`` is deleted.  The
candidate can later be retrained from the recorded config if it becomes part
of a combination.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results/rematch750_search_v4"
OUTPUT_DIR = ROOT / "outputs/rematch750_search_v4"
V3_MACRO = 0.7234723567962646
V3_MICRO = 0.7341398000717163
DEFAULT_SIGNIFICANT_MACRO_PP = 1.0
DEFAULT_SIGNIFICANT_MICRO_PP = 1.0


def significant(result: dict, macro_pp: float, micro_pp: float) -> bool:
    macro = result.get("raw_macro")
    micro = result.get("raw_micro")
    if macro is None or micro is None:
        return False
    return (
        (float(macro) - V3_MACRO) * 100.0 >= float(macro_pp)
        and (float(micro) - V3_MICRO) * 100.0 >= float(micro_pp)
    )


def run_directory(trial_id: str) -> Path:
    return OUTPUT_DIR / f"RM_V4_{trial_id}" / "seed42"


def prune_result(result_path: Path, macro_pp: float, micro_pp: float) -> bool:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "passed_training":
        return False
    trial_id = str(result.get("trial_id", ""))
    run_dir = run_directory(trial_id)
    if not run_dir.exists():
        return False
    if significant(result, macro_pp, micro_pp):
        return False

    checkpoints = run_dir / "checkpoints"
    removed: list[str] = []
    if checkpoints.exists():
        for path in sorted(checkpoints.glob("*.pt")):
            path.unlink(missing_ok=True)
            removed.append(str(path.relative_to(run_dir)))
        for name in ("best.binding.json", "last.binding.json"):
            path = checkpoints / name
            if path.exists():
                path.unlink(missing_ok=True)
                removed.append(str(path.relative_to(run_dir)))
        marker = checkpoints / "PRUNED_NOT_SIGNIFICANT.json"
        marker.write_text(
            json.dumps(
                {
                    "trial_id": trial_id,
                    "reason": "below_v4_local_significance_threshold",
                    "macro_delta_pp": (float(result["raw_macro"]) - V3_MACRO) * 100.0,
                    "micro_delta_pp": (float(result["raw_micro"]) - V3_MICRO) * 100.0,
                    "required_macro_delta_pp": float(macro_pp),
                    "required_micro_delta_pp": float(micro_pp),
                    "removed": removed,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return bool(removed)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, ValueError):
        return False
    return True


def scan_once(macro_pp: float, micro_pp: float) -> int:
    removed = 0
    if not RESULT_DIR.exists():
        return removed
    for path in sorted(RESULT_DIR.glob("*.json")):
        try:
            removed += int(prune_result(path, macro_pp, micro_pp))
        except Exception as exc:  # keep the watcher alive and visible
            print(f"prune warning: {path}: {exc}", flush=True)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--until-pid",
        type=int,
        action="append",
        default=[],
        help="Stop when every listed PID has exited; may be repeated.",
    )
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--macro-pp", type=float, default=DEFAULT_SIGNIFICANT_MACRO_PP)
    parser.add_argument("--micro-pp", type=float, default=DEFAULT_SIGNIFICANT_MICRO_PP)
    args = parser.parse_args()

    while True:
        removed = scan_once(args.macro_pp, args.micro_pp)
        if removed:
            print(f"pruned {removed} non-significant checkpoint(s)", flush=True)
        if not args.watch:
            return 0
        if args.until_pid and not any(pid_alive(int(pid)) for pid in args.until_pid):
            # One final scan after every queue exits, then stop.
            removed = scan_once(args.macro_pp, args.micro_pp)
            if removed:
                print(f"pruned {removed} non-significant checkpoint(s)", flush=True)
            return 0
        time.sleep(max(float(args.interval_seconds), 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
