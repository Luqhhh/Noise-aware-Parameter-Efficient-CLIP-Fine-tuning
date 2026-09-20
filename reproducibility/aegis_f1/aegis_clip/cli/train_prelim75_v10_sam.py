"""Run one fixed PRELIM75 v10 preflight, gate, train, diagnostic, or delivery action."""
from __future__ import annotations

import argparse
import json

from aegis_clip.prelim75_sam import (
    CANDIDATES,
    a0_v10,
    deliver_v10,
    evaluate_v10,
    load_v10_plan,
    prepare_v10,
    preflight_v10,
    smoke_v10,
    train_v10,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--action",
        choices=["preflight", "prepare", "a0", "smoke", "train", "evaluate", "deliver"],
        required=True,
    )
    parser.add_argument("--candidate", choices=CANDIDATES)
    args = parser.parse_args()
    if args.action in ("train", "evaluate", "deliver") and args.candidate is None:
        parser.error(f"--candidate is required for {args.action}")
    if args.action not in ("train", "evaluate", "deliver") and args.candidate is not None:
        parser.error(f"--candidate is not accepted for {args.action}")
    plan = load_v10_plan(args.config)
    if args.action == "preflight":
        result = preflight_v10(plan)
    elif args.action == "prepare":
        result = prepare_v10(plan)
    elif args.action == "a0":
        result = a0_v10(plan)
    elif args.action == "smoke":
        result = smoke_v10(plan)
    elif args.action == "train":
        result = {"candidate_path": str(train_v10(plan, args.candidate))}
    elif args.action == "evaluate":
        result = evaluate_v10(plan, args.candidate)
    else:
        result = deliver_v10(plan, args.candidate)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
