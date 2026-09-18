"""Run one read-only, prepare, smoke, train, evaluate or deliver v7 action."""
from __future__ import annotations

import argparse
import json

from aegis_clip.prelim75_cooldown import (
    deliver_v7,
    evaluate_v7,
    load_v7_plan,
    prepare_v7,
    preflight_v7,
    smoke_v7,
    train_cooldown,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--action",
        choices=["preflight", "prepare", "smoke", "train", "evaluate", "deliver"],
        required=True,
    )
    parser.add_argument("--candidate", choices=["L0", "L1"])
    args = parser.parse_args()
    plan = load_v7_plan(args.config)
    if args.action in ("train", "evaluate", "deliver") and args.candidate is None:
        parser.error(f"--candidate is required for {args.action}")
    if args.action not in ("train", "evaluate", "deliver") and args.candidate is not None:
        parser.error(f"--candidate is not accepted for {args.action}")
    if args.action == "preflight":
        result = preflight_v7(plan)
    elif args.action == "prepare":
        result = prepare_v7(plan)
    elif args.action == "smoke":
        result = smoke_v7(plan)
    elif args.action == "train":
        result = {"candidate_path": str(train_cooldown(plan, args.candidate))}
    elif args.action == "evaluate":
        result = evaluate_v7(plan, args.candidate)
    else:
        result = deliver_v7(plan, args.candidate)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
