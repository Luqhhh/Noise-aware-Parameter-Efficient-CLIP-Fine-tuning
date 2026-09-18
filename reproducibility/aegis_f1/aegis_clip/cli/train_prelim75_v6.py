"""Run one read-only, D0, smoke, train, evaluate or deliver v6 action."""
from __future__ import annotations

import argparse
import json

from aegis_clip.prelim75_online_geometry import (
    deliver_v6,
    evaluate_v6,
    load_v6_plan,
    prepare_online_geometry,
    preflight_v6,
    run_d0,
    smoke_v6,
    train_online_geometry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--action",
        choices=["preflight", "prepare", "d0", "smoke", "train", "evaluate", "deliver"],
        required=True,
    )
    parser.add_argument("--candidate", choices=["G0", "G1"])
    args = parser.parse_args()
    plan = load_v6_plan(args.config)
    if args.action in ("train", "evaluate", "deliver") and args.candidate is None:
        parser.error(f"--candidate is required for {args.action}")
    if args.action not in ("train", "evaluate", "deliver") and args.candidate is not None:
        parser.error(f"--candidate is not accepted for {args.action}")
    if args.action == "preflight":
        result = preflight_v6(plan)
    elif args.action == "prepare":
        result = prepare_online_geometry(plan)
    elif args.action == "d0":
        result = run_d0(plan)
    elif args.action == "smoke":
        result = smoke_v6(plan)
    elif args.action == "train":
        result = {"candidate_path": str(train_online_geometry(plan, args.candidate))}
    elif args.action == "evaluate":
        result = evaluate_v6(plan, args.candidate)
    else:
        result = deliver_v6(plan, args.candidate)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
