"""Prepare, smoke-test, train, diagnose, or deliver the fixed PRELIM75 v4."""
from __future__ import annotations

import argparse
import json

from aegis_clip.prelim75_trusted_ce import (
    deliver_trusted_ce,
    evaluate_trusted_ce,
    load_trusted_ce_plan,
    prepare_trusted_assets,
    smoke_v4,
    train_trusted_ce,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--action", choices=["prepare", "smoke", "train", "evaluate", "deliver"], required=True)
    parser.add_argument("--candidate", choices=["C0", "C1"])
    args = parser.parse_args()
    plan = load_trusted_ce_plan(args.config)
    if args.action in ("train", "evaluate", "deliver") and args.candidate is None:
        parser.error(f"--candidate is required for {args.action}")
    if args.action in ("prepare", "smoke") and args.candidate is not None:
        parser.error(f"--candidate is not accepted for {args.action}")
    if args.action == "prepare":
        result = prepare_trusted_assets(plan)
    elif args.action == "smoke":
        result = smoke_v4(plan)
    elif args.action == "train":
        result = {"candidate_path": str(train_trusted_ce(plan, args.candidate))}
    elif args.action == "evaluate":
        result = evaluate_trusted_ce(plan, args.candidate)
    else:
        result = deliver_trusted_ce(plan, args.candidate)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
