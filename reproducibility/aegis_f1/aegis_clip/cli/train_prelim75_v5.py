"""Prepare, smoke-test, train, diagnose, or deliver fixed PRELIM75 v5."""
from __future__ import annotations

import argparse
import json

from aegis_clip.prelim75_fusion import (
    deliver_fusion,
    evaluate_fusion,
    load_fusion_plan,
    prepare_fusion_assets,
    smoke_v5,
    train_fusion,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--action", choices=["prepare", "smoke", "train", "evaluate", "deliver"], required=True)
    parser.add_argument("--candidate", choices=["F0", "F1"])
    args = parser.parse_args()
    plan = load_fusion_plan(args.config)
    if args.action in ("train", "evaluate", "deliver") and args.candidate is None:
        parser.error(f"--candidate is required for {args.action}")
    if args.action in ("prepare", "smoke") and args.candidate is not None:
        parser.error(f"--candidate is not accepted for {args.action}")
    if args.action == "prepare":
        result = prepare_fusion_assets(plan)
    elif args.action == "smoke":
        result = smoke_v5(plan)
    elif args.action == "train":
        result = {"candidate_path": str(train_fusion(plan, args.candidate))}
    elif args.action == "evaluate":
        result = evaluate_fusion(plan, args.candidate)
    else:
        result = deliver_fusion(plan, args.candidate)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
