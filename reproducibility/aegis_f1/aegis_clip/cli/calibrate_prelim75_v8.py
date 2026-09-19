"""Run one fixed PRELIM75 v8 source preparation, cache, fit, or delivery action."""
from __future__ import annotations

import argparse
import json

from aegis_clip.prelim75_source_bias import (
    cache_v8_scores,
    deliver_v8,
    fit_v8_bias,
    load_v8_plan,
    preflight_v8,
    prepare_v8_source,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--action",
        choices=["preflight", "prepare", "cache", "fit", "deliver"],
        required=True,
    )
    parser.add_argument("--candidate", choices=["B0", "B1"])
    args = parser.parse_args()
    plan = load_v8_plan(args.config)
    if args.action in {"cache", "fit", "deliver"} and args.candidate is None:
        parser.error(f"--candidate is required for {args.action}")
    if args.action in {"preflight", "prepare"} and args.candidate is not None:
        parser.error(f"--candidate is not accepted for {args.action}")
    if args.action == "preflight":
        result = preflight_v8(plan)
    elif args.action == "prepare":
        result = prepare_v8_source(plan)
    elif args.action == "cache":
        result = cache_v8_scores(plan, args.candidate)
    elif args.action == "fit":
        result = fit_v8_bias(plan, args.candidate)
    else:
        result = deliver_v8(plan, args.candidate)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
