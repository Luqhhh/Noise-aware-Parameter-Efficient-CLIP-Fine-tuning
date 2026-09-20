"""Execute one fixed PRELIM75 v9 audit or inference action."""
from __future__ import annotations

import argparse
import json

from aegis_clip.prelim75_source_recrop import (
    diagnostic_v9,
    infer_v9,
    load_v9_plan,
    preflight_v9,
    run_d0,
    smoke_v9,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--action",
        required=True,
        choices=("preflight", "d0", "smoke", "diagnostic", "infer"),
    )
    args = parser.parse_args()
    plan = load_v9_plan(args.config)
    actions = {
        "preflight": preflight_v9,
        "d0": run_d0,
        "smoke": smoke_v9,
        "diagnostic": diagnostic_v9,
        "infer": infer_v9,
    }
    result = actions[args.action](plan)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
