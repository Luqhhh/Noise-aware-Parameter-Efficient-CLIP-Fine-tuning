"""Run the current-stage noise pipeline from a threshold manifest."""

from __future__ import annotations

import argparse
import json

from aegis_clip.stage_pipeline import run_stage_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    run_path = run_stage_pipeline(args.manifest)
    print(json.dumps({"pipeline_run": str(run_path)}))


if __name__ == "__main__":
    main()
