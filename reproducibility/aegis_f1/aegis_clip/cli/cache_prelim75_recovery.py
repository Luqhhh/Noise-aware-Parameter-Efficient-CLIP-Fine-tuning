"""Execute PRELIM75 v3 D0 once; never overwrite or touch test images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis_clip.prelim75_recovery import cache_recovery_teacher, load_recovery_plan, preflight
from aegis_clip.runtime import atomic_json_dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    plan = load_recovery_plan(args.config)
    output = Path(plan["output"])
    if output.exists():
        # The queue creates the root and preflight before launching this entry.
        allowed = {"preflight.json", "queue_registration.json", "training_source", "D0.log", "queue_status.json"}
        unexpected = {path.name for path in output.iterdir()} - allowed
        if unexpected:
            raise FileExistsError(f"Refusing existing v3 output content: {sorted(unexpected)}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    preflight_path = output / "preflight.json"
    if not preflight_path.exists():
        atomic_json_dump(preflight(plan), preflight_path)
    report = cache_recovery_teacher(plan)
    final = {
        "status": report["status"],
        "d0_complete": True,
        "gate_passed": report["gate_passed"],
        "training_started": False,
        "platform_packages_created": 0,
        "online_accuracy": None,
    }
    atomic_json_dump(final, output / "final_status.json")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
