#!/usr/bin/env python3
"""Report F05 evidence-transfer deltas against the bound same-revision HL00 control."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AEGIS_ROOT = ROOT / "reproducibility" / "aegis_f1"
if str(AEGIS_ROOT) not in sys.path:
    sys.path.insert(0, str(AEGIS_ROOT))

from aegis_clip.config import load_config  # noqa: E402

MANIFEST = ROOT / "configs/rematch750_f05_transfer/manifest.json"
BINDING = ROOT / "configs/rematch750_f05_transfer/hl00_control_binding.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _report_path(config: dict, experiment_id: str) -> Path:
    root = Path(config["output"]["root"])
    return (
        root
        / experiment_id
        / f"seed{int(config['project'].get('seed', 42))}"
        / "checkpoints/selected_report.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path, default=None)
    args = parser.parse_args()

    manifest = _load_json(MANIFEST)
    binding = _load_json(BINDING)
    control = binding["metrics"]
    rows = []
    for trial in manifest["trials"]:
        config = load_config(ROOT / trial["config"])
        report_path = _report_path(config, trial["experiment_id"])
        if not report_path.is_file():
            rows.append({"trial": trial["name"], "status": "pending", "report": str(report_path)})
            continue
        report = _load_json(report_path)
        macro = float(report["raw_macro"])
        micro = float(report["raw_micro"])
        macro_delta_pp = (macro - float(control["raw_macro"])) * 100.0
        micro_delta_pp = (micro - float(control["raw_micro"])) * 100.0
        promote = macro_delta_pp >= float(manifest["promotion"]["macro_delta_pp"]) and micro_delta_pp >= -float(
            manifest["promotion"]["micro_max_regression_pp"]
        )
        rows.append(
            {
                "trial": trial["name"],
                "status": "complete",
                "selected_epoch": report.get("selected_epoch"),
                "raw_macro": macro,
                "raw_micro": micro,
                "macro_delta_pp_vs_hl00": macro_delta_pp,
                "micro_delta_pp_vs_hl00": micro_delta_pp,
                "promote": promote,
                "report": str(report_path),
            }
        )

    payload = {
        "protocol": manifest["protocol"],
        "control": {
            "trial": binding["trial"],
            "raw_macro": control["raw_macro"],
            "raw_micro": control["raw_micro"],
            "selected_report_sha256": binding["selected_report_sha256"],
        },
        "gate": manifest["promotion"],
        "rows": rows,
        "result": {"complete": sum(r["status"] == "complete" for r in rows), "pending": sum(r["status"] == "pending" for r in rows)},
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
