"""Copy the independently verified fixed raw/bias pair to new Windows folders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from aegis_clip.runtime import atomic_json_dump, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    report = json.loads((args.source / "report.json").read_text())
    verified = json.loads((args.source / "independent_verification.json").read_text())
    assert report["status"] == "completed_verified_delivery" and verified["status"] == "passed"
    assert report["binding"]["config_sha256"] == sha256_file(args.config)
    assert report["reduction"] == "sum_view_logits" and verified["full_float64_decision_agreement"]
    destination = Path(cfg["delivery_root"])
    files = []
    packages = {}

    def copy_checked(source, target):
        expected = sha256_file(source)
        if target.exists():
            if sha256_file(target) != expected:
                raise ValueError(f"Refusing to overwrite different existing artifact: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        assert sha256_file(target) == expected
        files.append(dict(source=str(source.resolve()), destination=str(target), sha256=expected,
                          bytes=target.stat().st_size))

    for name in ("submission_raw", "submission_bias"):
        package = args.source / name
        target = destination / (name + "_sixview_logit_sum")
        expected = verified["packages"][name]
        assert sha256_file(package / "pred_results.csv") == expected["csv_sha256"]
        assert sha256_file(package / "submission.zip") == expected["zip_sha256"]
        for filename in ("pred_results.csv", "submission.zip", "manifest.json", "submission_check.log"):
            copy_checked(package / filename, target / filename)
        copy_checked(args.source / "independent_verification.json", target / "independent_verification.json")
        stage = json.loads((Path(cfg["stage_root"]) / "dataset_manifest.json").read_text())
        checked = subprocess.run([sys.executable, str(ROOT / "scripts/check_submission.py"),
            "--test_dir", stage["test_root"], "--class-mapping", str(Path(cfg["stage_root"]) / "class_to_idx.json"),
            "--csv", str(target / "pred_results.csv"), "--zip", str(target / "submission.zip")],
            capture_output=True, text=True, check=True)
        log = checked.stdout + checked.stderr
        assert "All checks passed" in log
        (target / "windows_submission_check.log").write_text(log)
        packages[name] = dict(path=str(target / "submission.zip"), **expected, windows_nine_checks_passed=True)
    audit = destination / "sixview_logit_sum_delivery"
    for name in ("report.json", "independent_verification.json", "binding.json", "native_replay.json", "cold_replay.json"):
        copy_checked(args.source / name, audit / name)
    if (args.source / "verification_revision.json").is_file():
        copy_checked(args.source / "verification_revision.json", audit / "verification_revision.json")
    copy_checked(args.config, audit / "config.json")
    receipt = dict(status="verified_windows_delivery", experiment_id=cfg["experiment_id"], packages=packages,
                   files=files, platform_score=None, promotion_baseline_percent=cfg["promotion_baseline_percent"])
    atomic_json_dump(receipt, args.receipt)
    atomic_json_dump(receipt, audit / "windows_copy_receipt.json")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
