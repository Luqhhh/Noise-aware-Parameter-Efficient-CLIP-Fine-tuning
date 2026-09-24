#!/usr/bin/env python3
"""Build the N1 high-precision reject sidecar and audit report.

The manifest uses the preliminary ``CL + fold-safe kNN strict consensus`` rule
plus cross-label exact duplicate conflicts.  Rejected samples receive weight 0;
all labels remain untouched.  The audit intentionally reports reject-rate
diagnostics but never tunes the threshold toward the historical 1%.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AEGIS_ROOT = ROOT / "reproducibility" / "aegis_f1"
for location in (str(ROOT), str(AEGIS_ROOT)):
    if location not in sys.path:
        sys.path.insert(0, location)

from analysis.noisy_labels.high_precision_drop import (  # noqa: E402
    load_quality_and_issues,
    select_high_precision_drop,
)
from aegis_clip.runtime import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-quality", required=True)
    parser.add_argument("--oof-logits")
    parser.add_argument("--issues-csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output")
    parser.add_argument("--knn-agreement-max", type=float, default=0.20)
    parser.add_argument("--class-margin-quantile", type=float, default=0.75)
    parser.add_argument("--max-class-reject-rate", type=float, default=0.10)
    parser.add_argument("--max-global-reject-rate", type=float, default=0.10)
    args = parser.parse_args()

    quality, issues = load_quality_and_issues(
        args.sample_quality,
        args.issues_csv,
        oof_logits_path=args.oof_logits,
        confident_joint_max_class_reject_rate=args.max_class_reject_rate,
        confident_joint_max_global_reject_rate=args.max_global_reject_rate,
    )
    result = select_high_precision_drop(
        quality,
        issues,
        knn_agreement_max=args.knn_agreement_max,
        class_margin_quantile=args.class_margin_quantile,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result["manifest"].to_csv(output, index=False)

    audit = dict(result["audit"])
    audit.update(
        {
            "manifest": str(output),
            "manifest_sha256": sha256_file(output),
            "source_quality": str(Path(args.sample_quality).resolve()),
            "source_quality_sha256": sha256_file(args.sample_quality),
            "issues_csv": (
                str(Path(args.issues_csv).resolve()) if args.issues_csv else None
            ),
            "issues_csv_sha256": (
                sha256_file(args.issues_csv) if args.issues_csv else None
            ),
            "oof_logits": str(Path(args.oof_logits).resolve())
            if args.oof_logits
            else None,
            "oof_logits_sha256": (
                sha256_file(args.oof_logits) if args.oof_logits else None
            ),
            "selection_rule": (
                "confident_joint_issue AND oof_top1!=label AND "
                "knn_top1!=label AND oof_top1==knn_top1 AND "
                "margin>=class_q75 AND knn_agreement<=0.20; "
                "plus cross-label exact duplicate conflict"
            ),
            "labels_modified": False,
            "pseudo_labels_created": False,
            "soft_repair_used": False,
        }
    )
    audit_output = (
        Path(args.audit_output).resolve()
        if args.audit_output
        else output.with_name(output.stem + "_audit.json")
    )
    audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
