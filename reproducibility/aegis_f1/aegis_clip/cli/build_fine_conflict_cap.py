"""Build the cross-fitted FINE conflict-cap trust bundle."""

from __future__ import annotations

import argparse
import json

from aegis_clip.fine_trust import build_fine_conflict_cap_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--base-bundle", required=True)
    parser.add_argument("--feature-tensor", required=True)
    parser.add_argument("--feature-paths", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output-bundle", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--num-classes", type=int, default=500)
    parser.add_argument("--power-iterations", type=int, default=20)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--expected-conflicts", type=int)
    parser.add_argument("--expected-changed", type=int)
    args = parser.parse_args()
    result = build_fine_conflict_cap_bundle(
        source_csv=args.source_csv,
        base_bundle_path=args.base_bundle,
        feature_tensor_path=args.feature_tensor,
        feature_paths_path=args.feature_paths,
        feature_manifest_path=args.feature_manifest,
        output_bundle_path=args.output_bundle,
        output_audit_path=args.output_audit,
        num_classes=args.num_classes,
        power_iterations=args.power_iterations,
        expected_samples=args.expected_samples,
        expected_conflicts=args.expected_conflicts,
        expected_changed=args.expected_changed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
