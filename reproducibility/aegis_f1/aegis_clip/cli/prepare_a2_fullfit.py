"""Prepare the conservative A2 fixed full-fit assets."""

from __future__ import annotations

import argparse
import json

from aegis_clip.a2_gate import prepare_a2_fullfit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2-train-csv", required=True)
    parser.add_argument("--a2-val-csv", required=True)
    parser.add_argument("--content-groups", required=True)
    parser.add_argument("--trust-bundle", required=True)
    parser.add_argument("--a2-rejected-paths", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clean-threshold", type=float, default=0.70)
    parser.add_argument("--expected-classes", type=int, default=500)
    args = parser.parse_args()
    manifest = prepare_a2_fullfit(
        a2_train_csv=args.a2_train_csv,
        a2_val_csv=args.a2_val_csv,
        content_groups_json=args.content_groups,
        trust_bundle_path=args.trust_bundle,
        a2_rejected_paths=args.a2_rejected_paths,
        output_dir=args.output_dir,
        clean_threshold=args.clean_threshold,
        expected_classes=args.expected_classes,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
