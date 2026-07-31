"""Prepare the exact A2-kept split for trust-aligned visual adaptation."""

from __future__ import annotations

import argparse
import json

from aegis_clip.a2_kept import prepare_a2_kept_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--purification-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-classes", type=int, default=500)
    args = parser.parse_args()
    result = prepare_a2_kept_split(
        args.split_csv,
        args.purification_csv,
        args.validation_csv,
        args.output_dir,
        expected_classes=args.expected_classes,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
