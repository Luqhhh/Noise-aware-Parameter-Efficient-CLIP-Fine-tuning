"""Build the strict KTA anchor trust bundle."""

from __future__ import annotations

import argparse
import json

from aegis_clip.kta_curriculum import build_kta_curriculum_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bundle", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--issues", required=True)
    parser.add_argument("--output-bundle", required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()
    result = build_kta_curriculum_bundle(
        base_bundle_path=args.base_bundle,
        quality_path=args.quality,
        issues_path=args.issues,
        output_bundle_path=args.output_bundle,
        output_manifest_path=args.output_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
