"""Build paired curriculum-control and structured-correction train artifacts."""

from __future__ import annotations

import argparse
import json

from aegis_clip.structured_allocation import build_structured_allocation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--oof-logits", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--base-trust", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--curriculum-budget", type=float, default=0.30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sinkhorn-iterations", type=int, default=100)
    parser.add_argument("--minimum-class-support", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = build_structured_allocation(
        args.assignments,
        args.oof_logits,
        args.quality,
        args.base_trust,
        args.output_dir,
        curriculum_budget=args.curriculum_budget,
        temperature=args.temperature,
        sinkhorn_iterations=args.sinkhorn_iterations,
        minimum_class_support=args.minimum_class_support,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
