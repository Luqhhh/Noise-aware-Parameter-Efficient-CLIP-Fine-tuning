#!/usr/bin/env python3
import argparse, json, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.pair_protocol_audit import audit_experiment_pair

logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Audit experiment pair protocol")
    p.add_argument("--reference-config", required=True)
    p.add_argument("--candidate-config", required=True)
    p.add_argument("--reference-ckpt", required=True)
    p.add_argument("--candidate-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-confounded-analysis", action="store_true")
    return p.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    # Check files exist
    for path_str, label in [(args.reference_config, "ref config"), (args.candidate_config, "cand config"),
                              (args.reference_ckpt, "ref ckpt"), (args.candidate_ckpt, "cand ckpt")]:
        if not Path(path_str).exists():
            logger.error(f"Missing {label}: {path_str}")
            sys.exit(4)
    try:
        result = audit_experiment_pair(
            reference_config_path=args.reference_config, candidate_config_path=args.candidate_config,
            reference_ckpt_path=args.reference_ckpt, candidate_ckpt_path=args.candidate_ckpt,
            output_path=args.output)
    except Exception as e:
        logger.error(f"Audit failed: {e}")
        sys.exit(4)
    print(json.dumps({"paired_valid": result.paired_valid, "causal_claim_allowed": result.causal_claim_allowed,
                       "sample_classification": result.sample_classification,
                       "max_visual_abs_diff": result.max_visual_abs_diff,
                       "unexpected_differences": len(result.unexpected_differences),
                       "warnings": len(result.warnings)}, indent=2))
    if not result.paired_valid:
        if args.allow_confounded_analysis:
            logger.warning("Proceeding with confounded analysis")
            sys.exit(2)
        logger.error("Experiments not paired. Use --allow-confounded-analysis.")
        sys.exit(3)
    if result.warnings: sys.exit(2)
    logger.info("Paired audit PASSED."); sys.exit(0)

if __name__ == "__main__": main()
