"""Run the fixed overlap diagnostic for one PRELIM75 v3 candidate."""
from __future__ import annotations

import argparse

from aegis_clip.prelim75_recovery import evaluate_recovery, load_recovery_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate", choices=["S0", "S1"], required=True)
    args = parser.parse_args()
    evaluate_recovery(load_recovery_plan(args.config), args.candidate)


if __name__ == "__main__":
    main()
