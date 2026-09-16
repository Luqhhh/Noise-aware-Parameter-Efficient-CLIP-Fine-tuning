"""Train one fixed PRELIM75 v3 candidate independently from V1."""
from __future__ import annotations

import argparse

from aegis_clip.prelim75_recovery import load_recovery_plan, train_recovery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate", choices=["S0", "S1"], required=True)
    args = parser.parse_args()
    train_recovery(load_recovery_plan(args.config), args.candidate)


if __name__ == "__main__":
    main()
