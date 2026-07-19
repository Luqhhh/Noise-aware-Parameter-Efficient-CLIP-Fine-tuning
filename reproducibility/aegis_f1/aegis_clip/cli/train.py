"""Train an AegisCLIP experiment."""

from __future__ import annotations

import argparse

from aegis_clip.config import load_config
from aegis_clip.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    best = train(
        load_config(args.config),
        resume=args.resume,
        init_checkpoint=args.init_checkpoint,
        overwrite=args.overwrite,
    )
    print(best)


if __name__ == "__main__":
    main()
