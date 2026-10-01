"""Run the fixed pair once, then independently verify completed artifacts."""
import argparse
import ctypes
import os
from pathlib import Path

from strong_aug_pair.runtime import run
from verify_strong_aug_pair import verify_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    # Keep the local machine awake only while this explicitly requested run is alive.
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        run(args.plan)
        if (Path(args.plan).parent / 'report.json').exists():
            verify_report(args.plan)
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
