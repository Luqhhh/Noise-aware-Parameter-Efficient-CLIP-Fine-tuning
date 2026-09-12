"""Cache the fixed FULLFT_DUAL pre-head base/residual features."""
import argparse
from aegis_clip.prelim75 import cache_features, load_plan

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cache_features(load_plan(args.config))

if __name__ == '__main__':
    main()
