"""Train H0/H1 once with the fixed parent and matched feature cache."""
import argparse
from aegis_clip.prelim75 import load_plan, train_head

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--candidate', choices=['H0', 'H1'], required=True)
    args = parser.parse_args()
    train_head(load_plan(args.config), args.candidate)

if __name__ == '__main__':
    main()
