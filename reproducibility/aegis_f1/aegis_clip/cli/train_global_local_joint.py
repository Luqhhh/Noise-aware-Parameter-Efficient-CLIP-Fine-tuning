"""Execute V0/V1 once, from the original complete composite parent."""
import argparse
from aegis_clip.prelim75 import load_plan
from aegis_clip.prelim75_joint import train_joint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--candidate', choices=['V0', 'V1'], required=True)
    args = parser.parse_args()
    train_joint(load_plan(args.config), args.candidate)

if __name__ == '__main__':
    main()
