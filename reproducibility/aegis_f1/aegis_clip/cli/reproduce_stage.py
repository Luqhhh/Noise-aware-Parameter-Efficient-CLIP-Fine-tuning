"""Audit a bound recipe by default; --execute runs approved synthetic nodes."""
import argparse
import json
from pathlib import Path
from aegis_clip.reproduction import run_recipe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--repository-root',default=str(Path(__file__).resolve().parents[4]))
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--audit-only',action='store_true')
    mode.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.resume and not args.execute:parser.error('--resume requires --execute')
    result=run_recipe(args.manifest,args.repository_root,execute=args.execute,resume=args.resume)
    print(json.dumps(result,indent=2))
    if result.get('status')=='blocked':raise SystemExit(2)

if __name__=='__main__':main()
