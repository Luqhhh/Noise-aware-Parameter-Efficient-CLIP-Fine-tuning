"""Explicit execution of the fixed machine-A protocol."""
import argparse
import json
from pathlib import Path
from .inputs import Context
from .runtime import complete, replay_parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--locations',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--action',choices=['replay','run'],default='run')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--allow-local-parent-baseline',action='store_true',help='Recorded local-execution ruling; never claims source replay passed')
    args=parser.parse_args()
    if not args.execute:parser.error('Fixed GPU computation requires --execute')
    context=Context(args.locations)
    if args.action=='replay':
        report=replay_parent(context,args.output)
    else:
        report=complete(context,args.output,args.allow_local_parent_baseline)
    print(json.dumps({key:report[key] for key in ('status','decision','mismatch_count') if key in report}),flush=True)


if __name__=='__main__':main()