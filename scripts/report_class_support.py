"""Export fixed class support, bound to actual source-file hashes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproducibility/aegis_f1'))
from aegis_clip.class_support import summarize_support


def main():
    parser = argparse.ArgumentParser()
    for name in ('raw-csv','fit-csv','validation-csv','groups','mapping','output-dir','root-name'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError('Report output already exists')
    def canonical(path):
        path = path.replace('\\','/')
        prefix = args.root_name+'/'
        if path.startswith(prefix): path = path[len(prefix):]
        if Path(path).is_absolute() or '..' in Path(path).parts:
            raise ValueError('Expected canonical stage-relative identity')
        return path
    def rows(path):
        with open(path) as stream:
            return [dict(image_path=canonical(r['image_path']),label=int(r['label'])) for r in csv.DictReader(stream)]
    groups = json.loads(Path(args.groups).read_text())
    mapping = json.loads(Path(args.mapping).read_text())
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError('Noncontiguous class mapping')
    names = [k for k,v in sorted(mapping.items(),key=lambda item:item[1])]
    report = summarize_support(rows(args.raw_csv),rows(args.fit_csv),rows(args.validation_csv),groups,names)
    bindings = {key: {'path':value,'sha256':hashlib.sha256(Path(value).read_bytes()).hexdigest()}
                for key,value in vars(args).items() if key not in {'output_dir','root_name'}}
    output.mkdir(parents=True)
    with (output/'class_support.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(report[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(report)
    (output/'manifest.json').write_text(json.dumps({'sources':bindings,'argv':sys.argv,
        'classes':len(report),'raw_samples':sum(r['raw_samples'] for r in report),
        'validation_samples':sum(r['validation_support'] for r in report),
        'validation_samples_in_current_fit_groups':sum(r['validation_groups_seen_by_current_fit_samples'] for r in report),
        'limits':'Group file identities are recorded source declarations; ancestor exposure, per-class accuracy and true noise rates are not inferred. Empty metrics mean unavailable, not zero.'},indent=2))


if __name__ == '__main__':
    main()
