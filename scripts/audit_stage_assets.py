"""Inventory declared stage assets without downloading or modifying inputs.

Unknown lineage stays unknown. A verified byte hash is not provenance approval.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.aegis_f1.aegis_clip.runtime import atomic_json_dump, sha256_file

FIELDS = ('artifact_id artifact_type stage dataset_id scope path exists byte_size sha256 '
          'producer_commit producer_command parent_artifact_ids learned_from_group_set_id '
          'encoded_group_set_id selected_using_group_set_id class_mapping_sha256 '
          'inference_protocol_sha256 status').split()


def audit_assets(records: list[dict], *, base_dir: Path) -> list[dict]:
    result = []
    seen = set()
    for record in records:
        identity = record.get('artifact_id')
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError('Each artifact needs a unique nonempty artifact_id')
        seen.add(identity)
        item = {key: record.get(key) for key in FIELDS}
        value = record.get('path')
        if value is not None and (not isinstance(value, str) or '://' in value):
            raise ValueError('Asset paths must be local filesystem paths or null')
        path = Path(value) if value else None
        if path is not None and not path.is_absolute(): path = base_dir / path
        exists = path is not None and path.is_file()
        item.update(path=str(path) if path is not None else None, exists=exists,
                    byte_size=path.stat().st_size if exists else None,
                    sha256=sha256_file(path) if exists else None)
        expected = record.get('sha256')
        item['declared_sha256'] = expected
        if not exists: item['status'] = 'missing'
        elif expected and item['sha256'] != expected: item['status'] = 'hash_mismatch'
        elif not item['producer_command']: item['status'] = 'missing_producer'
        else: item['status'] = 'bytes_verified_provenance_unverified'
        result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    payload = json.loads(manifest.read_text())
    records = audit_assets(payload['artifacts'], base_dir=manifest.parent)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json_dump({'schema_version':1,'manifest_sha256':sha256_file(manifest),'artifacts':records}, output/'assets_inventory.json')
    with (output/'missing_assets.csv').open('w', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=['artifact_id','artifact_type','stage','path','status'])
        writer.writeheader()
        for row in records:
            if row['status'] in {'missing','hash_mismatch','missing_producer'}:
                writer.writerow({k:row[k] for k in writer.fieldnames})

if __name__ == '__main__': main()
