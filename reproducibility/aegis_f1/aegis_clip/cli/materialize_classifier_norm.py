"""Materialize an explicitly requested, inference-only norm-aligned checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from aegis_clip.checkpoint import _atomic_torch_save
from aegis_clip.classifier_norm import aligned_inference_checkpoint
from aegis_clip.runtime import atomic_json_dump, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--authorization', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists(): raise FileExistsError(output)
    authorization = json.loads(Path(args.authorization).read_text())
    parent_sha = sha256_file(args.checkpoint)
    if parent_sha != authorization.get('parent_sha256'):
        raise ValueError('Authorized parent hash mismatch')
    parent = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    candidate = aligned_inference_checkpoint(parent, parent_sha256=parent_sha, authorization=authorization)
    output.mkdir(parents=True, exist_ok=False)
    target = output/'candidate.pt'
    _atomic_torch_save(candidate, target)
    restored = torch.load(target, map_location='cpu', weights_only=False)
    for name, value in parent['model_state_dict'].items():
        if name != 'classifier.weight' and not torch.equal(value, restored['model_state_dict'][name]):
            raise RuntimeError(f'Unexpected changed tensor: {name}')
    for adapter in ('local_feature_adapter', 'part_token_adapter'):
        if adapter in parent:
            for name, value in parent[adapter]['state_dict'].items():
                if not torch.equal(value, restored[adapter]['state_dict'][name]):
                    raise RuntimeError(f'Unexpected adapter change: {adapter}.{name}')
    atomic_json_dump({'status':'materialized_user_requested_candidate', 'inference_only':True,
                     'parent_sha256':parent_sha, 'checkpoint_sha256':sha256_file(target),
                     'authorization_sha256':sha256_file(args.authorization),
                     'changed_model_parameters':['classifier.weight'],
                     'diagnostic_status':'rejected; preserved, no promotion'}, output/'manifest.json')
    print(target)

if __name__ == '__main__': main()
