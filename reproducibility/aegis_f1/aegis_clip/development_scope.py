"""Select development fitting rows before any learned feature statistics."""
import json
from pathlib import Path
import pandas as pd
import torch
from aegis_clip.runtime import sha256_file


def development_rows(split_dir, root_name, binding, *, base_dir):
    path = Path(binding['path'])
    if not path.is_absolute(): path = Path(base_dir)/path
    if sha256_file(path) != binding['sha256']:
        raise ValueError('Development group binding changed')
    values = json.loads(path.read_text())
    if not isinstance(values,list) or not values or not all(isinstance(x,str) for x in values) or len(set(values)) != len(values):
        raise ValueError('Expected unique registered development groups')
    allowed = set(values)
    groups = json.loads((Path(split_dir)/'content_groups.json').read_text())
    def canonical(p):
        p = str(p).replace('\\','/')
        if p.startswith(root_name+'/'): p=p[len(root_name)+1:]
        if Path(p).is_absolute() or '..' in Path(p).parts:
            raise ValueError('Invalid development sample identity')
        return p
    def read(name):
        frame = pd.read_csv(Path(split_dir)/name)
        paths = [canonical(p) for p in frame['image_path']]
        if len(set(paths)) != len(paths) or any(p not in groups for p in paths):
            raise ValueError('Duplicate sample or missing content group')
        return dict(zip(paths,frame['label'].astype(int))), {str(groups[p]) for p in paths}
    train,train_groups = read('train.csv');_,val_groups = read('val.csv')
    if train_groups != allowed:
        raise ValueError('Development fitting groups differ from registration')
    if train_groups & val_groups:
        raise ValueError('Development train and validation share content groups')
    return train, allowed, canonical


def select_development_features(features, labels, paths, train, canonical):
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(paths) or len(labels) != len(paths):
        raise ValueError('Feature cache shape mismatch')
    keys = [canonical(p) for p in paths]
    if len(set(keys)) != len(keys):raise ValueError('Duplicate cache identity')
    index = {key:i for i,key in enumerate(keys)}
    if not set(train) <= set(index):raise ValueError('Missing development features')
    selected = [index[key] for key in train]
    if any(int(labels[index[key]]) != label for key,label in train.items()):
        raise ValueError('Development cache label mismatch')
    indices = torch.tensor(selected,dtype=torch.long,device=features.device)
    return features[indices],labels[indices.to(labels.device)],[paths[i] for i in selected]
