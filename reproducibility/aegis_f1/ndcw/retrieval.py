"""Bounded-memory exact top-k, following the existing v1 chunked kNN backend."""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn.functional as F

from v2.plan import require


@torch.no_grad()
def neighbors(features, k=10, query_chunk=128, gallery_chunk=8192, *, progress=None, max_seconds=7200):
    require(k == 10 and len(features) > k, 'ND-CW first round fixes k=10')
    require(query_chunk > 0 and gallery_chunk > 0 and max_seconds > 0, 'Invalid kNN bounds')
    require(features.ndim == 2 and torch.isfinite(features).all()
            and (features.norm(dim=1) > 0).all(), 'Invalid frozen feature tensor')
    features = F.normalize(features.float(), dim=1)
    indices = np.empty((len(features), k), dtype=np.int64)
    scores = np.empty((len(features), k), dtype=np.float32)
    started = time.monotonic()
    for qs in range(0, len(features), query_chunk):
        require(time.monotonic() - started < max_seconds, 'A0 retrieval budget exhausted')
        q = features[qs:qs + query_chunk]
        values = q.new_full((len(q), k), -torch.inf)
        chosen = torch.zeros((len(q), k), device=q.device, dtype=torch.long)
        for gs in range(0, len(features), gallery_chunk):
            sim = q @ features[gs:gs + gallery_chunk].t()
            qi = torch.arange(qs, qs + len(q), device=q.device)
            gi = torch.arange(gs, gs + sim.shape[1], device=q.device)
            sim.masked_fill_(qi[:, None] == gi[None, :], -torch.inf)
            # Stable sort resolves exact ties by smallest original index.
            order = sim.argsort(dim=1, descending=True, stable=True)[:, :min(k, sim.shape[1])]
            local_values = sim.gather(1, order)
            merged = torch.cat([values, local_values], dim=1)
            all_indices = torch.cat([chosen, order + gs], dim=1)
            pick = merged.argsort(dim=1, descending=True, stable=True)[:, :k]
            values = merged.gather(1, pick)
            chosen = all_indices.gather(1, pick)
        end = qs + len(q)
        indices[qs:end] = chosen.cpu().numpy()
        scores[qs:end] = values.cpu().numpy()
        if progress is not None:
            progress(end, len(features), time.monotonic() - started)
    return indices, scores


def candidate_pairs(rows, indices, scores, minimum=.94):
    require(indices.shape == scores.shape == (len(rows), 10), 'Wrong kNN result dimensions')
    directed = [set(map(int, row)) for row in indices]
    seen = set()
    for i in range(len(rows)):
        for j, s in zip(indices[i], scores[i]):
            j = int(j)
            require(0 <= j < len(rows) and i != j and np.isfinite(s), 'Invalid retrieved neighbor')
            if float(s) < minimum:
                continue
            a, b = sorted((i, j))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            ra, rb = rows[a], rows[b]
            yield dict(i=a, j=b, image_i=ra['image_path'], image_j=rb['image_path'],
                       label_i=ra['label'], label_j=rb['label'],
                       cosine_similarity=float(s), content_group_i=ra['content_group'],
                       content_group_j=rb['content_group'], mutual_nn=i in directed[j],
                       exact=ra['content_group'] == rb['content_group'])
