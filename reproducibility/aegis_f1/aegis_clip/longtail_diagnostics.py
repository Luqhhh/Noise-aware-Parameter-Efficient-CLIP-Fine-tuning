"""Observe realized supervision; never modify targets, sampling or loss."""
import torch


class SupervisionLedger:
    def __init__(self, sample_labels, num_classes, *, frequency_segments=None):
        self.labels = torch.as_tensor(sample_labels, dtype=torch.long).cpu()
        self.num_classes = num_classes
        if self.labels.ndim != 1 or not len(self.labels) or (self.labels < 0).any() or (self.labels >= num_classes).any():
            raise ValueError('invalid sample labels')
        if frequency_segments is not None and len(frequency_segments) != num_classes:
            raise ValueError('fixed frequency membership must cover class mapping')
        self.segments = tuple(frequency_segments) if frequency_segments is not None else (None,) * num_classes
        self.draws = torch.zeros(len(self.labels), dtype=torch.int64)
        self.target_mass = torch.zeros(num_classes, dtype=torch.float64)
        self.weighted_mass = self.target_mass.clone()
        self.corrected_in = self.target_mass.clone()
        self.corrected_out = self.target_mass.clone()
        self.zero_weight = self.target_mass.clone()
        self.denominators = []

    def update(self, indices, targets, weights, *, denominator):
        i = torch.as_tensor(indices).detach().cpu().long()
        t = torch.as_tensor(targets).detach().cpu().double()
        w = torch.as_tensor(weights).detach().cpu().double()
        if i.ndim != 1 or t.shape != (len(i), self.num_classes) or w.shape != (len(i),):
            raise ValueError('supervision batch shapes mismatch')
        if (i < 0).any() or (i >= len(self.labels)).any():raise ValueError('sample index out of bounds')
        if not torch.isfinite(t).all() or not torch.isfinite(w).all() or (t < 0).any() or (w < 0).any():
            raise ValueError('nonfinite or negative supervision')
        if not torch.allclose(t.sum(1), torch.ones(len(i), dtype=t.dtype), atol=1e-6, rtol=1e-6):
            raise ValueError('soft targets must sum to one')
        denominator = float(denominator)
        if not 0 < denominator < float('inf'):raise ValueError('invalid actual loss denominator')
        original = torch.nn.functional.one_hot(self.labels[i], self.num_classes).double()
        difference = t-original
        self.draws.index_add_(0, i, torch.ones_like(i))
        self.target_mass += t.sum(0)
        self.weighted_mass += (t*w[:, None]).sum(0)
        self.corrected_in += difference.clamp_min(0).sum(0)
        self.corrected_out += (-difference).clamp_min(0).sum(0)
        self.zero_weight += original[w == 0].sum(0)
        self.denominators.append(denominator)

    def report(self):
        rows = []
        for c in range(self.num_classes):
            mask = self.labels == c
            rows.append(dict(class_id=c, frequency_segment=self.segments[c],
                fit_samples=int(mask.sum()), actual_draws=int(self.draws[mask].sum()),
                actual_unique_samples=int((self.draws[mask]>0).sum()),
                zero_weight_draws=int(self.zero_weight[c]),
                original_label_target_mass=int(self.draws[mask].sum()),
                corrected_target_mass_in=float(self.corrected_in[c]),
                corrected_target_mass_out=float(self.corrected_out[c]),
                actual_target_mass=float(self.target_mass[c]),
                weighted_supervision_mass=float(self.weighted_mass[c])))
        return dict(classes=rows, optimizer_steps=len(self.denominators),
                    actual_draws=int(self.draws.sum()), loss_denominators=self.denominators,
                    interpretation='Observed classification target exposure, not gradient magnitude or true noise rate. MixUp and relabeling both contribute target transfer.',
                    frequency_definition='explicit fixed membership' if self.segments[0] is not None else 'unregistered; no grouping inferred',
                    auxiliary_losses_included=False)
