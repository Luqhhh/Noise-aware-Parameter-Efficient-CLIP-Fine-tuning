"""Training-only logit L2 penalty with auditable activation counters."""
import math
import torch


class LogitPenalty:
    def __init__(self, coefficient):
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError('Invalid coefficient')
        self.coefficient = float(coefficient)
        self.calls = self.examples = 0
        self.penalty_sum = 0.

    def apply(self, original, logits):
        if self.coefficient == 0 or not torch.is_grad_enabled() or not logits.requires_grad:
            return original
        penalty = .5 * self.coefficient * logits.float().square().mean(dim=1)
        if original.shape != penalty.shape:
            raise ValueError('Expected one classification loss per image')
        if not torch.isfinite(penalty).all():
            raise ValueError('Nonfinite logit penalty')
        self.calls += 1
        self.examples += len(logits)
        self.penalty_sum += float(penalty.detach().sum())
        return original + penalty

    def state_dict(self):
        return {'coefficient': self.coefficient, 'calls': self.calls,
                'examples': self.examples, 'penalty_sum': self.penalty_sum}

    def load_state_dict(self, state):
        if state['coefficient'] != self.coefficient:
            raise ValueError('Resume coefficient mismatch')
        self.calls, self.examples = int(state['calls']), int(state['examples'])
        self.penalty_sum = float(state['penalty_sum'])
