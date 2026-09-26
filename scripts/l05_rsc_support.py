"""Training-only RSC at the input of an existing linear classifier."""
import torch
from torch import nn
from torch.nn import functional as F


def challenge_mask(features, classifier, labels, fraction=1/3):
    """Signed target-logit gradients; decisions use detached float32 values."""
    if type(classifier) is not nn.Linear:
        raise TypeError('RSC requires the existing plain linear classifier')
    with torch.no_grad(), torch.autocast(device_type=features.device.type, enabled=False):
        x, w = features.detach().float(), classifier.weight.detach().float()
        b = None if classifier.bias is None else classifier.bias.detach().float()
        gradient = w[labels]  # exact derivative of each target logit wrt head input
        threshold = torch.quantile(gradient, 1-fraction, dim=1, keepdim=True)
        feature_keep = gradient < threshold
        original = F.linear(x, w, b).softmax(1).gather(1, labels[:, None]).squeeze(1)
        muted = F.linear(x * feature_keep, w, b).softmax(1).gather(1, labels[:, None]).squeeze(1)
        changes = original - muted
        batch_keep = changes < torch.quantile(changes, 1-fraction)
        return feature_keep | batch_keep[:, None]


class RepresentationChallenge:
    def __init__(self, fraction=1/3):
        if fraction != 1/3:
            raise ValueError('Only the preregistered one-third recipe is supported')
        self.fraction = fraction
        self.calls = self.examples = self.muted_examples = self.muted_channels = 0
        self.classifier = None
        self.pending = None

    def install(self, model):
        classifier = model.classifier
        if type(classifier) is not nn.Linear:
            raise TypeError('Expected unmodified linear head')
        self.classifier = classifier
        def capture(module, args, output):
            self.pending = (args[0], output) if module.training and torch.is_grad_enabled() else None
        return classifier.register_forward_hook(capture)

    def apply(self, logits, targets):
        if self.classifier is None:
            raise RuntimeError('RSC hook not installed')
        if not self.classifier.training or not torch.is_grad_enabled():
            self.pending = None
            return logits
        if self.pending is None or self.pending[1] is not logits:
            raise RuntimeError('Classification loss does not match captured head forward')
        features, _ = self.pending
        self.pending = None
        labels = targets.argmax(1)
        if not torch.equal(targets, F.one_hot(labels, targets.shape[1]).to(targets)):
            raise ValueError('This recipe requires unmixed one-hot labels')
        mask = challenge_mask(features, self.classifier, labels, self.fraction)
        self.calls += 1
        self.examples += len(labels)
        self.muted_examples += int((~mask).any(1).sum())
        self.muted_channels += int((~mask).sum())
        # Functional linear avoids re-entering the capture hook. Do not re-normalize.
        return F.linear(features * mask, self.classifier.weight, self.classifier.bias)

    def state_dict(self):
        return {k: getattr(self, k) for k in ('fraction', 'calls', 'examples', 'muted_examples', 'muted_channels')}

    def load_state_dict(self, state):
        if state['fraction'] != self.fraction:
            raise ValueError('RSC fraction changed on resume')
        for key in ('calls', 'examples', 'muted_examples', 'muted_channels'):
            value = state[key]
            if not isinstance(value, int) or value < 0:
                raise ValueError('Invalid RSC counter')
            setattr(self, key, value)
