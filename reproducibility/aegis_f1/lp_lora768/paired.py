"""Independent sample, augmentation, mixup and model random streams for both arms."""
from __future__ import annotations
import contextlib
import hashlib
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from aegis_clip.runtime import seed_worker
from aegis_clip.v1_pipeline import Images, image_transform
from aegis_clip.v1_strategy import target_probabilities
from v1_continuation.runtime import logical_update


def stream_seed(seed, stream, epoch, index=0):
    return int.from_bytes(hashlib.blake2b(f"{seed}:{stream}:{epoch}:{index}".encode(), digest_size=8).digest(), "little") % (2**63-1)


class PairedImages(Images):
    def __init__(self, context, epoch):
        self.active = context.active
        super().__init__(context.train_root, [context.train[i] for i in self.active], image_transform(448, training=True, crop_min=.8))
        self.seed, self.epoch = context.protocol["seed"], epoch

    def __getitem__(self, index):
        original = int(self.active[index])
        seed = stream_seed(self.seed, "augmentation", self.epoch, original)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            # CPU-only default_generator avoids reseeding/initializing CUDA in workers.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed % (2**32))
                image, _ = super().__getitem__(index)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        return image, original


def training_loader(context, epoch):
    seed = context.protocol["seed"]
    order = torch.randperm(len(context.active), generator=torch.Generator().manual_seed(stream_seed(seed, "sampling", epoch)))
    worker_rng = torch.Generator().manual_seed(stream_seed(seed, "workers", epoch))
    return DataLoader(PairedImages(context, epoch), sampler=order.tolist(), batch_size=32,
                      num_workers=2, worker_init_fn=seed_worker, generator=worker_rng, drop_last=False)


def mix_plan(context, epoch, batch, size):
    seed = context.protocol["seed"]
    permutation = torch.randperm(size, generator=torch.Generator().manual_seed(stream_seed(seed, "mix_permutation", epoch, batch)))
    rng = np.random.default_rng(stream_seed(seed, "mix_beta", epoch, batch))
    lam = float(rng.beta(.2, .2))
    return permutation, lam


def batch_identity(context, images, indices, epoch, batch):
    permutation, lam = mix_plan(context, epoch, batch, len(indices))
    h = hashlib.sha256()
    for value in (images, indices, permutation):
        h.update(value.contiguous().cpu().numpy().tobytes())
    h.update(float(lam).hex().encode())
    weights = context.targets["weights"][indices]
    return dict(sha256=h.hexdigest(), lam=lam, reliability_mass=float(weights.sum()),
                indices=indices.tolist(), permutation=permutation.tolist())


def optimizer_for(model, context):
    groups = []
    adapters = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    if adapters:
        groups.append(dict(params=adapters, lr=5e-5, weight_decay=0., scope="lora"))
    head = [p for n, p in model.named_parameters() if n.startswith("head.") and p.requires_grad]
    groups.append(dict(params=head, lr=2.5e-4, weight_decay=.01, scope="head"))
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = {id(p) for g in groups for p in g["params"]}
    if expected != actual:
        raise ValueError("Unexpected trainable visual parameter outside LoRA")
    return torch.optim.AdamW(groups)


def scheduler_for(optimizer, steps):
    return torch.optim.lr_scheduler.OneCycleLR(optimizer, [g["lr"] for g in optimizer.param_groups],
                                               total_steps=4*steps, pct_start=.1, anneal_strategy="cos")


def update_once(context, model, optimizer, images, indices, *, epoch, batch, scaler=None):
    amp = str(context.device).startswith("cuda")
    scaler = scaler if scaler is not None else torch.amp.GradScaler("cuda", enabled=amp)
    target = context.targets
    classes = len(context.classes)
    probabilities = target_probabilities(target["targets"][indices], target["labels"][indices],
        target["original_alpha"][indices], classes, .1).to(context.device)
    weights = target["weights"][indices].to(context.device)
    permutation, lam = mix_plan(context, epoch, batch, len(indices))
    devices = [torch.device(context.device).index or 0] if amp else []
    with torch.random.fork_rng(devices=devices):
        seed = stream_seed(context.protocol["seed"], "model", epoch, batch)
        torch.random.default_generator.manual_seed(seed)
        if amp:
            torch.cuda.manual_seed(seed)
        return logical_update(model, optimizer, images.to(context.device), probabilities,
                              weights, permutation.to(context.device), lam, 8, scaler, amp, 1.)


@torch.no_grad()
def check_real_mixup_loss(context, model, images, indices, epoch, batch):
    """Verify full logical reliability against the original v1 loss on real images."""
    from aegis_clip.v1_strategy import weighted_mixup_loss
    probabilities=target_probabilities(context.targets['targets'][indices],context.targets['labels'][indices],
        context.targets['original_alpha'][indices],len(context.classes),.1).to(context.device)
    weights=context.targets['weights'][indices].to(context.device)
    permutation,lam=mix_plan(context,epoch,batch,len(indices))
    permutation=permutation.to(context.device)
    images=images.to(context.device)
    mixed_images=lam*images+(1-lam)*images[permutation]
    mode=model.training
    model.eval()
    logits=torch.cat([model(part).float() for part in mixed_images.split(8)])
    expected=weighted_mixup_loss(logits,probabilities,weights,permutation,lam)
    mass=weights[:,None]*probabilities
    mixed_mass=lam*mass+(1-lam)*mass[permutation]
    accumulated=sum(-(mixed_mass[i:i+8]*logits[i:i+8].log_softmax(1)).sum()/weights.sum().clamp_min(1e-8)
                    for i in range(0,len(indices),8))
    torch.testing.assert_close(accumulated,expected,rtol=1e-6,atol=1e-6)
    model.train(mode)
    return dict(weighted_mixup_loss=float(expected),micro_accumulated_loss=float(accumulated),equivalent=True)