"""Stage-bound v1 target preparation, training, calibration, and inference."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageFile
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from aegis_clip.v1_strategy import (
    CosineHead, WeightAverage, accuracy_report, build_classifier, class_top_keep,
    denoised_targets, fit_training_bias, knn_signals, load_trainable_state,
    target_probabilities, trainable_state, using_weights, weighted_mixup_loss,
)
from aegis_clip.runtime import atomic_json_dump, seed_worker, sha256_file

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
ROOT = Path(__file__).resolve().parents[3]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_recipe(path, source_root=None):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if set(config) != {"project", "source", "model", "denoise", "train", "decode", "output"}:
        raise ValueError("Unexpected v1 recipe sections")
    if config["project"]["stage"] != "repechage":
        raise ValueError("This adapter is scoped to the current rematch stage")
    m, d, t = config["model"], config["denoise"], config["train"]
    if (m["backbone"] != "ViT-B/32" or m["pretrained"] != "openai"
            or m["image_size"] % 32 or m["image_size"] < 32
            or not 1 <= m["blocks"] <= 12 or m["rank"] < 1 or m["alpha"] <= 0):
        raise ValueError("Invalid v1 OpenAI CLIP architecture")
    if config["source"]["partition"] not in ("train_dev", "full_train"):
        raise ValueError("Unsupported training partition")
    feature_path = m.get("feature_path", "projection")
    if feature_path not in ("projection", "pre_projection"):
        raise ValueError("Unsupported visual feature path")
    if feature_path == "pre_projection" and not config["source"].get("preprojection_cache"):
        raise ValueError("Pre-projection training requires a separately bound feature cache")
    if feature_path == "projection" and config["source"].get("preprojection_cache"):
        raise ValueError("Pre-projection cache cannot initialize a projected head")
    decoder = config["decode"]
    if decoder["bias_source"] == "test_uniform_experimental":
        if decoder.get("experimental_test_bias") is not True:
            raise ValueError("Test bias requires the explicit experimental research switch")
        if decoder.get("logit_reduction") != "sum":
            raise ValueError("The external-comparison test bias uses summed view logits")
    elif decoder["bias_source"] != "val_dev" or decoder.get("experimental_test_bias"):
        raise ValueError("Bias must use val_dev or the explicit test-uniform research mode")
    if (not config["decode"]["scales"] or any(s < m["image_size"] for s in config["decode"]["scales"])
            or config["decode"]["bias_iterations"] < 1):
        raise ValueError("Invalid fixed decoder")
    if (not 0 < d["keep_ratio"] <= 1 or d["teacher_epochs"] < 1 or d["knn_k"] < 1
            or not 0 <= d["minimum_weight"] <= 1 or not 0 <= d["pseudo_threshold"] <= 1
            or not 0 <= d["pseudo_margin"] <= 1 or not 0 <= d["pseudo_soft_alpha"] <= 1):
        raise ValueError("Invalid denoising recipe")
    if (min(d["teacher_batch_size"], d["query_chunk"], d["gallery_chunk"]) < 1
            or d["teacher_lr"] <= 0 or not 0 <= d["high_agreement_floor"] <= 1
            or not 0 <= config["decode"]["bias_strength"] <= 1):
        raise ValueError("Invalid denoising or bias dimensions")
    if (t["epochs"] < 1 or t["batch_size"] < 1 or t["num_workers"] < 0
            or t["mixup_alpha"] < 0 or not 0 <= t["label_smoothing"] < 1
            or not 0 <= t["ema_decay"] < 1 or not 0 < t["crop_min_scale"] <= 1
            or not 1 <= t["swa_start"] <= t["epochs"]
            or min(t["lora_lr"], t["head_lr"], t["grad_clip"]) <= 0):
        raise ValueError("Invalid training recipe")
    config["source"]["root"] = str(Path(source_root or config["source"]["root"]).expanduser().resolve())
    out = Path(config["output"]["root"]).expanduser()
    config["output"]["root"] = str((path.parent / out).resolve() if not out.is_absolute() else out.resolve())
    config["_config_path"] = str(path)
    return config


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def validate_rows(train, val, classes):
    for rows in (train, val):
        names = [r["image_path"] for r in rows]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate split paths")
        for r in rows:
            name = Path(r["image_path"])
            if (name.is_absolute() or ".." in name.parts or len(name.parts) < 3
                    or name.parts[0] != "train"):
                raise ValueError("Training rows must stay inside the official train root")
            label = int(r["label"])
            if label < 0 or label >= len(classes) or name.parts[1] != classes[label]:
                raise ValueError("Class mapping and split label disagree")
    if {r["image_path"] for r in train} & {r["image_path"] for r in val}:
        raise ValueError("Training/validation path overlap")
    if {r["content_group"] for r in train} & {r["content_group"] for r in val}:
        raise ValueError("Training/validation content overlap")


class StageContext:
    def __init__(self, config):
        from aegis_clip.config import load_config
        from aegis_clip.rematch_assets import validate_dataset, validate_cache, official_weight_hash
        self.config = config
        reference_path = Path(config["source"]["root"]) / config["source"]["reference_config"]
        self.reference = load_config(reference_path)
        self.manifest = validate_dataset(self.reference)
        self.cache_manifest = validate_cache(self.reference, self.manifest)
        base = Path(self.reference["data"]["dataset_manifest"]).parent
        mapping = json.loads((base / "class_to_idx.json").read_text())
        self.classes = [k for k, _ in sorted(mapping.items(), key=lambda pair: pair[1])]
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError("Noncontiguous class mapping")
        if self.manifest["data_version"] != str(config["project"]["data_version"]):
            raise ValueError("Recipe belongs to another data version")
        self.full = read_rows(base / "full_train.csv")
        train_dev, val_dev = read_rows(base / "train_dev.csv"), read_rows(base / "val_dev.csv")
        validate_rows(train_dev, val_dev, self.classes)
        if {r["image_path"] for r in self.full} != {r["image_path"] for r in train_dev + val_dev}:
            raise ValueError("Split does not cover full official training set")
        self.train = self.full if config["source"]["partition"] == "full_train" else train_dev
        self.val = [] if config["source"]["partition"] == "full_train" else val_dev
        self.calibration = val_dev
        self.train_root = Path(self.reference["data"]["train_root"])
        self.test_root = Path(self.reference["data"]["test_root"])
        self.official = Path(self.reference["model"]["official_checkpoint"])
        # Bind only declared, effective strategy inputs. Paths remain portable;
        # hashes bind actual data, cached features, and official initialization.
        recipe = {k: v for k, v in config.items() if k not in ("_config_path", "source", "output")}
        recipe["partition"] = config["source"]["partition"]
        self.binding = dict(stage=self.manifest["stage"], data_version=self.manifest["data_version"],
            dataset_manifest_sha256=sha256_file(base / "dataset_manifest.json"),
            class_mapping_sha256=sha256_file(base / "class_to_idx.json"),
            feature_manifest_sha256=sha256_file(base / "features/manifest.json"),
            official_checkpoint_sha256=official_weight_hash(self.reference),
            recipe_sha256=fingerprint(recipe),
            implementation_sha256=fingerprint({name: sha256_file(ROOT / name) for name in (
                "reproducibility/aegis_f1/aegis_clip/v1_strategy.py",
                "reproducibility/aegis_f1/aegis_clip/v1_pipeline.py",
                "reproducibility/aegis_f1/aegis_clip/model.py",
                "reproducibility/aegis_f1/aegis_clip/submission.py")}))
        self.preprojection_cache = None
        if config["model"].get("feature_path") == "pre_projection":
            cache = config["source"]["preprojection_cache"]
            self.preprojection_cache = Path(cache["tensor_path"]).expanduser().resolve()
            protocol_path = Path(cache["protocol_path"]).expanduser().resolve()
            if (sha256_file(self.preprojection_cache) != cache["tensor_sha256"]
                    or sha256_file(protocol_path) != cache["protocol_sha256"]):
                raise ValueError("Pre-projection cache or source protocol checksum mismatch")
            protocol = json.loads(protocol_path.read_text())
            identity = ("stage", "data_version", "dataset_manifest_sha256",
                        "class_mapping_sha256", "official_checkpoint_sha256",
                        "feature_manifest_sha256")
            if (protocol.get("test_data_used") is not False
                    or any(protocol["binding"].get(k) != self.binding[k] for k in identity)):
                raise ValueError("Pre-projection cache belongs to another dataset or initialization")
            self.binding.update(feature_path="pre_projection",
                feature_tensor_sha256=cache["tensor_sha256"],
                feature_protocol_sha256=cache["protocol_sha256"])
        if config["decode"]["bias_source"] == "test_uniform_experimental":
            self.binding["test_bias_implementation_sha256"] = sha256_file(
                ROOT / "reproducibility/aegis_f1/aegis_clip/v1_test_bias.py")
            self.binding["prior_alignment_implementation_sha256"] = sha256_file(
                ROOT / "reproducibility/aegis_f1/aegis_clip/prior_alignment.py")

    def summary(self):
        return dict(status="implementation_ready", training_started=False, gpu_started=False,
            binding=self.binding, classes=len(self.classes), train_samples=len(self.train),
            val_samples=len(self.val), feature_source="audited current-stage 224 center cache",
            feature_path=self.config["model"].get("feature_path", "projection"),
            feature_dimension=768 if self.preprojection_cache else 512,
            strategy_name="v1", strategy_origin="team_original",
            local_score=None, platform_score=None,
            rule_risk="same-trajectory SWA research only" if self.config["train"]["swa_enabled"] else None,
            test_bias_authorization=("user-relayed official confirmation 2026-10-01"
                if self.config["decode"]["bias_source"] == "test_uniform_experimental" else None))

    def features(self):
        dimension = 512
        if self.preprojection_cache:
            cache = torch.load(self.preprojection_cache, map_location="cpu", weights_only=True)
            if cache["image_paths"] != [r["image_path"] for r in self.full]:
                raise ValueError("Pre-projection feature rows do not match current full_train")
            if tuple(cache["projection"].shape) != (768, 512):
                raise ValueError("Unexpected official cache projection")
            tensor, dimension = cache["preprojection"], 768
        else:
            tensor = torch.load(self.reference["features"]["tensor_path"], map_location="cpu", weights_only=True)
        if tuple(tensor.shape) != (len(self.full), dimension) or not torch.isfinite(tensor).all():
            raise ValueError("Unexpected feature tensor")
        index = {r["image_path"]: i for i, r in enumerate(self.full)}
        return tensor[[index[r["image_path"]] for r in self.train]].float()


def save_artifact(path, payload, binding):
    path = Path(path)
    if path.exists() or path.with_suffix(".sha256.json").exists():
        raise FileExistsError(f"Artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**payload, "binding": binding}, path)
    atomic_json_dump(dict(sha256=sha256_file(path), binding=binding), path.with_suffix(".sha256.json"))


def load_artifact(path, binding):
    path = Path(path)
    meta = json.loads(path.with_suffix(".sha256.json").read_text())
    if meta["binding"] != binding or meta["sha256"] != sha256_file(path):
        raise ValueError("Artifact checksum or current-stage binding mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["binding"] != binding:
        raise ValueError("Artifact payload binding mismatch")
    return payload


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_targets(context, device):
    cfg = context.config["denoise"]
    seed_training(context.config["project"]["seed"])
    features = F.normalize(context.features().to(device), dim=1)
    labels = torch.tensor([int(r["label"]) for r in context.train], device=device)
    groups = {name: i for i, name in enumerate(sorted({r["content_group"] for r in context.train}))}
    group_ids = torch.tensor([groups[r["content_group"]] for r in context.train], device=device)
    started = time.monotonic()
    print(f"group-excluded kNN: {len(features)} training rows", flush=True)
    agreement, prediction = knn_signals(features, labels, features, labels, len(context.classes),
        k=cfg["knn_k"], query_groups=group_ids, gallery_groups=group_ids,
        query_chunk=cfg["query_chunk"], gallery_chunk=cfg["gallery_chunk"])
    knn_seconds = time.monotonic() - started
    print(f"kNN complete in {knn_seconds:.1f}s; preparing teacher", flush=True)
    initial = class_top_keep(agreement, labels, len(context.classes), cfg["keep_ratio"])
    initial |= agreement >= cfg["high_agreement_floor"]
    teacher = CosineHead(features.shape[1], len(context.classes), dropout=.1).to(device)
    opt = torch.optim.AdamW(teacher.parameters(), lr=cfg["teacher_lr"], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["teacher_epochs"],
                                                         eta_min=cfg["teacher_lr"] * .1)
    selected = torch.where(initial)[0]
    for epoch in range(cfg["teacher_epochs"]):
        teacher.train()
        order = selected[torch.randperm(len(selected), device=device)]
        for indices in order.split(cfg["teacher_batch_size"]):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(teacher(features[indices]), labels[indices], label_smoothing=.1)
            loss.backward()
            opt.step()
        scheduler.step()
        print(f"teacher epoch {epoch + 1}/{cfg['teacher_epochs']}", flush=True)
    teacher.eval()
    with torch.no_grad():
        probabilities = torch.cat([teacher(f).softmax(1) for f in features.split(cfg["teacher_batch_size"])])
    targets = denoised_targets(probabilities, labels, agreement, prediction,
        minimum_weight=cfg["minimum_weight"], pseudo_threshold=cfg["pseudo_threshold"],
        pseudo_margin=cfg["pseudo_margin"], pseudo_soft_alpha=cfg["pseudo_soft_alpha"])
    targets = {k: v.cpu() for k, v in targets.items()}
    targets.update(teacher_state={k: v.cpu() for k, v in teacher.state_dict().items()},
                   image_paths=[r["image_path"] for r in context.train], class_names=context.classes)
    targets["stats"] = dict(initial_kept=int(initial.sum()), kept=int(targets["kept"].sum()),
        pseudo=int(targets["pseudo"].sum()), used=int((targets["weights"] > 0).sum()),
        original_samples=len(labels), clean_labels_known=False)
    path = Path(context.config["output"]["root"]) / "targets.pt"
    save_artifact(path, targets, context.binding)
    coverage = torch.bincount(targets["targets"][targets["weights"] > 0], minlength=len(context.classes))
    atomic_json_dump(dict(targets["stats"], knn_seconds=knn_seconds,
        preparation_seconds=time.monotonic() - started, classes_with_targets=int((coverage > 0).sum()),
        target_counts=coverage.tolist()), path.parent / "target_report.json")
    return path


def image_transform(size, *, training=False, scale=None, crop_min=.8):
    if training:
        ops = [T.RandomResizedCrop(size, scale=(crop_min, 1.0), interpolation=T.InterpolationMode.BICUBIC),
               T.RandomHorizontalFlip(), T.RandAugment(num_ops=2, magnitude=7)]
    else:
        ops = [T.Resize(scale or size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(MEAN, STD)])


class Images(Dataset):
    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = Path(root), rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        relative = Path(row["image_path"])
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2:
            raise ValueError("Image path escaped official dataset")
        path = self.root.joinpath(*relative.parts[1:])
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Image symlink escaped official dataset")
        # Current-stage decode audit was strict. Never substitute placeholders.
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with Image.open(path) as image:
            image.load()
            value = self.transform(image.convert("RGB"))
        return value, index


def loader_for(context, rows, transform, *, shuffle=False, root=None):
    cfg = context.config["train"]
    return DataLoader(Images(root or context.train_root, rows, transform),
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], shuffle=shuffle,
        pin_memory=False, worker_init_fn=seed_worker, persistent_workers=False)


@torch.no_grad()
def collect_logits(model, loader, device, classes, *, flip=False):
    model.eval()
    result = torch.empty((len(loader.dataset), classes))
    for images, indices in loader:
        images = images.to(device)
        logits = model(images)
        if flip:
            logits = (logits + model(images.flip(3))) / 2
        result[indices] = logits.float().cpu()
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite inference logits")
    return result


def collect_tta(context, model, rows, device, root=None):
    cfg = context.config
    logits = torch.zeros((len(rows), len(context.classes)))
    for scale in cfg["decode"]["scales"]:
        loader = loader_for(context, rows, image_transform(cfg["model"]["image_size"], scale=scale), root=root)
        logits += collect_logits(model, loader, device, len(context.classes), flip=cfg["decode"]["flip"])
    return logits / len(cfg["decode"]["scales"])


def train_loop(model, loader, targets, cfg, binding, output, *, device="cpu", resume=None,
               evaluation=None, stop_after=None):
    """Resumable single-trajectory training; CPU toy tests use the same loop."""
    output = Path(output)
    seed_training(cfg["seed"])
    opt = torch.optim.AdamW([
        dict(params=[p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("head.")],
             lr=cfg["lora_lr"], weight_decay=0.0),
        dict(params=list(model.head.parameters()), lr=cfg["head_lr"], weight_decay=.01)])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, [cfg["lora_lr"], cfg["head_lr"]],
        total_steps=cfg["epochs"] * len(loader), pct_start=.1, anneal_strategy="cos")
    amp = str(device).startswith("cuda") and cfg["amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    initial = trainable_state(model, cpu=False)
    ema = WeightAverage(initial, cfg["ema_decay"]) if cfg["ema_decay"] else None
    swa = WeightAverage(initial) if cfg["swa_enabled"] else None
    start, history, trajectory = 1, [], fingerprint(dict(binding=binding, seed=cfg["seed"]))
    if resume:
        payload = load_artifact(resume, binding)
        if payload["trajectory"] != trajectory:
            raise ValueError("Resume belongs to another training trajectory")
        if payload.get("targets_sha256") != cfg.get("targets_sha256"):
            raise ValueError("Resume targets have changed")
        if payload["optimizer"] is None or payload["rng"] is None:
            raise ValueError("Only a complete epoch checkpoint can resume training")
        load_trainable_state(model, payload["raw_state"])
        ema = WeightAverage.restore(payload["ema"]) if payload["ema"] else None
        swa = WeightAverage.restore(payload["swa"]) if payload["swa"] else None
        for average in (ema, swa):
            if average:
                average.state = {name: value.to(device) for name, value in average.state.items()}
        opt.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        torch.set_rng_state(payload["rng"]["torch"])
        np.random.set_state(payload["rng"]["numpy"])
        random.setstate(payload["rng"]["python"])
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state(payload["rng"]["cuda"], device)
        start, history = payload["epoch"] + 1, payload["history"]
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training directory is not empty: {output}")
    if start > cfg["epochs"] + 1:
        raise ValueError("Resume epoch exceeds the fixed training schedule")
    output.mkdir(parents=True, exist_ok=True)
    tensor_targets = {k: targets[k].to(device) for k in ("targets", "labels", "weights", "original_alpha")}
    classes = int(model.head.weight.shape[0])
    probabilities = target_probabilities(tensor_targets["targets"], tensor_targets["labels"],
        tensor_targets["original_alpha"], classes, cfg["label_smoothing"])
    current = ema.state if ema else trainable_state(model, cpu=False)
    run_started = time.monotonic()
    for epoch in range(start, cfg["epochs"] + 1):
        model.train()
        losses, steps = [], 0
        epoch_started, last_report = time.monotonic(), 0.0
        for batch, (images, indices) in enumerate(loader, 1):
            images, indices = images.to(device), indices.to(device)
            weights = tensor_targets["weights"][indices]
            permutation = torch.randperm(len(images), device=device)
            lam = float(np.random.beta(cfg["mixup_alpha"], cfg["mixup_alpha"])) if cfg["mixup_alpha"] else 1.0
            images = lam * images + (1 - lam) * images[permutation]
            opt.zero_grad(set_to_none=True)
            autocast = torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
            with autocast:
                loss = weighted_mixup_loss(model(images), probabilities[indices], weights, permutation, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            old_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
                steps += 1
                if ema:
                    ema.update(trainable_state(model, cpu=False))
            losses.append(float(loss.detach()))
            now = time.monotonic()
            if batch == 1 or batch == len(loader) or now - last_report >= 60:
                progress = dict(status="training", epoch=epoch, epochs=cfg["epochs"],
                    batch=batch, batches=len(loader), optimizer_steps=steps,
                    loss=float(np.mean(losses)), epoch_seconds=now - epoch_started,
                    elapsed_seconds=now - run_started)
                atomic_json_dump(progress, output / "progress.json")
                print(json.dumps(progress), flush=True)
                last_report = now
        current = ema.state if ema else trainable_state(model, cpu=False)
        if swa and epoch >= cfg["swa_start"]:
            swa.update(current)
        metric = {}
        if evaluation:
            atomic_json_dump(dict(status="evaluating", epoch=epoch, epochs=cfg["epochs"],
                batch=len(loader), batches=len(loader)), output / "progress.json")
            with using_weights(model, current):
                metric = evaluation(model)
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), optimizer_steps=steps, metrics=metric))
        rng = dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())
        if str(device).startswith("cuda"):
            rng["cuda"] = torch.cuda.get_rng_state(device)
        payload = dict(epoch=epoch, trajectory=trajectory, raw_state=trainable_state(model),
            selected_state=current, selected_policy="ema" if ema else "last",
            ema=ema.payload() if ema else None, swa=swa.payload() if swa else None,
            optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            rng=rng, history=history, classes=targets["class_names"], model_config=cfg["model_config"],
            target_stats=targets["stats"], targets_sha256=cfg.get("targets_sha256"),
            effective_training_config=cfg)
        checkpoint = output / f"epoch_{epoch:02d}.pt"
        save_artifact(checkpoint, payload, binding)
        atomic_json_dump(history, output / "history.json")
        print(json.dumps(history[-1]), flush=True)
        if stop_after is not None and epoch >= stop_after:
            return checkpoint
    selected = swa.state if swa and swa.count else current
    selected_metrics = None
    if evaluation:
        with using_weights(model, selected):
            selected_metrics = evaluation(model)
    payload.update(selected_state=selected, selected_policy="swa_ema" if swa and swa.count else payload["selected_policy"],
        swa_epochs=list(range(cfg["swa_start"], cfg["epochs"] + 1)) if swa and swa.count else [],
        selected_metrics=selected_metrics, optimizer=None, scheduler=None, scaler=None, rng=None)
    final = output / "selected.pt"
    save_artifact(final, payload, binding)
    atomic_json_dump(dict(status="training_complete", epochs=cfg["epochs"],
        selected_checkpoint=str(final), elapsed_seconds=time.monotonic() - run_started),
        output / "progress.json")
    return final


def train(context, device, resume=None):
    output = Path(context.config["output"]["root"])
    target_path = output / "targets.pt"
    targets = load_artifact(target_path, context.binding)
    if targets["image_paths"] != [r["image_path"] for r in context.train] or targets["class_names"] != context.classes:
        raise ValueError("Targets do not match the current-stage training rows")
    used = torch.where(targets["weights"] > 0)[0].tolist()
    if not used:
        raise ValueError("No training samples survived automatic denoising")
    seed_training(context.config["project"]["seed"])
    model = build_classifier(context.official, len(context.classes), context.config["model"], device)
    model.head.load_state_dict(targets["teacher_state"])
    cfg = context.config["train"]
    loader = loader_for(context, [context.train[i] for i in used], image_transform(
        context.config["model"]["image_size"], training=True, crop_min=cfg["crop_min_scale"]), shuffle=True)
    subset = {k: targets[k][used] for k in ("targets", "labels", "weights", "original_alpha")}
    subset.update(class_names=context.classes, stats=targets["stats"])
    train_counts = torch.bincount(torch.tensor([int(r["label"]) for r in context.train]), minlength=len(context.classes))
    def evaluate(current):
        val_loader = loader_for(context, context.val, image_transform(context.config["model"]["image_size"]))
        logits = collect_logits(current, val_loader, device, len(context.classes))
        return accuracy_report(logits, torch.tensor([int(r["label"]) for r in context.val]), train_counts)
    training_cfg = dict(cfg, seed=context.config["project"]["seed"], model_config=context.config["model"],
                        targets_sha256=sha256_file(target_path),
                        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                        effective_strategy=context.config)
    return train_loop(model, loader, subset, training_cfg, context.binding, output / "training", device=device,
                      resume=resume, evaluation=evaluate if context.val else None)


def load_selected(context, checkpoint, device):
    payload = load_artifact(checkpoint, context.binding)
    if payload["classes"] != context.classes or payload["model_config"] != context.config["model"]:
        raise ValueError("Selected checkpoint architecture or classes differ")
    model = build_classifier(context.official, len(context.classes), context.config["model"], device)
    load_trainable_state(model, payload["selected_state"])
    model.eval()
    return model, payload


def calibrate(context, checkpoint, device):
    if context.config["decode"].get("bias_source", "val_dev") != "val_dev":
        raise ValueError("Test-uniform research calibration is performed in its explicit inference path")
    model, _ = load_selected(context, checkpoint, device)
    rows = context.calibration
    logits = collect_tta(context, model, rows, device)
    labels = torch.tensor([int(r["label"]) for r in rows])
    bias = fit_training_bias(logits, labels, iterations=context.config["decode"]["bias_iterations"])
    artifact = dict(bias=bias, checkpoint_sha256=sha256_file(checkpoint),
                    decoder=context.config["decode"], source="official_training_val_dev",
                    test_data_used=False, samples=len(rows),
                    logits_sha256=hashlib.sha256(logits.contiguous().numpy().tobytes()).hexdigest())
    path = Path(context.config["output"]["root"]) / "calibration.pt"
    save_artifact(path, artifact, context.binding)
    train_counts = torch.bincount(torch.tensor([int(r["label"]) for r in context.train]),
                                 minlength=len(context.classes))
    calibrated = logits + bias * context.config["decode"]["bias_strength"]
    full_training = context.config["source"]["partition"] == "full_train"
    atomic_json_dump(dict(samples=len(rows), partition="val_dev", views=context.config["decode"],
        raw=accuracy_report(logits, labels, train_counts),
        calibrated=accuracy_report(calibrated, labels, train_counts),
        student_training_partition=context.config["source"]["partition"],
        calibration_split_in_training_population=full_training,
        score_scope="training_included_diagnostic" if full_training else "development_split",
        bias_fitted_on_this_split=True, independent_test_score=False, platform_score=None),
        path.parent / "validation_report.json")
    with (path.parent / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "label", "raw_prediction", "calibrated_prediction"])
        writer.writerows(zip([r["image_path"] for r in rows], labels.tolist(),
                            logits.argmax(1).tolist(), calibrated.argmax(1).tolist()))
    return path


def validate_calibration(payload, context, checkpoint):
    if (payload["source"] != "official_training_val_dev" or payload["test_data_used"] is not False
            or payload["checkpoint_sha256"] != sha256_file(checkpoint)
            or payload["decoder"] != context.config["decode"]
            or tuple(payload["bias"].shape) != (len(context.classes),)
            or not torch.isfinite(payload["bias"]).all()):
        raise ValueError("Calibration is not bound to this checkpoint and decoder")


def infer(context, checkpoint, device):
    if context.config["decode"].get("bias_source") == "test_uniform_experimental":
        from aegis_clip.v1_test_bias import infer_test_uniform
        return infer_test_uniform(context, checkpoint, device)
    from aegis_clip.submission import create_submission
    calibration = load_artifact(Path(context.config["output"]["root"]) / "calibration.pt", context.binding)
    validate_calibration(calibration, context, checkpoint)
    model, payload = load_selected(context, checkpoint, device)
    base = Path(context.reference["data"]["dataset_manifest"]).parent
    rows = read_rows(base / "test_manifest.csv")
    names = [Path(r["image_path"]).name for r in rows]
    if len(names) != context.manifest["test_samples"]:
        raise ValueError("Test manifest count changed")
    logits = collect_tta(context, model, rows, device, root=context.test_root)
    logits += calibration["bias"] * context.config["decode"]["bias_strength"]
    predictions = [(name, context.classes[index]) for name, index in zip(names, logits.argmax(1).tolist())]
    out = Path(context.config["output"]["root"]) / "submission"
    create_submission(predictions, names, out, checkpoint, inference_mode="v1_fixed_multiscale_flip",
        tta_risk_acknowledged=True, valid_labels=set(context.classes), space_after_comma=True,
        extra_manifest=dict(binding=context.binding, decoder=context.config["decode"],
            calibration_sha256=sha256_file(Path(context.config["output"]["root"]) / "calibration.pt"),
            selected_policy=payload["selected_policy"], status="unpromoted_research_candidate",
            platform_score=None, test_usage="inference_only"))
    checked = subprocess.run([sys.executable, str(ROOT / "scripts/check_submission.py"),
        "--test_dir", str(context.test_root), "--class-mapping", context.reference["data"]["class_mapping"],
        "--csv", str(out / "pred_results.csv"), "--zip", str(out / "submission.zip")],
        capture_output=True, text=True)
    (out / "submission_check.log").write_text(checked.stdout + checked.stderr)
    checked.check_returncode()
    return out
