"""Prepare, replay, freeze groups, probe, train four epochs, and export both arms."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from aegis_clip.runtime import atomic_json_dump, sha256_file, seed_worker
from aegis_clip.v1_pipeline import Images, image_transform, read_rows, save_artifact, load_artifact, seed_training
from aegis_clip.v1_strategy import WeightAverage, target_probabilities, trainable_state, using_weights, load_trainable_state
from aegis_clip.submission import create_submission
from v1_continuation.runtime import logical_update
from .source import ROOT, Source, check_protocol, implementation_hashes
from .core import (ARMS, PairedImages, epoch_order, mixup_draw, group_excluded_maximum,
                   require, metrics, paired, review_decision)


def dump(value, path):
    atomic_json_dump(value, path)


def progress(output, phase, **fields):
    event = dict(phase=phase, **fields)
    dump(event, output / "progress.json")
    print(json.dumps(event), flush=True)


def prepare(locations_path, output):
    locations = json.loads(Path(locations_path).read_text())
    protocol = json.loads((ROOT / "configs/team_exploration_20261001/machine_b.json").read_text())
    check_protocol(protocol)
    source = Source(locations)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    counts = np.bincount([int(r["label"]) for r in source.train], minlength=750)
    plan = dict(status="prepared_not_started", protocol=protocol, locations=source.locations,
        source_binding=source.binding, implementation_hashes=implementation_hashes(),
        train_rows=133815, active_train_rows=119074, val_rows=14880,
        logical_batches_per_epoch=math.ceil(len(source.active)/32),
        tail_classes=sorted(range(750), key=lambda c: (int(counts[c]), c))[:75],
        sampler="per_epoch_uniform_permutation_of_unchanged_active_v1_population",
        randomness="per_image_epoch_augmentation_scope_and_independent_per_batch_mixup",
        group_quantile=dict(q=.1, method="numpy_linear", comparison="strict_less_than"),
        group_compute=dict(query_chunk=256, gallery_chunk=8192, dtype="float32", tf32=False),
        other_large_group_rule="complement_and_tail75_net_nonnegative_vs_control_and_parent",
        quality_stop_micro_pp=-2, wall_clock_time_limits=False,
        environment=dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__),
        original_artifacts_modified=False, training_started=False, platform_score=None)
    dump(plan, output / "plan.json")
    dump(dict(original_parent_binding=source.parent_binding, local_location_map=source.locations,
              local_source_binding=source.binding), output / "location_map.json")
    return output / "plan.json"


def verify(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    check_protocol(plan["protocol"])
    require(plan["implementation_hashes"] == implementation_hashes(), "Prepared implementation changed")
    source = Source(plan["locations"])
    require(source.binding == plan["source_binding"], "Prepared source changed")
    return plan, source


def resource_check(output):
    require(torch.cuda.is_available(), "CUDA unavailable")
    smi = subprocess.run(["nvidia-smi", "-q", "-x"], check=True, capture_output=True, text=True)
    import xml.etree.ElementTree as ET
    xml = ET.fromstring(smi.stdout)
    windows_processes = {}
    if os.name == "nt":
        inventory = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process | Select-Object ProcessId,Name | ConvertTo-Json -Compress"],
            check=True, capture_output=True, text=True)
        windows_processes = {int(row["ProcessId"]): row["Name"] for row in json.loads(inventory.stdout)}
    processes = []
    known_display = {"chatgpt.exe", "applicationframehost.exe", "shellhost.exe", "explorer.exe",
        "crossdeviceresume.exe", "nahimicsvc64.exe", "qq.exe", "searchhost.exe", "startmenuexperiencehost.exe",
        "widgetboard.exe", "msedge.exe", "phoneexperiencehost.exe", "msedgewebview2.exe", "nahimic3.exe",
        "nvidia overlay.exe", "flclash.exe", "textinputhost.exe", "shellexperiencehost.exe", "wallpaper32.exe",
        "codex-computer-use-swift.exe", "razerappengine.exe", "globalpresenter.exe", "codex.exe", "dwm.exe",
        "systemsettings.exe", "quark.exe", "legionzone.exe", "media_container.exe"}
    for process in xml.findall(".//process_info"):
        pid = int(process.findtext("pid"))
        name, kind = process.findtext("process_name"), process.findtext("type")
        if pid == os.getpid():
            continue
        if os.name == "nt":
            if pid not in windows_processes:
                processes.append(dict(pid=pid, name=name, type=kind, process_already_exited=True))
                continue
            name = windows_processes[pid]
        processes.append(dict(pid=pid, name=name, type=kind))
        require(kind in ("G", "C+G") and Path(name).name.lower() in known_display,
                f"Existing or unknown GPU workload; no preemption: {pid} {name}")
    report = dict(gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda, processes=processes,
        no_other_training_process_in_gpu_inventory=True, no_processes_terminated=True,
        windows_wddm_display_contexts_explicitly_recorded=True)
    dump(report, output / "resources.json")
    return report


def eval_loader(source, rows, size, scale=None, root=None):
    return DataLoader(Images(root or source.train_root, rows, image_transform(size, scale=scale)),
        batch_size=32, num_workers=2, shuffle=False, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(42), timeout=0)


@torch.no_grad()
def predict(model, source, rows, output, phase, *, size=448, scales=None, flip=False, root=None):
    model.eval()
    logits = torch.zeros(len(rows), len(source.classes))
    center_prediction, flip_prediction = None, None
    started, last = time.monotonic(), 0.
    scales = scales or [size]
    if len(scales) == 1 and flip:
        center_prediction, flip_prediction = torch.empty(len(rows), dtype=torch.long), torch.empty(len(rows), dtype=torch.long)
    for view, scale in enumerate(scales, 1):
        loader = eval_loader(source, rows, size, scale, root)
        for batch, (images, indices) in enumerate(loader, 1):
            images = images.to("cuda")
            result = model(images)
            if flip:
                flipped = model(images.flip(3))
                if len(scales) == 1:
                    center_prediction[indices] = result.argmax(1).cpu()
                    flip_prediction[indices] = flipped.argmax(1).cpu()
                result = result + flipped
            require(torch.isfinite(result).all(), "Nonfinite evaluation")
            logits[indices] += result.cpu()
            now = time.monotonic()
            if batch == 1 or batch == len(loader) or now-last >= 30:
                progress(output, phase, view=view, views=len(scales), scale=scale, batch=batch,
                         batches=len(loader), elapsed_seconds=now-started)
                last = now
    return logits.numpy(), (None if center_prediction is None else dict(
        center=center_prediction.numpy(), flip=flip_prediction.numpy(),
        agreement=float((center_prediction == flip_prediction).float().mean()))), time.monotonic()-started


def save_predictions(path, source, logits, diagnostic=None):
    data = dict(image_paths=np.array([r["image_path"] for r in source.val]),
                labels=np.array([int(r["label"]) for r in source.val]),
                predictions=logits.argmax(1), logits=logits)
    if diagnostic:
        data.update(center_predictions=diagnostic["center"], flip_predictions=diagnostic["flip"])
    require(not path.exists(), "Prediction overwrite")
    np.savez_compressed(path, **data)


def freeze_groups(plan, source, output):
    start = time.monotonic()
    train_features, val_features = source.features()
    gallery = train_features.cuda()
    groups = {g: i for i, g in enumerate(sorted({r["content_group"] for r in source.train}))}
    ids = torch.tensor([groups[r["content_group"]] for r in source.train], device="cuda")
    last = [0.]
    def callback(phase):
        def report(done, total):
            now = time.monotonic()
            if done == total or now-last[0] >= 30:
                progress(output, phase, rows_done=done, rows=total, elapsed_seconds=now-start)
                last[0] = now
        return report
    maxima_train = group_excluded_maximum(gallery, gallery, ids, ids, progress=callback("freeze_train_similarity"))
    maxima_val = group_excluded_maximum(val_features.cuda(), gallery, None, None,
                                       progress=callback("freeze_val_similarity"))
    threshold = float(np.quantile(maxima_train, .1, method="linear"))
    labels = np.array([int(r["label"]) for r in source.val])
    target = maxima_val < threshold
    masks = dict(all=np.ones(len(labels), dtype=bool), target=target, complement=~target,
                 tail75=np.isin(labels, plan["tail_classes"]))
    path = output / "frozen_groups.csv"
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["partition", "image_path", "label", "content_group", "max_cosine", "target", "tail75"])
        for partition, rows, maximum in (("train_dev", source.train, maxima_train), ("val_dev", source.val, maxima_val)):
            for row, value in zip(rows, maximum):
                writer.writerow([partition, row["image_path"], row["label"], row["content_group"],
                    repr(float(value)), int(value < threshold), int(int(row["label"]) in plan["tail_classes"])])
    parent_errors = int((target & (source.parent_predictions != labels)).sum())
    frozen = dict(threshold=threshold, quantile=plan["group_quantile"],
        cache=source.binding, algorithm=plan["group_compute"], gallery="train_dev_only",
        val_in_gallery=False, parent_target_errors=parent_errors,
        masks={name: int(mask.sum()) for name, mask in masks.items()},
        group_file_sha256=sha256_file(path), elapsed_seconds=time.monotonic()-start,
        training_rows_per_second=len(source.train)/(time.monotonic()-start),
        before_formal_training=True, clean_labels_known=False, true_source_shift_known=False)
    dump(frozen, output / "frozen_groups.json")
    del gallery, ids
    torch.cuda.empty_cache()
    return masks, frozen


def training_loader(source, arm, epoch):
    rows = [source.train[i] for i in source.active]
    return DataLoader(PairedImages(source.train_root, rows, arm, epoch, 42), batch_size=32,
        sampler=epoch_order(len(rows), 42, epoch), drop_last=False, num_workers=2,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(42+epoch), timeout=0)


def train_arm(plan, source, output, arm, masks, baseline, *, probe=False):
    directory = output / ("probe" if probe else "arms") / arm
    directory.mkdir(parents=True, exist_ok=False)
    seed_training(42)
    model = source.model().cuda()
    optimizer = torch.optim.AdamW([
        dict(params=[p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("head.")],
             lr=5e-5, weight_decay=0.),
        dict(params=list(model.head.parameters()), lr=2.5e-4, weight_decay=.01)])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, [5e-5, 2.5e-4],
        total_steps=4*plan["logical_batches_per_epoch"], pct_start=.1, anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    live = lambda: {name: p.detach() for name, p in model.named_parameters() if p.requires_grad}
    ema = WeightAverage(live(), .999)
    average = None
    supervision = {key: torch.as_tensor(source.supervision[key][source.active], device="cuda")
                   for key in ("targets", "labels", "weights", "original_alpha")}
    probabilities = target_probabilities(supervision["targets"], supervision["labels"],
                                          supervision["original_alpha"], 750, .1)
    labels = np.array([int(r["label"]) for r in source.val])
    log_path, history, updates, last = directory / "draws.jsonl", [], 0, 0.
    start, update_seconds = time.monotonic(), []
    torch.cuda.reset_peak_memory_stats()
    binding = dict(plan_sha256=sha256_file(output / "plan.json"), source_binding=source.binding,
                   frozen_groups_sha256=sha256_file(output / "frozen_groups.csv"), arm=arm)
    dump(dict(status="probing" if probe else "training", epochs_per_arm=4), directory / "status.json")
    logical_wall_start = time.monotonic()
    with log_path.open("x", encoding="utf-8") as draws:
        for epoch in range(1, 2 if probe else 5):
            model.train()
            losses = []
            loader = training_loader(source, arm, epoch)
            for batch, (images, indices) in enumerate(loader, 1):
                clock = time.monotonic()
                images = images.cuda()
                indices = indices.cuda()
                permutation, lam = mixup_draw(len(images), 42, epoch, batch)
                permutation = permutation.cuda()
                loss, numeric = logical_update(model, optimizer, images, probabilities[indices],
                    supervision["weights"][indices], permutation, lam, 8, scaler, True, 1.)
                scheduler.step()
                ema.update(live())
                updates += 1
                torch.cuda.synchronize()
                update_seconds.append(time.monotonic()-clock)
                losses.append(loss)
                draws.write(json.dumps(dict(epoch=epoch, batch=batch,
                    indices=indices.cpu().tolist(), permutation=permutation.cpu().tolist(), lam=lam,
                    lr=scheduler.get_last_lr(), logical_weight_mass=float(supervision["weights"][indices].sum()),
                    update=updates, numeric=numeric), sort_keys=True)+"\n")
                draws.flush()
                now = time.monotonic()
                if batch == 1 or batch == len(loader) or now-last >= 30:
                    progress(output, "cost_probe" if probe else "training", arm=arm, epoch=epoch,
                        epochs=4, batch=batch, batches=len(loader), updates=updates,
                        loss=loss, elapsed_seconds=now-start, gradient_norm=numeric["gradient_norm"])
                    dump(dict(status="probing" if probe else "training", epoch=epoch, epochs_per_arm=4,
                              batch=batch, batches=len(loader), updates=updates), directory / "status.json")
                    last = now
                if probe and batch == 5:
                    break
            if probe:
                break
            # Export complete epoch states before evaluating any stop condition.
            if epoch >= 2:
                if average is None:
                    average = WeightAverage(ema.state)
                average.update(ema.state)
            clock = time.monotonic()
            save_artifact(directory / f"epoch_{epoch:02d}.pt", dict(epoch=epoch, raw_state=trainable_state(model),
                selected_state={k: v.cpu().clone() for k, v in ema.state.items()},
                selected_policy="ema", ema=ema.payload(), average=average.payload() if average else None,
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                updates=updates, classes=source.classes, complete=epoch == 4, bias=None), binding)
            io_seconds = time.monotonic()-clock
            with using_weights(model, ema.state):
                logits, _, seconds = predict(model, source, source.val, output, f"{arm}_epoch{epoch}_ema")
            save_predictions(directory / f"val_epoch{epoch:02d}_ema.npz", source, logits)
            result = paired(labels, baseline, logits.argmax(1), 750, masks)
            history.append(dict(epoch=epoch, optimizer_updates=updates, loss=float(np.mean(losses)),
                                validation_seconds=seconds, checkpoint_seconds=io_seconds, paired_vs_parent=result))
            dump(history, directory / "history.json")
            require(result["all"]["candidate"]["micro"]-result["all"]["baseline"]["micro"] >= -.02,
                    f"Quality stop: {arm} epoch{epoch} micro below parent by more than 2pp")
    if probe:
        logical_wall_seconds = time.monotonic()-logical_wall_start
        logits, _, validation_seconds = predict(model, source, source.val, output, f"{arm}_probe_full_validation")
        checkpoint_start = time.monotonic()
        save_artifact(directory / "discarded_probe.pt", dict(selected_state=trainable_state(model),
                      optimizer_updates=5, formal_training=False, bias=None), binding)
        cost = dict(updates=5, update_seconds=update_seconds,
            mean_compute_update_seconds=float(np.mean(update_seconds)),
            mean_update_seconds=logical_wall_seconds/5, logical_wall_seconds=logical_wall_seconds,
            validation_seconds=validation_seconds, checkpoint_seconds=time.monotonic()-checkpoint_start,
            total_probe_seconds=time.monotonic()-start, peak_memory_bytes=torch.cuda.max_memory_allocated(),
            weights_discarded=True, formal_training_restarts_from_original_parent=True)
        dump(cost, directory / "cost.json")
        del model, optimizer, ema
        torch.cuda.empty_cache()
        return cost
    require(updates == 4*plan["logical_batches_per_epoch"] and average.count == 3, "Incomplete four-epoch training")
    for name, state in (("last_raw", trainable_state(model)), ("last_ema", ema.state), ("ema_swa_2_4", average.state)):
        save_artifact(directory / f"{name}.pt", dict(selected_state={k: v.cpu().clone() for k, v in state.items()},
            selected_policy=name, classes=source.classes, average_epochs=[2, 3, 4] if name == "ema_swa_2_4" else None,
            complete=True, epoch=4, optimizer_updates=updates, bias=None), binding)
    raw_logits, _, _ = predict(model, source, source.val, output, f"{arm}_last_raw_diagnostic")
    save_predictions(directory / "val_last_raw.npz", source, raw_logits)
    dump(dict(last_raw=metrics(labels, raw_logits.argmax(1), 750), last_ema=history[-1]["paired_vs_parent"]),
         directory / "diagnostics.json")
    dump(dict(status="four_epochs_complete", updates=updates, epochs=4), directory / "status.json")
    del model, optimizer, ema, average
    torch.cuda.empty_cache()
    return binding


def check_draws(output):
    paths = [output / "arms" / arm / "draws.jsonl" for arm in ARMS]
    fields = ("epoch", "batch", "indices", "permutation", "lam", "lr", "logical_weight_mass", "update")
    count = 0
    from itertools import zip_longest
    with paths[0].open() as a, paths[1].open() as b:
        for x, y in zip_longest(a, b):
            require(x is not None and y is not None, "Paired update counts differ")
            x, y = json.loads(x), json.loads(y)
            require(all(x[key] == y[key] for key in fields), "Paired sampling/Mixup/LR/supervision differed")
            count += 1
    return dict(paired_updates=count, exact_draws_match=True,
                draw_file_sha256={arm: sha256_file(path) for arm, path in zip(ARMS, paths)})


def export_arm(source, output, arm, binding):
    directory = output / "arms" / arm
    checkpoint = directory / "ema_swa_2_4.pt"
    payload = load_artifact(checkpoint, binding)
    require(payload["complete"] and payload["average_epochs"] == [2, 3, 4] and payload["bias"] is None,
            "Incorrect primary checkpoint")
    seed_training(42)
    model = source.model().cuda()
    load_trainable_state(model, payload["selected_state"])
    # Independently cold-load each single checkpoint. Center and flip diagnostic
    # are obtained from the same two forwards; center is the primary comparison.
    logits_sum, diagnostic, _ = predict(model, source, source.val, output, f"{arm}_primary_center_flip", flip=True)
    # Center logits need a separate pass because sum logits are only the diagnostic decoder.
    center, _, _ = predict(model, source, source.val, output, f"{arm}_primary_center")
    require(np.array_equal(center.argmax(1), diagnostic["center"]), "Cold-loaded center replay mismatch")
    save_predictions(directory / "val_primary.npz", source, center, diagnostic)
    six, _, _ = predict(model, source, source.val, output, f"{arm}_validation_six_views",
                          scales=[448, 512, 576], flip=True)
    save_predictions(directory / "val_six_views.npz", source, six)
    labels = np.array([int(r["label"]) for r in source.val])
    rows = read_rows(source.stage / "test_manifest.csv")
    names = [Path(r["image_path"]).name for r in rows]
    require(len(names) == 37444 and set(names) == {p.name for p in source.test_root.iterdir() if p.is_file()},
            "Test population changed")
    test_logits, _, seconds = predict(model, source, rows, output, f"{arm}_test_six_views",
        scales=[448, 512, 576], flip=True, root=source.test_root)
    package = directory / "submission"
    create_submission([(name, source.classes[index]) for name, index in zip(names, test_logits.argmax(1))],
        names, package, checkpoint, inference_mode="single_checkpoint_fixed_six_views_no_bias",
        tta_risk_acknowledged=True, valid_labels=set(source.classes), space_after_comma=True,
        extra_manifest=dict(binding=binding, decoder=dict(input_size=448, scales=[448, 512, 576],
                            flip=True, reduction="sum_logits", bias=None), platform_score=None))
    result = subprocess.run([sys.executable, str(ROOT / "scripts/check_submission.py"),
        "--test_dir", str(source.test_root), "--class-mapping", str(source.stage / "class_to_idx.json"),
        "--csv", str(package / "pred_results.csv"), "--zip", str(package / "submission.zip")],
        capture_output=True, text=True)
    (package / "submission_check.log").write_text(result.stdout+result.stderr, encoding="utf-8")
    result.check_returncode()
    del model
    torch.cuda.empty_cache()
    return center.argmax(1), dict(flip_consistency=diagnostic["agreement"], package=str(package),
        validation_center_metrics=metrics(labels, center.argmax(1), 750),
        validation_six_view_metrics=metrics(labels, six.argmax(1), 750),
        final_weight_diagnostics=json.loads((directory / "diagnostics.json").read_text()),
        test_inference_seconds=seconds, checkpoint_sha256=sha256_file(checkpoint),
        csv_sha256=sha256_file(package / "pred_results.csv"), zip_sha256=sha256_file(package / "submission.zip"))


def run(plan_path):
    plan, source = verify(plan_path)
    output = Path(plan_path).resolve().parent
    require(not (output / "status.json").exists(), "No automatic restart or overwrite")
    resources = resource_check(output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    dump(dict(status="running_preflight", pid=os.getpid(), training_started=False, epochs_per_arm=4), output / "status.json")
    try:
        seed_training(42)
        model = source.model().cuda()
        parent_logits, _, seconds = predict(model, source, source.val, output, "parent_original_center_replay")
        require(np.array_equal(parent_logits.argmax(1), source.parent_predictions),
                f"Parent full replay mismatch: {int((parent_logits.argmax(1)!=source.parent_predictions).sum())} rows")
        save_predictions(output / "baseline_center.npz", source, parent_logits)
        baseline = parent_logits.argmax(1)
        labels = np.array([int(r["label"]) for r in source.val])
        _, parent_flip, flip_seconds = predict(model, source, source.val, output, "parent_flip_diagnostic", flip=True)
        dump(dict(exact_matches=14880, rows=14880, decoder="448_center_no_bias", original_feature_dimension=512,
                  metrics=metrics(labels, baseline, 750), seconds=seconds,
                  flip_consistency=parent_flip["agreement"]), output / "parent_replay.json")
        del model
        torch.cuda.empty_cache()
        masks, frozen = freeze_groups(plan, source, output)
        if frozen["parent_target_errors"] < 75:
            dump(dict(status="closed_target_error_gate", training_started=False,
                      parent_target_errors=frozen["parent_target_errors"], minimum=75), output / "status.json")
            return
        probes = {}
        for arm in ARMS:
            resource_check(output)
            probes[arm] = train_arm(plan, source, output, arm, masks, baseline, probe=True)
        group_seconds = frozen["elapsed_seconds"]
        estimated_train = sum(p["mean_update_seconds"] * 4*plan["logical_batches_per_epoch"] for p in probes.values())
        full_validation = max(p["validation_seconds"] for p in probes.values())
        estimated_package = full_validation * (37444/14880) * 6 * 2
        dump(dict(probes=probes, group_seconds=group_seconds, estimated_training_seconds=estimated_train,
            estimated_evaluation_seconds=full_validation*(8+2*9), estimated_two_packages_seconds=estimated_package,
            estimated_checkpoint_io_seconds=sum(p["checkpoint_seconds"]*7 for p in probes.values()),
            estimated_total_seconds=group_seconds+estimated_train+full_validation*26+estimated_package,
            includes_submission_checks=True, wall_clock_cutoff=None, no_time_based_cancellation=True), output / "cost.json")
        dump(dict(status="formal_training", epochs_per_arm=4, pid=os.getpid(), training_started=True), output / "status.json")
        bindings = {}
        for arm in ARMS:
            resource_check(output)
            bindings[arm] = train_arm(plan, source, output, arm, masks, baseline)
        draws = check_draws(output)
        predictions, exports = {}, {}
        for arm in ARMS:
            resource_check(output)
            predictions[arm], exports[arm] = export_arm(source, output, arm, bindings[arm])
        control, candidate = [predictions[arm] for arm in ARMS]
        vs_control = paired(labels, control, candidate, 750, masks)
        vs_parent = paired(labels, baseline, candidate, 750, masks)
        control_parent = paired(labels, baseline, control, 750, masks)
        decision = review_decision(vs_control, vs_parent)
        with (output / "paired.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "label", "parent", "control", "candidate", "target", "tail75",
                             "correction_vs_control", "regression_vs_control", "correction_vs_parent", "regression_vs_parent"])
            for i, row in enumerate(source.val):
                writer.writerow([row["image_path"], int(labels[i]), int(baseline[i]), int(control[i]), int(candidate[i]),
                    int(masks["target"][i]), int(masks["tail75"][i]),
                    int(control[i]!=labels[i] and candidate[i]==labels[i]), int(control[i]==labels[i] and candidate[i]!=labels[i]),
                    int(baseline[i]!=labels[i] and candidate[i]==labels[i]), int(baseline[i]==labels[i] and candidate[i]!=labels[i])])
        report = dict(status="completed_pending_independent_verification", epochs_per_arm=4,
            parent_replay_matches=14880, frozen_groups=frozen, source_binding=source.binding,
            candidate_vs_control=vs_control, candidate_vs_parent=vs_parent, control_vs_parent=control_parent,
            draws=draws, exports=exports, decision=decision, resources=resources,
            error_overlap=dict(parent_control=int(((baseline!=labels)&(control!=labels)).sum()),
                parent_candidate=int(((baseline!=labels)&(candidate!=labels)).sum()),
                control_candidate=int(((control!=labels)&(candidate!=labels)).sum()),
                all_three=int(((baseline!=labels)&(control!=labels)&(candidate!=labels)).sum())),
            elapsed_seconds=time.monotonic()-start, test_in_training=False,
            original_noisy_labels=True, clean_labels_known=False, platform_score=None,
            automatic_full_training=False, original_artifacts_modified=False)
        report["artifact_sha256"] = {str(path.relative_to(output)): sha256_file(path)
            for path in output.rglob("*") if path.is_file()
            and path.name not in ("progress.json", "status.json", "report.json", "run.log", "run.err")}
        dump(report, output / "report.json")
        dump(dict(status="completed_pending_independent_verification", training_started=True, epochs_per_arm=4,
                  elapsed_seconds=time.monotonic()-start), output / "status.json")
    except BaseException as error:
        dump(dict(status="aborted", error_type=type(error).__name__, error=str(error),
                  elapsed_seconds=time.monotonic()-start, completed_candidate=False,
                  automatic_restart=False), output / "status.json")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify", "run"])
    parser.add_argument("--locations")
    parser.add_argument("--output")
    parser.add_argument("--plan")
    args = parser.parse_args()
    if args.action == "prepare":
        require(args.locations and args.output, "Prepare needs locations and a fresh output")
        print(prepare(args.locations, args.output))
    elif args.action == "verify":
        plan, _ = verify(args.plan)
        print(json.dumps(dict(status="verified", epochs_per_arm=4, active_train_rows=plan["active_train_rows"])))
    else:
        require(args.plan, "Run needs the frozen plan")
        run(args.plan)


if __name__ == "__main__":
    main()
