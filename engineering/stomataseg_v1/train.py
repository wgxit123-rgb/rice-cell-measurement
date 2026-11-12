from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .common import (FIXED_TRAINING, artifact_path, check_sha, frozen_gate, get_run,
                     load_config, load_model_class, normalize_rgb, prepared_manifest,
                     resolve, sha256, source_receipt, write_json)


def centernet_focal_loss(logits, target, valid):
    """Exact normalized CenterNet focal form from representation_train_centernet_v2."""
    probability = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    positive = (target >= 1.0 - 1e-6).float() * valid
    negative = (target < 1.0 - 1e-6).float() * valid
    positive_loss = -torch.log(probability) * (1.0 - probability).pow(2) * positive
    negative_loss = -torch.log(1.0 - probability) * probability.pow(2) * (1.0 - target).pow(4) * negative
    return positive_loss.sum() / positive.sum().clamp_min(1.0) + negative_loss.sum() / valid.sum().clamp_min(1.0)


def compute_loss(logits, targets, ignore):
    valid = (~ignore).float()
    semantic_logits, boundary_logits, center_logits = logits[:, 0], logits[:, 1], logits[:, 2]
    semantic_target, boundary_target, center_target = targets[:, 0], targets[:, 1], targets[:, 2]
    masked_mean = lambda values: (values * valid).sum() / valid.sum().clamp_min(1.0)
    semantic_bce = masked_mean(F.binary_cross_entropy_with_logits(semantic_logits, semantic_target, reduction="none"))
    probability = torch.sigmoid(semantic_logits) * valid
    target_valid = semantic_target * valid
    intersection = (probability * target_valid).sum(dim=(1, 2))
    denominator = probability.sum(dim=(1, 2)) + target_valid.sum(dim=(1, 2))
    dice = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    positive = (boundary_target * valid).sum()
    negative = valid.sum() - positive
    boundary_bce = F.binary_cross_entropy_with_logits(
        boundary_logits, boundary_target, reduction="none",
        pos_weight=(negative / positive.clamp_min(1.0)).clamp(1.0, 20.0))
    return semantic_bce + dice + 2.0 * masked_mean(boundary_bce) + centernet_focal_loss(center_logits, center_target, valid)


class CropSampler:
    """Cache only explicit selected TRAIN images. Targets originate on full work images."""

    def __init__(self, records, manifest_path: Path, seed: int):
        self.rng = np.random.default_rng(seed)
        self.items = []
        self.input_sha256 = {}
        for record in records:
            paths = {}
            # Hash all train artifacts, but load only the arrays needed for learning.
            for name, item in record["artifacts"].items():
                path = artifact_path(manifest_path, record, name)
                paths[name] = path
                self.input_sha256[str(path)] = item["sha256"]
            original = Path(record["source_image_path"])
            check_sha(original, record["source_image_sha256"])
            self.input_sha256[str(original)] = record["source_image_sha256"]
            with Image.open(paths["rgb_work"]) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            targets = np.load(paths["targets"], allow_pickle=False)
            valid = np.load(paths["train_valid"], allow_pickle=False)
            labels = np.load(paths["instances_work"], allow_pickle=False)
            shape = tuple(record["work_shape"])
            if (rgb.shape[:2] != shape or targets.shape != (3, *shape)
                    or valid.shape != shape or labels.shape != shape or valid.dtype != bool):
                raise ValueError("Prepared train array shape/dtype mismatch")
            if np.any((labels > 0) & ~valid):
                raise ValueError("An annotated positive is invalid outside declared conflict removal")
            self.items.append((rgb, targets, valid, np.argwhere(targets[2] >= 1.0 - 1e-6)))

    def sample(self, size: int = 384, foreground_probability: float = 0.75):
        rgb, target, valid, positives = self.items[int(self.rng.integers(len(self.items)))]
        h, w = valid.shape
        if len(positives) and self.rng.random() < foreground_probability:
            cy, cx = positives[int(self.rng.integers(len(positives)))]
        else:
            cy, cx = self.rng.integers(h), self.rng.integers(w)
        top, left = int(cy) - size // 2, int(cx) - size // 2
        bottom, right = top + size, left + size
        padding = ((max(0, -top), max(0, bottom - h)), (max(0, -left), max(0, right - w)))
        image_pad = np.pad(rgb, (*padding, (0, 0)), mode="edge")
        target_pad = np.pad(target, ((0, 0), *padding), mode="constant")
        valid_pad = np.pad(valid, padding, mode="constant", constant_values=False)
        top, left = max(0, top), max(0, left)
        rgb_crop = image_pad[top:top + size, left:left + size]
        target_crop = target_pad[:, top:top + size, left:left + size]
        valid_crop = valid_pad[top:top + size, left:left + size]
        if self.rng.random() < 0.5:
            rgb_crop, target_crop, valid_crop = rgb_crop[:, ::-1], target_crop[:, :, ::-1], valid_crop[:, ::-1]
        if self.rng.random() < 0.5:
            rgb_crop, target_crop, valid_crop = rgb_crop[::-1], target_crop[:, ::-1], valid_crop[::-1]
        return normalize_rgb(rgb_crop), np.ascontiguousarray(target_crop), np.ascontiguousarray(~valid_crop)

    def batch(self):
        images, targets, ignores = zip(*(self.sample() for _ in range(FIXED_TRAINING["batch_size"])))
        return tuple(torch.from_numpy(np.stack(values)) for values in (images, targets, ignores))


def verify_inputs(mapping):
    for filename, expected in mapping.items():
        check_sha(Path(filename), expected)


def amp_optimizer_update(model, optimizer, scaler):
    """GradScaler owns overflow handling; report whether an effective update occurred."""
    scaler.unscale_(optimizer)
    gradients_finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                           for parameter in model.parameters())
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    scale_after = scaler.get_scale()
    skipped = not gradients_finite or scale_after < scale_before
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError("Nonfinite parameter after AMP optimizer attempt")
    return bool(skipped), scale_before, scale_after


def train_run(config_path: Path, run_id: str):
    config, base, config_sha = load_config(config_path)
    sources = source_receipt(config, base)
    freeze, freeze_sha = frozen_gate(config, base, config_sha)
    run = get_run(config, run_id)
    manifest_path, manifest = prepared_manifest(config, base, "train", freeze["prepared_manifest_sha256"]["train"])
    if manifest["config_sha256"] != config_sha or manifest["source_sha256"] != sources:
        raise ValueError("Prepared train data does not bind the current frozen config/source")
    by_id = {record["image_id"]: record for record in manifest["images"]}
    if set(run["image_ids"]) - set(by_id):
        raise ValueError("A run image is missing from prepared train data")
    records = [by_id[image_id] for image_id in run["image_ids"]]
    weights = None
    weight_sha = None
    if run["initialization"] != "random":
        item = config["weights"][run["initialization"]]
        weights = resolve(base, item["path"])
        weight_sha = check_sha(weights, item["sha256"])
    sampler = CropSampler(records, manifest_path, run["seed"])
    out_dir = resolve(base, run["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "run_id": run_id, "status": "running", **run,
               "config_sha256": config_sha, "source_sha256": sources,
               "training_freeze_sha256": freeze_sha, "prepared_manifest_sha256": sha256(manifest_path),
               "input_sha256": sampler.input_sha256, "initialization_weights_sha256": weight_sha,
               "training": FIXED_TRAINING, "real_data_split_read": "train_only",
               "checkpoint_policy": "fixed_final_step300_model_only_no_optimizer"}
    write_json(out_dir / "STARTED.json", receipt)
    start = time.monotonic()
    attempt = step_done = overflow_skips = 0
    try:
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[key] = "4"
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "1"):
            raise ValueError("This experiment reserves physical GPU1 only")
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.set_num_threads(4)
        random.seed(run["seed"])
        np.random.seed(run["seed"])
        torch.manual_seed(run["seed"])
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("A single visible CUDA device is required; no CPU/device fallback")
        torch.cuda.manual_seed_all(run["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda:0")
        model = load_model_class(config, base)(run["initialization"], weights).to(device)
        model.train()
        optimizer = torch.optim.AdamW([
            {"params": model.encoder_parameters, "lr": 1e-4},
            {"params": model.decoder_parameters, "lr": 1e-3},
        ], weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=4096.0)
        with (out_dir / "steps.jsonl").open("x", encoding="utf-8", buffering=1) as log:
            while step_done < 300:
                attempt += 1
                batch = tuple(value.to(device, non_blocking=False) for value in sampler.batch())
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    logits = model(batch[0])
                if logits.shape != batch[1].shape:
                    raise ValueError("Model/target shape mismatch")
                with torch.autocast(device_type="cuda", enabled=False):
                    loss = compute_loss(logits.float(), batch[1].float(), batch[2])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at attempt {attempt}, step_done={step_done}")
                scaler.scale(loss).backward()
                skipped, scale_before, scale_after = amp_optimizer_update(model, optimizer, scaler)
                if skipped:
                    overflow_skips += 1
                else:
                    step_done += 1
                import json
                log.write(json.dumps({"attempt": attempt, "step": step_done, "step_done": step_done,
                                      "status": "overflow_skip" if skipped else "optimizer_update",
                                      "overflow_skips": overflow_skips, "scale_before": scale_before,
                                      "scale_after": scale_after, "loss": float(loss.detach()),
                                      "elapsed_seconds": time.monotonic() - start}) + "\n")
                if overflow_skips > 20:
                    raise FloatingPointError("AMP overflow skip limit exceeded (maximum 20)")
        verify_inputs(sampler.input_sha256)
        if source_receipt(config, base) != sources or sha256(config_path) != config_sha:
            raise ValueError("Sources/config changed during training")
        check_sha(manifest_path, receipt["prepared_manifest_sha256"])
        check_sha(resolve(base, config["training_freeze"]["path"]), freeze_sha)
        if weights is not None:
            check_sha(weights, weight_sha)
        checkpoint = out_dir / "final.pt"
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save({"model_state": state, "run_id": run_id, "step": 300,
                    "initialization": run["initialization"], "config_sha256": config_sha,
                    "source_sha256": sources}, checkpoint)
        receipt.update(status="completed", step=300, elapsed_seconds=time.monotonic() - start,
                       checkpoint={"path": "final.pt", "sha256": sha256(checkpoint)},
                       model_load_report=model.load_report, attempts=attempt,
                       effective_optimizer_updates=step_done, amp_overflow_skips=overflow_skips,
                       attempted_sample_exposures=attempt * 4,
                       exposure_note="Overflow attempts add computation and sampled batches; elapsed time/exposure are not equal across runs")
        write_json(out_dir / "run.json", receipt)
    except Exception as error:
        receipt.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic() - start,
                       attempts=attempt, effective_optimizer_updates=step_done,
                       amp_overflow_skips=overflow_skips, attempted_sample_exposures=attempt * 4)
        write_json(out_dir / "FAILURE.json", receipt)
        raise
    return out_dir / "run.json"


def main():
    parser = argparse.ArgumentParser(description="Run exactly one frozen 300-step train-only condition.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(train_run(args.config, args.run_id))


if __name__ == "__main__":
    main()
