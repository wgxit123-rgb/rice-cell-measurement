from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage.segmentation import watershed
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .instance_metrics import instance_metrics
from .representation_model import ResNet50InstanceNet
from .utils import choose_device, load_config, seed_everything, seed_worker, sha256_file, write_json


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def make_targets(instance_map: np.ndarray) -> np.ndarray:
    semantic = (instance_map > 0).astype(np.float32)
    boundary = np.zeros_like(instance_map, dtype=np.uint8)
    centers = np.zeros_like(instance_map, dtype=np.float32)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for instance_id in np.unique(instance_map):
        if instance_id == 0:
            continue
        mask = (instance_map == instance_id).astype(np.uint8)
        boundary |= mask & ~cv2.erode(mask, kernel, iterations=1)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(distance.argmax()), distance.shape)
        centers[y, x] = 1.0
    centers = cv2.GaussianBlur(centers, (0, 0), sigmaX=3.0, sigmaY=3.0)
    if centers.max() > 0:
        centers /= centers.max()
    return np.stack([semantic, boundary.astype(np.float32), centers], axis=0)


class ReviewedInstanceDataset(Dataset):
    def __init__(
        self,
        root: Path,
        manifest: Path,
        instance_dir: Path,
        ignore_dir: Path,
        split: str,
        augment: bool,
        allowed_ids: set[str] | None = None,
    ) -> None:
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if allowed_ids is not None:
            rows = [row for row in rows if row["id"] in allowed_ids]
        self.rows = sorted(rows, key=lambda row: int(row["id"]))
        if not self.rows:
            raise ValueError(f"No reviewed instances for split={split}")
        self.root = root
        self.instance_dir = instance_dir
        self.ignore_dir = ignore_dir
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(self.root / row["blind_image_copy"]) as handle:
            image = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(self.instance_dir / f"{row['id']}.tif") as handle:
            instances = np.asarray(handle, dtype=np.uint16)
        with Image.open(self.ignore_dir / f"{row['id']}.png") as handle:
            ignore = np.asarray(handle.convert("L"), dtype=np.uint8) > 0
        if self.augment and np.random.rand() < 0.5:
            image, instances, ignore = image[:, ::-1], instances[:, ::-1], ignore[:, ::-1]
        if self.augment and np.random.rand() < 0.5:
            image, instances, ignore = image[::-1, :], instances[::-1, :], ignore[::-1, :]
        target = make_targets(instances)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "id": row["id"],
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "instances": torch.from_numpy(np.ascontiguousarray(instances.astype(np.int32))),
            "ignore": torch.from_numpy(np.ascontiguousarray(ignore)),
        }


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (values * valid).sum() / valid.sum().clamp_min(1.0)


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, ignore: torch.Tensor) -> torch.Tensor:
    valid = (~ignore).float()
    semantic_logits, boundary_logits, center_logits = logits[:, 0], logits[:, 1], logits[:, 2]
    semantic_target, boundary_target, center_target = targets[:, 0], targets[:, 1], targets[:, 2]
    semantic_bce = _masked_mean(F.binary_cross_entropy_with_logits(semantic_logits, semantic_target, reduction="none"), valid)
    semantic_prob = torch.sigmoid(semantic_logits) * valid
    semantic_target_valid = semantic_target * valid
    intersection = (semantic_prob * semantic_target_valid).sum(dim=(1, 2))
    denominator = semantic_prob.sum(dim=(1, 2)) + semantic_target_valid.sum(dim=(1, 2))
    semantic_dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    positive = (boundary_target * valid).sum()
    negative = valid.sum() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
    boundary_bce = F.binary_cross_entropy_with_logits(
        boundary_logits, boundary_target, reduction="none", pos_weight=pos_weight
    )
    center_mse = (torch.sigmoid(center_logits) - center_target).square()
    return semantic_bce + semantic_dice_loss + 2.0 * _masked_mean(boundary_bce, valid) + _masked_mean(center_mse, valid)


def logits_to_instances(
    logits: torch.Tensor,
    semantic_threshold: float,
    center_threshold: float,
    boundary_weight: float,
    min_instance_pixels: int,
) -> np.ndarray:
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    foreground = probabilities[0] >= semantic_threshold
    if not foreground.any():
        return np.zeros(foreground.shape, dtype=np.uint16)
    center = probabilities[2]
    local_max = center == cv2.dilate(center, np.ones((21, 21), np.uint8))
    peaks = local_max & (center >= center_threshold) & foreground
    markers, marker_count = ndimage.label(peaks)
    distance = cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 5)
    if marker_count == 0:
        distance_max = distance == cv2.dilate(distance, np.ones((21, 21), np.uint8))
        fallback = distance_max & (distance >= 0.35 * max(float(distance.max()), 1e-6)) & foreground
        markers, marker_count = ndimage.label(fallback)
    if marker_count == 0:
        markers, _ = ndimage.label(foreground)
    elevation = -distance + boundary_weight * probabilities[1]
    labels = watershed(elevation, markers=markers, mask=foreground).astype(np.int32)
    output = np.zeros_like(labels, dtype=np.uint16)
    next_id = 1
    for instance_id in np.unique(labels):
        if instance_id == 0:
            continue
        mask = labels == instance_id
        if int(mask.sum()) < min_instance_pixels:
            continue
        output[mask] = next_id
        next_id += 1
    return output


def train_epoch(model, loader, device, optimizer, scaler, amp: bool) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        ignore = batch["ignore"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = compute_loss(logits, targets, ignore)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item()) * images.shape[0]
    return total / len(loader.dataset)


@torch.inference_mode()
def evaluate(model, loader, device, postprocess: dict[str, Any], prediction_dir: Path | None = None) -> dict[str, Any]:
    model.eval()
    per_image: list[dict[str, Any]] = []
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        for index, sample_id in enumerate(batch["id"]):
            predicted = logits_to_instances(
                logits[index],
                semantic_threshold=float(postprocess["semantic_threshold"]),
                center_threshold=float(postprocess["center_threshold"]),
                boundary_weight=float(postprocess["boundary_weight"]),
                min_instance_pixels=int(postprocess["min_instance_pixels"]),
            )
            reference = batch["instances"][index].numpy().astype(np.uint16)
            ignore = batch["ignore"][index].numpy().astype(bool)
            metrics = instance_metrics(
                predicted,
                reference,
                ignore=ignore,
                iou_threshold=float(postprocess["match_iou_threshold"]),
                boundary_tolerance_px=int(postprocess["boundary_tolerance_px"]),
            )
            compact = {key: value for key, value in metrics.items() if key != "matches"}
            compact["id"] = str(sample_id)
            per_image.append(compact)
            if prediction_dir is not None:
                Image.fromarray(predicted).save(prediction_dir / f"{sample_id}.tif")
    numeric_keys = ("pq", "dq", "sq", "f1")
    aggregate: dict[str, Any] = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in per_image if row[key] is not None]
        aggregate[f"macro_{key}"] = None if not values else float(np.mean(values))
    semantic_values = [float(row["semantic"]["f1"]) for row in per_image if row["semantic"]["f1"] is not None]
    boundary_values = [float(row["boundary"]["f1"]) for row in per_image if row["boundary"]["f1"] is not None]
    aggregate["macro_semantic_f1"] = None if not semantic_values else float(np.mean(semantic_values))
    aggregate["macro_boundary_f1"] = None if not boundary_values else float(np.mean(boundary_values))
    aggregate["total_true_positive"] = int(sum(row["true_positive"] for row in per_image))
    aggregate["total_false_positive"] = int(sum(row["false_positive"] for row in per_image))
    aggregate["total_false_negative"] = int(sum(row["false_negative"] for row in per_image))
    aggregate["images"] = len(per_image)
    return {"aggregate": aggregate, "per_image": per_image}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    root = Path(config.get("project_root", ".")).resolve()
    manifest = (root / config["manifest"]).resolve()
    instance_dir = (root / config["instance_dir"]).resolve()
    ignore_dir = (root / config["ignore_dir"]).resolve()
    output = (root / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    seed_everything(seed)
    device = choose_device(str(config.get("device", "auto")))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    allowed_ids = {str(value) for value in config.get("train_ids", [])} or None
    train_data = ReviewedInstanceDataset(root, manifest, instance_dir, ignore_dir, "train", True, allowed_ids)
    val_data = ReviewedInstanceDataset(root, manifest, instance_dir, ignore_dir, "val", False)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
    initialization = str(config["initialization"])
    weights_path = (root / config["weights_path"]).resolve() if config.get("weights_path") else None
    model = ResNet50InstanceNet(initialization, weights_path).to(device)
    optimizer = AdamW(
        [
            {"params": list(model.encoder_parameters), "lr": float(config["encoder_learning_rate"])},
            {"params": list(model.decoder_parameters), "lr": float(config["decoder_learning_rate"])},
        ],
        weight_decay=float(config["weight_decay"]),
    )
    amp = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    history: list[dict[str, Any]] = []
    best_key = (-1.0, -1.0)
    best_epoch = 0
    started = time.time()
    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss = train_epoch(model, train_loader, device, optimizer, scaler, amp)
        validation = evaluate(model, val_loader, device, config["postprocess"])
        aggregate = validation["aggregate"]
        record = {"epoch": epoch, "train_loss": train_loss, "val": aggregate}
        history.append(record)
        print(record, flush=True)
        pq = float(aggregate["macro_pq"] or 0.0)
        semantic = float(aggregate["macro_semantic_f1"] or 0.0)
        if (pq, semantic) > best_key:
            best_key = (pq, semantic)
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val": validation,
                    "config": config,
                    "initialization_report": model.load_report,
                },
                output / "best.pt",
            )
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    best_validation = evaluate(model, val_loader, device, config["postprocess"], output / "val_predictions")
    weights_hash = sha256_file(weights_path) if weights_path else None
    payload: dict[str, Any] = {
        "result_status": "development_result",
        "config": config,
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest),
        "weights_sha256": weights_hash,
        "initialization_report": model.load_report,
        "source_sha256": {
            "representation_train.py": sha256_file(Path(__file__)),
            "representation_model.py": sha256_file(Path(__file__).with_name("representation_model.py")),
            "instance_metrics.py": sha256_file(Path(__file__).with_name("instance_metrics.py")),
        },
        "device": str(device),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_ids": [row["id"] for row in train_data.rows],
        "val_ids": [row["id"] for row in val_data.rows],
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples_used": 0,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": best_epoch,
        "best_selection_key": {"macro_pq": best_key[0], "macro_semantic_f1": best_key[1]},
        "best_validation": best_validation,
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    if device.type == "cuda":
        payload.update(
            {
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        )
    write_json(output / "run.json", payload)
    print(f"Saved {output / 'best.pt'} and {output / 'run.json'}")


if __name__ == "__main__":
    main()
