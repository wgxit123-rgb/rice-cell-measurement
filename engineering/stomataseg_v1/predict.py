from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from torch.nn import functional as F

from .common import (FIXED_DECODER, artifact_path, check_sha, frozen_gate, get_run,
                     load_config, load_model_class, normalize_rgb, prepared_manifest,
                     read_json, resize_nearest, resolve, sha256, source_receipt, write_json)


def decode_instances(probabilities: np.ndarray) -> np.ndarray:
    """Fixed local-center NMS plus one distance maximum per unseeded foreground component."""
    if probabilities.ndim != 3 or probabilities.shape[0] != 3 or not np.isfinite(probabilities).all():
        raise ValueError("Expected finite [foreground,boundary,center] probability maps")
    foreground, boundary, center = probabilities
    mask = foreground >= 0.5
    if not mask.any():
        return np.zeros(mask.shape, np.uint16)
    peaks = (center >= 0.2) & (center == ndi.maximum_filter(center, size=9, mode="constant")) & mask
    peak_regions, count = ndi.label(peaks, structure=np.ones((3, 3), bool))
    markers = np.zeros(mask.shape, np.int32)
    marker_id = 0
    for region, slices in enumerate(ndi.find_objects(peak_regions), start=1):
        if slices is None:
            continue
        locations = np.argwhere(peak_regions[slices] == region)
        # Flat NMS plateaus get one deterministic row-major maximum.
        scores = center[slices][locations[:, 0], locations[:, 1]]
        row, column = locations[int(np.argmax(scores))]
        row, column = row + slices[0].start, column + slices[1].start
        marker_id += 1
        markers[row, column] = marker_id
    components, component_count = ndi.label(mask, structure=np.ones((3, 3), bool))
    distance = ndi.distance_transform_edt(mask)
    seeded = np.zeros(component_count + 1, bool)
    seeded[components[markers > 0]] = True
    for component, slices in enumerate(ndi.find_objects(components), start=1):
        if slices is not None and not seeded[component]:
            current = components[slices] == component
            row, column = np.unravel_index(np.argmax(np.where(current, distance[slices], -1)), current.shape)
            row, column = row + slices[0].start, column + slices[1].start
            marker_id += 1
            markers[row, column] = marker_id
    if marker_id > 65535:
        raise ValueError("Prediction has more than 65535 instances; uint16 would truncate")
    decoded = watershed(-distance + 3.0 * boundary, markers=markers, mask=mask, connectivity=2)
    areas = np.bincount(decoded.ravel(), minlength=marker_id + 1)
    kept = np.flatnonzero(areas >= 16)
    kept = kept[kept != 0]
    lookup = np.zeros(marker_id + 1, np.uint16)
    lookup[kept] = np.arange(1, len(kept) + 1, dtype=np.uint16)
    return lookup[decoded]


def pad_to_32(image):
    h, w = image.shape[-2:]
    return F.pad(image, (0, (-w) % 32, 0, (-h) % 32), mode="replicate"), (h, w)


def predict_run(config_path: Path, run_id: str):
    config, base, config_sha = load_config(config_path)
    sources = source_receipt(config, base)
    freeze, freeze_sha = frozen_gate(config, base, config_sha, evaluation=True)
    run = get_run(config, run_id)
    expected_runs = {item["id"] for item in config["runs"]}
    if set(freeze.get("checkpoints_sha256", {})) != expected_runs:
        raise ValueError("Evaluation must bind all 18 final checkpoints before any test input is read")
    # Check every terminal receipt/checkpoint before opening the TEST manifest or image.
    for other in config["runs"]:
        other_dir = resolve(base, other["output_dir"])
        other_receipt = read_json(other_dir / "run.json")
        if (other_receipt.get("status") != "completed" or other_receipt.get("step") != 300
                or other_receipt.get("run_id") != other["id"]
                or other_receipt.get("config_sha256") != config_sha
                or other_receipt.get("source_sha256") != sources):
            raise ValueError("All 18 runs must be complete and bound to frozen config/source")
        expected = freeze["checkpoints_sha256"][other["id"]]
        if other_receipt["checkpoint"]["path"] != "final.pt" or other_receipt["checkpoint"]["sha256"] != expected:
            raise ValueError("Terminal receipt checkpoint mismatch")
        check_sha(other_dir / "final.pt", expected)
    manifest_path, manifest = prepared_manifest(config, base, "test", freeze["prepared_manifest_sha256"]["test"])
    if (len(manifest["images"]) != 33 or manifest["config_sha256"] != config_sha
            or manifest["source_sha256"] != sources):
        raise ValueError("Evaluation requires all 33 test images and current config/source")
    out_dir = resolve(base, run["prediction_dir"])
    out_dir.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "status": "running", "run_id": run_id,
               "config_sha256": config_sha, "source_sha256": sources,
               "evaluation_freeze_sha256": freeze_sha, "prepared_manifest_sha256": sha256(manifest_path),
               "checkpoint_sha256": freeze["checkpoints_sha256"][run_id], "decoder": FIXED_DECODER,
               "ground_truth_arrays_read": False, "reference_metadata_present_in_manifest": True,
               "reference_metadata_used_for_prediction": False, "images": []}
    write_json(out_dir / "STARTED.json", receipt)
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "1"):
            raise ValueError("This experiment reserves physical GPU1 only")
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        torch.set_num_threads(4)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Single visible CUDA GPU1 is required; no device fallback")
        model = load_model_class(config, base)("random", None)
        checkpoint_path = resolve(base, run["output_dir"]) / "final.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (checkpoint["step"] != 300 or checkpoint["run_id"] != run_id
                or checkpoint["config_sha256"] != config_sha or checkpoint["source_sha256"] != sources):
            raise ValueError("Checkpoint metadata mismatch")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model = model.to("cuda:0").eval()
        rgb_hashes = {}
        with torch.inference_mode():
            for index, record in enumerate(manifest["images"]):
                rgb_path = artifact_path(manifest_path, record, "rgb_work")
                rgb_hashes[str(rgb_path)] = record["artifacts"]["rgb_work"]["sha256"]
                with Image.open(rgb_path) as image:
                    rgb = np.asarray(image.convert("RGB")).copy()
                if list(rgb.shape[:2]) != record["work_shape"]:
                    raise ValueError("Test image/work shape mismatch")
                tensor = torch.from_numpy(normalize_rgb(rgb)).unsqueeze(0).to("cuda:0")
                padded, shape = pad_to_32(tensor)
                logits = model(padded)[..., :shape[0], :shape[1]]
                probabilities = torch.sigmoid(logits[0].float()).cpu().numpy()
                prediction = decode_instances(probabilities)
                native = resize_nearest(prediction, tuple(record["native_shape"]))
                if native.dtype != np.uint16 or int(native.max()) > 65535:
                    raise ValueError("Native prediction must remain uint16 without ID truncation")
                output = out_dir / f"image_{index:04d}.tif"
                Image.fromarray(native).save(output, compression="tiff_deflate")
                receipt["images"].append({"image_id": record["image_id"], "file_name": record["file_name"],
                    "native_shape": record["native_shape"], "work_shape": record["work_shape"],
                    "instance_count": int(prediction.max()), "prediction": {"path": output.name, "sha256": sha256(output)}})
        for filename, expected in rgb_hashes.items():
            check_sha(Path(filename), expected)
        check_sha(checkpoint_path, freeze["checkpoints_sha256"][run_id])
        check_sha(manifest_path, receipt["prepared_manifest_sha256"])
        check_sha(resolve(base, config["evaluation_freeze"]["path"]), freeze_sha)
        if source_receipt(config, base) != sources or sha256(config_path) != config_sha:
            raise ValueError("Sources/config changed during prediction")
        receipt["status"] = "completed"
        write_json(out_dir / "predictions.json", receipt)
    except Exception as error:
        receipt.update(status="failed", error=repr(error))
        write_json(out_dir / "FAILURE.json", receipt)
        raise
    return out_dir / "predictions.json"


def main():
    parser = argparse.ArgumentParser(description="Predict all test images only after explicit full evaluation freeze.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(predict_run(args.config, args.run_id))


if __name__ == "__main__":
    main()
