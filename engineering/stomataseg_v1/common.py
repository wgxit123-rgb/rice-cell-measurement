from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

INITIALIZATIONS = ("random", "supervised_imagenet", "dino_self_supervised")
SEEDS = (20261026, 20261027, 20261028)
FIXED_TRAINING = {
    "steps": 300, "batch_size": 4, "crop_size": 384,
    "foreground_crop_probability": 0.75, "encoder_lr": 1e-4,
    "decoder_lr": 1e-3, "weight_decay": 1e-4,
    "cpu_threads": 4, "amp": True, "horizontal_flip": True,
    "vertical_flip": True, "loss_dtype": "float32", "amp_init_scale": 4096.0,
    "max_amp_overflow_skips": 20, "step_definition": "effective_optimizer_updates",
}
FIXED_DECODER = {
    "foreground_threshold": 0.5, "center_threshold": 0.2,
    "nms_size": 9, "boundary_weight": 3.0, "min_instance_area": 16,
}
POLICY = {
    "category_id": 3173368342, "long_side": 0, "supervision_halo": 3,
    "rasterization": "skimage_polygon_pixel_centers",
    "overlap": "same_category_conflict_ignore",
    "train_supervision": "annotated_foreground_and_inferred_local_negative_halo",
    "halo_interpretation": "local_negative_assumption_not_human_confirmed_background",
    "evaluation_supervision": "all_pixels_except_overlap_or_explicit_invalid",
    "unsupported_reference_policy": "train_strict_test_preserve_unscorable_annotation_records",
    "unscorable_interpretation": "current_pixel_reference_support_not_human_annotation_error",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sha(path: Path, expected: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Missing/invalid SHA-256 for {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}: expected {expected}, actual {actual}")
    return actual


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, content: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def resolve(base: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("An explicit non-empty path is required")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> tuple[dict[str, Any], Path, str]:
    path = path.resolve()
    config = read_json(path)
    if config.get("schema_version") != 1:
        raise ValueError("Expected config schema_version=1")
    if config.get("training") != FIXED_TRAINING:
        raise ValueError("Training settings differ from the fixed design")
    if config.get("decoder") != FIXED_DECODER:
        raise ValueError("Decoder settings differ from the fixed design")
    if config.get("preprocessing") != POLICY:
        raise ValueError("Preprocessing/partial-supervision policy differs from the fixed design")
    validate_runs(config["runs"])
    return config, path.parent, sha256(path)


def validate_runs(runs: list[dict[str, Any]]) -> None:
    expected = {(init, budget, seed) for init in INITIALIZATIONS for budget in (4, 16) for seed in SEEDS}
    actual = {(run["initialization"], run["budget"], run["seed"]) for run in runs}
    if len(runs) != 18 or actual != expected or len({run["id"] for run in runs}) != 18:
        raise ValueError("Exactly 18 unique initialization/budget/seed runs are required")
    selected: dict[int, list[Any]] = {}
    outputs = []
    for run in runs:
        ids = run["image_ids"]
        if len(ids) != run["budget"] or len(set(ids)) != len(ids):
            raise ValueError("Explicit unique full-image IDs must equal the budget")
        if run["budget"] in selected and selected[run["budget"]] != ids:
            raise ValueError("Image selection must be identical across seeds and initialization")
        selected[run["budget"]] = ids
        outputs.append(run["output_dir"])
    if selected[4] != selected[16][:4] or len(set(outputs)) != 18:
        raise ValueError("Budgets must be nested in fixed order, with distinct output directories")


def source_receipt(config: dict[str, Any], base: Path) -> dict[str, str]:
    sources = config.get("source_sha256", {})
    if not sources:
        raise ValueError("An explicit frozen source_sha256 mapping is required")
    declared_paths = {resolve(base, filename) for filename in sources}
    required_paths = {Path(__file__).with_name(filename).resolve() for filename in
                      ("__init__.py", "common.py", "prepare.py", "train.py", "predict.py")}
    if not required_paths.issubset(declared_paths):
        raise ValueError("source_sha256 must bind every executed engine module")
    for filename, expected in sources.items():
        check_sha(resolve(base, filename), expected)
    return dict(sources)


def frozen_gate(config: dict[str, Any], base: Path, config_sha: str, evaluation: bool = False):
    key = "evaluation_freeze" if evaluation else "training_freeze"
    path = resolve(base, config[key]["path"])
    freeze = read_json(path)
    if freeze.get("schema_version") != 1 or freeze.get("status") != "frozen":
        raise ValueError(f"An explicit frozen {key} is required")
    if freeze.get("config_sha256") != config_sha or freeze.get("source_sha256") != config["source_sha256"]:
        raise ValueError("Freeze does not bind this config and source mapping")
    if set(freeze.get("run_ids", [])) != {run["id"] for run in config["runs"]}:
        raise ValueError("Freeze must cover all 18 runs")
    if evaluation and freeze.get("allow_test_inference") is not True:
        raise ValueError("Evaluation freeze must explicitly allow test inference")
    return freeze, sha256(path)


def get_run(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    matches = [run for run in config["runs"] if run["id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate run_id: {run_id}")
    return matches[0]


def prepared_manifest(config: dict[str, Any], base: Path, split: str, expected_sha: str):
    path = resolve(base, config["prepared"][split]["output_dir"]) / "manifest.json"
    check_sha(path, expected_sha)
    manifest = read_json(path)
    if manifest.get("schema_version") != 1 or manifest.get("split") != split:
        raise ValueError("Prepared manifest split/schema mismatch")
    if manifest.get("policy") != POLICY:
        raise ValueError("Prepared manifest policy mismatch")
    return path, manifest


def artifact_path(manifest_path: Path, record: dict[str, Any], name: str) -> Path:
    item = record["artifacts"][name]
    path = resolve(manifest_path.parent, item["path"])
    if not path.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError("Prepared artifact escaped its manifest directory")
    check_sha(path, item["sha256"])
    return path


def resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    ys = np.minimum(((np.arange(h) + 0.5) * array.shape[0] / h).astype(int), array.shape[0] - 1)
    xs = np.minimum(((np.arange(w) + 0.5) * array.shape[1] / w).astype(int), array.shape[1] - 1)
    return array[ys[:, None], xs[None, :]]


def image_to_work(image: Image.Image, long_side: int = 0) -> np.ndarray:
    image = image.convert("RGB")
    if long_side == 0:
        return np.asarray(image).copy()
    if long_side < 0:
        raise ValueError("long_side must be zero (native) or positive")
    factor = min(1.0, long_side / max(image.size))
    shape = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
    return np.asarray(image.resize(shape, Image.Resampling.BILINEAR)).copy()


def require_retained_objects(native: np.ndarray, work: np.ndarray) -> None:
    native_ids = set(int(value) for value in np.unique(native) if value != 0)
    work_ids = set(int(value) for value in np.unique(work) if value != 0)
    if native_ids != work_ids:
        raise ValueError(f"Object lost during preprocessing: missing IDs {sorted(native_ids - work_ids)}")


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    image = rgb.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return np.ascontiguousarray(image.transpose(2, 0, 1))


def supervision_masks(labels: np.ndarray, overlap: np.ndarray, explicit_invalid: np.ndarray | None = None):
    invalid = overlap.copy()
    if explicit_invalid is not None:
        invalid |= explicit_invalid
    foreground = labels > 0
    # Chebyshev-distance three work-pixel outer ring; positives always take priority.
    local_region = ndi.binary_dilation(foreground, structure=np.ones((3, 3), bool), iterations=3)
    evaluation_valid = ~invalid
    train_valid = (foreground | local_region) & evaluation_valid
    return train_valid, evaluation_valid


def make_targets(labels: np.ndarray) -> np.ndarray:
    targets = np.zeros((3, *labels.shape), np.float32)
    targets[0] = labels > 0
    ids, indexed = np.unique(labels, return_inverse=True)
    indexed = indexed.reshape(labels.shape)
    if ids[0] != 0:
        indexed += 1
        ids = np.concatenate((np.zeros(1, labels.dtype), ids))
    objects = ndi.find_objects(indexed)
    for local_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        annotation_id = ids[local_id]
        # A real one-pixel image margin preserves global erosion/EDT at image edges.
        y0, y1 = max(0, slices[0].start - 1), min(labels.shape[0], slices[0].stop + 1)
        x0, x1 = max(0, slices[1].start - 1), min(labels.shape[1], slices[1].stop + 1)
        mask = labels[y0:y1, x0:x1] == annotation_id
        boundary = mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3), bool), border_value=0)
        targets[1, y0:y1, x0:x1] = np.maximum(targets[1, y0:y1, x0:x1], boundary)
        distance = ndi.distance_transform_edt(mask)
        cy, cx = np.unravel_index(np.argmax(distance), distance.shape)
        cy, cx = cy + y0, cx + x0
        sigma = max(2.0, np.sqrt(mask.sum() / np.pi) / 4.0)
        radius = int(np.ceil(4.3 * sigma))
        gy0, gy1 = max(0, cy - radius), min(labels.shape[0], cy + radius + 1)
        gx0, gx1 = max(0, cx - radius), min(labels.shape[1], cx + radius + 1)
        yy, xx = np.ogrid[gy0:gy1, gx0:gx1]
        center = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        center[center < 1e-4] = 0
        targets[2, gy0:gy1, gx0:gx1] = np.maximum(targets[2, gy0:gy1, gx0:gx1], center.astype(np.float32))
    return targets


def load_model_class(config: dict[str, Any], base: Path):
    item = config["architecture_source"]
    path = resolve(base, item["path"])
    check_sha(path, item["sha256"])
    if config["source_sha256"].get(item["path"]) != item["sha256"]:
        raise ValueError("Architecture source must also be in source_sha256")
    spec = importlib.util.spec_from_file_location("stomataseg_frozen_r50", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the explicit architecture source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ResNet50InstanceNet
