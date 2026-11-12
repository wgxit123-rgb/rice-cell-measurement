from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.draw import polygon

from .common import (POLICY, check_sha, image_to_work, load_config, make_targets,
                     read_json, require_retained_objects, resize_nearest, resolve, sha256, source_receipt,
                     supervision_masks, write_json)


def rasterize_annotations(annotations: list[dict], shape: tuple[int, int], allow_unscorable: bool = False):
    """Union every polygon belonging to an annotation before detecting inter-ID conflicts."""
    labels = np.zeros(shape, dtype=np.uint64)
    overlap = np.zeros(shape, dtype=bool)
    seen: set[int] = set()
    original_areas = {}
    for annotation in annotations:
        annotation_id = annotation["id"]
        if not isinstance(annotation_id, int) or annotation_id <= 0 or annotation_id in seen:
            raise ValueError("Annotation IDs must be unique positive integers")
        seen.add(annotation_id)
        if annotation.get("iscrowd", 0) != 0 or not isinstance(annotation["segmentation"], list):
            raise ValueError("Only explicit non-crowd COCO polygon instances are supported")
        mask = np.zeros(shape, bool)
        for coordinates in annotation["segmentation"]:
            points = np.asarray(coordinates, dtype=np.float64)
            if points.ndim != 1 or len(points) < 6 or len(points) % 2 or not np.isfinite(points).all():
                raise ValueError(f"Invalid polygon for annotation {annotation_id}")
            points = points.reshape(-1, 2)
            rows, columns = polygon(points[:, 1] - 0.5, points[:, 0] - 0.5, shape=shape)
            mask[rows, columns] = True
        if not mask.any() and not allow_unscorable:
            raise ValueError(f"Annotation {annotation_id} has no rasterized native pixels")
        original_areas[annotation_id] = int(mask.sum())
        overlap |= mask & (labels != 0)
        labels[mask & (labels == 0)] = annotation_id
    labels[overlap] = 0
    for annotation_id in seen:
        if not np.any(labels == annotation_id) and not allow_unscorable:
            raise ValueError(f"Annotation {annotation_id} has no valid native pixels after conflict removal")
    return labels, overlap, original_areas


def prepare_split(config_path: Path, split: str):
    config, base, config_sha = load_config(config_path)
    sources_before = source_receipt(config, base)
    catalog_path = resolve(base, config["catalog"]["path"])
    check_sha(catalog_path, config["catalog"]["sha256"])
    annotation_path = resolve(base, config["annotations"][split]["path"])
    annotation_sha = check_sha(annotation_path, config["annotations"][split]["sha256"])
    catalog = read_json(catalog_path)
    coco = read_json(annotation_path)
    category = [row for row in coco["categories"] if row["id"] == POLICY["category_id"]]
    if len(category) != 1:
        raise ValueError("Required original complex-area category is absent or duplicated")
    catalog_rows = [row for row in catalog["images"] if row["split"] == split]
    if len({row["image_id"] for row in catalog_rows}) != len(catalog_rows):
        raise ValueError("Duplicate catalog image IDs within split")
    images = {row["id"]: row for row in coco["images"]}
    if len(images) != len(coco["images"]):
        raise ValueError("Duplicate COCO image IDs")
    by_image: dict[object, list[dict]] = {image_id: [] for image_id in images}
    for annotation in coco["annotations"]:
        if annotation["image_id"] not in images:
            raise ValueError("COCO annotation references an absent image")
        if annotation["category_id"] == POLICY["category_id"]:
            by_image[annotation["image_id"]].append(annotation)
    paired = [row for row in catalog_rows if row["paired"] is True]
    excluded = [row["image_id"] for row in catalog_rows if row["paired"] is not True]
    if split == "train":
        wanted = list(dict.fromkeys(image_id for run in config["runs"] for image_id in run["image_ids"]))
        by_catalog_id = {row["image_id"]: row for row in paired}
        if set(wanted) - set(by_catalog_id):
            raise ValueError("A frozen train image ID is absent from the exact paired pool")
        selected = [by_catalog_id[image_id] for image_id in wanted]
    else:
        selected = paired
    if split == "test" and (len(catalog_rows) != 33 or excluded or set(images) != {row["image_id"] for row in selected}):
        raise ValueError("Test preparation must retain all 33 original, exactly paired images")
    if not selected:
        raise ValueError("No exactly paired images in this split")
    out_dir = resolve(base, config["prepared"][split]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "split": split, "config_sha256": config_sha,
                "catalog_sha256": config["catalog"]["sha256"], "annotation_sha256": annotation_sha,
                "policy": POLICY, "source_sha256": sources_before,
                "excluded_unpaired_image_ids": excluded,
                "unselected_paired_image_ids": [row["image_id"] for row in paired if row not in selected],
                "images": []}
    try:
        for index, row in enumerate(selected):
            image_id = row["image_id"]
            if image_id not in images:
                raise ValueError(f"Paired catalog ID absent from COCO: {image_id}")
            reference = images[image_id]
            if (row["file_name"] != reference["file_name"] or row["width"] != reference["width"]
                    or row["height"] != reference["height"]):
                raise ValueError(f"Exact COCO/catalog pairing failed: {image_id}")
            source_path = resolve(catalog_path.parent, row["path"])
            check_sha(source_path, row["sha256"])
            if source_path.stat().st_size != row["bytes"]:
                raise ValueError("Catalog byte size mismatch")
            with Image.open(source_path) as image:
                image = image.convert("RGB")
                if image.size != (row["width"], row["height"]):
                    raise ValueError("Decoded image dimensions mismatch")
                rgb_digest = hashlib.sha256(struct.pack(">II", image.width, image.height) + np.asarray(image).tobytes()).hexdigest()
                if rgb_digest != row["rgb_sha256"]:
                    raise ValueError("Catalog decoded RGB SHA-256 mismatch")
                rgb = image_to_work(image, long_side=POLICY["long_side"])
            native, conflict, original_areas = rasterize_annotations(
                by_image[image_id], (row["height"], row["width"]), allow_unscorable=split == "test")
            work_shape = rgb.shape[:2]
            work = resize_nearest(native, work_shape)
            conflict_work = resize_nearest(conflict, work_shape)
            work[conflict_work] = 0
            require_retained_objects(native, work)
            instances = []
            for annotation_id, polygon_area in sorted(original_areas.items()):
                native_area = int(np.count_nonzero(native == annotation_id))
                work_area = int(np.count_nonzero(work == annotation_id))
                if native_area and not work_area:
                    raise ValueError(f"Object lost during preprocessing: image={image_id}, annotation={annotation_id}")
                scorable = native_area > 0 and work_area > 0
                reason = None if scorable else (
                    "native_zero_pixel_support" if polygon_area == 0 else "fully_conflict_ignored")
                instances.append({"annotation_id": annotation_id, "native_area": native_area,
                                  "native_polygon_area_before_overlap": polygon_area, "work_area": work_area,
                                  "scorable": scorable, "reason": reason})
            train_valid, evaluation_valid = supervision_masks(work, conflict_work)
            targets = make_targets(work)
            arrays = {"instances_native": native, "instances_work": work,
                      "train_valid": train_valid, "evaluation_valid": evaluation_valid,
                      "evaluation_valid_native": ~conflict, "targets": targets}
            artifacts = {}
            image_dir = out_dir / f"image_{index:04d}"
            image_dir.mkdir()
            for name, array in arrays.items():
                target = image_dir / f"{name}.npy"
                with target.open("xb") as stream:
                    np.save(stream, array, allow_pickle=False)
                artifacts[name] = {"path": target.relative_to(out_dir).as_posix(), "sha256": sha256(target)}
            rgb_path = image_dir / "rgb_work.png"
            Image.fromarray(rgb).save(rgb_path)
            artifacts["rgb_work"] = {"path": rgb_path.relative_to(out_dir).as_posix(), "sha256": sha256(rgb_path)}
            manifest["images"].append({"image_id": image_id, "file_name": row["file_name"],
                "source_image_path": str(source_path), "source_image_sha256": row["sha256"],
                "native_shape": list(native.shape), "work_shape": list(work_shape),
                "scale_yx": [work_shape[0] / native.shape[0], work_shape[1] / native.shape[1]],
                "instances": instances, "source_annotation_count": len(original_areas),
                "scorable_count": sum(instance["scorable"] for instance in instances),
                "unscorable_records": [instance for instance in instances if not instance["scorable"]],
                "overlap_native_pixels": int(conflict.sum()),
                "overlap_work_pixels": int(conflict_work.sum()), "artifacts": artifacts})
            check_sha(source_path, row["sha256"])
        check_sha(catalog_path, config["catalog"]["sha256"])
        check_sha(annotation_path, annotation_sha)
        if source_receipt(config, base) != sources_before or sha256(config_path) != config_sha:
            raise ValueError("Sources/config changed during preparation")
        manifest["source_annotation_count"] = sum(record["source_annotation_count"] for record in manifest["images"])
        manifest["scorable_count"] = sum(record["scorable_count"] for record in manifest["images"])
        manifest["unscorable_count"] = manifest["source_annotation_count"] - manifest["scorable_count"]
        write_json(out_dir / "manifest.json", manifest)
    except Exception as error:
        write_json(out_dir / "FAILURE.json", {"status": "failed", "error": repr(error), "split": split})
        raise
    return out_dir / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description="Prepare exact original complex-instance polygons, without selection.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    args = parser.parse_args()
    print(prepare_split(args.config, args.split))


if __name__ == "__main__":
    main()
