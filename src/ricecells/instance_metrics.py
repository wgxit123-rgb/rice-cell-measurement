"""Instance-level metrics with explicit empty-set and matching semantics.

The legacy semantic Dice helper is intentionally not reused here.  In
particular, an empty prediction and an empty target are reported as an
``empty_both`` case instead of silently becoming a perfect score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def _safe_f1(tp: int, fp: int, fn: int) -> tuple[float | None, str]:
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return None, "empty_both"
    return float(2 * tp / denominator), "non_empty"


def binary_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Return binary precision/recall/F1 without empty-set inflation."""

    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if pred.shape != target.shape:
        raise ValueError("pred and target shapes differ")
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    f1, status = _safe_f1(tp, fp, fn)
    precision = None if tp + fp == 0 else float(tp / (tp + fp))
    recall = None if tp + fn == 0 else float(tp / (tp + fn))
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "predicted_positive": int(pred.sum()),
        "target_positive": int(target.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "status": status,
    }


def _ids(instance_map: np.ndarray, valid: np.ndarray) -> list[int]:
    return [int(value) for value in np.unique(instance_map[valid]) if int(value) != 0]


def _iou_matrix(
    predicted: np.ndarray,
    reference: np.ndarray,
    predicted_ids: list[int],
    reference_ids: list[int],
    valid: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((len(predicted_ids), len(reference_ids)), dtype=np.float64)
    for row, predicted_id in enumerate(predicted_ids):
        predicted_mask = (predicted == predicted_id) & valid
        for col, reference_id in enumerate(reference_ids):
            reference_mask = (reference == reference_id) & valid
            union = int(np.logical_or(predicted_mask, reference_mask).sum())
            if union:
                matrix[row, col] = float(np.logical_and(predicted_mask, reference_mask).sum() / union)
    return matrix


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask)
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    return mask & ~eroded


def _tolerant_boundary_f1(predicted: np.ndarray, reference: np.ndarray, tolerance_px: int) -> dict[str, Any]:
    predicted_boundary = _boundary(predicted)
    reference_boundary = _boundary(reference)
    if not predicted_boundary.any() and not reference_boundary.any():
        return {"precision": None, "recall": None, "f1": None, "status": "empty_both"}
    if not predicted_boundary.any() or not reference_boundary.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "status": "one_empty"}
    # Distance transforms give the nearest boundary distance in pixels.
    distance_to_reference = cv2.distanceTransform((~reference_boundary).astype(np.uint8), cv2.DIST_L2, 3)
    distance_to_predicted = cv2.distanceTransform((~predicted_boundary).astype(np.uint8), cv2.DIST_L2, 3)
    precision = float((distance_to_reference[predicted_boundary] <= tolerance_px).mean())
    recall = float((distance_to_predicted[reference_boundary] <= tolerance_px).mean())
    f1 = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))
    return {"precision": precision, "recall": recall, "f1": f1, "status": "non_empty"}


def instance_metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    ignore: np.ndarray | None = None,
    iou_threshold: float = 0.5,
    boundary_tolerance_px: int = 2,
    event_iou_threshold: float = 0.1,
) -> dict[str, Any]:
    """Compute one-to-one instance metrics using deterministic Hungarian matching.

    ``iou_threshold`` is the frozen match threshold used for TP.  The lower
    ``event_iou_threshold`` is only a diagnostic: a predicted instance
    overlapping multiple reference instances suggests a merge, while a
    reference instance overlapping multiple predictions suggests a split.
    Ignore pixels are excluded from both masks before matching.
    """

    predicted = np.asarray(predicted)
    reference = np.asarray(reference)
    if predicted.shape != reference.shape:
        raise ValueError("predicted and reference shapes differ")
    if ignore is None:
        valid = np.ones(predicted.shape, dtype=bool)
    else:
        ignore = np.asarray(ignore, dtype=bool)
        if ignore.shape != predicted.shape:
            raise ValueError("ignore shape differs from predicted/reference")
        valid = ~ignore

    predicted_ids = _ids(predicted, valid)
    reference_ids = _ids(reference, valid)
    ious = _iou_matrix(predicted, reference, predicted_ids, reference_ids, valid)
    matches: list[dict[str, Any]] = []
    if ious.size:
        rows, cols = linear_sum_assignment(-ious)
        for row, col in zip(rows.tolist(), cols.tolist()):
            score = float(ious[row, col])
            if score >= iou_threshold:
                matches.append({"predicted_id": predicted_ids[row], "reference_id": reference_ids[col], "iou": score})
    tp = len(matches)
    fp = len(predicted_ids) - tp
    fn = len(reference_ids) - tp
    f1, status = _safe_f1(tp, fp, fn)
    precision = None if tp + fp == 0 else float(tp / (tp + fp))
    recall = None if tp + fn == 0 else float(tp / (tp + fn))
    sq = None if not matches else float(np.mean([match["iou"] for match in matches]))
    dq = None if status == "empty_both" else float(2 * tp / (2 * tp + fp + fn))
    # Standard PQ is zero whenever the image is non-empty but has no matched
    # instances.  Only the genuinely empty/empty case is undefined.
    pq = None if status == "empty_both" else (0.0 if not matches else float(sq * dq))

    merge_events = 0
    for row in range(len(predicted_ids)):
        if int((ious[row] >= event_iou_threshold).sum()) >= 2:
            merge_events += 1
    split_events = 0
    for col in range(len(reference_ids)):
        if int((ious[:, col] >= event_iou_threshold).sum()) >= 2:
            split_events += 1

    predicted_union = np.logical_and(predicted > 0, valid)
    reference_union = np.logical_and(reference > 0, valid)
    semantic = binary_metrics(predicted_union, reference_union)
    boundary = _tolerant_boundary_f1(predicted_union, reference_union, boundary_tolerance_px)
    return {
        "matching": "hungarian_max_iou",
        "iou_threshold": float(iou_threshold),
        "event_iou_threshold": float(event_iou_threshold),
        "boundary_tolerance_px": int(boundary_tolerance_px),
        "predicted_instances": len(predicted_ids),
        "reference_instances": len(reference_ids),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "status": status,
        "sq": sq,
        "dq": dq,
        "pq": pq,
        "merge_event_count": merge_events,
        "split_event_count": split_events,
        "matches": matches,
        "semantic": semantic,
        "boundary": boundary,
    }


__all__ = ["binary_metrics", "instance_metrics"]
