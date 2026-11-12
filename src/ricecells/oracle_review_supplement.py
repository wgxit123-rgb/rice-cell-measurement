"""Object-budget review simulation; never an estimate of human editing time.

Score construction has no reference-label argument. Outcomes are attached only
after scoring. Missed references remain an uncorrectable floor in this model.
"""
from __future__ import annotations

import math
from collections import defaultdict
import cv2
import numpy as np
from scipy.stats import rankdata
from .instance_metrics import instance_metrics

FEATURES = ("log_area", "log_perimeter", "log_aspect", "solidity")
STRATEGIES = ("random_expected", "uncertainty", "morphology", "combined", "size_impact", "measurement_risk", "oracle_upper_bound")


def object_features(instance_map, instance_id):
    mask = (instance_map == instance_id).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("cannot measure empty object")
    contour = max(contours, key=cv2.contourArea)
    area = float(mask.sum())
    perimeter = float(cv2.arcLength(contour, True))
    _, (width, height), _ = cv2.minAreaRect(contour)
    # Preserves the historical preflight convention, not calibrated 3-D shape.
    return {"area": area, "log_area": math.log(max(area, 1.0)),
            "log_perimeter": math.log(max(perimeter, 1.0)),
            "log_aspect": math.log(max(max(width, height) / max(min(width, height), 1e-6), 1.0)),
            "solidity": min(area / max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0), 2.0)}


def robust_reference(rows):
    if not rows:
        raise ValueError("training morphology reference is empty")
    result = {}
    for key in FEATURES:
        values = np.asarray([r[key] for r in rows], dtype=float)
        median = float(np.median(values))
        result[key] = (median, max(1.4826 * float(np.median(np.abs(values - median))), 1e-3))
    return result


def entropy(probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    if not np.isfinite(p).all():
        raise ValueError("nonfinite probabilities")
    return -(p * np.log(p) + (1-p) * np.log(1-p)) / math.log(2)


def score_candidates(predicted, ignore, foreground_probability, boundary_probability, morphology_reference):
    """Reference-free ranking. Cellpose uses foreground entropy also at edges."""
    valid = ~np.asarray(ignore, dtype=bool)
    predicted = np.where(valid, predicted, 0)
    foreground_entropy = entropy(foreground_probability)
    edge_entropy = entropy(foreground_probability if boundary_probability is None else boundary_probability)
    if foreground_entropy.shape != predicted.shape or edge_entropy.shape != predicted.shape:
        raise ValueError("probability shape mismatch")
    rows = []
    for predicted_id in sorted(int(v) for v in np.unique(predicted) if v):
        mask = predicted == predicted_id
        edge = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        feature = object_features(predicted, predicted_id)
        rows.append({"predicted_id": predicted_id, "predicted_area": feature["area"],
                     "uncertainty": .5 * (float(foreground_entropy[mask].mean()) + (float(edge_entropy[edge].mean()) if edge.any() else 1.0)),
                     "morphology": max(abs(feature[k]-morphology_reference[k][0])/morphology_reference[k][1] for k in FEATURES)})
    if rows:
        uncertainty_rank = rankdata([r["uncertainty"] for r in rows]) / len(rows)
        morphology_rank = rankdata([r["morphology"] for r in rows]) / len(rows)
        for row, u, m in zip(rows, uncertainty_rank, morphology_rank):
            row.update({"uncertainty_rank": float(u), "morphology_rank": float(m), "combined": float((u+m)/2),
                        "size_impact": row["predicted_area"], "measurement_risk": float((u+m)/2 * row["predicted_area"])})
    return rows


def attach_outcomes(candidates, predicted, reference, ignore):
    valid = ~np.asarray(ignore, dtype=bool)
    metrics = instance_metrics(predicted, reference, ignore=ignore)
    matches = {int(r["predicted_id"]): int(r["reference_id"]) for r in metrics["matches"]}
    matched_ref = set(matches.values())
    ref_areas = {int(i): float(((reference == i) & valid).sum()) for i in np.unique(reference[valid]) if i}
    rows = []
    for candidate in candidates:
        row = dict(candidate)
        rid = matches.get(row["predicted_id"])
        error = row["predicted_area"] if rid is None else abs(row["predicted_area"] - ref_areas[rid])
        row.update({"reference_id": rid, "correctable_area_error": error, "oracle_upper_bound": error})
        rows.append(row)
    floor = sum(a for i, a in ref_areas.items() if i not in matched_ref)
    return rows, {"missed_reference_area_pixels": floor, "reference_area_pixels": sum(ref_areas.values()),
                  "initial_error_pixels": floor + sum(r["correctable_area_error"] for r in rows)}


def simulate_budgets(rows, totals, fractions):
    n = len(rows)
    error = np.asarray([r["correctable_area_error"] for r in rows], dtype=float)
    output = []
    for fraction in fractions:
        if not 0 <= fraction <= 1:
            raise ValueError("review fraction must be in [0,1]")
        k = min(n, math.ceil(fraction*n))
        for strategy in STRATEGIES:
            if strategy == "random_expected":
                removed = float(error.sum()) * k/n if n else 0.0
            else:
                order = np.argsort(-np.asarray([r[strategy] for r in rows]), kind="stable")
                removed = float(error[order[:k]].sum())
            denominator = totals["reference_area_pixels"]
            initial = totals["initial_error_pixels"]
            residual = max(initial-removed, 0.0)
            output.append({"strategy": strategy, "budget_fraction": fraction, "review_count": k,
                           "candidate_count": n, "removed_area_error_pixels": removed,
                           "removed_correctable_error_fraction": removed/float(error.sum()) if error.sum() else None,
                           "removed_total_error_fraction": removed/initial if initial else None,
                           "residual_area_error_fraction": residual/denominator if denominator else None,
                           **totals})
    return output


def hierarchical_summary(budget_rows):
    """Average optimization seeds within each view before describing views."""
    keys = ("system", "image_id", "strategy", "budget_fraction")
    metrics = ("removed_correctable_error_fraction", "removed_total_error_fraction", "residual_area_error_fraction")
    grouped = defaultdict(list)
    for row in budget_rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    per_view = []
    for key, rows in sorted(grouped.items()):
        if len(rows) != 3 or len({r["seed"] for r in rows}) != 3:
            raise ValueError("each view requires three unique training seeds")
        row = dict(zip(keys, key)); row["training_seeds"] = 3
        for metric in metrics:
            values = [r[metric] for r in rows if r[metric] is not None]
            row[metric] = float(np.mean(values)) if values else None
            row[metric+"_defined_seeds"] = len(values)
        per_view.append(row)
    grouped = defaultdict(list)
    for row in per_view:
        grouped[(row["system"], row["strategy"], row["budget_fraction"])].append(row)
    summary = []
    for key, rows in sorted(grouped.items()):
        if len(rows) != 5 or len({r["image_id"] for r in rows}) != 5:
            raise ValueError("summary requires five distinct validation views")
        row = dict(zip(("system", "strategy", "budget_fraction"), key)); row["validation_views"] = 5
        for metric in metrics:
            values = [r[metric] for r in rows if r[metric] is not None]
            row[metric+"_mean"] = float(np.mean(values)) if values else None
            row[metric+"_view_sd"] = float(np.std(values, ddof=1)) if len(values)>1 else None
            row[metric+"_defined_views"] = len(values)
        summary.append(row)
    return per_view, summary
