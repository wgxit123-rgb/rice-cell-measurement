"""Positive-reference measurement endpoints; unmatched predictions are NOT FP.

No claim of exhaustive image annotation or absolute physical area is made.
Hungarian assignment precedes the inclusive IoU threshold. Contingency counts
are saved as integers to permit independent scalar reconstruction.
"""
from fractions import Fraction

import numpy as np
from scipy.optimize import linear_sum_assignment


def contingency(prediction, reference, ignore=None):
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    if prediction.ndim != 2 or prediction.shape != reference.shape:
        raise ValueError("Expected equal-shape two-dimensional label maps")
    if not np.issubdtype(prediction.dtype, np.integer) or not np.issubdtype(reference.dtype, np.integer):
        raise ValueError("Labels must be integer-valued arrays")
    if np.any(prediction < 0) or np.any(reference < 0):
        raise ValueError("Negative labels are invalid")
    valid = np.ones(reference.shape, dtype=bool)
    if ignore is not None:
        ignore = np.asarray(ignore, dtype=bool)
        if ignore.shape != reference.shape:
            raise ValueError("Ignore shape mismatch")
        valid = ~ignore
    p_values, p_inverse = np.unique(prediction[valid], return_inverse=True)
    r_values, r_inverse = np.unique(reference[valid], return_inverse=True)
    if len(p_values) and len(r_values):
        counts = np.bincount(p_inverse * len(r_values) + r_inverse,
                             minlength=len(p_values) * len(r_values)).reshape(len(p_values), len(r_values))
    else:
        counts = np.zeros((len(p_values), len(r_values)), dtype=np.int64)
    p_nonzero = np.flatnonzero(p_values != 0)
    r_nonzero = np.flatnonzero(r_values != 0)
    p_ids = [int(value) for value in p_values[p_nonzero]]
    r_ids = [int(value) for value in r_values[r_nonzero]]
    p_areas = counts.sum(axis=1)[p_nonzero]
    r_areas = counts.sum(axis=0)[r_nonzero]
    intersections = counts[np.ix_(p_nonzero, r_nonzero)]
    unions = p_areas[:, None] + r_areas[None, :] - intersections
    ious = np.divide(intersections, unions, out=np.zeros(intersections.shape, dtype=float), where=unions > 0)
    assignment = []
    if ious.size:
        rows, cols = linear_sum_assignment(-ious)
        for p, r in zip(rows, cols):
            assignment.append({"predicted_id": p_ids[int(p)], "reference_id": r_ids[int(r)],
                               "intersection": int(intersections[p, r]), "union": int(unions[p, r])})
    return {
        "predicted_areas": {str(key): int(area) for key, area in zip(p_ids, p_areas)},
        "reference_areas": {str(key): int(area) for key, area in zip(r_ids, r_areas)},
        "positive_intersections": [{"predicted_id": p_ids[int(p)], "reference_id": r_ids[int(r)], "pixels": int(intersections[p, r])}
                                   for p, r in zip(*np.nonzero(intersections))],
        "hungarian_assignment": assignment,
        "valid_pixels": int(valid.sum()),
        "ignored_pixels": int((~valid).sum()),
        "scope": "original-annotation positive reference; unmatched predictions not certified false positives",
    }


def score_contingency(table, threshold=0.5):
    theta = Fraction(str(threshold))
    if not 0 < theta <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    predicted = {int(key): int(value) for key, value in table["predicted_areas"].items()}
    reference = {int(key): int(value) for key, value in table["reference_areas"].items()}
    matches = [row for row in table["hungarian_assignment"]
               if row["union"] > 0 and row["intersection"] * theta.denominator >= row["union"] * theta.numerator]
    by_reference = {row["reference_id"]: row for row in matches}
    predicted_used = {row["predicted_id"] for row in matches}
    matched_area_abs = 0
    missed_area = 0
    errors = []
    rows = []
    for ref_id, area in sorted(reference.items()):
        match = by_reference.get(ref_id)
        if match is None:
            missed_area += area
            rows.append({"reference_id": ref_id, "reference_area_pixels": area, "recovered": False,
                         "predicted_id": None, "predicted_area_pixels": None, "absolute_area_error_pixels": None,
                         "relative_area_error": None, "intersection": None, "union": None})
        else:
            pred_area = predicted[match["predicted_id"]]
            absolute = abs(pred_area - area)
            relative = absolute / area
            matched_area_abs += absolute
            errors.append(relative)
            rows.append({"reference_id": ref_id, "reference_area_pixels": area, "recovered": True,
                         "predicted_id": match["predicted_id"], "predicted_area_pixels": pred_area,
                         "absolute_area_error_pixels": absolute, "relative_area_error": relative,
                         "intersection": match["intersection"], "union": match["union"]})
    total = sum(reference.values())
    tp = len(matches)
    union_reference = total
    union_prediction = sum(predicted.values())
    return {
        "matching": "maximum_total_IoU_Hungarian_then_inclusive_threshold",
        "iou_threshold": float(theta),
        "reference_count": len(reference), "predicted_count": len(predicted), "recovered_reference_count": tp,
        "unrecovered_reference_count": len(reference) - tp,
        "unmatched_predictions_not_certified_false_positive": len(predicted) - len(predicted_used),
        "positive_reference_recovery_fraction": tp / len(reference) if reference else None,
        "reference_area_pixels": total, "matched_absolute_area_error_pixels": matched_area_abs,
        "missed_reference_area_pixels": missed_area,
        "positive_reference_area_residual_fraction": (matched_area_abs + missed_area) / total if total else None,
        "missed_reference_area_fraction": missed_area / total if total else None,
        "conditional_mean_relative_area_error": sum(errors) / len(errors) if errors else None,
        "conditional_median_relative_area_error": float(np.median(errors)) if errors else None,
        "conditional_error_denominator": len(errors),
        "annotation_relative_total_area_difference_fraction": abs(union_prediction - union_reference) / total if total else None,
        "predicted_total_area_pixels": union_prediction,
        "objects": rows,
        "full_image_FP_PQ_count_accuracy_claimed": False,
        "no_reference_status": "no_positive_reference" if not reference else "positive_reference_present",
    }


def score_prediction(prediction, reference, ignore=None, threshold=0.5):
    table = contingency(prediction, reference, ignore)
    return {"contingency": table, "endpoints": score_contingency(table, threshold)}
