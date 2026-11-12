from __future__ import annotations

import numpy as np

from ricecells.instance_metrics import binary_metrics, instance_metrics


def test_empty_binary_case_is_not_perfect():
    result = binary_metrics(np.zeros((4, 4), bool), np.zeros((4, 4), bool))
    assert result["status"] == "empty_both"
    assert result["f1"] is None
    assert result["target_positive"] == 0


def test_hungarian_matches_instances_and_reports_pq():
    reference = np.zeros((8, 8), np.uint16)
    reference[1:4, 1:4] = 1
    reference[4:7, 4:7] = 2
    predicted = np.zeros_like(reference)
    predicted[1:4, 1:4] = 10
    predicted[4:7, 4:7] = 20
    result = instance_metrics(predicted, reference)
    assert result["true_positive"] == 2
    assert result["false_positive"] == 0
    assert result["false_negative"] == 0
    assert result["f1"] == 1.0
    assert result["pq"] == 1.0
    assert result["matches"] == [
        {"predicted_id": 10, "reference_id": 1, "iou": 1.0},
        {"predicted_id": 20, "reference_id": 2, "iou": 1.0},
    ]


def test_merge_and_split_diagnostics_are_not_truth_labels():
    reference = np.zeros((8, 8), np.uint16)
    reference[1:4, 1:4] = 1
    reference[4:7, 4:7] = 2
    predicted = np.zeros_like(reference)
    predicted[1:7, 1:7] = 10
    result = instance_metrics(predicted, reference, event_iou_threshold=0.1)
    assert result["true_positive"] == 0
    assert result["merge_event_count"] == 1
    assert result["pq"] == 0.0


def test_ignore_pixels_are_excluded_from_matching():
    reference = np.zeros((5, 5), np.uint16)
    predicted = np.zeros_like(reference)
    reference[1:4, 1:4] = 1
    predicted[1:4, 1:4] = 2
    ignore = np.zeros_like(reference, bool)
    ignore[1:4, 1:4] = True
    result = instance_metrics(predicted, reference, ignore=ignore)
    assert result["status"] == "empty_both"
    assert result["predicted_instances"] == 0
