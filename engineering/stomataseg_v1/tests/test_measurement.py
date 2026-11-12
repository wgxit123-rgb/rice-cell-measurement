import numpy as np
import pytest

from engineering.stomataseg_v1.measurement import contingency, score_contingency, score_prediction


def test_empty_is_not_perfect():
    x = np.zeros((4, 4), dtype=np.uint16)
    row = score_prediction(x, x)["endpoints"]
    assert row["positive_reference_recovery_fraction"] is None
    assert row["conditional_mean_relative_area_error"] is None
    assert row["positive_reference_area_residual_fraction"] is None


def test_missing_all_is_unit_residual_not_zero_error():
    gt = np.ones((4, 4), dtype=np.uint16)
    row = score_prediction(np.zeros_like(gt), gt)["endpoints"]
    assert row["positive_reference_recovery_fraction"] == 0
    assert row["positive_reference_area_residual_fraction"] == 1
    assert row["conditional_mean_relative_area_error"] is None


def test_perfect_subset_has_zero_conditional_error_but_missing_mass():
    gt = np.zeros((4, 8), dtype=np.uint16)
    gt[:, :3], gt[:, 5:] = 4, 9
    p = np.zeros_like(gt)
    p[:, :3] = 66
    row = score_prediction(p, gt)["endpoints"]
    assert row["positive_reference_recovery_fraction"] == 0.5
    assert row["conditional_mean_relative_area_error"] == 0
    assert row["positive_reference_area_residual_fraction"] == 0.5


def test_extra_predictions_not_automatically_false_positive():
    gt = np.zeros((4, 8), dtype=np.uint16)
    gt[:, :3] = 1
    p = gt.copy()
    p[:, 5:] = 2
    row = score_prediction(p, gt)["endpoints"]
    assert row["positive_reference_recovery_fraction"] == 1
    assert row["unmatched_predictions_not_certified_false_positive"] == 1
    assert row["positive_reference_area_residual_fraction"] == 0
    assert row["annotation_relative_total_area_difference_fraction"] == 1
    assert "false_positive" not in row


def test_inclusive_half_iou_boundary():
    gt = np.ones((2, 2), dtype=np.uint16)
    p = np.zeros_like(gt)
    p[0] = 1
    table = contingency(p, gt)
    assert score_contingency(table, 0.5)["recovered_reference_count"] == 1
    assert score_contingency(table, 0.75)["recovered_reference_count"] == 0


def test_hungarian_precedes_threshold():
    table = {"predicted_areas": {"1": 10, "2": 10}, "reference_areas": {"8": 10, "9": 10},
             "hungarian_assignment": [{"predicted_id": 1, "reference_id": 8, "intersection": 6, "union": 14},
                                       {"predicted_id": 2, "reference_id": 9, "intersection": 9, "union": 11}]}
    row = score_contingency(table, 0.5)
    assert row["recovered_reference_count"] == 1
    assert row["objects"][0]["recovered"] is False


def test_ignore_applies_to_both_but_not_all_unlabelled():
    gt = np.ones((2, 4), dtype=np.uint16)
    p = np.ones_like(gt)
    p[:, 2:] = 2
    ignore = np.zeros_like(gt, dtype=bool)
    ignore[:, 2:] = True
    row = score_prediction(p, gt, ignore)["endpoints"]
    assert row["reference_area_pixels"] == 4
    assert row["predicted_count"] == 1
    assert row["positive_reference_area_residual_fraction"] == 0


def test_overlarge_prediction_penalized_on_unlabelled_background():
    gt = np.zeros((4, 8), dtype=np.uint16)
    gt[:, :2] = 1
    p = np.ones_like(gt)
    row = score_prediction(p, gt)["endpoints"]
    assert row["recovered_reference_count"] == 0
    assert row["positive_reference_area_residual_fraction"] == 1


def test_sparse_large_ids_and_integer_intersection():
    gt = np.zeros((4, 8), dtype=np.uint64)
    gt[:, :2] = 3173368342
    p = gt.copy()
    table = contingency(p, gt)
    assert table["positive_intersections"][0]["pixels"] == 8
    assert table["reference_areas"] == {"3173368342": 8}
    assert score_contingency(table)["positive_reference_area_residual_fraction"] == 0


@pytest.mark.parametrize("bad", [np.ones((3, 3), dtype=np.float32), -np.ones((3, 3), dtype=np.int16)])
def test_invalid_labels_rejected(bad):
    with pytest.raises(ValueError):
        contingency(bad, np.ones((3, 3), dtype=np.int16))


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        contingency(np.zeros((3, 3), dtype=int), np.zeros((2, 2), dtype=int))


def test_all_ignored_is_undefined():
    x = np.ones((3, 3), dtype=np.uint16)
    row = score_prediction(x, x, np.ones_like(x, dtype=bool))["endpoints"]
    assert row["reference_count"] == 0
    assert row["positive_reference_recovery_fraction"] is None


def test_decomposition_and_three_threshold_monotonicity():
    gt = np.zeros((4, 12), dtype=np.uint16)
    gt[:, :4], gt[:, 8:] = 2, 7
    p = np.zeros_like(gt)
    p[:, :3] = 8
    table = contingency(p, gt)
    rows = [score_contingency(table, t) for t in (0.25, 0.5, 0.75)]
    for row in rows:
        assert row["positive_reference_area_residual_fraction"] == (row["matched_absolute_area_error_pixels"] + row["missed_reference_area_pixels"]) / row["reference_area_pixels"]
    assert rows[0]["recovered_reference_count"] >= rows[1]["recovered_reference_count"] >= rows[2]["recovered_reference_count"]
