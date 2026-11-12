import pytest

from ricecells.measurement_fidelity import analyze_objects, mean_with_coverage, quantiles


def obj(kind, rid, pid, ra, pa, truncated=False):
    return {'kind': kind, 'reference_id': rid, 'predicted_id': pid,
            'reference_area_pixels': ra, 'predicted_area_pixels': pa,
            'error_area_pixels': abs(pa-ra), 'truncated_reference': truncated if rid else None,
            'retained_sensitivity': not truncated if rid else True}


def test_perfect_objects():
    out = analyze_objects([obj('matched', 1, 2, 100, 100)])['metrics']
    assert out['signed_count_error'] == 0
    assert out['residual_area_error_fraction'] == 0
    assert out['reference_area_coverage'] == 1


def test_total_area_cancellation_does_not_erase_instance_residual():
    out = analyze_objects([obj('matched', 1, 1, 100, 120), obj('matched', 2, 2, 100, 80)])['metrics']
    assert out['absolute_total_area_error_fraction'] == 0
    assert out['residual_area_error_fraction'] == .2
    assert out['absolute_total_vs_instance_residual_gap_fraction'] == .2
    assert out['matched_mean_signed_relative_area_error'] == 0


def test_equal_area_merge_bookkeeping_still_counts_missing_reference():
    out = analyze_objects([obj('matched', 1, 1, 100, 200), obj('missed_reference', 2, None, 100, 0)])['metrics']
    assert out['signed_count_error'] == -1
    assert out['signed_total_area_error_fraction'] == 0
    assert out['residual_area_error_fraction'] == 1
    assert out['reference_object_coverage'] == .5


def test_unmatched_objects_never_become_zero_area_samples():
    result = analyze_objects([obj('missed_reference', 1, None, 100, 0), obj('false_prediction', None, 2, 0, 150)])
    assert result['area_distributions_px2']['reference_all']['n'] == 1
    assert result['area_distributions_px2']['predicted_all']['n'] == 1
    assert result['metrics']['matched_median_absolute_relative_area_error'] is None
    assert result['metrics']['reference_area_coverage'] == 0


def test_empty_both_is_undefined_not_perfect():
    out = analyze_objects([])['metrics']
    assert out['reference_object_coverage'] is None
    assert out['residual_area_error_fraction'] is None
    assert out['signed_count_error'] == 0


def test_empty_reference_but_prediction_retained():
    out = analyze_objects([obj('false_prediction', None, 1, 0, 50)])['metrics']
    assert out['signed_count_error'] == 1
    assert out['residual_area_error_fraction'] is None
    assert out['prediction_object_coverage'] == 0


def test_exclusion_removes_matched_partner_but_keeps_all_false_predictions():
    rows = [obj('matched', 1, 10, 100, 100, True), obj('missed_reference', 2, None, 70, 0, True),
            obj('false_prediction', None, 20, 0, 50), obj('matched', 3, 30, 200, 200)]
    out = analyze_objects(rows, 'exclude_truncated_pairs')['metrics']
    assert out['reference_instances'] == 1
    assert out['predicted_instances'] == 2
    assert out['unmatched_prediction_instances'] == 1


@pytest.mark.parametrize('bad', [-1, float('nan'), float('inf'), .2])
def test_invalid_area_rejected(bad):
    with pytest.raises(ValueError):
        analyze_objects([obj('matched', 1, 1, bad, 50)])


def test_duplicate_id_rejected():
    with pytest.raises(ValueError):
        analyze_objects([obj('matched', 1, 1, 50, 50), obj('matched', 1, 2, 50, 50)])


def test_changed_truncation_rejected():
    row = obj('matched', 1, 1, 50, 50, True)
    row['retained_sensitivity'] = True
    with pytest.raises(ValueError):
        analyze_objects([row])


def test_quantiles_are_linear_and_empty_coverage_visible():
    assert quantiles([10, 30])['quantiles'] == [12, 15, 20, 25, 28]
    assert quantiles([])['quantiles'] == [None]*5
    assert mean_with_coverage([1., None, 3.]) == {'mean': 2., 'defined': 2, 'expected': 3}


def test_incorrect_archived_error_rejected():
    row = obj('matched', 1, 1, 50, 70)
    row['error_area_pixels'] = 0
    with pytest.raises(ValueError):
        analyze_objects([row])
