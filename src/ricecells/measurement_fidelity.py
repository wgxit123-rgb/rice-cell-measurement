"""Descriptive count/area fidelity from frozen per-object assignments.

No rematching, no new reference labels and no assumption of biological
independence. Empty conditional subsets retain explicit undefined values.
"""
import math
import statistics

import numpy as np


def number(value):
    value = float(value)
    if not math.isfinite(value) or value < 0 or not value.is_integer():
        raise ValueError('Pixel areas must be finite nonnegative integers')
    return value


def truth(value):
    if value in (True, 'True', 'true'):
        return True
    if value in (False, 'False', 'false'):
        return False
    raise ValueError('Unknown Boolean flag')


def quantiles(values):
    return {'n': len(values), 'quantiles': [float(v) for v in np.quantile(values, [.1, .25, .5, .75, .9], method='linear')]
            if values else [None] * 5}


def validate_objects(rows):
    pids, rids = set(), set()
    for row in rows:
        kind = row['kind']
        if kind not in ('matched', 'missed_reference', 'false_prediction'):
            raise ValueError('Unknown assignment kind')
        p, r = number(row['predicted_area_pixels']), number(row['reference_area_pixels'])
        for key, ids, area in (('predicted_id', pids, p), ('reference_id', rids, r)):
            value = row[key]
            if value not in (None, ''):
                value = int(value)
                if value <= 0 or value in ids or area <= 0:
                    raise ValueError('Invalid or duplicated object ID/area')
                ids.add(value)
            elif area != 0:
                raise ValueError('Absent object must have zero bookkeeping area')
        if ((kind == 'matched' and (p <= 0 or r <= 0))
                or (kind == 'missed_reference' and (r <= 0 or p != 0))
                or (kind == 'false_prediction' and (p <= 0 or r != 0))):
            raise ValueError('Assignment kind conflicts with object areas')
        expected_retained = True if kind == 'false_prediction' else not truth(row['truncated_reference'])
        if truth(row['retained_sensitivity']) != expected_retained:
            raise ValueError('Changed paired-truncation policy')
        if not math.isclose(float(row['error_area_pixels']), abs(p-r), abs_tol=1e-10):
            raise ValueError('Archived object error does not reproduce')


def analyze_objects(rows, scope='all_cells'):
    validate_objects(rows)
    if scope not in ('all_cells', 'exclude_truncated_pairs'):
        raise ValueError('Unknown scope')
    selected = rows if scope == 'all_cells' else [r for r in rows if truth(r['retained_sensitivity'])]
    matched = [r for r in selected if r['kind'] == 'matched']
    refs = [number(r['reference_area_pixels']) for r in selected if r['kind'] != 'false_prediction']
    preds = [number(r['predicted_area_pixels']) for r in selected if r['kind'] != 'missed_reference']
    mr = [number(r['reference_area_pixels']) for r in matched]
    mp = [number(r['predicted_area_pixels']) for r in matched]
    ref_area, pred_area = sum(refs), sum(preds)
    delta = pred_area - ref_area
    residual = sum(abs(number(r['predicted_area_pixels'])-number(r['reference_area_pixels'])) for r in selected)
    matched_relative = [(p-r)/r for p, r in zip(mp, mr)]
    ratio = lambda a, b: a/b if b else None
    metrics = {
        'reference_instances': len(refs), 'predicted_instances': len(preds), 'matched_instances': len(matched),
        'unmatched_reference_instances': len(refs)-len(matched), 'unmatched_prediction_instances': len(preds)-len(matched),
        'signed_count_error': len(preds)-len(refs), 'absolute_count_error': abs(len(preds)-len(refs)),
        'relative_signed_count_error': ratio(len(preds)-len(refs), len(refs)),
        'reference_total_area_pixels': ref_area, 'predicted_total_area_pixels': pred_area,
        'signed_total_area_error_fraction': ratio(delta, ref_area),
        'absolute_total_area_error_fraction': ratio(abs(delta), ref_area),
        'residual_area_error_fraction': ratio(residual, ref_area),
        'reference_object_coverage': ratio(len(matched), len(refs)),
        'prediction_object_coverage': ratio(len(matched), len(preds)),
        'reference_area_coverage': ratio(sum(mr), ref_area),
        'matched_mean_signed_relative_area_error': statistics.mean(matched_relative) if matched_relative else None,
        'matched_median_absolute_relative_area_error': statistics.median(map(abs, matched_relative)) if matched_relative else None,
        'predicted_to_reference_median_area_ratio': ratio(statistics.median(preds), statistics.median(refs)) if preds and refs else None,
        'absolute_total_vs_instance_residual_gap_fraction': ratio(residual-abs(delta), ref_area),
    }
    if residual + 1e-9 < abs(delta):
        raise ValueError('Triangle inequality violated by object accounting')
    return {'scope': scope, 'metrics': metrics,
            'area_distributions_px2': {'reference_all': quantiles(refs), 'predicted_all': quantiles(preds),
                                       'reference_matched': quantiles(mr), 'predicted_matched': quantiles(mp)},
            'quantile_probabilities': [.1, .25, .5, .75, .9]}


def mean_with_coverage(values):
    defined = [v for v in values if v is not None]
    if any(not math.isfinite(float(v)) for v in defined):
        raise ValueError('Nonfinite summary input')
    return {'mean': statistics.mean(defined) if defined else None, 'defined': len(defined), 'expected': len(values)}
