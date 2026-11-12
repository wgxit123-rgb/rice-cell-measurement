"""Fixed positive-reference aggregation; no array access or model selection.

``None`` means the mathematical denominator is empty. Three optimization seeds
are described by their mean and sample standard deviation, never biological n.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean, median, stdev
from typing import Any, Iterable

THRESHOLDS = (0.25, 0.5, 0.75)
ROLES = ("primary_confirmation", "secondary_confirmation")
INITIALIZATIONS = ("random", "supervised_imagenet", "dino_self_supervised")
PAIRS = ((INITIALIZATIONS[0], INITIALIZATIONS[1]),
         (INITIALIZATIONS[0], INITIALIZATIONS[2]),
         (INITIALIZATIONS[1], INITIALIZATIONS[2]))
IDENTITY_COLUMNS = ["run_id", "initialization", "budget", "seed", "image_id", "file_name",
                    "is_primary_confirmation", "is_secondary_confirmation", "iou_threshold"]
SCALAR_COLUMNS = [
    "matching", "reference_count", "predicted_count", "recovered_reference_count",
    "unrecovered_reference_count", "unmatched_predictions_not_certified_false_positive",
    "positive_reference_recovery_fraction", "reference_area_pixels",
    "matched_absolute_area_error_pixels", "missed_reference_area_pixels",
    "positive_reference_area_residual_fraction", "missed_reference_area_fraction",
    "conditional_mean_relative_area_error", "conditional_median_relative_area_error",
    "conditional_error_denominator", "annotation_relative_total_area_difference_fraction",
    "predicted_total_area_pixels", "full_image_FP_PQ_count_accuracy_claimed", "no_reference_status",
]
BOUND_METRICS = ["source_record_recovery_lower_bound", "source_record_recovery_upper_bound"]
AGG_METRICS = [
    "positive_reference_recovery_fraction", "positive_reference_area_residual_fraction",
    "missed_reference_area_fraction", "conditional_mean_relative_area_error",
    "conditional_median_relative_area_error", "annotation_relative_total_area_difference_fraction",
    *BOUND_METRICS,
]
FIELD_COLUMNS = IDENTITY_COLUMNS + ["G_total", "G_scorable", "U_unscorable"] + SCALAR_COLUMNS + BOUND_METRICS
OBJECT_COLUMNS = IDENTITY_COLUMNS + ["annotation_id", "scorable", "reason", "reference_area_pixels",
    "recovered", "predicted_id", "predicted_area_pixels", "absolute_area_error_pixels",
    "relative_area_error", "intersection", "union"]
SUM_COLUMNS = ["G_total", "G_scorable", "U_unscorable", "predicted_count", "recovered_reference_count",
    "unrecovered_reference_count", "unmatched_predictions_not_certified_false_positive",
    "reference_area_pixels", "matched_absolute_area_error_pixels", "missed_reference_area_pixels",
    "conditional_error_denominator", "predicted_total_area_pixels"]
RUN_ID_COLUMNS = ["run_id", "initialization", "budget", "seed", "role", "iou_threshold"]
RUN_METRIC_COLUMNS = [name for metric in AGG_METRICS
                      for name in (f"macro_{metric}", f"macro_{metric}_contributing_fields", f"pooled_{metric}")]
RUN_COLUMNS = RUN_ID_COLUMNS + ["field_count"] + SUM_COLUMNS + [
    "annotation_relative_total_area_difference_pixels", "conditional_relative_error_sum"] + RUN_METRIC_COLUMNS
SEED_COLUMNS = ["initialization", "budget", "role", "iou_threshold", "n_seeds", "seed_ids", "interpretation"] + [
    name for column in RUN_METRIC_COLUMNS if not column.endswith("_contributing_fields")
    for name in (f"{column}_mean", f"{column}_sample_sd", f"{column}_contributing_seeds")]
PAIR_VALUE_COLUMNS = [
    "common_pooled_left_mean_relative_area_error", "common_pooled_right_mean_relative_area_error",
    "common_pooled_right_minus_left", "common_macro_left_mean_relative_area_error",
    "common_macro_right_mean_relative_area_error", "common_macro_right_minus_left",
    "left_only_pooled_left_mean_relative_area_error", "right_only_pooled_right_mean_relative_area_error",
    "left_only_macro_left_mean_relative_area_error", "right_only_macro_right_mean_relative_area_error",
]
PAIRED_COLUMNS = ["left_run_id", "right_run_id", "left_initialization", "right_initialization", "budget",
    "seed", "role", "iou_threshold", "field_count", "G_total", "G_scorable", "U_unscorable",
    "common_recovered_count", "left_only_recovered_count", "right_only_recovered_count",
    "neither_recovered_count", "common_contributing_fields", "left_only_contributing_fields",
    "right_only_contributing_fields"] + PAIR_VALUE_COLUMNS


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    """Return a ratio, with no fabricated zero when support is empty."""
    return numerator / denominator if denominator else None


def average(values: Iterable[float | None]) -> float | None:
    """Average only defined values; their contributing count is stored separately."""
    supported = [value for value in values if value is not None]
    return mean(supported) if supported else None


def field_and_objects(record: dict[str, Any], endpoints: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Join immutable source identities to the existing measurement scalars."""
    identity = {key: record[key] for key in IDENTITY_COLUMNS if key != "iou_threshold"}
    identity["iou_threshold"] = endpoints["iou_threshold"]
    sources = record["source_records"]
    by_id = {row["reference_id"]: row for row in endpoints["objects"]}
    scorable_ids = {row["annotation_id"] for row in sources if row["scorable"]}
    if len(sources) != len({row["annotation_id"] for row in sources}) or set(by_id) != scorable_ids:
        raise ValueError("Source identities and scored reference identities differ")
    total, scored = len(sources), len(scorable_ids)
    if endpoints["reference_count"] != scored:
        raise ValueError("Reference count differs from source partition")
    unsupported = total - scored
    field = dict(identity, G_total=total, G_scorable=scored, U_unscorable=unsupported)
    field.update({key: endpoints[key] for key in SCALAR_COLUMNS})
    field[BOUND_METRICS[0]] = ratio(endpoints["recovered_reference_count"], total)
    field[BOUND_METRICS[1]] = ratio(endpoints["recovered_reference_count"] + unsupported, total)
    objects = []
    for source in sorted(sources, key=lambda row: row["annotation_id"]):
        item = dict(identity, annotation_id=source["annotation_id"], scorable=source["scorable"], reason=source["reason"])
        row = by_id.get(source["annotation_id"])
        for key in OBJECT_COLUMNS[len(IDENTITY_COLUMNS) + 3:]:
            item[key] = row[key] if row is not None else None
        objects.append(item)
    return field, objects


def in_role(row: dict[str, Any], role: str) -> bool:
    """Select one of the two nested, preassigned analysis roles."""
    if role not in ROLES:
        raise ValueError("Unknown analysis role")
    return bool(row[f"is_{role}"])


def run_summaries(fields: list[dict[str, Any]], objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report field macro and numerator/denominator pooled summaries together."""
    field_groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    object_groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in fields:
        field_groups[(row["run_id"], row["iou_threshold"])].append(row)
    for row in objects:
        object_groups[(row["run_id"], row["iou_threshold"])].append(row)
    output = []
    for (run_id, theta), all_fields in sorted(field_groups.items()):
        for role in ROLES:
            selected = [row for row in all_fields if in_role(row, role)]
            if not selected:
                raise ValueError("A preassigned role has no retained fields")
            selected_objects = [row for row in object_groups[(run_id, theta)] if in_role(row, role)]
            errors = [row["relative_area_error"] for row in selected_objects if row["scorable"] and row["recovered"]]
            item = {key: selected[0][key] for key in RUN_ID_COLUMNS if key != "role"}
            item.update(role=role, field_count=len(selected))
            item.update({key: sum(row[key] for row in selected) for key in SUM_COLUMNS})
            item["annotation_relative_total_area_difference_pixels"] = sum(
                abs(row["predicted_total_area_pixels"] - row["reference_area_pixels"]) for row in selected)
            item["conditional_relative_error_sum"] = sum(errors)
            if len(errors) != item["conditional_error_denominator"]:
                raise ValueError("Conditional denominator/object rows disagree")
            pooled = {
                "positive_reference_recovery_fraction": ratio(item["recovered_reference_count"], item["G_scorable"]),
                "positive_reference_area_residual_fraction": ratio(item["matched_absolute_area_error_pixels"] + item["missed_reference_area_pixels"], item["reference_area_pixels"]),
                "missed_reference_area_fraction": ratio(item["missed_reference_area_pixels"], item["reference_area_pixels"]),
                "conditional_mean_relative_area_error": average(errors),
                "conditional_median_relative_area_error": median(errors) if errors else None,
                "annotation_relative_total_area_difference_fraction": ratio(item["annotation_relative_total_area_difference_pixels"], item["reference_area_pixels"]),
                BOUND_METRICS[0]: ratio(item["recovered_reference_count"], item["G_total"]),
                BOUND_METRICS[1]: ratio(item["recovered_reference_count"] + item["U_unscorable"], item["G_total"]),
            }
            for metric in AGG_METRICS:
                item[f"macro_{metric}"] = average(row[metric] for row in selected)
                item[f"macro_{metric}_contributing_fields"] = sum(row[metric] is not None for row in selected)
                item[f"pooled_{metric}"] = pooled[metric]
            output.append(item)
    return output


def seed_summaries(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe all three optimization seeds, without inference or selection."""
    groups: dict[tuple[str, int, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        groups[(row["initialization"], row["budget"], row["role"], row["iou_threshold"])].append(row)
    output = []
    for (initialization, budget, role, theta), rows in sorted(groups.items()):
        seeds = sorted(row["seed"] for row in rows)
        if seeds != [20261026, 20261027, 20261028]:
            raise ValueError("Every seed summary must retain all three fixed optimization seeds")
        item = dict(initialization=initialization, budget=budget, role=role, iou_threshold=theta,
                    n_seeds=3, seed_ids="|".join(map(str, seeds)),
                    interpretation="optimization_repetitions_only_not_biological_n_or_significance")
        for column in RUN_METRIC_COLUMNS:
            if column.endswith("_contributing_fields"):
                continue
            values = [row[column] for row in rows if row[column] is not None]
            item[f"{column}_mean"] = mean(values) if values else None
            item[f"{column}_sample_sd"] = stdev(values) if len(values) > 1 else None
            item[f"{column}_contributing_seeds"] = len(values)
        output.append(item)
    return output


def paired_comparisons(objects: list[dict[str, Any]], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare common identities and retain left-only/right-only/neither sets."""
    by_run: dict[str, dict[tuple[int, int], dict[str, Any]]] = defaultdict(dict)
    for row in objects:
        if row["iou_threshold"] != 0.5:
            continue
        identity = (row["image_id"], row["annotation_id"])
        if identity in by_run[row["run_id"]]:
            raise ValueError("Duplicate reference identity in pair input")
        by_run[row["run_id"]][identity] = row
    lookup = {(run["initialization"], run["budget"], run["seed"]): run["id"] for run in runs}
    output = []
    for left_init, right_init in PAIRS:
        for budget in (4, 16):
            for seed in (20261026, 20261027, 20261028):
                left_id, right_id = lookup[(left_init, budget, seed)], lookup[(right_init, budget, seed)]
                left, right = by_run[left_id], by_run[right_id]
                if set(left) != set(right):
                    raise ValueError("Pair conditions do not retain identical source identities")
                for role in ROLES:
                    identities = [key for key in left if in_role(left[key], role)]
                    item = dict(left_run_id=left_id, right_run_id=right_id, left_initialization=left_init,
                                right_initialization=right_init, budget=budget, seed=seed, role=role,
                                iou_threshold=0.5, field_count=len({key[0] for key in identities}),
                                G_total=len(identities), G_scorable=0, U_unscorable=0)
                    categories: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
                        "common": [], "left_only": [], "right_only": [], "neither": []}
                    for identity in identities:
                        lrow, rrow = left[identity], right[identity]
                        if lrow["scorable"] != rrow["scorable"] or lrow["reference_area_pixels"] != rrow["reference_area_pixels"]:
                            raise ValueError("Pair reference support differs")
                        if not lrow["scorable"]:
                            item["U_unscorable"] += 1
                            continue
                        item["G_scorable"] += 1
                        category = ("common" if lrow["recovered"] and rrow["recovered"] else
                                    "left_only" if lrow["recovered"] else "right_only" if rrow["recovered"] else "neither")
                        categories[category].append((lrow, rrow))
                    for category in categories:
                        item[f"{category}_recovered_count"] = len(categories[category])
                    # The neither category is a support count, not an error assigned zero.
                    for category in ("common", "left_only", "right_only"):
                        item[f"{category}_contributing_fields"] = len({pair[0]["image_id"] for pair in categories[category]})
                    for category, side, index in (("common", "left", 0), ("common", "right", 1),
                                                  ("left_only", "left", 0), ("right_only", "right", 1)):
                        pairs = categories[category]
                        item[f"{category}_pooled_{side}_mean_relative_area_error"] = average(pair[index]["relative_area_error"] for pair in pairs)
                        by_field: dict[int, list[float]] = defaultdict(list)
                        for pair in pairs:
                            by_field[pair[index]["image_id"]].append(pair[index]["relative_area_error"])
                        item[f"{category}_macro_{side}_mean_relative_area_error"] = average(mean(values) for values in by_field.values())
                    for aggregation in ("pooled", "macro"):
                        left_value = item[f"common_{aggregation}_left_mean_relative_area_error"]
                        right_value = item[f"common_{aggregation}_right_mean_relative_area_error"]
                        item[f"common_{aggregation}_right_minus_left"] = right_value - left_value if left_value is not None and right_value is not None else None
                    output.append(item)
    return output
