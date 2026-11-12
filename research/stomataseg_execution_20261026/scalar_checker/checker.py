"""Independent integer/Fraction replay of saved StomataSeg scalars.

Only the saved scalar tables and integer contingencies are read. The saved
Hungarian assignment is a shared primitive, not independently optimized here.
No image, label, prediction, weight, production module or third-party library
is opened or imported. See README.md for the deliberately limited claim.
"""
import argparse
import csv
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys


THRESHOLDS = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
MATCHING = "maximum_total_IoU_Hungarian_then_inclusive_threshold"
SCOPE = "original-annotation positive reference; unmatched predictions not certified false positives"
ABS_TOL = Fraction(1, 10**12)
REL_TOL = Fraction(1, 10**12)
METRICS = (
    "matching", "reference_count", "predicted_count", "recovered_reference_count",
    "unrecovered_reference_count", "unmatched_predictions_not_certified_false_positive",
    "positive_reference_recovery_fraction", "reference_area_pixels",
    "matched_absolute_area_error_pixels", "missed_reference_area_pixels",
    "positive_reference_area_residual_fraction", "missed_reference_area_fraction",
    "conditional_mean_relative_area_error", "conditional_median_relative_area_error",
    "conditional_error_denominator", "annotation_relative_total_area_difference_fraction",
    "predicted_total_area_pixels", "full_image_FP_PQ_count_accuracy_claimed", "no_reference_status",
)
IDENTITY = ("run_id", "initialization", "budget", "seed", "image_id", "file_name",
            "is_primary_confirmation", "is_secondary_confirmation", "iou_threshold")
FIELD_COLUMNS = IDENTITY + ("G_total", "G_scorable", "U_unscorable") + METRICS + (
    "source_record_recovery_lower_bound", "source_record_recovery_upper_bound")
OBJECT_COLUMNS = IDENTITY + ("annotation_id", "scorable", "reason", "reference_area_pixels",
    "recovered", "predicted_id", "predicted_area_pixels", "absolute_area_error_pixels",
    "relative_area_error", "intersection", "union")
ROLES = ("primary_confirmation", "secondary_confirmation")
INITIALIZATIONS = ("random", "supervised_imagenet", "dino_self_supervised")
PAIRS = tuple(itertools.combinations(INITIALIZATIONS, 2))
AGG_METRICS = ("positive_reference_recovery_fraction", "positive_reference_area_residual_fraction",
    "missed_reference_area_fraction", "conditional_mean_relative_area_error",
    "conditional_median_relative_area_error", "annotation_relative_total_area_difference_fraction",
    "source_record_recovery_lower_bound", "source_record_recovery_upper_bound")
SUM_COLUMNS = ("G_total", "G_scorable", "U_unscorable", "predicted_count", "recovered_reference_count",
    "unrecovered_reference_count", "unmatched_predictions_not_certified_false_positive",
    "reference_area_pixels", "matched_absolute_area_error_pixels", "missed_reference_area_pixels",
    "conditional_error_denominator", "predicted_total_area_pixels")
RUN_ID_COLUMNS = ("run_id", "initialization", "budget", "seed", "role", "iou_threshold")
RUN_METRIC_COLUMNS = tuple(name for metric in AGG_METRICS for name in (
    "macro_" + metric, "macro_" + metric + "_contributing_fields", "pooled_" + metric))
RUN_COLUMNS = RUN_ID_COLUMNS + ("field_count",) + SUM_COLUMNS + (
    "annotation_relative_total_area_difference_pixels", "conditional_relative_error_sum") + RUN_METRIC_COLUMNS
SEED_COLUMNS = ("initialization", "budget", "role", "iou_threshold", "n_seeds", "seed_ids", "interpretation") + tuple(
    name for column in RUN_METRIC_COLUMNS if not column.endswith("_contributing_fields")
    for name in (column + "_mean", column + "_sample_sd", column + "_contributing_seeds"))
PAIR_VALUE_COLUMNS = ("common_pooled_left_mean_relative_area_error", "common_pooled_right_mean_relative_area_error",
    "common_pooled_right_minus_left", "common_macro_left_mean_relative_area_error",
    "common_macro_right_mean_relative_area_error", "common_macro_right_minus_left",
    "left_only_pooled_left_mean_relative_area_error", "right_only_pooled_right_mean_relative_area_error",
    "left_only_macro_left_mean_relative_area_error", "right_only_macro_right_mean_relative_area_error")
PAIRED_COLUMNS = ("left_run_id", "right_run_id", "left_initialization", "right_initialization", "budget",
    "seed", "role", "iou_threshold", "field_count", "G_total", "G_scorable", "U_unscorable",
    "common_recovered_count", "left_only_recovered_count", "right_only_recovered_count",
    "neither_recovered_count", "common_contributing_fields", "left_only_contributing_fields",
    "right_only_contributing_fields") + PAIR_VALUE_COLUMNS
ARTIFACT_NAMES = ("contingencies.jsonl", "field_endpoints.csv", "object_endpoints.csv", "run_summary.csv",
    "seed_summary.csv", "paired_identity_comparisons.csv", "CONFIG_V2.json", "TRAINING_FREEZE_V2.json",
    "EVALUATION_FREEZE_V2.json", "DATA_ROLES_FROM_FREEZE.json", "prepared_test_manifest.json")
EXPECTED_ROW_COUNTS = {"contingencies": 594, "field_endpoints": 1782, "object_endpoints": 80136,
    "run_summary": 108, "seed_summary": 36, "paired_identity_comparisons": 36}
DENOMINATORS = {"recovery": "G_scorable", "positive_reference_area_residual": "sum_scorable_reference_area",
    "conditional_errors": "recovered_scorable_reference_records", "support_bounds": "G_total",
    "macro": "nonempty_per_field_values_with_contributing_fields",
    "pooled_annotation_total_area_difference": "sum_field_abs(Parea-Garea)/sum_field_Garea",
    "paired": "identical_(image_id,annotation_id)_scorable_source_records"}


class CheckError(ValueError):
    """Input fails the frozen scalar contract."""


def require(condition, message):
    if not condition:
        raise CheckError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, name + " must be an integer >= " + str(minimum))
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + str(key))
        result[key] = value
    return result


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle, object_pairs_hook=unique_object,
                         parse_constant=lambda value: (_ for _ in ()).throw(CheckError("nonfinite JSON: " + value)))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def median(values):
    ordered = sorted(values)
    size = len(ordered)
    if not size:
        return None
    return ordered[size // 2] if size % 2 else (ordered[size // 2 - 1] + ordered[size // 2]) / 2


def mean(values):
    return sum(values, Fraction()) / len(values) if values else None


def divide(numerator, denominator):
    return Fraction(numerator, denominator) if denominator else None


def parse_areas(value, name):
    require(type(value) is dict, name + " must be an object")
    output = {}
    for key, area in value.items():
        require(type(key) is str and re.fullmatch(r"[1-9][0-9]*", key) is not None,
                name + " keys must be canonical positive integer strings")
        output[int(key)] = integer(area, name + "/" + key, 1)
    return output


def validate_contingency(table):
    require(set(table) == {"predicted_areas", "reference_areas", "positive_intersections",
        "hungarian_assignment", "valid_pixels", "ignored_pixels", "scope"}, "contingency fields differ")
    predicted = parse_areas(table["predicted_areas"], "predicted_areas")
    reference = parse_areas(table["reference_areas"], "reference_areas")
    valid = integer(table["valid_pixels"], "valid_pixels")
    integer(table["ignored_pixels"], "ignored_pixels")
    require(table["scope"] == SCOPE, "contingency scientific scope differs")
    require(sum(predicted.values()) <= valid and sum(reference.values()) <= valid,
            "positive area exceeds valid pixels")
    require(type(table["positive_intersections"]) is list, "positive_intersections must be a list")
    intersections = {}
    p_sums = {key: 0 for key in predicted}
    r_sums = {key: 0 for key in reference}
    for row in table["positive_intersections"]:
        require(set(row) == {"predicted_id", "reference_id", "pixels"}, "intersection fields differ")
        p = integer(row["predicted_id"], "intersection predicted_id", 1)
        r = integer(row["reference_id"], "intersection reference_id", 1)
        pixels = integer(row["pixels"], "intersection pixels", 1)
        require(p in predicted and r in reference, "intersection names an absent object")
        require((p, r) not in intersections, "duplicate positive intersection")
        require(pixels <= min(predicted[p], reference[r]), "intersection exceeds object area")
        intersections[p, r] = pixels
        p_sums[p] += pixels
        r_sums[r] += pixels
    require(all(p_sums[p] <= predicted[p] for p in predicted), "row intersections exceed predicted area")
    require(all(r_sums[r] <= reference[r] for r in reference), "column intersections exceed reference area")
    require(type(table["hungarian_assignment"]) is list, "assignment must be a list")
    assignment = []
    seen_p, seen_r = set(), set()
    for row in table["hungarian_assignment"]:
        require(set(row) == {"predicted_id", "reference_id", "intersection", "union"}, "assignment fields differ")
        p = integer(row["predicted_id"], "assignment predicted_id", 1)
        r = integer(row["reference_id"], "assignment reference_id", 1)
        intersection = integer(row["intersection"], "assignment intersection")
        union = integer(row["union"], "assignment union", 1)
        require(p in predicted and r in reference, "assignment names an absent object")
        require(p not in seen_p and r not in seen_r, "assignment is not one-to-one")
        require(intersection == intersections.get((p, r), 0), "assignment intersection differs from positive table")
        require(union == predicted[p] + reference[r] - intersection, "assignment union differs from areas")
        seen_p.add(p)
        seen_r.add(r)
        assignment.append((p, r, intersection, union))
    require(len(assignment) == min(len(predicted), len(reference)), "saved assignment cardinality differs")
    return predicted, reference, intersections, assignment


def reconstruct(table, threshold):
    """Rebuild endpoints using exact rational arithmetic and a saved assignment."""
    require(threshold in THRESHOLDS, "threshold is outside the frozen three-point set")
    predicted, reference, unused_intersections, assignment = validate_contingency(table)
    matches = {r: (p, intersection, union) for p, r, intersection, union in assignment
               if intersection * threshold.denominator >= union * threshold.numerator}
    objects = {}
    errors = []
    matched_absolute = missed = 0
    for ref_id, area in sorted(reference.items()):
        match = matches.get(ref_id)
        row = {"reference_area_pixels": area, "recovered": bool(match), "predicted_id": None,
               "predicted_area_pixels": None, "absolute_area_error_pixels": None,
               "relative_area_error": None, "intersection": None, "union": None}
        if match is None:
            missed += area
        else:
            p, intersection, union = match
            absolute = abs(predicted[p] - area)
            error = Fraction(absolute, area)
            matched_absolute += absolute
            errors.append(error)
            row.update(predicted_id=p, predicted_area_pixels=predicted[p],
                       absolute_area_error_pixels=absolute, relative_area_error=error,
                       intersection=intersection, union=union)
        objects[ref_id] = row
    total_reference = sum(reference.values())
    total_prediction = sum(predicted.values())
    recovered = len(matches)
    endpoints = dict(zip(METRICS, (
        MATCHING, len(reference), len(predicted), recovered, len(reference) - recovered,
        len(predicted) - recovered, divide(recovered, len(reference)), total_reference,
        matched_absolute, missed, divide(matched_absolute + missed, total_reference),
        divide(missed, total_reference), mean(errors), median(errors), len(errors),
        divide(abs(total_prediction - total_reference), total_reference), total_prediction,
        False, "positive_reference_present" if reference else "no_positive_reference")))
    return endpoints, objects


def as_fraction(value, context):
    try:
        decimal = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as error:
        raise CheckError(context + ": invalid numeric value " + repr(value)) from error
    require(decimal.is_finite(), context + ": nonfinite number")
    return Fraction(decimal)


def compare_cell(actual, expected, context, counter):
    counter[0] += 1
    if expected is None:
        require(actual == "", context + ": undefined must remain empty")
    elif type(expected) is bool:
        require(actual == ("true" if expected else "false"), context + ": boolean differs")
    elif type(expected) is int:
        require(re.fullmatch(r"-?(0|[1-9][0-9]*)", actual or "") is not None,
                context + ": expected canonical integer")
        require(int(actual) == expected, context + ": integer differs")
    elif isinstance(expected, Fraction):
        observed = as_fraction(actual, context)
        require(abs(observed - expected) <= ABS_TOL + REL_TOL * abs(expected), context + ": numeric differs")
    elif type(expected) is float:
        observed = as_fraction(actual, context)
        rational = Fraction(str(expected))
        require(abs(observed - rational) <= ABS_TOL + REL_TOL * abs(rational), context + ": numeric differs")
    else:
        require(actual == expected, context + ": text differs")


def read_csv(path, expected_columns):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, path.name + ": absent header")
        require(len(reader.fieldnames) == len(set(reader.fieldnames)), path.name + ": duplicate columns")
        require(tuple(reader.fieldnames) == tuple(expected_columns), path.name + ": columns/order differ")
        rows = list(reader)
    require(all(None not in row and all(value is not None for value in row.values()) for row in rows),
            path.name + ": malformed CSV width")
    return rows


def check_rows(path, columns, expected_rows, keys, counter):
    rows = read_csv(path, columns)
    expected = {}
    for row in expected_rows:
        key = tuple(str(row[name]) if name != "iou_threshold" else str(float(row[name])) for name in keys)
        require(key not in expected, "internal expected key duplicate in " + path.name)
        expected[key] = row
    require(len(rows) == len(expected), path.name + ": row count differs")
    seen = set()
    for line, row in enumerate(rows, 2):
        key = tuple(row[name] for name in keys)
        require(key not in seen, path.name + ": duplicate unit " + repr(key))
        require(key in expected, path.name + ": unexpected unit " + repr(key))
        seen.add(key)
        for column in columns:
            compare_cell(row[column], expected[key][column], path.name + ":" + str(line) + "/" + column, counter)
    require(seen == set(expected), path.name + ": missing units")
    return len(rows)


def load_contract():
    contract = read_json(Path(__file__).resolve().parent / "independent_contract.json")
    require(len(contract["runs"]) == 18 and len({row["run_id"] for row in contract["runs"]}) == 18,
            "checker contract must contain 18 unique frozen runs")
    require(len(contract["primary_image_ids"]) == 13 and len(set(contract["primary_image_ids"])) == 13,
            "checker contract primary IDs differ")
    require(len(contract["secondary_image_ids"]) == 33 and len(set(contract["secondary_image_ids"])) == 33,
            "checker contract secondary IDs differ")
    require(set(contract["primary_image_ids"]) < set(contract["secondary_image_ids"]), "roles are not nested")
    expected = set(itertools.product(INITIALIZATIONS, (4, 16), (20261026, 20261027, 20261028)))
    require({(r["initialization"], r["budget"], r["seed"]) for r in contract["runs"]} == expected,
            "checker contract optimization matrix differs")
    return contract


def validate_metadata(input_root, contract):
    """Read scalar frozen metadata only; array/weight paths stay opaque text."""
    for filename, expected_hash in (("CONFIG_V2.json", contract["config_sha256"]),
        ("TRAINING_FREEZE_V2.json", contract["frozen_training_sha256"]),
        ("prepared_test_manifest.json", contract["prepared_test_manifest_sha256"])):
        require(sha256(input_root / filename) == expected_hash, filename + ": frozen metadata SHA-256 differs")
    config = read_json(input_root / "CONFIG_V2.json")
    training = read_json(input_root / "TRAINING_FREEZE_V2.json")
    evaluation = read_json(input_root / "EVALUATION_FREEZE_V2.json")
    manifest = read_json(input_root / "prepared_test_manifest.json")
    summary = read_json(input_root / "SUMMARY.json")
    roles = read_json(input_root / "DATA_ROLES_FROM_FREEZE.json")
    require(training.get("status") == evaluation.get("status") == "frozen", "frozen metadata status differs")
    require(training.get("schema_version") == evaluation.get("schema_version") == config.get("schema_version") == 1,
            "metadata schema differs")
    require(evaluation.get("training_freeze_sha256") == contract["frozen_training_sha256"], "evaluation training-freeze binding differs")
    require(evaluation.get("allow_test_inference") is True, "evaluation inference gate absent")
    for freeze in (training, evaluation):
        require(freeze.get("config_sha256") == contract["config_sha256"], "freeze/config binding differs")
        require(freeze.get("source_sha256") == config.get("source_sha256"), "source binding differs")
        require(freeze.get("run_ids") == [row["run_id"] for row in contract["runs"]], "freeze run matrix differs")
    for name in ("data_roles", "reference_counts", "prepared_manifest_sha256", "endpoints", "data_roles_sha256"):
        require(evaluation.get(name) == training.get(name), "evaluation changed frozen " + name)
    require(training.get("data_roles_sha256") == contract["data_roles_sha256"], "role source hash differs")
    require(training.get("plan_sha256") == contract["plan_sha256"], "frozen plan hash differs")
    require(roles == training.get("data_roles"), "derived role metadata differs")
    for name, ids in (("primary_confirmation", contract["primary_image_ids"]),
                      ("secondary_confirmation", contract["secondary_image_ids"])):
        require(roles[name].get("image_ids") == ids and roles[name].get("count") == len(ids), "role identity differs: " + name)
    for key, expected in {"primary_images":13,"primary_source_records":652,"primary_scorable_records":651,
        "primary_unscorable_records":1,"all_images":33,"all_source_records":1484,
        "all_scorable_records":1482,"all_unscorable_records":2}.items():
        require(training["reference_counts"].get(key) == expected, "frozen source denominator differs: " + key)
    require(training["prepared_manifest_sha256"].get("test") == contract["prepared_test_manifest_sha256"], "test manifest binding differs")
    configured_runs = config.get("runs")
    require(type(configured_runs) is list and len(configured_runs) == 18, "config run matrix differs")
    configured = {run["id"]: run for run in configured_runs}
    require(set(configured) == {run["run_id"] for run in contract["runs"]}, "config run identities differ")
    images_by_budget = {}
    for run in contract["runs"]:
        actual = configured[run["run_id"]]
        require(all(actual[name] == run[name] for name in ("initialization", "budget", "seed")), "config run descriptor differs")
        ids = actual.get("image_ids")
        require(type(ids) is list and len(ids) == run["budget"] and len(set(ids)) == len(ids), "training image budget differs")
        require(not set(ids).intersection(contract["secondary_image_ids"]), "test image enters training budget")
        if run["budget"] in images_by_budget:
            require(ids == images_by_budget[run["budget"]], "training image selection changes by initialization/seed")
        images_by_budget[run["budget"]] = ids
    require(images_by_budget[4] == images_by_budget[16][:4], "4/16 training budgets are not nested")
    require(manifest.get("schema_version") == 1 and manifest.get("split") == "test", "prepared manifest schema differs")
    require(manifest.get("config_sha256") == contract["config_sha256"] and
            manifest.get("source_sha256") == config.get("source_sha256") and
            manifest.get("policy") == config.get("preprocessing"), "prepared manifest policy/binding differs")
    image_metadata = {}
    for image in manifest["images"]:
        iid = integer(image["image_id"], "manifest image_id", 1)
        require(iid not in image_metadata, "duplicate prepared manifest image")
        shape = image["native_shape"]
        require(type(shape) is list and len(shape) == 2 and all(type(v) is int and v > 0 for v in shape), "invalid native shape metadata")
        require(image.get("work_shape") == shape, "prepared work shape differs from native")
        ignored = integer(image["overlap_native_pixels"], "manifest ignored pixels")
        require(ignored <= shape[0] * shape[1], "ignored pixels exceed native image")
        sources, areas = [], {}
        source_ids = set()
        for source in image["instances"]:
            aid = integer(source["annotation_id"], "manifest annotation_id", 1)
            require(aid not in source_ids, "duplicate manifest source ID")
            source_ids.add(aid)
            require(type(source["scorable"]) is bool, "manifest scorable is not bool")
            area = integer(source["native_area"], "manifest native area")
            if source["scorable"]:
                require(source["reason"] is None and area > 0, "manifest scorable support differs")
                areas[str(aid)] = area
            else:
                require(source["reason"] in ("native_zero_pixel_support", "fully_conflict_ignored") and area == 0,
                        "manifest unsupported reason/area differs")
            sources.append({name: source[name] for name in ("annotation_id", "scorable", "reason")})
        require(image["source_annotation_count"] == len(sources) and image["scorable_count"] == len(areas), "manifest source partition differs")
        image_metadata[iid] = {"file_name": image["file_name"], "source_records": sorted(sources, key=lambda r:r["annotation_id"]),
            "reference_areas": areas, "valid_pixels": shape[0] * shape[1] - ignored, "ignored_pixels": ignored,
            "reference_sha256": image["artifacts"]["instances_native"]["sha256"],
            "evaluation_valid_sha256": image["artifacts"]["evaluation_valid_native"]["sha256"]}
    require(set(image_metadata) == set(contract["secondary_image_ids"]), "prepared test image identities differ")
    require(summary.get("schema_version") == 1 and summary.get("status") == "completed", "scoring summary is not completed")
    require(summary.get("config_sha256") == contract["config_sha256"] and
            summary.get("training_freeze_sha256") == contract["frozen_training_sha256"] and
            summary.get("evaluation_freeze_sha256") == sha256(input_root / "EVALUATION_FREEZE_V2.json"), "summary freeze/config binding differs")
    for key in ("source_sha256", "prepared_manifest_sha256", "reference_counts", "data_roles"):
        require(summary.get(key) == training.get(key), "summary frozen binding differs: " + key)
    require(summary.get("analysis_code_sha256") == evaluation.get("analysis_code_sha256"), "summary analysis-code binding differs")
    for field in ("checkpoints_sha256", "prediction_receipts_sha256"):
        hashes = summary.get(field)
        require(type(hashes) is dict and set(hashes) == set(configured), "summary missing condition binding: " + field)
        require(all(type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) for value in hashes.values()), "invalid opaque condition SHA-256")
    require(summary["checkpoints_sha256"] == evaluation.get("checkpoints_sha256"), "summary checkpoint binding differs")
    require(summary.get("row_counts") == EXPECTED_ROW_COUNTS, "summary complete table cardinality differs")
    require(summary.get("thresholds") == [0.25, 0.5, 0.75] and summary.get("primary_iou_threshold") == 0.5, "summary threshold contract differs")
    require(summary.get("denominators") == DENOMINATORS, "summary denominator definitions differ")
    for name in ("all_18_predictions_completed_before_first_array", "all_bindings_rechecked_at_end", "no_model_selection_or_threshold_search"):
        require(summary.get(name) is True, "summary required producer declaration missing: " + name)
    require(summary.get("scientific_arrays_read_before_preflight") is False, "summary producer preflight declaration differs")
    artifact_hashes = summary.get("artifact_sha256")
    require(type(artifact_hashes) is dict and set(artifact_hashes) == set(ARTIFACT_NAMES), "summary artifact manifest differs")
    for name in ARTIFACT_NAMES:
        require(sha256(input_root / name) == artifact_hashes[name], name + ": scalar artifact SHA-256 differs")
    return image_metadata


def expected_from_contingencies(input_root, contract, image_metadata=None):
    runs = contract["runs"]
    by_run = {row["run_id"]: row for row in runs}
    primary = set(contract["primary_image_ids"])
    secondary = set(contract["secondary_image_ids"])
    expected_units = set(itertools.product(by_run, secondary))
    field_rows, object_rows, tables = [], [], {}
    reference_signatures = {}
    seen = set()
    with (input_root / "contingencies.jsonl").open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            require(line.strip(), "empty contingency line")
            row = json.loads(line, object_pairs_hook=unique_object,
                             parse_constant=lambda value: (_ for _ in ()).throw(CheckError("nonfinite JSON: " + value)))
            require(type(row) is dict, "contingency line must be object")
            run_id = row.get("run_id")
            image_id = row.get("image_id")
            integer(image_id, "image_id", 1)
            key = (run_id, image_id)
            require(key in expected_units, "unexpected run/image unit: " + repr(key))
            require(key not in seen, "duplicate contingency unit: " + repr(key))
            seen.add(key)
            require(row.get("schema_version") == 1, "contingency schema version differs")
            for name in ("initialization", "budget", "seed"):
                require(row.get(name) == by_run[run_id][name], "run identity differs: " + name)
            require(type(row.get("is_primary_confirmation")) is bool and
                    row["is_primary_confirmation"] == (image_id in primary), "primary role differs")
            require(row.get("is_secondary_confirmation") is True, "secondary role differs")
            require(type(row.get("file_name")) is str and bool(row["file_name"]), "file_name missing")
            for prefix in ("reference", "evaluation_valid", "prediction"):
                require(type(row.get(prefix + "_path")) is str and row[prefix + "_path"], "opaque scalar path absent")
                require(re.fullmatch(r"[a-f0-9]{64}", row.get(prefix + "_sha256", "")) is not None,
                        "opaque scalar hash invalid")
            predicted, reference, intersections, assignment = validate_contingency(row["table"])
            sources = row.get("source_records")
            require(type(sources) is list, "source_records must be a list")
            source_ids, scorable_ids = [], []
            for source in sources:
                require(set(source) == {"annotation_id", "scorable", "reason"}, "source record fields differ")
                aid = integer(source["annotation_id"], "annotation_id", 1)
                source_ids.append(aid)
                require(type(source["scorable"]) is bool, "scorable must be boolean")
                if source["scorable"]:
                    require(source["reason"] is None, "scorable reference has unscorable reason")
                    scorable_ids.append(aid)
                else:
                    require(type(source["reason"]) is str and source["reason"], "unscorable reason missing")
            require(source_ids == sorted(set(source_ids)), "source records duplicate or not ID-sorted")
            require(set(scorable_ids) == set(reference), "scorable source IDs differ from reference areas")
            if image_metadata is not None:
                frozen = image_metadata[image_id]
                for name in ("file_name", "source_records", "reference_sha256", "evaluation_valid_sha256"):
                    require(row[name] == frozen[name], "contingency differs from frozen source metadata: " + name)
                for name in ("reference_areas", "valid_pixels", "ignored_pixels"):
                    require(row["table"][name] == frozen[name], "contingency differs from frozen reference metadata: " + name)
            for name, expected_value in (("G_total", len(sources)), ("G_scorable", len(reference)),
                                         ("U_unscorable", len(sources) - len(reference))):
                integer(row.get(name), name)
                require(row[name] == expected_value, name + " partition differs")
            signature = (row["file_name"], json.dumps(sources, sort_keys=True),
                         tuple(sorted(reference.items())), row["reference_sha256"], row["evaluation_valid_sha256"],
                         row["table"]["valid_pixels"], row["table"]["ignored_pixels"])
            if image_id in reference_signatures:
                require(signature == reference_signatures[image_id], "reference/role changed across runs")
            reference_signatures[image_id] = signature
            tables[key] = row
            counts = {name: row[name] for name in ("G_total", "G_scorable", "U_unscorable")}
            previous_recovered = None
            for threshold in THRESHOLDS:
                endpoints, objects = reconstruct(row["table"], threshold)
                recovered = endpoints["recovered_reference_count"]
                require(previous_recovered is None or recovered <= previous_recovered, "recovered count increased with threshold")
                previous_recovered = recovered
                identity = {name: row[name] for name in IDENTITY if name != "iou_threshold"}
                identity["iou_threshold"] = threshold
                field = dict(identity, **counts, **endpoints)
                field["source_record_recovery_lower_bound"] = divide(recovered, counts["G_total"])
                field["source_record_recovery_upper_bound"] = divide(recovered + counts["U_unscorable"], counts["G_total"])
                field_rows.append(field)
                for source in sources:
                    obj = dict(identity, **source)
                    if source["scorable"]:
                        obj.update(objects[source["annotation_id"]])
                    else:
                        obj.update({name: None for name in OBJECT_COLUMNS[len(IDENTITY) + 3:]})
                    object_rows.append(obj)
    require(seen == expected_units, "contingencies missing frozen run/image units")
    for run in by_run:
        for role, ids, expected_counts in (("primary", primary, (652, 651, 1)),
                                           ("secondary", secondary, (1484, 1482, 2))):
            totals = tuple(sum(tables[run, iid][name] for iid in ids) for name in ("G_total", "G_scorable", "U_unscorable"))
            require(totals == expected_counts, role + " complete source denominator differs")
    unique_source_ids = [source["annotation_id"] for iid in secondary for source in tables[runs[0]["run_id"], iid]["source_records"]]
    require(len(unique_source_ids) == len(set(unique_source_ids)), "source annotation ID reused across images")
    return field_rows, object_rows, tables


def check(input_root, contract):
    """Check only explicitly named scalar input files; never resolve array paths."""
    require(input_root.is_dir(), "input must be an existing scalar-results directory")
    counter = [0]
    image_metadata = validate_metadata(input_root, contract)
    fields, objects, tables = expected_from_contingencies(input_root, contract, image_metadata)
    counts = {
        "contingencies": len(tables),
        "field_endpoints": check_rows(input_root / "field_endpoints.csv", FIELD_COLUMNS, fields,
             ("run_id", "image_id", "iou_threshold"), counter),
        "object_endpoints": check_rows(input_root / "object_endpoints.csv", OBJECT_COLUMNS, objects,
             ("run_id", "image_id", "iou_threshold", "annotation_id"), counter),
    }
    # Full aggregation and pairing are deliberately implemented independently
    # below; production aggregation modules must not be imported.
    aggregate_counts = check_aggregate_tables(input_root, fields, objects, contract, counter)
    counts.update(aggregate_counts)
    return {"status": "passed", "schema_version": 1, "row_counts": counts,
            "scalar_cells_compared": counter[0], "arithmetic": "Python stdlib integers and Fraction; float CSV tolerance abs=1e-12,rel=1e-12",
            "scope": "saved integer areas/intersections and saved assignment to all scalar tables",
            "frozen_scalar_reference_metadata_sha256": contract["prepared_test_manifest_sha256"],
            "shared_primitive": "saved assignment declared maximum-total-IoU Hungarian by producer; optimality not independently certified",
            "producer_declarations": "completion/preflight/training-selection declarations checked for consistency, not independently observed execution",
            "not_certified": ["pixel contingency or reference truth", "independent Hungarian optimization",
                "full-image false positives/PQ/count accuracy", "biological independence or biological confidence intervals",
                "physical areas", "training replay or prediction replay"],
            "source_bounds": "extreme support bounds for unscorable source records; not a biological CI",
            "threshold_property": "recovered counts are nonincreasing; positive-reference area residual need not be monotone"}


def role_member(row, role):
    return row["is_" + role]


def summarize_runs(fields, objects, contract):
    object_groups = {}
    for row in objects:
        object_groups.setdefault((row["run_id"], row["iou_threshold"]), []).append(row)
    field_groups = {}
    for row in fields:
        field_groups.setdefault((row["run_id"], row["iou_threshold"]), []).append(row)
    output = []
    for (run_id, threshold), all_fields in sorted(field_groups.items()):
        for role in ROLES:
            selected = [row for row in all_fields if role_member(row, role)]
            selected_objects = [row for row in object_groups[run_id, threshold] if role_member(row, role)]
            errors = [row["relative_area_error"] for row in selected_objects if row["scorable"] and row["recovered"]]
            item = {name: selected[0][name] for name in RUN_ID_COLUMNS if name != "role"}
            item.update(role=role, field_count=len(selected))
            for name in SUM_COLUMNS:
                item[name] = sum(row[name] for row in selected)
            item["annotation_relative_total_area_difference_pixels"] = sum(
                abs(row["predicted_total_area_pixels"] - row["reference_area_pixels"]) for row in selected)
            item["conditional_relative_error_sum"] = sum(errors, Fraction())
            require(len(errors) == item["conditional_error_denominator"], "aggregate conditional denominator differs")
            pooled = {
                "positive_reference_recovery_fraction": divide(item["recovered_reference_count"], item["G_scorable"]),
                "positive_reference_area_residual_fraction": divide(item["matched_absolute_area_error_pixels"] + item["missed_reference_area_pixels"], item["reference_area_pixels"]),
                "missed_reference_area_fraction": divide(item["missed_reference_area_pixels"], item["reference_area_pixels"]),
                "conditional_mean_relative_area_error": mean(errors),
                "conditional_median_relative_area_error": median(errors),
                "annotation_relative_total_area_difference_fraction": divide(item["annotation_relative_total_area_difference_pixels"], item["reference_area_pixels"]),
                "source_record_recovery_lower_bound": divide(item["recovered_reference_count"], item["G_total"]),
                "source_record_recovery_upper_bound": divide(item["recovered_reference_count"] + item["U_unscorable"], item["G_total"]),
            }
            for metric in AGG_METRICS:
                values = [row[metric] for row in selected if row[metric] is not None]
                item["macro_" + metric] = mean(values)
                item["macro_" + metric + "_contributing_fields"] = len(values)
                item["pooled_" + metric] = pooled[metric]
            output.append(item)
    return output


def sample_sd(values):
    """Exact rational sample variance, then a 60-digit stdlib square root."""
    if len(values) < 2:
        return None
    average = mean(values)
    variance = sum(((value - average)**2 for value in values), Fraction()) / (len(values) - 1)
    with localcontext() as context:
        context.prec = 60
        return Fraction((Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt())


def summarize_seeds(runs):
    groups = {}
    for row in runs:
        key = tuple(row[name] for name in ("initialization", "budget", "role", "iou_threshold"))
        groups.setdefault(key, []).append(row)
    output = []
    for (initialization, budget, role, threshold), rows in sorted(groups.items()):
        seeds = sorted(row["seed"] for row in rows)
        require(seeds == [20261026, 20261027, 20261028], "seed summary lost a frozen optimization seed")
        item = dict(initialization=initialization, budget=budget, role=role, iou_threshold=threshold,
                    n_seeds=3, seed_ids="|".join(map(str, seeds)),
                    interpretation="optimization_repetitions_only_not_biological_n_or_significance")
        for column in RUN_METRIC_COLUMNS:
            if column.endswith("_contributing_fields"):
                continue
            values = [row[column] for row in rows if row[column] is not None]
            item[column + "_mean"] = mean(values)
            item[column + "_sample_sd"] = sample_sd(values)
            item[column + "_contributing_seeds"] = len(values)
        output.append(item)
    return output


def summarize_pairs(objects, contract):
    object_lookup = {}
    for row in objects:
        if row["iou_threshold"] == Fraction(1, 2):
            object_lookup.setdefault(row["run_id"], {})[row["image_id"], row["annotation_id"]] = row
    run_lookup = {(row["initialization"], row["budget"], row["seed"]): row["run_id"] for row in contract["runs"]}
    output = []
    for left_initialization, right_initialization in PAIRS:
        for budget, seed in itertools.product((4, 16), (20261026, 20261027, 20261028)):
            left_id = run_lookup[left_initialization, budget, seed]
            right_id = run_lookup[right_initialization, budget, seed]
            left, right = object_lookup[left_id], object_lookup[right_id]
            require(set(left) == set(right), "paired source identities differ")
            for role in ROLES:
                keys = [key for key, row in left.items() if role_member(row, role)]
                categories = {name: [] for name in ("common", "left_only", "right_only", "neither")}
                item = dict(left_run_id=left_id, right_run_id=right_id, left_initialization=left_initialization,
                    right_initialization=right_initialization, budget=budget, seed=seed, role=role,
                    iou_threshold=Fraction(1, 2), field_count=len({key[0] for key in keys}),
                    G_total=len(keys), G_scorable=0, U_unscorable=0)
                for key in keys:
                    lrow, rrow = left[key], right[key]
                    require((lrow["scorable"], lrow["reference_area_pixels"]) ==
                            (rrow["scorable"], rrow["reference_area_pixels"]), "paired reference support differs")
                    if not lrow["scorable"]:
                        item["U_unscorable"] += 1
                        continue
                    item["G_scorable"] += 1
                    category = ("common" if lrow["recovered"] and rrow["recovered"] else
                                "left_only" if lrow["recovered"] else "right_only" if rrow["recovered"] else "neither")
                    categories[category].append((lrow, rrow))
                for category, pairs in categories.items():
                    item[category + "_recovered_count"] = len(pairs)
                    if category != "neither":
                        item[category + "_contributing_fields"] = len({pair[0]["image_id"] for pair in pairs})
                for category, side, side_index in (("common", "left", 0), ("common", "right", 1),
                    ("left_only", "left", 0), ("right_only", "right", 1)):
                    rows = [pair[side_index] for pair in categories[category]]
                    item[category + "_pooled_" + side + "_mean_relative_area_error"] = mean([row["relative_area_error"] for row in rows])
                    by_field = {}
                    for row in rows:
                        by_field.setdefault(row["image_id"], []).append(row["relative_area_error"])
                    item[category + "_macro_" + side + "_mean_relative_area_error"] = mean([mean(values) for values in by_field.values()])
                for aggregation in ("pooled", "macro"):
                    lvalue = item["common_" + aggregation + "_left_mean_relative_area_error"]
                    rvalue = item["common_" + aggregation + "_right_mean_relative_area_error"]
                    item["common_" + aggregation + "_right_minus_left"] = rvalue - lvalue if lvalue is not None else None
                output.append(item)
    return output


def check_aggregate_tables(input_root, fields, objects, contract, counter):
    runs = summarize_runs(fields, objects, contract)
    seeds = summarize_seeds(runs)
    pairs = summarize_pairs(objects, contract)
    require((len(runs), len(seeds), len(pairs)) == (108, 36, 36), "aggregate matrix cardinality differs")
    return {
        "run_summary": check_rows(input_root / "run_summary.csv", RUN_COLUMNS, runs,
            ("run_id", "role", "iou_threshold"), counter),
        "seed_summary": check_rows(input_root / "seed_summary.csv", SEED_COLUMNS, seeds,
            ("initialization", "budget", "role", "iou_threshold"), counter),
        "paired_identity_comparisons": check_rows(input_root / "paired_identity_comparisons.csv", PAIRED_COLUMNS, pairs,
            ("left_run_id", "right_run_id", "role", "iou_threshold"), counter),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="new scalar result directory")
    parser.add_argument("--output", required=True, type=Path, help="new JSON receipt file; overwrite refused")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        parser.error("output already exists; refusing overwrite")
    if not output.parent.is_dir():
        parser.error("output parent must already exist")
    selected = ARTIFACT_NAMES + ("SUMMARY.json",)
    try:
        contract = load_contract()
        receipt = check(args.input.resolve(), contract)
        receipt["input_sha256"] = {name: sha256(args.input / name) for name in selected}
        receipt["checker_sha256"] = sha256(Path(__file__).resolve())
        receipt["contract_sha256"] = sha256(Path(__file__).resolve().parent / "independent_contract.json")
        code = 0
    except (CheckError, OSError, ValueError, KeyError, TypeError) as error:
        receipt = {"status": "failed", "schema_version": 1, "error_type": type(error).__name__, "error": str(error)}
        code = 1
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(output)}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
