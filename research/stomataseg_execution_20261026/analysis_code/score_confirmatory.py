"""One fixed confirmatory score pass, after every prediction is complete.

Preflight hashes and validates all 18 terminal prediction receipts and their 33
images before importing measurement or opening any GT/prediction array. Paths
come from CONFIG, its frozen manifests, and prediction receipts; no data glob.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aggregation as agg

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha256(path: Path) -> str:
    """Hash an explicit file without interpreting its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sha(path: Path, expected: str) -> str:
    """Require the exact frozen bytes, failing closed on missing hashes."""
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Invalid/missing SHA-256: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return actual


def read_json(path: Path) -> dict[str, Any]:
    """Read one explicitly named UTF-8 JSON document."""
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    """Create an audit JSON without replacing an existing artifact."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def resolve(base: Path, value: str) -> Path:
    """Resolve an explicit config or receipt path."""
    if not isinstance(value, str) or not value:
        raise ValueError("A nonempty explicit path is required")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def child_path(base: Path, value: str) -> Path:
    """Require a manifest/receipt artifact to stay inside its owning directory."""
    path = resolve(base, value)
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Artifact path escaped its receipt directory")
    return path


def verify_mapping(base: Path, mapping: dict[str, str]) -> dict[Path, str]:
    """Check an explicit, nonempty source-code mapping."""
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("An explicit source mapping is required")
    resolved = {}
    for filename, expected in mapping.items():
        path = resolve(base, filename)
        if path in resolved:
            raise ValueError("Duplicate resolved source path")
        check_sha(path, expected)
        resolved[path] = expected
    return resolved


def validate_runs(runs: list[dict[str, Any]]) -> None:
    """Require all fixed initialization/budget/seed combinations."""
    expected = {(init, budget, seed) for init in agg.INITIALIZATIONS for budget in (4, 16)
                for seed in (20261026, 20261027, 20261028)}
    actual = {(run["initialization"], run["budget"], run["seed"]) for run in runs}
    if len(runs) != 18 or actual != expected or len({run["id"] for run in runs}) != 18:
        raise ValueError("The complete fixed 18-run matrix is required")
    if len({resolve(Path.cwd(), run["prediction_dir"]) for run in runs}) != 18:
        raise ValueError("Prediction directories must be unique")


def preflight(config_path: Path) -> dict[str, Any]:
    """Bind every condition and artifact before any scientific array is read."""
    config_path = config_path.resolve()
    config = read_json(config_path)
    config_sha = sha256(config_path)
    base = config_path.parent
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported config schema")
    validate_runs(config["runs"])
    sources = config["source_sha256"]
    source_paths = verify_mapping(base, sources)
    if len(sources) != 8:
        raise ValueError("The frozen eight-source mapping may not be expanded")
    measurement_path = ROOT / "engineering/stomataseg_v1/measurement.py"
    if measurement_path not in source_paths:
        raise ValueError("Measurement source must be bound in the original source mapping")
    train_path = resolve(base, config["training_freeze"]["path"])
    eval_path = resolve(base, config["evaluation_freeze"]["path"])
    train, evaluation = read_json(train_path), read_json(eval_path)
    for freeze in (train, evaluation):
        if freeze.get("schema_version") != 1 or freeze.get("status") != "frozen":
            raise ValueError("Explicit frozen training/evaluation documents are required")
        if freeze.get("config_sha256") != config_sha or freeze.get("source_sha256") != sources:
            raise ValueError("Frozen config/source binding mismatch")
        run_ids = freeze.get("run_ids", [])
        if len(run_ids) != 18 or set(run_ids) != {run["id"] for run in config["runs"]}:
            raise ValueError("Freeze must bind exactly all 18 runs")
    check_sha(train_path, evaluation["training_freeze_sha256"])
    if evaluation.get("allow_test_inference") is not True:
        raise ValueError("Evaluation freeze must explicitly allow test inference")
    for key in ("data_roles", "reference_counts", "prepared_manifest_sha256", "endpoints", "data_roles_sha256"):
        if evaluation.get(key) != train[key]:
            raise ValueError(f"Evaluation freeze changed pretraining scientific contract: {key}")
    analysis_paths = verify_mapping(base, evaluation.get("analysis_code_sha256", {}))
    if not {Path(__file__).resolve(), Path(agg.__file__).resolve()}.issubset(analysis_paths):
        raise ValueError("Evaluation freeze must bind score driver and aggregation code")
    roles = train["data_roles"]
    primary = roles["primary_confirmation"]["image_ids"]
    secondary = roles["secondary_confirmation"]["image_ids"]
    if (len(primary) != 13 or len(set(primary)) != 13 or len(secondary) != 33 or len(set(secondary)) != 33
            or not set(primary).issubset(secondary)):
        raise ValueError("Expected nested13/complete33 preassigned roles")
    counts = train["reference_counts"]
    required_counts = dict(primary_images=13, primary_source_records=652, primary_scorable_records=651,
                           primary_unscorable_records=1, all_images=33, all_source_records=1484,
                           all_scorable_records=1482, all_unscorable_records=2)
    if any(counts.get(key) != value for key, value in required_counts.items()):
        raise ValueError("Unexpected frozen reference denominator contract")
    manifest_path = resolve(base, config["prepared"]["test"]["output_dir"]) / "manifest.json"
    manifest_sha = train["prepared_manifest_sha256"]["test"]
    check_sha(manifest_path, manifest_sha)
    manifest = read_json(manifest_path)
    if (manifest.get("schema_version") != 1 or manifest.get("split") != "test"
            or manifest.get("config_sha256") != config_sha or manifest.get("source_sha256") != sources
            or manifest.get("policy") != config["preprocessing"]):
        raise ValueError("Prepared manifest/config/policy binding mismatch")
    images = manifest["images"]
    if len(images) != 33 or {row["image_id"] for row in images} != set(secondary):
        raise ValueError("Prepared manifest must retain all 33 source images")
    image_by_id = {row["image_id"]: row for row in images}
    bindings: dict[Path, str] = {config_path: config_sha, train_path: sha256(train_path), eval_path: sha256(eval_path),
                               manifest_path: manifest_sha, **source_paths, **analysis_paths}
    for record in images:
        instances = record["instances"]
        ids = [row["annotation_id"] for row in instances]
        if (len(ids) != len(set(ids)) or any(type(value) is not int or value <= 0 for value in ids)
                or len(instances) != record["source_annotation_count"]):
            raise ValueError("Invalid source annotation identity metadata")
        for source in instances:
            if type(source["scorable"]) is not bool:
                raise ValueError("scorable must be an explicit boolean")
            if source["scorable"]:
                if source["reason"] is not None or source["native_area"] <= 0 or source["work_area"] <= 0:
                    raise ValueError("Invalid scorable support metadata")
            elif source["reason"] not in ("native_zero_pixel_support", "fully_conflict_ignored") or source["native_area"] != 0:
                raise ValueError("Unscorable support/reason is not retained explicitly")
        if (sum(source["scorable"] for source in instances) != record["scorable_count"]
                or [source for source in instances if not source["scorable"]] != record["unscorable_records"]
                or record["native_shape"] != record["work_shape"]):
            raise ValueError("Native support partition or no-resize policy mismatch")
        for artifact in ("instances_native", "evaluation_valid_native"):
            item = record["artifacts"][artifact]
            path = child_path(manifest_path.parent, item["path"])
            check_sha(path, item["sha256"])
            bindings[path] = item["sha256"]
    for selected, prefix in ((images, "all"), ([row for row in images if row["image_id"] in primary], "primary")):
        if (sum(row["source_annotation_count"] for row in selected) != counts[f"{prefix}_source_records"]
                or sum(row["scorable_count"] for row in selected) != counts[f"{prefix}_scorable_records"]
                or sum(len(row["unscorable_records"]) for row in selected) != counts[f"{prefix}_unscorable_records"]):
            raise ValueError("Manifest denominator totals differ from the frozen contract")
    checkpoints = evaluation.get("checkpoints_sha256", {})
    checkpoint_receipts = evaluation.get("checkpoint_receipts_sha256", {})
    expected_run_ids = {run["id"] for run in config["runs"]}
    if set(checkpoints) != expected_run_ids or set(checkpoint_receipts) != expected_run_ids:
        raise ValueError("Evaluation freeze must bind all 18 final checkpoints and terminal receipts")
    prediction_receipts: dict[str, dict[str, Any]] = {}
    prediction_paths: dict[tuple[str, int], Path] = {}
    evaluation_sha = bindings[eval_path]
    for run in config["runs"]:
        run_id = run["id"]
        run_dir = resolve(base, run["output_dir"])
        terminal_path = run_dir / "run.json"
        check_sha(terminal_path, checkpoint_receipts[run_id])
        terminal = read_json(terminal_path)
        if (terminal.get("status") != "completed" or terminal.get("step") != 300
                or terminal.get("run_id") != run_id or terminal.get("config_sha256") != config_sha
                or terminal.get("source_sha256") != sources or terminal.get("checkpoint") != {
                    "path": "final.pt", "sha256": checkpoints[run_id]}):
            raise ValueError("A fixed final300 terminal training receipt is missing or unbound")
        bindings[terminal_path] = checkpoint_receipts[run_id]
        checkpoint_path = run_dir / "final.pt"
        check_sha(checkpoint_path, checkpoints[run_id])
        bindings[checkpoint_path] = checkpoints[run_id]
        prediction_dir = resolve(base, run["prediction_dir"])
        receipt_path = prediction_dir / "predictions.json"
        receipt = read_json(receipt_path)
        if (receipt.get("schema_version") != 1 or receipt.get("status") != "completed"
                or receipt.get("run_id") != run_id or receipt.get("config_sha256") != config_sha
                or receipt.get("source_sha256") != sources or receipt.get("evaluation_freeze_sha256") != evaluation_sha
                or receipt.get("checkpoint_sha256") != checkpoints[run_id]
                or receipt.get("prepared_manifest_sha256") != manifest_sha or receipt.get("decoder") != config["decoder"]
                or receipt.get("ground_truth_arrays_read") is not False
                or receipt.get("reference_metadata_used_for_prediction") is not False):
            raise ValueError("All 18 completed prediction receipts must be exactly bound")
        if len(receipt["images"]) != 33 or {row["image_id"] for row in receipt["images"]} != set(secondary):
            raise ValueError("Every completed prediction must contain exactly all 33 source image IDs")
        bindings[receipt_path] = sha256(receipt_path)
        per_run_paths = set()
        for row in receipt["images"]:
            source = image_by_id[row["image_id"]]
            if any(row[key] != source[key] for key in ("file_name", "native_shape", "work_shape")):
                raise ValueError("Prediction image identity/shape differs from prepared metadata")
            item = row["prediction"]
            path = child_path(prediction_dir, item["path"])
            if path in per_run_paths or path.suffix.lower() not in (".tif", ".tiff"):
                raise ValueError("Prediction TIFF paths must be unique within run")
            per_run_paths.add(path)
            check_sha(path, item["sha256"])
            bindings[path] = item["sha256"]
            prediction_paths[(run_id, row["image_id"])] = path
        prediction_receipts[run_id] = receipt
    return dict(config=config, base=base, config_path=config_path, config_sha256=config_sha,
                train_path=train_path, evaluation_path=eval_path, training=train, evaluation=evaluation,
                manifest_path=manifest_path, manifest=manifest, bindings=bindings,
                prediction_receipts=prediction_receipts, prediction_paths=prediction_paths,
                measurement_path=measurement_path, analysis_paths=analysis_paths)


def csv_value(value: Any) -> Any:
    """Encode missing as blank and booleans consistently in portable CSV."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite scalar cannot be exported")
    return value


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Create a fixed-schema CSV, rejecting omitted or unexpected columns."""
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if set(row) != set(columns):
                raise ValueError(f"CSV schema mismatch: {path.name}: {set(row) ^ set(columns)}")
            writer.writerow({key: csv_value(row[key]) for key in columns})


def load_measurement(path: Path) -> Any:
    """Load the already SHA-checked original measurement module unchanged."""
    spec = importlib.util.spec_from_file_location("stomataseg_confirmatory_measurement", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the explicitly bound measurement module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(config_path: Path, output: Path) -> Path:
    """Execute the one fixed pass into a previously nonexistent directory."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Refusing an existing output directory")
    state = preflight(config_path)
    # All18 x all33 and every binding have passed. No array access occurs above.
    import numpy as np
    from PIL import Image

    measurement = load_measurement(state["measurement_path"])
    output.mkdir(parents=True, exist_ok=False)
    started = {"schema_version": 1, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
               "all_18_predictions_completed_before_first_array": True,
               "config_sha256": state["config_sha256"], "scientific_arrays_read_before_preflight": False}
    write_json(output / "STARTED.json", started)
    fields: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    contingency_rows = 0
    try:
        primary = set(state["training"]["data_roles"]["primary_confirmation"]["image_ids"])
        manifest_path = state["manifest_path"]
        images = state["manifest"]["images"]
        with (output / "contingencies.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for run in state["config"]["runs"]:
                run_id = run["id"]
                for image in images:
                    reference_item = image["artifacts"]["instances_native"]
                    valid_item = image["artifacts"]["evaluation_valid_native"]
                    reference_path = child_path(manifest_path.parent, reference_item["path"])
                    valid_path = child_path(manifest_path.parent, valid_item["path"])
                    prediction_path = state["prediction_paths"][(run_id, image["image_id"])]
                    for path in (reference_path, valid_path, prediction_path):
                        check_sha(path, state["bindings"][path])
                    reference = np.load(reference_path, allow_pickle=False)
                    evaluation_valid = np.load(valid_path, allow_pickle=False)
                    with Image.open(prediction_path) as prediction_image:
                        if prediction_image.format != "TIFF":
                            raise ValueError("Prediction must be native TIFF")
                        prediction = np.asarray(prediction_image).copy()
                    shape = tuple(image["native_shape"])
                    if (reference.shape != shape or evaluation_valid.shape != shape or prediction.shape != shape
                            or reference.dtype != np.uint64 or evaluation_valid.dtype != np.bool_
                            or prediction.dtype != np.uint16):
                        raise ValueError("Native reference/validity/uint16 prediction shape or dtype mismatch")
                    if int((~evaluation_valid).sum()) != image["overlap_native_pixels"]:
                        raise ValueError("Native conflict ignore pixels changed")
                    source_records = [{key: source[key] for key in ("annotation_id", "scorable", "reason")}
                                      for source in sorted(image["instances"], key=lambda row: row["annotation_id"])]
                    expected_areas = {str(source["annotation_id"]): source["native_area"]
                                      for source in image["instances"] if source["scorable"]}
                    # Exactly once per run/field: assignments do not change by theta.
                    table = measurement.contingency(prediction, reference, ignore=~evaluation_valid)
                    if table["reference_areas"] != expected_areas:
                        raise ValueError("Actual scorable reference IDs/areas differ from frozen source metadata")
                    if np.any(reference[~evaluation_valid] != 0):
                        raise ValueError("Conflict pixels must stay ignored and zero in native labels")
                    record = dict(schema_version=1, run_id=run_id, initialization=run["initialization"],
                                  budget=run["budget"], seed=run["seed"], image_id=image["image_id"],
                                  file_name=image["file_name"], is_primary_confirmation=image["image_id"] in primary,
                                  is_secondary_confirmation=True, G_total=len(source_records),
                                  G_scorable=len(expected_areas), U_unscorable=len(source_records) - len(expected_areas),
                                  source_records=source_records, reference_path=str(reference_path),
                                  reference_sha256=reference_item["sha256"], evaluation_valid_path=str(valid_path),
                                  evaluation_valid_sha256=valid_item["sha256"], prediction_path=str(prediction_path),
                                  prediction_sha256=state["bindings"][prediction_path], table=table)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    contingency_rows += 1
                    for theta in agg.THRESHOLDS:
                        endpoints = measurement.score_contingency(table, threshold=theta)
                        field, per_object = agg.field_and_objects(record, endpoints)
                        fields.append(field)
                        objects.extend(per_object)
        summaries = agg.run_summaries(fields, objects)
        seeds = agg.seed_summaries(summaries)
        pairs = agg.paired_comparisons(objects, state["config"]["runs"])
        expected_rows = dict(contingencies=594, field_endpoints=1782, object_endpoints=80136,
                             run_summary=108, seed_summary=36, paired_identity_comparisons=36)
        actual_rows = dict(contingencies=contingency_rows, field_endpoints=len(fields), object_endpoints=len(objects),
                           run_summary=len(summaries), seed_summary=len(seeds), paired_identity_comparisons=len(pairs))
        if actual_rows != expected_rows:
            raise ValueError("The full fixed analysis matrix was not preserved")
        for filename, columns, rows in (
            ("field_endpoints", agg.FIELD_COLUMNS, fields), ("object_endpoints", agg.OBJECT_COLUMNS, objects),
            ("run_summary", agg.RUN_COLUMNS, summaries), ("seed_summary", agg.SEED_COLUMNS, seeds),
            ("paired_identity_comparisons", agg.PAIRED_COLUMNS, pairs)):
            write_csv(output / f"{filename}.csv", columns, rows)
        for source_path, destination in ((state["config_path"], "CONFIG_V2.json"),
                                         (state["train_path"], "TRAINING_FREEZE_V2.json"),
                                         (state["evaluation_path"], "EVALUATION_FREEZE_V2.json"),
                                         (state["manifest_path"], "prepared_test_manifest.json")):
            shutil.copyfile(source_path, output / destination)
            check_sha(output / destination, state["bindings"][source_path])
        # An embedded role snapshot is explicitly derived, never relabelled original bytes.
        write_json(output / "DATA_ROLES_FROM_FREEZE.json", state["training"]["data_roles"])
        # Recheck all original code/config/freeze/manifests/arrays/receipts/checkpoints at end.
        for path, expected in state["bindings"].items():
            check_sha(path, expected)
        artifact_names = ["contingencies.jsonl", "field_endpoints.csv", "object_endpoints.csv", "run_summary.csv",
                          "seed_summary.csv", "paired_identity_comparisons.csv", "CONFIG_V2.json",
                          "TRAINING_FREEZE_V2.json", "EVALUATION_FREEZE_V2.json", "DATA_ROLES_FROM_FREEZE.json",
                          "prepared_test_manifest.json"]
        summary = dict(started, status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                       evaluation_freeze_sha256=state["bindings"][state["evaluation_path"]],
                       training_freeze_sha256=state["bindings"][state["train_path"]],
                       source_sha256=state["config"]["source_sha256"],
                       analysis_code_sha256=state["evaluation"]["analysis_code_sha256"],
                       prepared_manifest_sha256=state["training"]["prepared_manifest_sha256"],
                       checkpoints_sha256=state["evaluation"]["checkpoints_sha256"],
                       prediction_receipts_sha256={run["id"]: state["bindings"][resolve(state["base"], run["prediction_dir"]) / "predictions.json"]
                                                  for run in state["config"]["runs"]},
                       reference_counts=state["training"]["reference_counts"],
                       data_roles=state["training"]["data_roles"], thresholds=list(agg.THRESHOLDS),
                       primary_iou_threshold=0.5, row_counts=actual_rows,
                       artifact_sha256={name: sha256(output / name) for name in artifact_names},
                       all_bindings_rechecked_at_end=True,
                       no_model_selection_or_threshold_search=True,
                       interpretation={"scope": "visible_annotated_complex_positive_reference_only",
                                       "units": "native_pixels_and_dimensionless_ratios",
                                       "unmatched_predictions": "not_certified_false_positives",
                                       "unsupported_bounds": "source_record_support_sensitivity_not_biological_confidence_intervals",
                                       "seeds": "optimization_description_not_biological_n_or_significance",
                                       "roles": "nested_primary13_and_complete_secondary33_not_independent_cohorts",
                                       "filename_tokens": "not_verified_plant_identity",
                                       "conditional_errors": "depend_on_recovered_reference_identity_set",
                                       "annotation_total_area": "annotation_relative_not_absolute_measurement_accuracy"},
                       denominators={"recovery": "G_scorable",
                                     "positive_reference_area_residual": "sum_scorable_reference_area",
                                     "conditional_errors": "recovered_scorable_reference_records",
                                     "support_bounds": "G_total",
                                     "macro": "nonempty_per_field_values_with_contributing_fields",
                                     "pooled_annotation_total_area_difference": "sum_field_abs(Parea-Garea)/sum_field_Garea",
                                     "paired": "identical_(image_id,annotation_id)_scorable_source_records"})
        write_json(output / "SUMMARY.json", summary)
    except Exception as error:
        write_json(output / "FAILURE.json", dict(started, status="failed", error=repr(error),
                   completed_contingencies=contingency_rows, emitted_field_rows=len(fields), emitted_object_rows=len(objects)))
        raise
    return output / "SUMMARY.json"


def main() -> None:
    """Parse only explicit config and new output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(score(args.config, args.output))


if __name__ == "__main__":
    main()
