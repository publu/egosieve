"""Fail-closed validation for a Hugging Face model artifact."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .training.data import BOUNDARY_LABELS, ISSUE_LABELS, READINESS_LABELS
from .training.metrics import (
    expected_calibration_error,
    issue_metrics,
    readiness_metrics,
    selective_risk_curve,
    temporal_boundary_metrics,
)

REQUIRED_FILES = (
    "README.md",
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "metrics.json",
    "training_report.json",
    "splits.json",
    "test_predictions.json",
    "evidence.json",
)
REQUIRED_CODE_FILES = (
    "configuration_egosieve.py",
    "modeling_egosieve.py",
    "processing_egosieve.py",
)
EXPECTED_MODEL_AUTO_MAP = {
    "AutoConfig": "configuration_egosieve.EgoSieveConfig",
    "AutoModelForVideoClassification": "modeling_egosieve.EgoSieveModel",
}
EXPECTED_PROCESSOR_AUTO_MAP = {
    "AutoProcessor": "processing_egosieve.EgoSieveProcessor",
}
REQUIRED_METRICS = (
    "readiness.macro_f1",
    "issues.macro_auroc",
    "issues.macro_average_precision",
    "boundaries.f1",
    "boundaries.tolerance_s",
    "calibration.ece",
    "throughput.cpu_windows_per_second",
    "throughput.gpu_windows_per_second",
)
FORBIDDEN_FILES = ("UNTRAINED_HEADS",)
UNRESOLVED = re.compile(r"\{\{[A-Z0-9_]+\}\}")
SHA256 = re.compile(r"[0-9a-f]{64}")

TRAINING_REPORT_SCHEMA = "egosieve.training-report/v1"
SPLITS_SCHEMA = "egosieve.splits/v1"
TEST_PREDICTIONS_SCHEMA = "egosieve.test-predictions/v1"
RELEASE_EVIDENCE_SCHEMA = "egosieve.release-evidence/v1"
METRICS_SCHEMA = "egosieve.release-metrics/v1"
SPLIT_NAMES = ("train", "validation", "test")
PROVENANCE_KINDS = (
    "human",
    "human-derived",
    "programmatic-controlled-corruption",
    "unlabeled",
)
HUMAN_GROUNDED_PROVENANCE = frozenset({"human", "human-derived"})
READINESS_EVIDENCE_PROVENANCE = ("human", "human-derived")
ISSUE_EVIDENCE_PROVENANCE = (
    "human",
    "human-derived",
    "programmatic-controlled-corruption",
)
HASHED_RELEASE_FILES = tuple(name for name in REQUIRED_FILES if name != "evidence.json") + (
    *REQUIRED_CODE_FILES,
)


class ReleaseValidationError(ValueError):
    """The artifact is incomplete or cannot support its model-card claims."""


def _dig(obj: dict[str, Any], dotted: str) -> Any:
    value: Any = obj
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(dotted)
        value = value[key]
    return value


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ReleaseValidationError(f"invalid JSON artifact `{path.name}`: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseValidationError(f"`{path.name}` must contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _non_empty_string(value: Any, dotted: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseValidationError(f"`{dotted}` must be a non-empty string")
    return value.strip()


def _sha256(value: Any, dotted: str) -> str:
    result = _non_empty_string(value, dotted)
    if SHA256.fullmatch(result) is None:
        raise ReleaseValidationError(f"`{dotted}` must be a lowercase SHA-256 digest")
    return result


def _positive_integer(value: Any, dotted: str) -> int:
    result = _non_negative_integer(value, dotted)
    if result == 0:
        raise ReleaseValidationError(f"`{dotted}` must be positive")
    return result


def _expect_close(actual: Any, expected: Any, dotted: str) -> None:
    actual_number = _finite_non_negative(actual, dotted)
    if expected is None or not math.isfinite(float(expected)):
        raise ReleaseValidationError(f"recomputed metric `{dotted}` is undefined")
    if not math.isclose(actual_number, float(expected), rel_tol=1e-7, abs_tol=1e-9):
        raise ReleaseValidationError(
            f"metric `{dotted}` does not match test_predictions.json "
            f"(declared {actual_number:.12g}, recomputed {float(expected):.12g})"
        )


def _expect_integer(actual: Any, expected: int, dotted: str) -> None:
    actual_integer = _non_negative_integer(actual, dotted)
    if actual_integer != expected:
        raise ReleaseValidationError(
            f"metric `{dotted}` does not match test_predictions.json "
            f"(declared {actual_integer}, recomputed {expected})"
        )


def _finite_non_negative(value: Any, dotted: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReleaseValidationError(f"metric `{dotted}` must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ReleaseValidationError(f"metric `{dotted}` must be a finite non-negative number")
    return number


def _non_negative_integer(value: Any, dotted: str) -> int:
    number = _finite_non_negative(value, dotted)
    if not number.is_integer():
        raise ReleaseValidationError(f"metric `{dotted}` must be a non-negative integer")
    return int(number)


def _unit_metric(value: Any, dotted: str) -> float:
    number = _finite_non_negative(value, dotted)
    if number > 1:
        raise ReleaseValidationError(f"metric `{dotted}` must lie in [0, 1]")
    return number


def _validate_evidence(root: Path, evidence: dict[str, Any]) -> dict[str, str]:
    if evidence.get("schema") != RELEASE_EVIDENCE_SCHEMA:
        raise ReleaseValidationError(f"evidence.json schema must be `{RELEASE_EVIDENCE_SCHEMA}`")
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ReleaseValidationError("evidence.artifacts must be an object")

    hashes: dict[str, str] = {}
    for name in HASHED_RELEASE_FILES:
        declared = _sha256(artifacts.get(name), f"evidence.artifacts.{name}")
        actual = _sha256_file(root / name)
        if declared != actual:
            raise ReleaseValidationError(
                f"evidence hash mismatch for `{name}`: declared {declared}, actual {actual}"
            )
        hashes[name] = actual
    return hashes


def _validate_splits(splits: dict[str, Any]) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    if splits.get("schema") != SPLITS_SCHEMA:
        raise ReleaseValidationError(f"splits.json schema must be `{SPLITS_SCHEMA}`")
    examples = splits.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ReleaseValidationError("splits.examples must be a non-empty array")

    by_id: dict[str, dict[str, str]] = {}
    group_splits: dict[str, set[str]] = {}
    counts = {name: 0 for name in SPLIT_NAMES}
    for index, value in enumerate(examples):
        if not isinstance(value, dict):
            raise ReleaseValidationError(f"splits.examples.{index} must be an object")
        example_id = _non_empty_string(value.get("id"), f"splits.examples.{index}.id")
        group_id = _non_empty_string(value.get("group_id"), f"splits.examples.{index}.group_id")
        source = _non_empty_string(value.get("source"), f"splits.examples.{index}.source")
        split = _non_empty_string(value.get("split"), f"splits.examples.{index}.split")
        if split not in SPLIT_NAMES:
            raise ReleaseValidationError(
                f"splits.examples.{index}.split must be one of {SPLIT_NAMES}"
            )
        if example_id in by_id:
            raise ReleaseValidationError(f"splits.examples contains duplicate id `{example_id}`")
        by_id[example_id] = {
            "id": example_id,
            "group_id": group_id,
            "source": source,
            "split": split,
        }
        group_splits.setdefault(group_id, set()).add(split)
        counts[split] += 1

    leaked = sorted(group for group, memberships in group_splits.items() if len(memberships) > 1)
    if leaked:
        raise ReleaseValidationError(
            "splits.json leaks group(s) across train/validation/test: " + ", ".join(leaked)
        )
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise ReleaseValidationError(
            "splits.json must contain examples in every split; empty: " + ", ".join(empty)
        )
    return by_id, counts


def _validate_training_report(
    report: dict[str, Any], hashes: dict[str, str], split_counts: dict[str, int]
) -> None:
    if report.get("schema") != TRAINING_REPORT_SCHEMA:
        raise ReleaseValidationError(
            f"training_report.json schema must be `{TRAINING_REPORT_SCHEMA}`"
        )
    if report.get("completed") is not True:
        raise ReleaseValidationError("training_report.completed must be true")
    _non_empty_string(report.get("run_id"), "training_report.run_id")
    _non_empty_string(report.get("source_commit"), "training_report.source_commit")
    _positive_integer(report.get("optimizer_steps"), "training_report.optimizer_steps")

    backbone = report.get("backbone")
    if not isinstance(backbone, dict):
        raise ReleaseValidationError("training_report.backbone must be an object")
    _non_empty_string(backbone.get("model_id"), "training_report.backbone.model_id")
    _non_empty_string(backbone.get("revision"), "training_report.backbone.revision")

    declared_counts = report.get("split_counts")
    if not isinstance(declared_counts, dict):
        raise ReleaseValidationError("training_report.split_counts must be an object")
    for name in SPLIT_NAMES:
        declared = _positive_integer(
            declared_counts.get(name), f"training_report.split_counts.{name}"
        )
        if declared != split_counts[name]:
            raise ReleaseValidationError(
                f"training_report.split_counts.{name} does not match splits.json"
            )

    checkpoint_hash = _sha256(report.get("checkpoint_sha256"), "training_report.checkpoint_sha256")
    splits_hash = _sha256(report.get("splits_sha256"), "training_report.splits_sha256")
    if checkpoint_hash != hashes["model.safetensors"]:
        raise ReleaseValidationError(
            "training_report.checkpoint_sha256 does not match model.safetensors"
        )
    if splits_hash != hashes["splits.json"]:
        raise ReleaseValidationError("training_report.splits_sha256 does not match splits.json")

    retrieval = report.get("retrieval_objective")
    if not isinstance(retrieval, dict):
        raise ReleaseValidationError("training_report.retrieval_objective must be an object")
    _non_empty_string(retrieval.get("name"), "training_report.retrieval_objective.name")
    weight = _finite_non_negative(
        retrieval.get("weight"), "training_report.retrieval_objective.weight"
    )
    if weight <= 0:
        raise ReleaseValidationError("training_report.retrieval_objective.weight must be positive")
    _positive_integer(
        retrieval.get("positive_pairs"), "training_report.retrieval_objective.positive_pairs"
    )


def _probability_vector(value: Any, length: int, dotted: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ReleaseValidationError(f"{dotted} must contain exactly {length} probabilities")
    result = [_unit_metric(item, f"{dotted}.{index}") for index, item in enumerate(value)]
    if length == len(READINESS_LABELS) and not math.isclose(
        sum(result), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ReleaseValidationError(f"{dotted} must sum to 1")
    return result


def _boolean_vector(value: Any, length: int, dotted: str) -> list[bool]:
    if not isinstance(value, list) or len(value) != length:
        raise ReleaseValidationError(f"{dotted} must contain exactly {length} booleans")
    if any(not isinstance(item, bool) for item in value):
        raise ReleaseValidationError(f"{dotted} must contain only booleans")
    return value


def _binary_target_vector(value: Any, valid: list[bool], dotted: str) -> list[float]:
    if not isinstance(value, list) or len(value) != len(valid):
        raise ReleaseValidationError(
            f"{dotted} must contain exactly {len(valid)} binary targets or nulls"
        )
    result: list[float] = []
    for index, (item, is_valid) in enumerate(zip(value, valid, strict=True)):
        if is_valid and item not in (0, 1, False, True):
            raise ReleaseValidationError(f"{dotted}.{index} must be binary when valid")
        if not is_valid and item is not None and item not in (0, 1, False, True):
            raise ReleaseValidationError(f"{dotted}.{index} must be binary or null")
        result.append(float(item) if item is not None else -100.0)
    return result


def _timestamp_vector(value: Any, valid: list[bool], dotted: str) -> list[float]:
    if not isinstance(value, list) or len(value) != len(valid):
        raise ReleaseValidationError(f"{dotted} must contain exactly {len(valid)} timestamps")
    result: list[float] = []
    for index, (item, is_valid) in enumerate(zip(value, valid, strict=True)):
        if item is None:
            if is_valid:
                raise ReleaseValidationError(f"{dotted}.{index} cannot be null when valid")
            result.append(float("nan"))
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ReleaseValidationError(f"{dotted}.{index} must be numeric or null")
        number = float(item)
        if not math.isfinite(number) or number < 0:
            raise ReleaseValidationError(f"{dotted}.{index} must be finite and non-negative")
        result.append(number)
    return result


def _provenance_kind(value: Any, dotted: str) -> str:
    if not isinstance(value, dict):
        raise ReleaseValidationError(f"{dotted} must be an object with a `kind` field")
    kind = _non_empty_string(value.get("kind"), f"{dotted}.kind")
    if kind not in PROVENANCE_KINDS:
        raise ReleaseValidationError(f"{dotted}.kind must be one of {PROVENANCE_KINDS}")
    return kind


def _validate_predictions(
    document: dict[str, Any],
    split_by_id: dict[str, dict[str, str]],
    hashes: dict[str, str],
) -> dict[str, Any]:
    if document.get("schema") != TEST_PREDICTIONS_SCHEMA:
        raise ReleaseValidationError(
            f"test_predictions.json schema must be `{TEST_PREDICTIONS_SCHEMA}`"
        )
    if document.get("readiness_labels") != list(READINESS_LABELS):
        raise ReleaseValidationError("test_predictions.readiness_labels has an invalid order")
    if document.get("issue_labels") != list(ISSUE_LABELS):
        raise ReleaseValidationError("test_predictions.issue_labels has an invalid order")
    if document.get("boundary_labels") != list(BOUNDARY_LABELS):
        raise ReleaseValidationError("test_predictions.boundary_labels has an invalid order")

    checkpoint_hash = _sha256(
        document.get("checkpoint_sha256"), "test_predictions.checkpoint_sha256"
    )
    splits_hash = _sha256(document.get("splits_sha256"), "test_predictions.splits_sha256")
    if checkpoint_hash != hashes["model.safetensors"]:
        raise ReleaseValidationError(
            "test_predictions.checkpoint_sha256 does not match model.safetensors"
        )
    if splits_hash != hashes["splits.json"]:
        raise ReleaseValidationError("test_predictions.splits_sha256 does not match splits.json")

    examples = document.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ReleaseValidationError("test_predictions.examples must be a non-empty array")
    expected_test_ids = {
        example_id for example_id, row in split_by_id.items() if row["split"] == "test"
    }

    seen: set[str] = set()
    readiness_targets: list[int] = []
    readiness_probabilities: list[list[float]] = []
    readiness_valid: list[bool] = []
    issue_targets: list[list[float]] = []
    issue_probabilities: list[list[float]] = []
    issue_valid: list[list[bool]] = []
    boundary_reference: list[list[float]] = []
    boundary_prediction: list[list[float]] = []
    boundary_reference_valid: list[list[bool]] = []
    boundary_prediction_valid: list[list[bool]] = []
    sources: list[str] = []
    readiness_provenance_kinds: list[str] = []
    issue_provenance_kinds: list[str] = []

    for index, value in enumerate(examples):
        dotted = f"test_predictions.examples.{index}"
        if not isinstance(value, dict):
            raise ReleaseValidationError(f"{dotted} must be an object")
        example_id = _non_empty_string(value.get("id"), f"{dotted}.id")
        if example_id in seen:
            raise ReleaseValidationError(
                f"test_predictions.examples contains duplicate id `{example_id}`"
            )
        seen.add(example_id)
        split_row = split_by_id.get(example_id)
        if split_row is None or split_row["split"] != "test":
            raise ReleaseValidationError(
                f"test prediction `{example_id}` is not a member of the declared test split"
            )
        group_id = _non_empty_string(value.get("group_id"), f"{dotted}.group_id")
        source = _non_empty_string(value.get("source"), f"{dotted}.source")
        if group_id != split_row["group_id"] or source != split_row["source"]:
            raise ReleaseValidationError(
                f"test prediction `{example_id}` group/source does not match splits.json"
            )

        provenance = value.get("label_provenance")
        if not isinstance(provenance, dict):
            raise ReleaseValidationError(f"{dotted}.label_provenance must be an object")
        readiness_provenance = _provenance_kind(
            provenance.get("readiness"), f"{dotted}.label_provenance.readiness"
        )
        issue_provenance = _provenance_kind(
            provenance.get("issues"), f"{dotted}.label_provenance.issues"
        )
        boundary_provenance = _provenance_kind(
            provenance.get("boundaries"), f"{dotted}.label_provenance.boundaries"
        )

        if readiness_provenance in HUMAN_GROUNDED_PROVENANCE:
            review_count = _positive_integer(value.get("review_count"), f"{dotted}.review_count")
            if review_count < 2:
                raise ReleaseValidationError(f"{dotted}.review_count must be at least 2")
            _non_empty_string(value.get("rubric_version"), f"{dotted}.rubric_version")

        readiness_is_valid = value.get("readiness_valid")
        if not isinstance(readiness_is_valid, bool):
            raise ReleaseValidationError(f"{dotted}.readiness_valid must be a boolean")
        target_value = value.get("readiness_target")
        if readiness_is_valid:
            if readiness_provenance not in HUMAN_GROUNDED_PROVENANCE:
                raise ReleaseValidationError(
                    f"{dotted}.readiness_valid requires human or human-derived provenance"
                )
            target = _non_empty_string(target_value, f"{dotted}.readiness_target")
            if target not in READINESS_LABELS:
                raise ReleaseValidationError(
                    f"{dotted}.readiness_target must be one of {READINESS_LABELS}"
                )
            readiness_targets.append(READINESS_LABELS.index(target))
        else:
            if target_value is not None:
                raise ReleaseValidationError(
                    f"{dotted}.readiness_target must be null when readiness_valid is false"
                )
            if readiness_provenance == "programmatic-controlled-corruption":
                raise ReleaseValidationError(
                    f"{dotted}.label_provenance.readiness cannot claim a controlled "
                    "corruption without a readiness target"
                )
            readiness_targets.append(-100)
        readiness_valid.append(readiness_is_valid)
        readiness_provenance_kinds.append(readiness_provenance)
        readiness_probabilities.append(
            _probability_vector(
                value.get("readiness_probabilities"),
                len(READINESS_LABELS),
                f"{dotted}.readiness_probabilities",
            )
        )

        issue_mask = _boolean_vector(
            value.get("issue_valid"), len(ISSUE_LABELS), f"{dotted}.issue_valid"
        )
        if any(issue_mask) and issue_provenance == "unlabeled":
            raise ReleaseValidationError(
                f"{dotted}.label_provenance.issues cannot be unlabeled when issue targets are valid"
            )
        if not any(issue_mask) and issue_provenance != "unlabeled":
            raise ReleaseValidationError(
                f"{dotted}.label_provenance.issues must be unlabeled when no issue target is valid"
            )
        issue_valid.append(issue_mask)
        issue_provenance_kinds.append(issue_provenance)
        issue_targets.append(
            _binary_target_vector(value.get("issue_targets"), issue_mask, f"{dotted}.issue_targets")
        )
        issue_probabilities.append(
            _probability_vector(
                value.get("issue_probabilities"),
                len(ISSUE_LABELS),
                f"{dotted}.issue_probabilities",
            )
        )

        reference_mask = _boolean_vector(
            value.get("boundary_reference_valid"),
            len(BOUNDARY_LABELS),
            f"{dotted}.boundary_reference_valid",
        )
        prediction_mask = _boolean_vector(
            value.get("boundary_prediction_valid"),
            len(BOUNDARY_LABELS),
            f"{dotted}.boundary_prediction_valid",
        )
        if any(
            predicted and not reference
            for predicted, reference in zip(prediction_mask, reference_mask, strict=True)
        ):
            raise ReleaseValidationError(
                f"{dotted}.boundary_prediction_valid cannot evaluate an unknown reference"
            )
        if any(reference_mask) and boundary_provenance not in HUMAN_GROUNDED_PROVENANCE:
            raise ReleaseValidationError(
                f"{dotted}.boundary_reference_valid requires human or human-derived provenance"
            )
        if not any(reference_mask) and boundary_provenance != "unlabeled":
            raise ReleaseValidationError(
                f"{dotted}.label_provenance.boundaries must be unlabeled when no boundary "
                "target is valid"
            )
        boundary_reference_valid.append(reference_mask)
        boundary_prediction_valid.append(prediction_mask)
        boundary_reference.append(
            _timestamp_vector(
                value.get("boundary_reference_s"),
                reference_mask,
                f"{dotted}.boundary_reference_s",
            )
        )
        boundary_prediction.append(
            _timestamp_vector(
                value.get("boundary_prediction_s"),
                prediction_mask,
                f"{dotted}.boundary_prediction_s",
            )
        )
        sources.append(source)

    missing = sorted(expected_test_ids - seen)
    extra = sorted(seen - expected_test_ids)
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("extra " + ", ".join(extra))
        raise ReleaseValidationError(
            "test_predictions.examples must exactly cover the test split ("
            + "; ".join(details)
            + ")"
        )

    readiness_array = np.asarray(readiness_targets, dtype=np.int64)
    readiness_valid_array = np.asarray(readiness_valid, dtype=bool)
    for label_index, label in enumerate(READINESS_LABELS):
        if not np.any(readiness_valid_array & (readiness_array == label_index)):
            raise ReleaseValidationError(
                f"test_predictions.json has no valid `{label}` readiness target"
            )
    issue_target_array = np.asarray(issue_targets, dtype=np.float64)
    issue_valid_array = np.asarray(issue_valid, dtype=bool)
    for issue_index, issue_name in enumerate(ISSUE_LABELS):
        selected = issue_target_array[issue_valid_array[:, issue_index], issue_index]
        if not np.any(selected == 1) or not np.any(selected == 0):
            raise ReleaseValidationError(
                f"test_predictions.json needs positive and negative support for `{issue_name}`"
            )
    boundary_valid_array = np.asarray(boundary_reference_valid, dtype=bool)
    for boundary_index, boundary_name in enumerate(BOUNDARY_LABELS):
        if not np.any(boundary_valid_array[:, boundary_index]):
            raise ReleaseValidationError(
                f"test_predictions.json has no valid `{boundary_name}` boundary target"
            )

    issue_row_valid = np.any(issue_valid_array, axis=1)
    readiness_provenance_array = np.asarray(readiness_provenance_kinds, dtype=object)
    readiness_examples_by_provenance = {
        kind: int(np.count_nonzero(readiness_valid_array & (readiness_provenance_array == kind)))
        for kind in READINESS_EVIDENCE_PROVENANCE
    }
    issue_provenance_array = np.asarray(issue_provenance_kinds, dtype=object)
    issue_examples_by_provenance = {
        kind: int(np.count_nonzero(issue_row_valid & (issue_provenance_array == kind)))
        for kind in ISSUE_EVIDENCE_PROVENANCE
    }
    issues_controlled_corruptions = (
        issue_examples_by_provenance["programmatic-controlled-corruption"] > 0
    )
    if not issues_controlled_corruptions:
        raise ReleaseValidationError(
            "test_predictions.json needs at least one issue-valid row with "
            "programmatic-controlled-corruption provenance"
        )

    return {
        "readiness_targets": readiness_array,
        "readiness_probabilities": np.asarray(readiness_probabilities, dtype=np.float64),
        "readiness_valid": readiness_valid_array,
        "issue_targets": issue_target_array,
        "issue_probabilities": np.asarray(issue_probabilities, dtype=np.float64),
        "issue_valid": issue_valid_array,
        "boundary_reference": np.asarray(boundary_reference, dtype=np.float64),
        "boundary_prediction": np.asarray(boundary_prediction, dtype=np.float64),
        "boundary_reference_valid": boundary_valid_array,
        "boundary_prediction_valid": np.asarray(boundary_prediction_valid, dtype=bool),
        "sources": np.asarray(sources, dtype=object),
        "readiness_human_grounded": bool(np.any(readiness_valid_array)),
        "readiness_examples_by_provenance": readiness_examples_by_provenance,
        "issues_controlled_corruptions": issues_controlled_corruptions,
        "issue_examples_by_provenance": issue_examples_by_provenance,
    }


def _validate_metrics(metrics: dict[str, Any]) -> None:
    if metrics.get("schema") != METRICS_SCHEMA:
        raise ReleaseValidationError(f"metrics.json schema must be `{METRICS_SCHEMA}`")
    absent_metrics = []
    for dotted in REQUIRED_METRICS:
        try:
            value = _dig(metrics, dotted)
        except KeyError:
            absent_metrics.append(dotted)
            continue
        _finite_non_negative(value, dotted)
    if absent_metrics:
        raise ReleaseValidationError(f"metrics.json is missing: {', '.join(absent_metrics)}")
    for dotted in (
        "readiness.macro_f1",
        "issues.macro_auroc",
        "issues.macro_average_precision",
        "boundaries.f1",
        "calibration.ece",
    ):
        _unit_metric(_dig(metrics, dotted), dotted)

    issue_rows = metrics.get("issues", {}).get("per_issue")
    if not isinstance(issue_rows, dict):
        raise ReleaseValidationError("issues.per_issue must report every issue label")
    for label in ISSUE_LABELS:
        row = issue_rows.get(label)
        if not isinstance(row, dict):
            raise ReleaseValidationError(f"issues.per_issue is missing `{label}`")
        _unit_metric(row.get("auroc"), f"issues.per_issue.{label}.auroc")
        _unit_metric(
            row.get("average_precision"),
            f"issues.per_issue.{label}.average_precision",
        )
        _positive_integer(row.get("positives"), f"issues.per_issue.{label}.positives")
        _positive_integer(row.get("negatives"), f"issues.per_issue.{label}.negatives")

    per_class = metrics.get("readiness", {}).get("per_class")
    if not isinstance(per_class, dict):
        raise ReleaseValidationError("readiness.per_class must report KEEP, REVIEW, and REJECT")
    for label in ("KEEP", "REVIEW", "REJECT"):
        row = per_class.get(label)
        if not isinstance(row, dict):
            raise ReleaseValidationError(f"readiness.per_class is missing `{label}`")
        for name in ("precision", "recall", "f1", "support"):
            dotted = f"readiness.per_class.{label}.{name}"
            if name == "support":
                if _non_negative_integer(row.get(name), dotted) == 0:
                    raise ReleaseValidationError(f"metric `{dotted}` must be positive")
            else:
                _unit_metric(row.get(name), dotted)

    confusion = metrics.get("readiness", {}).get("confusion_matrix")
    if (
        not isinstance(confusion, list)
        or len(confusion) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in confusion)
    ):
        raise ReleaseValidationError("readiness.confusion_matrix must be a 3 by 3 array")
    for row_index, row in enumerate(confusion):
        for column_index, value in enumerate(row):
            _non_negative_integer(value, f"readiness.confusion_matrix.{row_index}.{column_index}")

    curve = metrics.get("calibration", {}).get("selective_risk")
    if not isinstance(curve, list) or not curve:
        raise ReleaseValidationError("calibration.selective_risk must contain measured points")
    for index, point in enumerate(curve):
        if not isinstance(point, dict):
            raise ReleaseValidationError(f"calibration.selective_risk.{index} must be an object")
        coverage = _finite_non_negative(
            point.get("coverage"), f"calibration.selective_risk.{index}.coverage"
        )
        risk = _finite_non_negative(point.get("risk"), f"calibration.selective_risk.{index}.risk")
        threshold = _finite_non_negative(
            point.get("threshold"), f"calibration.selective_risk.{index}.threshold"
        )
        if coverage > 1 or risk > 1 or threshold > 1:
            raise ReleaseValidationError(
                "selective-risk coverage, risk, and threshold must lie in [0, 1]"
            )
    _positive_integer(metrics.get("calibration", {}).get("n_bins"), "calibration.n_bins")

    per_boundary = metrics.get("boundaries", {}).get("per_boundary")
    if not isinstance(per_boundary, dict):
        raise ReleaseValidationError("boundaries.per_boundary must report start and end")
    for label in BOUNDARY_LABELS:
        row = per_boundary.get(label)
        if not isinstance(row, dict):
            raise ReleaseValidationError(f"boundaries.per_boundary is missing `{label}`")
        _unit_metric(row.get("f1"), f"boundaries.per_boundary.{label}.f1")
        for name in ("true_positives", "false_positives", "false_negatives"):
            _non_negative_integer(row.get(name), f"boundaries.per_boundary.{label}.{name}")

    evaluation = metrics.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ReleaseValidationError("metrics.json is missing evaluation provenance")
    if "human_reviewed" in evaluation:
        raise ReleaseValidationError(
            "evaluation.human_reviewed is too broad; declare task-specific provenance"
        )
    if evaluation.get("readiness_human_grounded") is not True:
        raise ReleaseValidationError("evaluation.readiness_human_grounded must be true")
    readiness_provenance_counts = evaluation.get("readiness_examples_by_provenance")
    if not isinstance(readiness_provenance_counts, dict) or set(readiness_provenance_counts) != set(
        READINESS_EVIDENCE_PROVENANCE
    ):
        raise ReleaseValidationError(
            "evaluation.readiness_examples_by_provenance must report human and human-derived"
        )
    for kind in READINESS_EVIDENCE_PROVENANCE:
        _non_negative_integer(
            readiness_provenance_counts[kind],
            f"evaluation.readiness_examples_by_provenance.{kind}",
        )
    if evaluation.get("issues_controlled_corruptions") is not True:
        raise ReleaseValidationError("evaluation.issues_controlled_corruptions must be true")
    issue_provenance_counts = evaluation.get("issue_examples_by_provenance")
    if not isinstance(issue_provenance_counts, dict) or set(issue_provenance_counts) != set(
        ISSUE_EVIDENCE_PROVENANCE
    ):
        raise ReleaseValidationError(
            "evaluation.issue_examples_by_provenance must report human, human-derived, "
            "and programmatic-controlled-corruption"
        )
    for kind in ISSUE_EVIDENCE_PROVENANCE:
        _non_negative_integer(
            issue_provenance_counts[kind],
            f"evaluation.issue_examples_by_provenance.{kind}",
        )
    if issue_provenance_counts["programmatic-controlled-corruption"] == 0:
        raise ReleaseValidationError(
            "evaluation.issue_examples_by_provenance must include a controlled-corruption row"
        )
    if evaluation.get("grouped_split") is not True:
        raise ReleaseValidationError("evaluation must declare a grouped split")
    if not isinstance(evaluation.get("per_source"), dict) or not evaluation["per_source"]:
        raise ReleaseValidationError(
            "evaluation.per_source must contain readiness source breakdowns"
        )
    for source, row in evaluation["per_source"].items():
        _non_empty_string(source, "evaluation.per_source key")
        if not isinstance(row, dict):
            raise ReleaseValidationError(f"evaluation.per_source.{source} must be an object")
        _positive_integer(
            row.get("readiness_examples"),
            f"evaluation.per_source.{source}.readiness_examples",
        )
        _unit_metric(
            row.get("readiness_macro_f1"),
            f"evaluation.per_source.{source}.readiness_macro_f1",
        )
    if (
        not isinstance(evaluation.get("annotation_guide"), str)
        or not evaluation["annotation_guide"].strip()
    ):
        raise ReleaseValidationError("evaluation.annotation_guide must identify the rubric")
    evaluation_counts = {}
    for name in (
        "test_examples",
        "readiness_examples",
        "issue_examples",
        "boundary_examples",
    ):
        evaluation_counts[name] = _positive_integer(evaluation.get(name), f"evaluation.{name}")
    if sum(issue_provenance_counts.values()) != evaluation_counts["issue_examples"]:
        raise ReleaseValidationError(
            "evaluation.issue_examples must equal issue_examples_by_provenance"
        )
    if sum(readiness_provenance_counts.values()) != evaluation_counts["readiness_examples"]:
        raise ReleaseValidationError(
            "evaluation.readiness_examples must equal readiness_examples_by_provenance"
        )

    datasets = metrics.get("data", {}).get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ReleaseValidationError("data.datasets must name every training dataset")
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict) or any(
            not isinstance(dataset.get(field), str) or not dataset[field].strip()
            for field in ("name", "license")
        ):
            raise ReleaseValidationError(
                f"data.datasets.{index} must contain non-empty name and license fields"
            )

    if _dig(metrics, "boundaries.tolerance_s") <= 0:
        raise ReleaseValidationError("boundaries.tolerance_s must be positive")
    for name in ("cpu_windows_per_second", "gpu_windows_per_second"):
        if _dig(metrics, f"throughput.{name}") <= 0:
            raise ReleaseValidationError(f"throughput.{name} must be positive")


def _validate_metric_linkages(metrics: dict[str, Any], hashes: dict[str, str]) -> None:
    links = metrics.get("evidence")
    if not isinstance(links, dict):
        raise ReleaseValidationError("metrics.evidence must link raw evaluation artifacts")
    expected = {
        "checkpoint_sha256": hashes["model.safetensors"],
        "splits_sha256": hashes["splits.json"],
        "test_predictions_sha256": hashes["test_predictions.json"],
    }
    for name, digest in expected.items():
        if _sha256(links.get(name), f"metrics.evidence.{name}") != digest:
            raise ReleaseValidationError(f"metrics.evidence.{name} does not match its artifact")


def _recompute_and_validate_metrics(metrics: dict[str, Any], predictions: dict[str, Any]) -> None:
    readiness = readiness_metrics(
        predictions["readiness_targets"],
        predictions["readiness_probabilities"],
        valid_mask=predictions["readiness_valid"],
    )
    _expect_close(metrics["readiness"]["macro_f1"], readiness["macro_f1"], "readiness.macro_f1")
    declared_confusion = metrics["readiness"]["confusion_matrix"]
    recomputed_confusion = readiness["confusion_matrix"].tolist()
    if declared_confusion != recomputed_confusion:
        raise ReleaseValidationError(
            "metric `readiness.confusion_matrix` does not match test_predictions.json"
        )
    for label in READINESS_LABELS:
        declared = metrics["readiness"]["per_class"][label]
        recomputed = readiness["per_class"][label]
        for name in ("precision", "recall", "f1"):
            _expect_close(declared[name], recomputed[name], f"readiness.per_class.{label}.{name}")
        _expect_integer(
            declared["support"],
            int(recomputed["support"]),
            f"readiness.per_class.{label}.support",
        )

    issues = issue_metrics(
        predictions["issue_targets"],
        predictions["issue_probabilities"],
        valid_mask=predictions["issue_valid"],
        issue_names=ISSUE_LABELS,
    )
    _expect_close(metrics["issues"]["macro_auroc"], issues["macro_auroc"], "issues.macro_auroc")
    _expect_close(
        metrics["issues"]["macro_average_precision"],
        issues["macro_average_precision"],
        "issues.macro_average_precision",
    )
    for label in ISSUE_LABELS:
        declared = metrics["issues"]["per_issue"][label]
        recomputed = issues["per_issue"][label]
        _expect_close(declared["auroc"], recomputed["auroc"], f"issues.per_issue.{label}.auroc")
        _expect_close(
            declared["average_precision"],
            recomputed["average_precision"],
            f"issues.per_issue.{label}.average_precision",
        )
        _expect_integer(
            declared["positives"],
            recomputed["positives"],
            f"issues.per_issue.{label}.positives",
        )
        _expect_integer(
            declared["negatives"],
            recomputed["negatives"],
            f"issues.per_issue.{label}.negatives",
        )

    tolerance = float(metrics["boundaries"]["tolerance_s"])
    boundaries = temporal_boundary_metrics(
        predictions["boundary_reference"],
        predictions["boundary_prediction"],
        tolerance_s=tolerance,
        reference_mask=predictions["boundary_reference_valid"],
        prediction_mask=predictions["boundary_prediction_valid"],
        boundary_names=BOUNDARY_LABELS,
    )
    _expect_close(metrics["boundaries"]["f1"], boundaries["micro_f1"], "boundaries.f1")
    for label in BOUNDARY_LABELS:
        declared = metrics["boundaries"]["per_boundary"][label]
        recomputed = boundaries["per_boundary"][label]
        _expect_close(declared["f1"], recomputed["f1"], f"boundaries.per_boundary.{label}.f1")
        for name in ("true_positives", "false_positives", "false_negatives"):
            _expect_integer(
                declared[name],
                recomputed[name],
                f"boundaries.per_boundary.{label}.{name}",
            )

    n_bins = _positive_integer(metrics["calibration"]["n_bins"], "calibration.n_bins")
    ece = expected_calibration_error(
        predictions["readiness_targets"],
        predictions["readiness_probabilities"],
        n_bins=n_bins,
        valid_mask=predictions["readiness_valid"],
    )
    _expect_close(metrics["calibration"]["ece"], ece, "calibration.ece")
    curve = selective_risk_curve(
        predictions["readiness_targets"],
        predictions["readiness_probabilities"],
        valid_mask=predictions["readiness_valid"],
    )
    declared_curve = metrics["calibration"]["selective_risk"]
    if len(declared_curve) != len(curve["coverage"]):
        raise ReleaseValidationError(
            "metric `calibration.selective_risk` does not match test_predictions.json"
        )
    for index, point in enumerate(declared_curve):
        for name in ("coverage", "risk", "threshold"):
            _expect_close(
                point[name], curve[name][index], f"calibration.selective_risk.{index}.{name}"
            )

    evaluation = metrics["evaluation"]
    _expect_integer(
        evaluation["test_examples"],
        len(predictions["readiness_targets"]),
        "evaluation.test_examples",
    )
    _expect_integer(
        evaluation["readiness_examples"],
        int(np.count_nonzero(predictions["readiness_valid"])),
        "evaluation.readiness_examples",
    )
    issue_rows = np.any(predictions["issue_valid"], axis=1)
    _expect_integer(
        evaluation["issue_examples"],
        int(np.count_nonzero(issue_rows)),
        "evaluation.issue_examples",
    )
    boundary_rows = np.any(predictions["boundary_reference_valid"], axis=1)
    _expect_integer(
        evaluation["boundary_examples"],
        int(np.count_nonzero(boundary_rows)),
        "evaluation.boundary_examples",
    )
    for name in ("readiness_human_grounded", "issues_controlled_corruptions"):
        if evaluation[name] is not predictions[name]:
            raise ReleaseValidationError(
                f"evaluation.{name} does not match test_predictions label provenance"
            )
    for kind in READINESS_EVIDENCE_PROVENANCE:
        _expect_integer(
            evaluation["readiness_examples_by_provenance"][kind],
            predictions["readiness_examples_by_provenance"][kind],
            f"evaluation.readiness_examples_by_provenance.{kind}",
        )
    for kind in ISSUE_EVIDENCE_PROVENANCE:
        _expect_integer(
            evaluation["issue_examples_by_provenance"][kind],
            predictions["issue_examples_by_provenance"][kind],
            f"evaluation.issue_examples_by_provenance.{kind}",
        )
    declared_sources = evaluation["per_source"]
    readiness_sources = predictions["sources"][predictions["readiness_valid"]]
    source_names = sorted(set(readiness_sources.tolist()))
    if set(declared_sources) != set(source_names):
        raise ReleaseValidationError(
            "evaluation.per_source keys must exactly match sources with valid readiness targets"
        )
    for source in source_names:
        members = predictions["readiness_valid"] & (predictions["sources"] == source)
        source_readiness = readiness_metrics(
            predictions["readiness_targets"][members],
            predictions["readiness_probabilities"][members],
        )
        row = declared_sources[source]
        if not isinstance(row, dict):
            raise ReleaseValidationError(f"evaluation.per_source.{source} must be an object")
        _expect_integer(
            row.get("readiness_examples"),
            int(np.count_nonzero(members)),
            f"evaluation.per_source.{source}.readiness_examples",
        )
        _expect_close(
            row.get("readiness_macro_f1"),
            source_readiness["macro_f1"],
            f"evaluation.per_source.{source}.readiness_macro_f1",
        )


def _validate_runtime(root: Path, config: dict[str, Any], processor_config: dict[str, Any]) -> None:
    """Load and execute a trusted release candidate through its public APIs."""

    try:
        import torch
        from PIL import Image
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForVideoClassification, AutoProcessor

        with safe_open(root / "model.safetensors", framework="pt", device="cpu") as weights:
            if not list(weights.keys()):
                raise ReleaseValidationError("model.safetensors contains no tensors")

        loaded_config = AutoConfig.from_pretrained(
            root, trust_remote_code=True, local_files_only=True
        )
        processor = AutoProcessor.from_pretrained(
            root, trust_remote_code=True, local_files_only=True
        )
        model = AutoModelForVideoClassification.from_pretrained(
            root, trust_remote_code=True, local_files_only=True
        ).eval()
        num_frames = int(processor_config["num_frames"])
        height = int(processor_config["size"]["height"])
        width = int(processor_config["size"]["width"])
        inputs = processor(
            videos=[Image.new("RGB", (width, height)) for _ in range(num_frames)],
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = model(
                pixel_values=inputs["pixel_values"],
                frame_mask=inputs["frame_mask"],
            )
    except ReleaseValidationError:
        raise
    except Exception as exc:
        raise ReleaseValidationError(f"artifact cannot load and run locally: {exc}") from exc

    if loaded_config.model_type != "egosieve":
        raise ReleaseValidationError("AutoConfig did not resolve EgoSieve")
    if getattr(loaded_config, "num_frames", None) != num_frames:
        raise ReleaseValidationError("model and processor num_frames do not match")
    expected_shapes = {
        "logits": (1, 3),
        "readiness_logits": (1, 3),
        "issue_logits": (1, len(ISSUE_LABELS)),
        "boundary_logits": (1, num_frames, 2),
    }
    for name, shape in expected_shapes.items():
        value = getattr(output, name, None)
        if value is None or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ReleaseValidationError(
                f"forward output `{name}` must be finite with shape {shape}"
            )
    embedding = getattr(output, "clip_embedding", None)
    if embedding is None or embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ReleaseValidationError("forward output `clip_embedding` has an invalid shape")
    if config.get("readiness_labels") != ["KEEP", "REVIEW", "REJECT"]:
        raise ReleaseValidationError("config readiness label order is invalid")
    if config.get("issue_labels") != list(ISSUE_LABELS):
        raise ReleaseValidationError("config issue label order is invalid")


def validate_release(directory: str | Path) -> dict[str, Any]:
    """Fail closed on incomplete, unmeasured, or unloadable release artifacts.

    This executes the candidate's local custom Transformers code. Only run it
    against an artifact produced by a trusted build pipeline.
    """

    root = Path(directory)
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise ReleaseValidationError(f"missing release files: {', '.join(missing)}")
    forbidden = [name for name in FORBIDDEN_FILES if (root / name).exists()]
    if forbidden:
        raise ReleaseValidationError(
            f"artifact is explicitly marked as non-release: {', '.join(forbidden)}"
        )
    empty = [name for name in REQUIRED_FILES if (root / name).stat().st_size == 0]
    if empty:
        raise ReleaseValidationError(f"release files cannot be empty: {', '.join(empty)}")
    missing_code = [name for name in REQUIRED_CODE_FILES if not (root / name).is_file()]
    if missing_code:
        raise ReleaseValidationError(f"missing custom model code: {', '.join(missing_code)}")
    empty_code = [name for name in REQUIRED_CODE_FILES if (root / name).stat().st_size == 0]
    if empty_code:
        raise ReleaseValidationError(f"custom model code cannot be empty: {', '.join(empty_code)}")

    card = (root / "README.md").read_text(encoding="utf-8")
    if UNRESOLVED.search(card):
        raise ReleaseValidationError("model card contains unresolved template variables")
    for metadata in (
        "license: apache-2.0",
        "library_name: transformers",
        "pipeline_tag: video-classification",
    ):
        if metadata not in card:
            raise ReleaseValidationError(f"model card is missing `{metadata}`")

    config = _load_object(root / "config.json")
    processor_config = _load_object(root / "preprocessor_config.json")
    metrics = _load_object(root / "metrics.json")
    training_report = _load_object(root / "training_report.json")
    splits = _load_object(root / "splits.json")
    test_predictions = _load_object(root / "test_predictions.json")
    evidence = _load_object(root / "evidence.json")

    if config.get("model_type") != "egosieve":
        raise ReleaseValidationError("config.json model_type must be `egosieve`")
    if config.get("auto_map") != EXPECTED_MODEL_AUTO_MAP:
        raise ReleaseValidationError("config.json has invalid custom AutoClass mappings")
    if processor_config.get("auto_map") != EXPECTED_PROCESSOR_AUTO_MAP:
        raise ReleaseValidationError("preprocessor_config.json has invalid AutoProcessor mapping")
    if (
        not isinstance(processor_config.get("num_frames"), int)
        or processor_config["num_frames"] <= 0
    ):
        raise ReleaseValidationError("preprocessor num_frames must be a positive integer")
    size = processor_config.get("size")
    if not isinstance(size, dict) or any(
        not isinstance(size.get(axis), int) or size[axis] <= 0 for axis in ("height", "width")
    ):
        raise ReleaseValidationError("preprocessor size must contain positive height and width")

    _validate_metrics(metrics)
    hashes = _validate_evidence(root, evidence)
    split_by_id, split_counts = _validate_splits(splits)
    _validate_training_report(training_report, hashes, split_counts)
    prediction_arrays = _validate_predictions(test_predictions, split_by_id, hashes)
    _validate_metric_linkages(metrics, hashes)
    _recompute_and_validate_metrics(metrics, prediction_arrays)
    _validate_runtime(root, config, processor_config)

    return {
        "config": config,
        "metrics": metrics,
        "training_report": training_report,
        "evidence": evidence,
        "files": list(REQUIRED_FILES + REQUIRED_CODE_FILES),
    }
