"""Versioned, mask-aware training-data parsing.

The JSONL contract deliberately keeps media out of the annotation file: one
line describes one source video and contains zero or more labelled windows.
This module has no dependency beyond NumPy and the Python standard library.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np

SCHEMA_VERSION = "egosieve.training/v1"
READINESS_LABELS = ("KEEP", "REVIEW", "REJECT")
ISSUE_LABELS = (
    "no_hands",
    "low_hand_activity",
    "hand_occlusion",
    "camera_instability",
    "blur",
    "exposure",
    "scene_cut",
    "duplicate_frames",
)
BOUNDARY_LABELS = ("start", "end")


class TrainingDataError(ValueError):
    """Raised when a training-data record violates the public contract."""


def _fail(path: str, message: str) -> None:
    raise TrainingDataError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, f"expected an object, received {type(value).__name__}")
    return value


def _non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "expected a non-empty string")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, f"expected a boolean, received {value!r}")
    return value


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, f"expected a finite number, received {value!r}")
    result = float(value)
    if not math.isfinite(result):
        _fail(path, f"expected a finite number, received {value!r}")
    return result


@dataclass(frozen=True)
class TrainingWindow(Mapping[str, Any]):
    """A validated temporal window.

    Label values may be absent.  A value is used only when both the value and
    its corresponding validity flag are present, which prevents an omitted
    issue from silently becoming a negative example.
    """

    start_s: float
    end_s: float
    readiness: str | None = None
    readiness_valid: bool = False
    issues: dict[str, bool | None] = field(default_factory=dict)
    issue_valid: dict[str, bool] = field(default_factory=dict)
    boundaries_s: dict[str, float | None] = field(default_factory=dict)
    boundary_valid: bool | dict[str, bool] = False
    annotator: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.extra)
        result.update({"start_s": self.start_s, "end_s": self.end_s})
        if self.readiness is not None:
            result["readiness"] = self.readiness
        result["readiness_valid"] = self.readiness_valid
        if self.issues or self.issue_valid:
            result["issues"] = dict(self.issues)
            result["issue_valid"] = dict(self.issue_valid)
        if self.boundaries_s or self.boundary_valid:
            result["boundaries_s"] = dict(self.boundaries_s)
            result["boundary_valid"] = (
                dict(self.boundary_valid)
                if isinstance(self.boundary_valid, dict)
                else self.boundary_valid
            )
        if self.annotator is not None:
            result["annotator"] = self.annotator
        return result

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class TrainingRecord(Mapping[str, Any]):
    """One validated source-video record from a training JSONL file."""

    id: str
    group_id: str
    video: str
    license: str
    windows: tuple[TrainingWindow, ...]
    schema: str = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.extra)
        result.update(
            {
                "schema": self.schema,
                "id": self.id,
                "group_id": self.group_id,
                "video": self.video,
                "license": self.license,
                "windows": [window.to_dict() for window in self.windows],
            }
        )
        return result

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


_WINDOW_FIELDS = {
    "start_s",
    "end_s",
    "readiness",
    "readiness_valid",
    "issues",
    "issue_valid",
    "boundaries_s",
    "boundary_valid",
    "annotator",
}
_RECORD_FIELDS = {"schema", "id", "group_id", "video", "license", "windows"}


def _issue_vocabulary(issue_names: Iterable[str], *, path: str) -> tuple[str, ...]:
    if isinstance(issue_names, (str, bytes)):
        _fail(path, "expected an iterable of issue names, not a string")
    try:
        names = tuple(issue_names)
    except TypeError as error:
        raise TrainingDataError(f"{path}: expected an iterable of issue names") from error
    if not names:
        _fail(path, "must contain at least one issue name")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        _fail(path, "must contain only non-empty strings")
    if len(set(names)) != len(names):
        _fail(path, "must not contain duplicates")
    return names


def parse_window(
    value: Mapping[str, Any],
    *,
    path: str = "window",
    issue_names: Iterable[str] = ISSUE_LABELS,
) -> TrainingWindow:
    """Validate and normalize a single window mapping."""

    raw = _mapping(value, path)
    vocabulary = _issue_vocabulary(issue_names, path="issue_names")
    allowed_issues = set(vocabulary)
    if "start_s" not in raw:
        _fail(path, "missing required field 'start_s'")
    if "end_s" not in raw:
        _fail(path, "missing required field 'end_s'")
    start_s = _finite_number(raw["start_s"], f"{path}.start_s")
    end_s = _finite_number(raw["end_s"], f"{path}.end_s")
    if start_s < 0:
        _fail(f"{path}.start_s", "must be non-negative")
    if end_s <= start_s:
        _fail(f"{path}.end_s", "must be greater than start_s")

    readiness_value = raw.get("readiness")
    readiness: str | None
    if readiness_value is None:
        readiness = None
    else:
        readiness = _non_empty_string(readiness_value, f"{path}.readiness")
        if readiness not in READINESS_LABELS:
            _fail(
                f"{path}.readiness",
                f"expected one of {READINESS_LABELS}, received {readiness_value!r}",
            )
    readiness_valid = _boolean(
        raw.get("readiness_valid", readiness is not None),
        f"{path}.readiness_valid",
    )
    if readiness_valid and readiness is None:
        _fail(f"{path}.readiness_valid", "cannot be true when readiness is missing")

    issues_raw = _mapping(raw.get("issues", {}), f"{path}.issues")
    issues: dict[str, bool | None] = {}
    for name, label in issues_raw.items():
        issue_name = _non_empty_string(name, f"{path}.issues key")
        if issue_name not in allowed_issues:
            _fail(
                f"{path}.issues.{issue_name}",
                f"unknown issue; expected one of {vocabulary}. "
                "Pass an explicit issue_names vocabulary for a custom schema.",
            )
        if label is not None and not isinstance(label, bool):
            _fail(f"{path}.issues.{issue_name}", "expected a boolean or null")
        issues[issue_name] = label

    issue_valid_value = raw.get("issue_valid")
    if issue_valid_value is None:
        issue_valid = {name: label is not None for name, label in issues.items()}
    else:
        issue_valid_raw = _mapping(issue_valid_value, f"{path}.issue_valid")
        issue_valid = {}
        for name, valid in issue_valid_raw.items():
            issue_name = _non_empty_string(name, f"{path}.issue_valid key")
            if issue_name not in allowed_issues:
                _fail(
                    f"{path}.issue_valid.{issue_name}",
                    f"unknown issue; expected one of {vocabulary}. "
                    "Pass an explicit issue_names vocabulary for a custom schema.",
                )
            issue_valid[issue_name] = _boolean(valid, f"{path}.issue_valid.{issue_name}")
        for name, valid in issue_valid.items():
            if valid and issues.get(name) is None:
                _fail(
                    f"{path}.issue_valid.{name}",
                    "cannot be true when the corresponding issue label is missing",
                )

    boundaries_raw = _mapping(raw.get("boundaries_s", {}), f"{path}.boundaries_s")
    boundaries_s: dict[str, float | None] = {}
    for name, timestamp in boundaries_raw.items():
        if name not in BOUNDARY_LABELS:
            _fail(
                f"{path}.boundaries_s",
                f"unknown boundary {name!r}; expected one of {BOUNDARY_LABELS}",
            )
        boundaries_s[name] = (
            None if timestamp is None else _finite_number(timestamp, f"{path}.boundaries_s.{name}")
        )
        if boundaries_s[name] is not None and not start_s <= boundaries_s[name] <= end_s:
            _fail(
                f"{path}.boundaries_s.{name}",
                f"must fall within the window [{start_s}, {end_s}]",
            )

    boundary_valid_value = raw.get("boundary_valid")
    if boundary_valid_value is None:
        boundary_valid: bool | dict[str, bool] = {
            name: timestamp is not None for name, timestamp in boundaries_s.items()
        }
    elif isinstance(boundary_valid_value, bool):
        boundary_valid = boundary_valid_value
        if boundary_valid and not any(value is not None for value in boundaries_s.values()):
            _fail(f"{path}.boundary_valid", "cannot be true when boundaries_s is missing")
    else:
        boundary_valid_raw = _mapping(boundary_valid_value, f"{path}.boundary_valid")
        boundary_valid = {}
        for name, valid in boundary_valid_raw.items():
            if name not in BOUNDARY_LABELS:
                _fail(
                    f"{path}.boundary_valid",
                    f"unknown boundary {name!r}; expected one of {BOUNDARY_LABELS}",
                )
            boundary_valid[name] = _boolean(valid, f"{path}.boundary_valid.{name}")
            if boundary_valid[name] and boundaries_s.get(name) is None:
                _fail(
                    f"{path}.boundary_valid.{name}",
                    "cannot be true when the corresponding timestamp is missing",
                )
    start_boundary = boundaries_s.get("start")
    end_boundary = boundaries_s.get("end")
    if start_boundary is not None and end_boundary is not None and end_boundary < start_boundary:
        _fail(f"{path}.boundaries_s.end", "must not precede the start boundary")

    annotator_value = raw.get("annotator")
    annotator = (
        None if annotator_value is None else _non_empty_string(annotator_value, f"{path}.annotator")
    )
    extra = {key: item for key, item in raw.items() if key not in _WINDOW_FIELDS}
    return TrainingWindow(
        start_s=start_s,
        end_s=end_s,
        readiness=readiness,
        readiness_valid=readiness_valid,
        issues=issues,
        issue_valid=issue_valid,
        boundaries_s=boundaries_s,
        boundary_valid=boundary_valid,
        annotator=annotator,
        extra=extra,
    )


def parse_record(
    value: Mapping[str, Any],
    *,
    path: str = "record",
    issue_names: Iterable[str] = ISSUE_LABELS,
) -> TrainingRecord:
    """Validate a mapping as an ``egosieve.training/v1`` record."""

    vocabulary = _issue_vocabulary(issue_names, path="issue_names")
    if isinstance(value, TrainingRecord):
        # Revalidate issue keys when a pre-parsed record crosses an API
        # boundary with a potentially different explicit vocabulary.
        for index, window in enumerate(value.windows):
            parse_window(
                window,
                path=f"{path}.windows[{index}]",
                issue_names=vocabulary,
            )
        return value
    raw = _mapping(value, path)
    required = ("schema", "id", "group_id", "video", "license", "windows")
    for name in required:
        if name not in raw:
            _fail(path, f"missing required field {name!r}")
    schema = _non_empty_string(raw["schema"], f"{path}.schema")
    if schema != SCHEMA_VERSION:
        _fail(
            f"{path}.schema",
            f"unsupported schema {schema!r}; expected {SCHEMA_VERSION!r}",
        )
    record_id = _non_empty_string(raw["id"], f"{path}.id")
    group_id = _non_empty_string(raw["group_id"], f"{path}.group_id")
    video = _non_empty_string(raw["video"], f"{path}.video")
    license_name = _non_empty_string(raw["license"], f"{path}.license")
    windows_value = raw["windows"]
    if not isinstance(windows_value, list):
        _fail(f"{path}.windows", "expected an array")
    windows = tuple(
        parse_window(
            window,
            path=f"{path}.windows[{index}]",
            issue_names=vocabulary,
        )
        for index, window in enumerate(windows_value)
    )
    extra = {key: item for key, item in raw.items() if key not in _RECORD_FIELDS}
    return TrainingRecord(
        schema=schema,
        id=record_id,
        group_id=group_id,
        video=video,
        license=license_name,
        windows=windows,
        extra=extra,
    )


def validate_record(
    value: Mapping[str, Any], *, issue_names: Iterable[str] = ISSUE_LABELS
) -> TrainingRecord:
    """Alias for :func:`parse_record`, useful at API boundaries."""

    return parse_record(value, issue_names=issue_names)


def loads_jsonl(
    text: str,
    *,
    source: str = "<string>",
    issue_names: Iterable[str] = ISSUE_LABELS,
) -> list[TrainingRecord]:
    """Parse training records from JSONL text, skipping empty lines."""

    return _read_jsonl_lines(text.splitlines(), source=source, issue_names=issue_names)


def _read_jsonl_lines(
    lines: Iterable[str],
    *,
    source: str,
    issue_names: Iterable[str] = ISSUE_LABELS,
) -> list[TrainingRecord]:
    vocabulary = _issue_vocabulary(issue_names, path="issue_names")
    records: list[TrainingRecord] = []
    seen_ids: dict[str, int] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        location = f"{source}:{line_number}"
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise TrainingDataError(
                f"{location}: invalid JSON ({error.msg} at column {error.colno})"
            ) from error
        try:
            record = parse_record(value, path=location, issue_names=vocabulary)
        except TrainingDataError:
            raise
        previous_line = seen_ids.get(record.id)
        if previous_line is not None:
            raise TrainingDataError(
                f"{location}: duplicate record id {record.id!r} "
                f"(first seen on line {previous_line})"
            )
        seen_ids[record.id] = line_number
        records.append(record)
    return records


def load_jsonl(
    source: str | Path | TextIO,
    *,
    issue_names: Iterable[str] = ISSUE_LABELS,
) -> list[TrainingRecord]:
    """Load and validate a JSONL path or open text stream."""

    vocabulary = _issue_vocabulary(issue_names, path="issue_names")
    if hasattr(source, "read"):
        stream = source
        name = str(getattr(stream, "name", "<stream>"))
        return _read_jsonl_lines(stream, source=name, issue_names=vocabulary)
    path = Path(source)
    with path.open("r", encoding="utf-8") as stream:
        return _read_jsonl_lines(stream, source=str(path), issue_names=vocabulary)


# Friendly synonym for callers that prefer the verb ``read``.
read_jsonl = load_jsonl


@dataclass(frozen=True)
class TargetArrays(Mapping[str, np.ndarray]):
    """Dense clip targets plus absolute boundary timestamps.

    ``boundary_times_s`` is an intermediate annotation representation, not a
    model label.  Use :func:`egosieve.training.build_sampled_targets` to create
    per-frame ``boundary_labels`` and ``boundary_label_mask`` arrays.
    """

    readiness_labels: np.ndarray
    readiness_label_mask: np.ndarray
    issue_labels: np.ndarray
    issue_label_mask: np.ndarray
    boundary_times_s: np.ndarray
    boundary_time_mask: np.ndarray
    issue_names: tuple[str, ...]
    boundary_names: tuple[str, ...] = BOUNDARY_LABELS

    @property
    def readiness_valid(self) -> np.ndarray:
        return self.readiness_label_mask

    @property
    def issue_valid(self) -> np.ndarray:
        return self.issue_label_mask

    @property
    def boundary_valid(self) -> np.ndarray:
        return self.boundary_time_mask

    @property
    def boundary_labels(self) -> np.ndarray:
        """Backward-compatible alias for the intermediate timestamp array."""

        return self.boundary_times_s

    @property
    def boundary_label_mask(self) -> np.ndarray:
        """Backward-compatible alias for :attr:`boundary_time_mask`."""

        return self.boundary_time_mask

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "readiness_labels": self.readiness_labels,
            "readiness_label_mask": self.readiness_label_mask,
            "issue_labels": self.issue_labels,
            "issue_label_mask": self.issue_label_mask,
            "boundary_times_s": self.boundary_times_s,
            "boundary_time_mask": self.boundary_time_mask,
        }

    def __getitem__(self, key: str) -> np.ndarray:
        aliases = {
            "readiness_valid": "readiness_label_mask",
            "issue_valid": "issue_label_mask",
            "boundary_valid": "boundary_time_mask",
            "boundary_labels": "boundary_times_s",
            "boundary_label_mask": "boundary_time_mask",
        }
        try:
            return getattr(self, aliases.get(key, key))
        except AttributeError as error:
            raise KeyError(key) from error

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())


def encode_windows(
    windows: Iterable[TrainingWindow | Mapping[str, Any]],
    *,
    issue_names: Iterable[str] = ISSUE_LABELS,
    ignore_index: int = -100,
) -> TargetArrays:
    """Convert sparse window annotations to dense NumPy arrays and masks.

    Unknown categorical labels use ``ignore_index``.  Unknown binary and
    timestamp targets are filled with zero; their masks are always false, so
    the fill values cannot turn missing annotations into training targets.
    """

    issues = _issue_vocabulary(issue_names, path="issue_names")
    normalized = tuple(
        parse_window(window, path=f"windows[{index}]", issue_names=issues)
        for index, window in enumerate(windows)
    )

    count = len(normalized)
    readiness_labels = np.full(count, int(ignore_index), dtype=np.int64)
    readiness_mask = np.zeros(count, dtype=bool)
    issue_labels = np.zeros((count, len(issues)), dtype=np.float32)
    issue_mask = np.zeros((count, len(issues)), dtype=bool)
    boundary_times = np.zeros((count, len(BOUNDARY_LABELS)), dtype=np.float32)
    boundary_time_mask = np.zeros((count, len(BOUNDARY_LABELS)), dtype=bool)

    readiness_to_id = {name: index for index, name in enumerate(READINESS_LABELS)}
    for row, window in enumerate(normalized):
        if window.readiness_valid and window.readiness is not None:
            readiness_labels[row] = readiness_to_id[window.readiness]
            readiness_mask[row] = True

        for column, name in enumerate(issues):
            label = window.issues.get(name)
            valid = window.issue_valid.get(name, False) and label is not None
            if valid:
                issue_labels[row, column] = float(label)
                issue_mask[row, column] = True

        for column, name in enumerate(BOUNDARY_LABELS):
            timestamp = window.boundaries_s.get(name)
            if isinstance(window.boundary_valid, bool):
                declared_valid = window.boundary_valid
            else:
                declared_valid = window.boundary_valid.get(name, False)
            if declared_valid and timestamp is not None:
                boundary_times[row, column] = timestamp
                boundary_time_mask[row, column] = True

    return TargetArrays(
        readiness_labels=readiness_labels,
        readiness_label_mask=readiness_mask,
        issue_labels=issue_labels,
        issue_label_mask=issue_mask,
        boundary_times_s=boundary_times,
        boundary_time_mask=boundary_time_mask,
        issue_names=issues,
    )


def encode_records(
    records: Iterable[TrainingRecord | Mapping[str, Any]],
    *,
    issue_names: Iterable[str] = ISSUE_LABELS,
    ignore_index: int = -100,
) -> TargetArrays:
    """Flatten record windows and encode them with :func:`encode_windows`."""

    issues = _issue_vocabulary(issue_names, path="issue_names")
    windows: list[TrainingWindow] = []
    for index, value in enumerate(records):
        record = parse_record(value, path=f"records[{index}]", issue_names=issues)
        windows.extend(record.windows)
    return encode_windows(windows, issue_names=issues, ignore_index=ignore_index)


# Compact aliases for common call sites.
windows_to_arrays = encode_windows
records_to_arrays = encode_records


__all__ = [
    "BOUNDARY_LABELS",
    "ISSUE_LABELS",
    "READINESS_LABELS",
    "SCHEMA_VERSION",
    "TargetArrays",
    "TrainingDataError",
    "TrainingRecord",
    "TrainingWindow",
    "encode_records",
    "encode_windows",
    "load_jsonl",
    "loads_jsonl",
    "parse_record",
    "parse_window",
    "read_jsonl",
    "records_to_arrays",
    "validate_record",
    "windows_to_arrays",
]
