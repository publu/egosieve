"""Build mask-aware model targets on exact sampled frame timestamps."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import (
    BOUNDARY_LABELS,
    ISSUE_LABELS,
    TargetArrays,
    TrainingWindow,
    encode_windows,
    parse_window,
)


def _tolerance(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("boundary_tolerance_s must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("boundary_tolerance_s must be finite and non-negative")
    return result


def _timestamp_rows(values: Any, window_count: int) -> tuple[np.ndarray, ...]:
    if isinstance(values, np.ndarray):
        if values.ndim == 1:
            if window_count != 1:
                raise ValueError(
                    "one-dimensional sampled_timestamps_s is valid only for one window"
                )
            candidates: list[Any] = [values]
        elif values.ndim == 2:
            candidates = list(values)
        else:
            raise ValueError("sampled_timestamps_s must be one- or two-dimensional")
    else:
        try:
            outer = list(values)
        except TypeError as error:
            raise ValueError("sampled_timestamps_s must be an iterable") from error
        is_flat = bool(outer) and all(np.asarray(item).ndim == 0 for item in outer)
        candidates = [outer] if window_count == 1 and is_flat else outer

    if len(candidates) != window_count:
        raise ValueError(
            "sampled_timestamps_s must contain exactly one timestamp sequence per window; "
            f"received {len(candidates)} for {window_count} windows"
        )
    rows: list[np.ndarray] = []
    for index, candidate in enumerate(candidates):
        raw = np.asarray(candidate)
        if raw.dtype == np.bool_:
            raise ValueError(f"sampled_timestamps_s[{index}] must contain numeric timestamps")
        try:
            row = raw.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"sampled_timestamps_s[{index}] must contain numeric timestamps"
            ) from error
        if row.ndim != 1 or len(row) == 0:
            raise ValueError(
                f"sampled_timestamps_s[{index}] must be a non-empty one-dimensional sequence"
            )
        if np.any(~np.isfinite(row)) or np.any(row < 0):
            raise ValueError(
                f"sampled_timestamps_s[{index}] must contain finite non-negative timestamps"
            )
        if np.any(np.diff(row) <= 0):
            raise ValueError(f"sampled_timestamps_s[{index}] must be strictly increasing")
        rows.append(row)
    return tuple(rows)


def _inside_window(timestamp: float, window: TrainingWindow) -> bool:
    return (
        window.start_s <= timestamp <= window.end_s
        or math.isclose(timestamp, window.start_s, rel_tol=0.0, abs_tol=1e-12)
        or math.isclose(timestamp, window.end_s, rel_tol=0.0, abs_tol=1e-12)
    )


def _boundary_is_valid(window: TrainingWindow, name: str) -> bool:
    if isinstance(window.boundary_valid, bool):
        return window.boundary_valid
    return window.boundary_valid.get(name, False)


@dataclass(frozen=True)
class SampledTargetArrays(Mapping[str, np.ndarray]):
    """Targets aligned to a padded ``[windows, frames]`` timestamp matrix."""

    readiness_labels: np.ndarray
    readiness_label_mask: np.ndarray
    issue_labels: np.ndarray
    issue_label_mask: np.ndarray
    boundary_labels: np.ndarray
    boundary_label_mask: np.ndarray
    frame_mask: np.ndarray
    sampled_timestamps_s: np.ndarray
    boundary_matched_mask: np.ndarray
    boundary_distance_s: np.ndarray
    issue_names: tuple[str, ...]
    boundary_names: tuple[str, ...] = BOUNDARY_LABELS

    def as_model_inputs(self) -> dict[str, np.ndarray]:
        """Return only keyword arguments accepted by ``EgoSieveModel``."""

        return {
            "frame_mask": self.frame_mask,
            "readiness_labels": self.readiness_labels,
            "readiness_label_mask": self.readiness_label_mask,
            "issue_labels": self.issue_labels,
            "issue_label_mask": self.issue_label_mask,
            "boundary_labels": self.boundary_labels,
            "boundary_label_mask": self.boundary_label_mask,
        }

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            **self.as_model_inputs(),
            "sampled_timestamps_s": self.sampled_timestamps_s,
            "boundary_matched_mask": self.boundary_matched_mask,
            "boundary_distance_s": self.boundary_distance_s,
        }

    def __getitem__(self, key: str) -> np.ndarray:
        return self.as_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())


def build_sampled_targets(
    windows: Iterable[TrainingWindow | Mapping[str, Any]],
    sampled_timestamps_s: Sequence[Sequence[float]] | np.ndarray,
    *,
    boundary_tolerance_s: float,
    issue_names: Iterable[str] = ISSUE_LABELS,
    ignore_index: int = -100,
) -> SampledTargetArrays:
    """Rasterize sparse labels onto exact per-window sampled timestamps.

    Each valid annotated boundary is assigned to its nearest sampled frame if
    and only if the absolute timestamp error is at most
    ``boundary_tolerance_s``.  An exact distance tie chooses the earlier frame
    (the first index).  Once assigned, that boundary channel is valid over all
    real frames in the window: the nearest frame is positive and the others
    are known negatives.  Missing, invalid, and out-of-tolerance annotations
    leave the entire channel masked.
    """

    tolerance = _tolerance(boundary_tolerance_s)
    window_values = tuple(windows)
    base: TargetArrays = encode_windows(
        window_values,
        issue_names=issue_names,
        ignore_index=ignore_index,
    )
    normalized = tuple(
        value
        if isinstance(value, TrainingWindow)
        else parse_window(value, path=f"windows[{index}]", issue_names=base.issue_names)
        for index, value in enumerate(window_values)
    )

    timestamp_rows = _timestamp_rows(sampled_timestamps_s, len(normalized))
    for index, (window, timestamps) in enumerate(zip(normalized, timestamp_rows, strict=True)):
        outside = [timestamp for timestamp in timestamps if not _inside_window(timestamp, window)]
        if outside:
            raise ValueError(
                f"sampled_timestamps_s[{index}] contains {outside[0]} outside "
                f"window [{window.start_s}, {window.end_s}]"
            )

    max_frames = max((len(row) for row in timestamp_rows), default=0)
    count = len(normalized)
    timestamps_array = np.full((count, max_frames), np.nan, dtype=np.float64)
    frame_mask = np.zeros((count, max_frames), dtype=bool)
    boundary_labels = np.zeros((count, max_frames, len(BOUNDARY_LABELS)), dtype=np.float32)
    boundary_mask = np.zeros_like(boundary_labels, dtype=bool)
    boundary_matched = np.zeros((count, len(BOUNDARY_LABELS)), dtype=bool)
    boundary_distance = np.full((count, len(BOUNDARY_LABELS)), np.nan, dtype=np.float64)

    for row, (window, timestamps) in enumerate(zip(normalized, timestamp_rows, strict=True)):
        frame_count = len(timestamps)
        timestamps_array[row, :frame_count] = timestamps
        frame_mask[row, :frame_count] = True
        for column, name in enumerate(BOUNDARY_LABELS):
            annotated = window.boundaries_s.get(name)
            if not _boundary_is_valid(window, name) or annotated is None:
                continue
            distances = np.abs(timestamps - annotated)
            nearest = int(np.argmin(distances))
            distance = float(distances[nearest])
            boundary_distance[row, column] = distance
            within_tolerance = distance <= tolerance or math.isclose(
                distance,
                tolerance,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            if not within_tolerance:
                continue
            boundary_labels[row, nearest, column] = 1.0
            boundary_mask[row, :frame_count, column] = True
            boundary_matched[row, column] = True

    return SampledTargetArrays(
        readiness_labels=base.readiness_labels,
        readiness_label_mask=base.readiness_label_mask,
        issue_labels=base.issue_labels,
        issue_label_mask=base.issue_label_mask,
        boundary_labels=boundary_labels,
        boundary_label_mask=boundary_mask,
        frame_mask=frame_mask,
        sampled_timestamps_s=timestamps_array,
        boundary_matched_mask=boundary_matched,
        boundary_distance_s=boundary_distance,
        issue_names=base.issue_names,
    )


@dataclass(frozen=True)
class TrainingTargetBuilder:
    """Reusable configuration for exact-timestamp target construction."""

    boundary_tolerance_s: float
    issue_names: tuple[str, ...] = ISSUE_LABELS
    ignore_index: int = -100

    def __post_init__(self) -> None:
        _tolerance(self.boundary_tolerance_s)
        # An empty build validates the vocabulary without manufacturing data.
        empty = encode_windows((), issue_names=self.issue_names, ignore_index=self.ignore_index)
        object.__setattr__(self, "issue_names", empty.issue_names)

    def __call__(
        self,
        windows: Iterable[TrainingWindow | Mapping[str, Any]],
        sampled_timestamps_s: Sequence[Sequence[float]] | np.ndarray,
    ) -> SampledTargetArrays:
        return build_sampled_targets(
            windows,
            sampled_timestamps_s,
            boundary_tolerance_s=self.boundary_tolerance_s,
            issue_names=self.issue_names,
            ignore_index=self.ignore_index,
        )


@dataclass(frozen=True)
class TrainingCollator:
    """Pad sampled features and attach model-ready masked NumPy targets.

    Each example is a mapping with ``window``, ``sampled_timestamps_s``, and
    optionally either ``frame_embeddings`` or ``pixel_values``.  The optional
    feature array's leading dimension must match its exact timestamps.
    """

    boundary_tolerance_s: float
    issue_names: tuple[str, ...] = ISSUE_LABELS
    ignore_index: int = -100

    def __post_init__(self) -> None:
        builder = TrainingTargetBuilder(
            boundary_tolerance_s=self.boundary_tolerance_s,
            issue_names=self.issue_names,
            ignore_index=self.ignore_index,
        )
        object.__setattr__(self, "issue_names", builder.issue_names)

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
        items = list(examples)
        if not items:
            raise ValueError("TrainingCollator requires at least one example")
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise TypeError(f"examples[{index}] must be a mapping")
            if "window" not in item or "sampled_timestamps_s" not in item:
                raise ValueError(
                    f"examples[{index}] must contain 'window' and 'sampled_timestamps_s'"
                )
        windows = [item["window"] for item in items]
        timestamps = [item["sampled_timestamps_s"] for item in items]
        targets = build_sampled_targets(
            windows,
            timestamps,
            boundary_tolerance_s=self.boundary_tolerance_s,
            issue_names=self.issue_names,
            ignore_index=self.ignore_index,
        )
        batch = targets.as_model_inputs()

        feature_keys = [
            key
            for key in ("frame_embeddings", "pixel_values")
            if any(key in item for item in items)
        ]
        if len(feature_keys) > 1:
            raise ValueError("examples must provide only frame_embeddings or only pixel_values")
        if not feature_keys:
            return batch
        feature_key = feature_keys[0]
        if any(feature_key not in item for item in items):
            raise ValueError(f"every example must provide {feature_key}")

        arrays = [np.asarray(item[feature_key]) for item in items]
        trailing_shape = arrays[0].shape[1:]
        for index, (array, timestamps_row) in enumerate(zip(arrays, timestamps, strict=True)):
            if array.ndim < 2:
                raise ValueError(f"examples[{index}].{feature_key} must have a frame dimension")
            if array.shape[0] != len(timestamps_row):
                raise ValueError(
                    f"examples[{index}].{feature_key} has {array.shape[0]} frames but "
                    f"{len(timestamps_row)} sampled timestamps"
                )
            if array.shape[1:] != trailing_shape:
                raise ValueError(f"all {feature_key} arrays must share their non-frame shape")
        dtype = np.result_type(*(array.dtype for array in arrays))
        padded = np.zeros(
            (len(arrays), targets.frame_mask.shape[1], *trailing_shape),
            dtype=dtype,
        )
        for row, array in enumerate(arrays):
            padded[row, : len(array)] = array
        batch[feature_key] = padded
        return batch


def collate_training_examples(
    examples: Sequence[Mapping[str, Any]],
    *,
    boundary_tolerance_s: float,
    issue_names: tuple[str, ...] = ISSUE_LABELS,
    ignore_index: int = -100,
) -> dict[str, np.ndarray]:
    """Functional wrapper around :class:`TrainingCollator`."""

    return TrainingCollator(
        boundary_tolerance_s=boundary_tolerance_s,
        issue_names=issue_names,
        ignore_index=ignore_index,
    )(examples)


build_frame_targets = build_sampled_targets
MaskAwareTrainingCollator = TrainingCollator


__all__ = [
    "MaskAwareTrainingCollator",
    "SampledTargetArrays",
    "TrainingCollator",
    "TrainingTargetBuilder",
    "build_frame_targets",
    "build_sampled_targets",
    "collate_training_examples",
]
