"""Deterministic sliding-window and frame timestamp planning."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import floor, isfinite
from typing import TypeVar

from .models import FrameSample, SamplingPlan, SamplingWindow

T = TypeVar("T")

_TIMESTAMP_DIGITS = 12
_EPSILON = 10 ** (-_TIMESTAMP_DIGITS)


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return value


def _canonical(value: float) -> float:
    # A fixed precision makes independently computed overlap points share one
    # decode request without sacrificing useful timestamp precision.
    return round(float(value), _TIMESTAMP_DIGITS)


def sliding_windows(
    duration_s: float,
    window_duration_s: float,
    stride_s: float,
    *,
    include_tail: bool = True,
) -> tuple[tuple[float, float], ...]:
    """Return full sliding windows and, optionally, one end-anchored tail.

    A source shorter than one window still produces a single clipped window.
    For longer sources, regular windows are laid out at ``stride_s``.  If they
    do not reach the end and ``include_tail`` is true, exactly one final window
    is anchored at the source end; this avoids a sequence of ever-shorter tail
    windows while ensuring the final frames are represented.
    """

    duration_s = _positive("duration_s", duration_s)
    window_duration_s = _positive("window_duration_s", window_duration_s)
    stride_s = _positive("stride_s", stride_s)

    if duration_s <= window_duration_s + _EPSILON:
        return ((0.0, _canonical(duration_s)),)

    last_regular_start = duration_s - window_duration_s
    count = floor((last_regular_start + _EPSILON) / stride_s) + 1
    starts = [_canonical(index * stride_s) for index in range(count)]
    windows = [(start, _canonical(min(duration_s, start + window_duration_s))) for start in starts]

    if include_tail and windows[-1][1] < duration_s - _EPSILON:
        tail_start = _canonical(duration_s - window_duration_s)
        if abs(tail_start - windows[-1][0]) > _EPSILON:
            windows.append((tail_start, _canonical(duration_s)))

    return tuple(windows)


def sample_timestamps(
    start_s: float,
    end_s: float,
    count: int,
    *,
    strategy: str = "center",
) -> tuple[float, ...]:
    """Choose exactly ``count`` deterministic timestamps inside an interval.

    ``center`` samples the center of equal-width bins and is the robust
    default: unlike end-point sampling it never requests the exact EOF.
    ``start`` samples the left edge of each bin. ``inclusive`` includes both
    ends and is mainly useful for callers that know their decoder's EOF rules.
    """

    start_s = float(start_s)
    end_s = float(end_s)
    if not isfinite(start_s) or not isfinite(end_s) or start_s < 0 or end_s <= start_s:
        raise ValueError("times must be finite and satisfy 0 <= start_s < end_s")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if strategy not in {"center", "start", "inclusive"}:
        raise ValueError("strategy must be 'center', 'start', or 'inclusive'")

    span = end_s - start_s
    if strategy == "center":
        values = (start_s + (index + 0.5) * span / count for index in range(count))
    elif strategy == "start":
        values = (start_s + index * span / count for index in range(count))
    elif count == 1:
        values = ((start_s + end_s) / 2.0,)
    else:
        values = (start_s + index * span / (count - 1) for index in range(count))
    return tuple(_canonical(value) for value in values)


def plan_frame_samples(
    duration_s: float,
    *,
    window_duration_s: float,
    stride_s: float,
    frames_per_window: int,
    start_time_s: float = 0.0,
    include_tail: bool = True,
    strategy: str = "center",
) -> SamplingPlan:
    """Build a plan that decodes every unique timestamp at most once.

    Each window always has ``frames_per_window`` references.  Identical
    timestamps in overlapping windows point to the same global sample index,
    so a consumer can decode :attr:`SamplingPlan.samples` once and gather by
    ``sample_indices`` for model batches.
    """

    duration_s = _positive("duration_s", duration_s)
    window_duration_s = _positive("window_duration_s", window_duration_s)
    stride_s = _positive("stride_s", stride_s)
    start_time_s = float(start_time_s)
    if not isfinite(start_time_s):
        raise ValueError("start_time_s must be finite")
    if isinstance(frames_per_window, bool) or not isinstance(frames_per_window, int):
        raise ValueError("frames_per_window must be a positive integer")
    if frames_per_window <= 0:
        raise ValueError("frames_per_window must be a positive integer")

    intervals = sliding_windows(
        duration_s,
        window_duration_s,
        stride_s,
        include_tail=include_tail,
    )
    keys_by_window: list[tuple[float, ...]] = []
    timestamp_by_key: dict[float, float] = {}
    for start_s, end_s in intervals:
        timestamps = sample_timestamps(start_s, end_s, frames_per_window, strategy=strategy)
        keys_by_window.append(timestamps)
        for timestamp in timestamps:
            timestamp_by_key.setdefault(timestamp, timestamp)

    ordered_timestamps = sorted(timestamp_by_key)
    index_by_timestamp = {timestamp: index for index, timestamp in enumerate(ordered_timestamps)}
    samples = tuple(
        FrameSample(
            index=index,
            timestamp_s=timestamp,
            source_timestamp_s=_canonical(start_time_s + timestamp),
        )
        for index, timestamp in enumerate(ordered_timestamps)
    )
    windows = tuple(
        SamplingWindow(
            index=index,
            start_s=start_s,
            end_s=end_s,
            source_start_s=_canonical(start_time_s + start_s),
            source_end_s=_canonical(start_time_s + end_s),
            sample_indices=tuple(index_by_timestamp[key] for key in keys_by_window[index]),
        )
        for index, (start_s, end_s) in enumerate(intervals)
    )
    return SamplingPlan(
        duration_s=duration_s,
        start_time_s=start_time_s,
        window_duration_s=window_duration_s,
        stride_s=stride_s,
        frames_per_window=frames_per_window,
        samples=samples,
        windows=windows,
        sampling_strategy=strategy,
        include_tail=include_tail,
    )


def sample_once(timestamps_s: Sequence[float], sampler: Callable[[float], T]) -> tuple[T, ...]:
    """Invoke ``sampler`` once for each unique timestamp, preserving order.

    The function is useful independently of :func:`plan_frame_samples` when a
    caller supplies timestamps from another planner. Repeated timestamps reuse
    the first sampled value.
    """

    cache: dict[float, T] = {}
    values: list[T] = []
    for raw_timestamp in timestamps_s:
        if not isfinite(float(raw_timestamp)) or float(raw_timestamp) < 0:
            raise ValueError("timestamps must be finite and non-negative")
        timestamp = _canonical(raw_timestamp)
        if timestamp not in cache:
            cache[timestamp] = sampler(timestamp)
        values.append(cache[timestamp])
    return tuple(values)


def materialize_plan(plan: SamplingPlan, sampler: Callable[[float], T]) -> tuple[T, ...]:
    """Materialize the unique samples in a plan exactly once each."""

    return tuple(sampler(sample.timestamp_s) for sample in plan.samples)


def gather_window_values(plan: SamplingPlan, values: Sequence[T]) -> tuple[tuple[T, ...], ...]:
    """Gather unique decoded values into fixed-size per-window batches."""

    if len(values) != len(plan.samples):
        raise ValueError("values must contain one item for every unique sample")
    return tuple(
        tuple(values[sample_index] for sample_index in window.sample_indices)
        for window in plan.windows
    )


# A concise alias for callers that naturally think in terms of a sampling plan.
plan_sampling = plan_frame_samples
