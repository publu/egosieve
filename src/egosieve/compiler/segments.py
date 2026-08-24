"""Turn noisy sliding-window scores into stable, routed time segments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isclose, isfinite
from typing import Any

KEEP = "keep"
REVIEW = "review"
DISCARD = "discard"
ROUTES = frozenset({KEEP, REVIEW, DISCARD})
# Highest to lowest. Ambiguous evidence must remain reviewable; an established
# keep run may bridge a contradictory discard window under ``merge_gap_s``.
ROUTE_PRECEDENCE = (REVIEW, KEEP, DISCARD)
_UNSET = object()


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class WindowScore:
    """A model score associated with one source-relative time window."""

    index: int
    start_s: float
    end_s: float
    score: float
    uncertainty: float | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("window index cannot be negative")
        object.__setattr__(self, "start_s", _finite("start_s", self.start_s))
        object.__setattr__(self, "end_s", _finite("end_s", self.end_s))
        object.__setattr__(self, "score", _finite("score", self.score))
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("window times must satisfy 0 <= start_s < end_s")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between zero and one")
        if self.uncertainty is not None:
            uncertainty = _finite("uncertainty", self.uncertainty)
            if not 0.0 <= uncertainty <= 1.0:
                raise ValueError("uncertainty must be between zero and one")
            object.__setattr__(self, "uncertainty", uncertainty)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, fallback_index: int) -> WindowScore:
        return cls(
            index=int(value.get("index", fallback_index)),
            start_s=float(value["start_s"]),
            end_s=float(value["end_s"]),
            score=float(value["score"]),
            uncertainty=(None if value.get("uncertainty") is None else float(value["uncertainty"])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "score": self.score,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True)
class Segment:
    """A merged compiler decision spanning one or more scored windows."""

    start_s: float
    end_s: float
    route: str
    mean_score: float
    min_score: float
    max_score: float
    window_indices: tuple[int, ...]
    reason: str = "stable"

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_s", _finite("start_s", self.start_s))
        object.__setattr__(self, "end_s", _finite("end_s", self.end_s))
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("segment times must satisfy 0 <= start_s < end_s")
        if self.route not in ROUTES:
            raise ValueError(f"route must be one of {sorted(ROUTES)}")
        for name in ("mean_score", "min_score", "max_score"):
            value = _finite(name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if not self.window_indices:
            raise ValueError("a segment must refer to at least one window")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "route": self.route,
            "reason": self.reason,
            "mean_score": self.mean_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "window_indices": list(self.window_indices),
        }


@dataclass(frozen=True)
class SegmentCompilerConfig:
    """Thresholds and routing policy for :func:`compile_segments`."""

    enter_threshold: float = 0.7
    exit_threshold: float = 0.5
    merge_gap_s: float = 0.5
    min_duration_s: float = 1.0
    uncertainty_threshold: float | None = 0.5
    uncertainty_route: str = REVIEW
    short_segment_route: str = DISCARD
    include_discard: bool = False

    def __post_init__(self) -> None:
        enter = _finite("enter_threshold", self.enter_threshold)
        exit_ = _finite("exit_threshold", self.exit_threshold)
        if not 0.0 <= exit_ <= enter <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= exit <= enter <= 1")
        for name in ("merge_gap_s", "min_duration_s"):
            if _finite(name, getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.uncertainty_threshold is not None:
            uncertainty = _finite("uncertainty_threshold", self.uncertainty_threshold)
            if not 0.0 <= uncertainty <= 1.0:
                raise ValueError("uncertainty_threshold must be between zero and one")
        if self.uncertainty_route not in ROUTES:
            raise ValueError(f"uncertainty_route must be one of {sorted(ROUTES)}")
        if self.short_segment_route not in ROUTES:
            raise ValueError(f"short_segment_route must be one of {sorted(ROUTES)}")


def hysteresis_mask(
    scores: Sequence[float],
    *,
    enter_threshold: float,
    exit_threshold: float,
    initially_active: bool = False,
) -> tuple[bool, ...]:
    """Apply a Schmitt-trigger style state machine to noisy scores.

    An inactive sequence enters at ``score >= enter_threshold``. Once active,
    it remains active while ``score >= exit_threshold``. Equality therefore
    has deterministic, non-flickering behavior at both boundaries.
    """

    config = SegmentCompilerConfig(
        enter_threshold=enter_threshold,
        exit_threshold=exit_threshold,
        merge_gap_s=0,
        min_duration_s=0,
        uncertainty_threshold=None,
    )
    active = bool(initially_active)
    result: list[bool] = []
    for raw_score in scores:
        score = _finite("score", raw_score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("scores must be between zero and one")
        active = score >= config.exit_threshold if active else score >= config.enter_threshold
        result.append(active)
    return tuple(result)


def _coerce_windows(
    windows: Iterable[WindowScore | Mapping[str, Any] | Sequence[float]],
) -> tuple[WindowScore, ...]:
    coerced: list[WindowScore] = []
    for fallback_index, value in enumerate(windows):
        if isinstance(value, WindowScore):
            window = value
        elif isinstance(value, Mapping):
            window = WindowScore.from_mapping(value, fallback_index=fallback_index)
        else:
            fields = tuple(value)
            if len(fields) == 3:
                start_s, end_s, score = fields
                uncertainty = None
            elif len(fields) == 4:
                start_s, end_s, score, uncertainty = fields
            else:
                raise ValueError("window tuples must be (start, end, score[, uncertainty])")
            window = WindowScore(
                index=fallback_index,
                start_s=float(start_s),
                end_s=float(end_s),
                score=float(score),
                uncertainty=None if uncertainty is None else float(uncertainty),
            )
        coerced.append(window)
    ordered = tuple(sorted(coerced, key=lambda item: (item.start_s, item.end_s, item.index)))
    if len({window.index for window in ordered}) != len(ordered):
        raise ValueError("window indices must be unique")
    return ordered


def _segment_from_windows(windows: Sequence[WindowScore], route: str, reason: str) -> Segment:
    scores = [window.score for window in windows]
    return Segment(
        start_s=min(window.start_s for window in windows),
        end_s=max(window.end_s for window in windows),
        route=route,
        mean_score=sum(scores) / len(scores),
        min_score=min(scores),
        max_score=max(scores),
        window_indices=tuple(window.index for window in windows),
        reason=reason,
    )


def _merge_routed_windows(
    routed: Sequence[tuple[WindowScore, str, str]], merge_gap_s: float
) -> list[Segment]:
    grouped: dict[tuple[str, str], list[WindowScore]] = {}
    segments: list[Segment] = []
    # Grouping by route first permits sliding windows on either side of a
    # review decision to reconnect when their actual time intervals overlap.
    for window, route, reason in routed:
        key = (route, reason)
        group = grouped.setdefault(key, [])
        if group and window.start_s > max(item.end_s for item in group) + merge_gap_s:
            segments.append(_segment_from_windows(group, route, reason))
            grouped[key] = [window]
        else:
            group.append(window)
    for (route, reason), group in grouped.items():
        if group:
            segments.append(_segment_from_windows(group, route, reason))
    return segments


def merge_segments(segments: Iterable[Segment], *, max_gap_s: float) -> tuple[Segment, ...]:
    """Merge same-route segments separated by no more than ``max_gap_s``."""

    max_gap_s = _finite("max_gap_s", max_gap_s)
    if max_gap_s < 0:
        raise ValueError("max_gap_s cannot be negative")
    ordered = sorted(segments, key=lambda item: (item.route, item.start_s, item.end_s))
    merged: list[Segment] = []
    for segment in ordered:
        if (
            merged
            and merged[-1].route == segment.route
            and segment.start_s <= merged[-1].end_s + max_gap_s
        ):
            previous = merged.pop()
            prior_count = len(previous.window_indices)
            next_count = len(segment.window_indices)
            total_count = prior_count + next_count
            merged.append(
                Segment(
                    start_s=min(previous.start_s, segment.start_s),
                    end_s=max(previous.end_s, segment.end_s),
                    route=segment.route,
                    mean_score=(previous.mean_score * prior_count + segment.mean_score * next_count)
                    / total_count,
                    min_score=min(previous.min_score, segment.min_score),
                    max_score=max(previous.max_score, segment.max_score),
                    window_indices=tuple(
                        dict.fromkeys(previous.window_indices + segment.window_indices)
                    ),
                    reason=(previous.reason if previous.reason == segment.reason else "merged"),
                )
            )
        else:
            merged.append(segment)
    return tuple(sorted(merged, key=lambda item: (item.start_s, item.end_s, item.route)))


def _segment_for_interval(
    start_s: float,
    end_s: float,
    route: str,
    window_indices: Iterable[int],
    windows_by_index: Mapping[int, WindowScore],
    reason: str,
) -> Segment:
    requested_indices = frozenset(window_indices)
    indices = tuple(
        window.index for window in windows_by_index.values() if window.index in requested_indices
    )
    if not indices:
        raise ValueError("a partition interval must retain supporting windows")
    scores = [windows_by_index[index].score for index in indices]
    return Segment(
        start_s=start_s,
        end_s=end_s,
        route=route,
        mean_score=sum(scores) / len(scores),
        min_score=min(scores),
        max_score=max(scores),
        window_indices=indices,
        reason=reason,
    )


def _merge_adjacent_partition(
    segments: Sequence[Segment], windows_by_index: Mapping[int, WindowScore]
) -> tuple[Segment, ...]:
    """Coalesce adjacent equal routes without jumping across another route."""

    merged: list[Segment] = []
    for segment in sorted(segments, key=lambda item: (item.start_s, item.end_s)):
        if (
            merged
            and merged[-1].route == segment.route
            and isclose(merged[-1].end_s, segment.start_s, rel_tol=0.0, abs_tol=1e-9)
        ):
            previous = merged.pop()
            merged.append(
                _segment_for_interval(
                    previous.start_s,
                    segment.end_s,
                    segment.route,
                    (*previous.window_indices, *segment.window_indices),
                    windows_by_index,
                    (previous.reason if previous.reason == segment.reason else "merged"),
                )
            )
        else:
            merged.append(segment)
    return tuple(merged)


def _partition_routed_segments(
    segments: Sequence[Segment], windows: Sequence[WindowScore]
) -> tuple[Segment, ...]:
    """Resolve overlaps into an exclusive partition of the routed time union."""

    if not segments:
        return ()
    windows_by_index = {window.index: window for window in windows}
    boundaries = sorted(
        {value for segment in segments for value in (segment.start_s, segment.end_s)}
    )
    atoms: list[Segment] = []
    for start_s, end_s in zip(boundaries, boundaries[1:], strict=False):
        if end_s <= start_s:
            continue
        covering = [
            segment for segment in segments if segment.start_s <= start_s and segment.end_s >= end_s
        ]
        if not covering:
            continue
        winning_route = min({segment.route for segment in covering}, key=ROUTE_PRECEDENCE.index)
        winners = [segment for segment in covering if segment.route == winning_route]
        supporting = {
            index
            for segment in winners
            for index in segment.window_indices
            if windows_by_index[index].start_s < end_s and windows_by_index[index].end_s > start_s
        }
        # A merge-gap bridge intentionally covers time with no direct window.
        # Keep its endpoint evidence attached instead of fabricating a score.
        if not supporting:
            supporting = {index for segment in winners for index in segment.window_indices}
        reasons = {segment.reason for segment in winners}
        atoms.append(
            _segment_for_interval(
                start_s,
                end_s,
                winning_route,
                supporting,
                windows_by_index,
                reasons.pop() if len(reasons) == 1 else "resolved",
            )
        )
    return _merge_adjacent_partition(atoms, windows_by_index)


def compile_segments(
    windows: Iterable[WindowScore | Mapping[str, Any] | Sequence[float]],
    *,
    config: SegmentCompilerConfig | None = None,
    enter_threshold: float | None = None,
    exit_threshold: float | None = None,
    merge_gap_s: float | None = None,
    min_duration_s: float | None = None,
    uncertainty_threshold: float | None | object = _UNSET,
    uncertainty_route: str | None = None,
    short_segment_route: str | None = None,
    include_discard: bool | None = None,
) -> tuple[Segment, ...]:
    """Compile scored windows into stable ``keep``/``review`` segments.

    The binary selection state is stabilized first with hysteresis. A window is
    uncertain when either its explicit uncertainty reaches the configured
    threshold, or its score lies strictly between the exit and enter thresholds.
    Such windows follow ``uncertainty_route``. Same-route windows are merged,
    then overlaps are split into a non-overlapping timeline partition using
    the explicit ``review > keep > discard`` precedence. Keep fragments shorter
    than ``min_duration_s`` follow ``short_segment_route``. Discard segments are
    omitted by default; their omission can leave gaps but never overlaps.

    Individual keyword overrides are convenient for small scripts; passing a
    config keeps production policies explicit. Passing
    ``uncertainty_threshold=None`` disables explicit uncertainty routing while
    retaining score-band routing.
    """

    base = config or SegmentCompilerConfig()
    overrides: dict[str, Any] = {}
    for name, value in (
        ("enter_threshold", enter_threshold),
        ("exit_threshold", exit_threshold),
        ("merge_gap_s", merge_gap_s),
        ("min_duration_s", min_duration_s),
        ("uncertainty_route", uncertainty_route),
        ("short_segment_route", short_segment_route),
        ("include_discard", include_discard),
    ):
        if value is not None:
            overrides[name] = value
    if uncertainty_threshold is not _UNSET:
        overrides["uncertainty_threshold"] = uncertainty_threshold
    policy = replace(base, **overrides) if overrides else base

    ordered = _coerce_windows(windows)
    if not ordered:
        return ()
    active = hysteresis_mask(
        [window.score for window in ordered],
        enter_threshold=policy.enter_threshold,
        exit_threshold=policy.exit_threshold,
    )

    routed: list[tuple[WindowScore, str, str]] = []
    for window, selected in zip(ordered, active, strict=True):
        in_threshold_band = policy.exit_threshold < window.score < policy.enter_threshold
        explicit_uncertainty = (
            policy.uncertainty_threshold is not None
            and window.uncertainty is not None
            and window.uncertainty >= policy.uncertainty_threshold
        )
        if in_threshold_band or explicit_uncertainty:
            route, reason = policy.uncertainty_route, "uncertain"
        elif selected:
            route, reason = KEEP, "hysteresis"
        else:
            route, reason = DISCARD, "below_threshold"
        routed.append((window, route, reason))

    candidates = _merge_routed_windows(routed, policy.merge_gap_s)
    candidates = list(merge_segments(candidates, max_gap_s=policy.merge_gap_s))
    segments = list(_partition_routed_segments(candidates, ordered))
    routed_short: list[Segment] = []
    for segment in segments:
        if segment.route == KEEP and segment.duration_s < policy.min_duration_s:
            routed_short.append(replace(segment, route=policy.short_segment_route, reason="short"))
        else:
            routed_short.append(segment)
    windows_by_index = {window.index: window for window in ordered}
    segments = list(_merge_adjacent_partition(routed_short, windows_by_index))
    if not policy.include_discard:
        segments = [segment for segment in segments if segment.route != DISCARD]
    return tuple(sorted(segments, key=lambda item: (item.start_s, item.end_s, item.route)))


# More domain-specific spelling for pipeline call sites.
stabilize_segments = compile_segments
