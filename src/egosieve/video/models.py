"""Value objects shared by the video ingestion pipeline.

The objects in this module deliberately contain no decoder or model dependencies.
They are safe to serialize, compare in tests, and pass between worker processes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite
from typing import Any


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _integer(name: str, value: int, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        converted = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not isfinite(numeric) or numeric != converted or converted < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return converted


@dataclass(frozen=True)
class VideoMetadata:
    """Normalized metadata for the first video stream in a source file.

    ``width`` and ``height`` are the encoded pixel dimensions.  The display
    dimensions account for quarter-turn rotation metadata, while the raw
    rotation value remains available in ``rotation_degrees``.
    """

    source_path: str
    duration_s: float
    width: int
    height: int
    display_width: int
    display_height: int
    rotation_degrees: float = 0.0
    source_sha256: str | None = None
    source_size_bytes: int | None = None
    start_time_s: float = 0.0
    fps: float | None = None
    frame_count: int | None = None
    codec_name: str | None = None
    pixel_format: str | None = None
    time_base: str | None = None
    format_name: str | None = None
    stream_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", str(self.source_path))
        object.__setattr__(self, "duration_s", _finite("duration_s", self.duration_s))
        object.__setattr__(self, "start_time_s", _finite("start_time_s", self.start_time_s))
        object.__setattr__(
            self, "rotation_degrees", _finite("rotation_degrees", self.rotation_degrees) % 360.0
        )
        if self.duration_s <= 0:
            raise ValueError("duration_s must be greater than zero")
        for name in ("width", "height", "display_width", "display_height"):
            object.__setattr__(self, name, _integer(name, getattr(self, name), minimum=1))
        if self.fps is not None:
            fps = _finite("fps", self.fps)
            if fps <= 0:
                raise ValueError("fps must be greater than zero when provided")
            object.__setattr__(self, "fps", fps)
        if self.frame_count is not None:
            object.__setattr__(
                self, "frame_count", _integer("frame_count", self.frame_count, minimum=0)
            )
        if self.source_size_bytes is not None:
            object.__setattr__(
                self,
                "source_size_bytes",
                _integer("source_size_bytes", self.source_size_bytes, minimum=0),
            )
        object.__setattr__(self, "stream_index", _integer("stream_index", self.stream_index))

    @property
    def end_time_s(self) -> float:
        """Timestamp on the source stream's time line at the end of the file."""

        return self.start_time_s + self.duration_s

    @property
    def encoded_dimensions(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def display_dimensions(self) -> tuple[int, int]:
        return (self.display_width, self.display_height)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation with explicit dimensions."""

        return {
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "duration_s": self.duration_s,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "width": self.width,
            "height": self.height,
            "display_width": self.display_width,
            "display_height": self.display_height,
            "rotation_degrees": self.rotation_degrees,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "codec_name": self.codec_name,
            "pixel_format": self.pixel_format,
            "time_base": self.time_base,
            "format_name": self.format_name,
            "stream_index": self.stream_index,
        }


@dataclass(frozen=True)
class FrameSample:
    """One unique frame request on both relative and source time lines."""

    index: int
    timestamp_s: float
    source_timestamp_s: float

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("sample index cannot be negative")
        object.__setattr__(self, "timestamp_s", _finite("timestamp_s", self.timestamp_s))
        object.__setattr__(
            self, "source_timestamp_s", _finite("source_timestamp_s", self.source_timestamp_s)
        )
        if self.timestamp_s < 0:
            raise ValueError("timestamp_s cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp_s": self.timestamp_s,
            "source_timestamp_s": self.source_timestamp_s,
        }


@dataclass(frozen=True)
class SamplingWindow:
    """A sliding time window referring to unique samples by integer index."""

    index: int
    start_s: float
    end_s: float
    source_start_s: float
    source_end_s: float
    sample_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("window index cannot be negative")
        object.__setattr__(self, "start_s", _finite("start_s", self.start_s))
        object.__setattr__(self, "end_s", _finite("end_s", self.end_s))
        object.__setattr__(self, "source_start_s", _finite("source_start_s", self.source_start_s))
        object.__setattr__(self, "source_end_s", _finite("source_end_s", self.source_end_s))
        if self.end_s <= self.start_s:
            raise ValueError("window end_s must be greater than start_s")
        if self.source_end_s <= self.source_start_s:
            raise ValueError("window source_end_s must be greater than source_start_s")
        if not self.sample_indices:
            raise ValueError("a sampling window must reference at least one sample")
        if any(index < 0 for index in self.sample_indices):
            raise ValueError("sample indices cannot be negative")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "source_start_s": self.source_start_s,
            "source_end_s": self.source_end_s,
            "duration_s": self.duration_s,
            "sample_indices": list(self.sample_indices),
        }


@dataclass(frozen=True)
class SamplingPlan:
    """All unique frame requests and their sliding-window membership."""

    duration_s: float
    start_time_s: float
    window_duration_s: float
    stride_s: float
    frames_per_window: int
    samples: tuple[FrameSample, ...]
    windows: tuple[SamplingWindow, ...]
    sampling_strategy: str = "center"
    include_tail: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration_s", _finite("duration_s", self.duration_s))
        object.__setattr__(self, "start_time_s", _finite("start_time_s", self.start_time_s))
        object.__setattr__(
            self, "window_duration_s", _finite("window_duration_s", self.window_duration_s)
        )
        object.__setattr__(self, "stride_s", _finite("stride_s", self.stride_s))
        if self.duration_s <= 0 or self.window_duration_s <= 0 or self.stride_s <= 0:
            raise ValueError("durations and stride must be greater than zero")
        if self.frames_per_window <= 0:
            raise ValueError("frames_per_window must be greater than zero")
        if self.sampling_strategy not in {"center", "start", "inclusive"}:
            raise ValueError("sampling_strategy must be 'center', 'start', or 'inclusive'")
        if tuple(sample.index for sample in self.samples) != tuple(range(len(self.samples))):
            raise ValueError("samples must have contiguous, ordered indices")
        if tuple(window.index for window in self.windows) != tuple(range(len(self.windows))):
            raise ValueError("windows must have contiguous, ordered indices")
        for window in self.windows:
            if len(window.sample_indices) != self.frames_per_window:
                raise ValueError("each window must contain exactly frames_per_window references")
            if any(index >= len(self.samples) for index in window.sample_indices):
                raise ValueError("window refers to a sample outside this plan")

    @property
    def timestamps_s(self) -> tuple[float, ...]:
        return tuple(sample.timestamp_s for sample in self.samples)

    @property
    def source_timestamps_s(self) -> tuple[float, ...]:
        return tuple(sample.source_timestamp_s for sample in self.samples)

    def samples_for_window(self, window: int | SamplingWindow) -> tuple[FrameSample, ...]:
        target = self.windows[window] if isinstance(window, int) else window
        return tuple(self.samples[index] for index in target.sample_indices)

    def iter_window_samples(self) -> Iterator[tuple[SamplingWindow, tuple[FrameSample, ...]]]:
        for window in self.windows:
            yield window, self.samples_for_window(window)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "start_time_s": self.start_time_s,
            "window_duration_s": self.window_duration_s,
            "stride_s": self.stride_s,
            "frames_per_window": self.frames_per_window,
            "sampling_strategy": self.sampling_strategy,
            "include_tail": self.include_tail,
            "unique_sample_count": len(self.samples),
            "window_count": len(self.windows),
            "samples": [sample.to_dict() for sample in self.samples],
            "windows": [window.to_dict() for window in self.windows],
        }


@dataclass(frozen=True)
class ExtractedFrame:
    """A materialized frame corresponding to a unique :class:`FrameSample`."""

    sample: FrameSample
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {**self.sample.to_dict(), "path": self.path}
