"""Orchestration for probe -> sample plan -> unique frame extraction."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Any

from .frames import extract_plan_frames, frames_to_tensor
from .models import ExtractedFrame, SamplingPlan, VideoMetadata
from .probe import probe_video
from .sampling import gather_window_values, plan_frame_samples


@dataclass(frozen=True)
class VideoProcessingConfig:
    window_duration_s: float = 8.0
    stride_s: float = 4.0
    frames_per_window: int = 8
    include_tail: bool = True
    sampling_strategy: str = "center"
    output_size: tuple[int, int] | None = None
    image_format: str = "jpg"
    probe_timeout_s: float | None = 30.0
    frame_timeout_s: float | None = 120.0
    extraction_batch_size: int = 64

    def __post_init__(self) -> None:
        for name in ("window_duration_s", "stride_s"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if (
            isinstance(self.frames_per_window, bool)
            or not isinstance(self.frames_per_window, int)
            or self.frames_per_window <= 0
        ):
            raise ValueError("frames_per_window must be a positive integer")
        if self.sampling_strategy not in {"center", "start", "inclusive"}:
            raise ValueError("sampling_strategy must be 'center', 'start', or 'inclusive'")
        if self.output_size is not None:
            if len(self.output_size) != 2:
                raise ValueError("output_size must contain width and height")
            for value in self.output_size:
                if (
                    isinstance(value, bool)
                    or not isfinite(float(value))
                    or int(value) != float(value)
                    or int(value) <= 0
                ):
                    raise ValueError("output_size values must be positive integers")
        for name in ("probe_timeout_s", "frame_timeout_s"):
            value = getattr(self, name)
            if value is not None and (not isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f"{name} must be finite and positive, or None")
        if (
            isinstance(self.extraction_batch_size, bool)
            or not isinstance(self.extraction_batch_size, int)
            or self.extraction_batch_size <= 0
        ):
            raise ValueError("extraction_batch_size must be a positive integer")


@dataclass(frozen=True)
class PreparedVideo:
    """Source metadata, immutable sample plan, and unique extracted frames."""

    metadata: VideoMetadata
    plan: SamplingPlan
    frames: tuple[ExtractedFrame, ...]

    def __post_init__(self) -> None:
        if len(self.frames) != len(self.plan.samples):
            raise ValueError("prepared video must have one frame per unique sample")
        for expected, frame in zip(self.plan.samples, self.frames, strict=True):
            if frame.sample != expected:
                raise ValueError("prepared frames must follow the sampling plan order")

    def frame_paths_by_window(self) -> tuple[tuple[str, ...], ...]:
        return gather_window_values(self.plan, [frame.path for frame in self.frames])

    def tensors_by_window(self, **transform_options: Any) -> tuple[Any, ...]:
        """Load each fixed-size window as a lightweight NumPy batch tensor."""

        unique_batch = frames_to_tensor([frame.path for frame in self.frames], **transform_options)
        return tuple(unique_batch[list(window.sample_indices)] for window in self.plan.windows)


class VideoProcessor:
    """Prepare model-agnostic video windows with injectable subprocess calls."""

    def __init__(
        self,
        config: VideoProcessingConfig | None = None,
        *,
        ffprobe_bin: os.PathLike[str] | str = "ffprobe",
        ffmpeg_bin: os.PathLike[str] | str = "ffmpeg",
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.config = config or VideoProcessingConfig()
        self.ffprobe_bin = ffprobe_bin
        self.ffmpeg_bin = ffmpeg_bin
        self.runner = runner

    def probe(
        self, source_path: os.PathLike[str] | str, *, calculate_hash: bool = True
    ) -> VideoMetadata:
        return probe_video(
            source_path,
            ffprobe_bin=self.ffprobe_bin,
            calculate_hash=calculate_hash,
            timeout_s=self.config.probe_timeout_s,
            runner=self.runner,
        )

    def plan(self, metadata: VideoMetadata) -> SamplingPlan:
        return plan_frame_samples(
            metadata.duration_s,
            start_time_s=metadata.start_time_s,
            window_duration_s=self.config.window_duration_s,
            stride_s=self.config.stride_s,
            frames_per_window=self.config.frames_per_window,
            include_tail=self.config.include_tail,
            strategy=self.config.sampling_strategy,
        )

    def extract(
        self,
        source_path: os.PathLike[str] | str,
        plan: SamplingPlan,
        output_dir: os.PathLike[str] | str,
        *,
        overwrite: bool = False,
        verify_outputs: bool = True,
    ) -> tuple[ExtractedFrame, ...]:
        return extract_plan_frames(
            source_path,
            plan,
            output_dir,
            ffmpeg_bin=self.ffmpeg_bin,
            image_format=self.config.image_format,
            output_size=self.config.output_size,
            overwrite=overwrite,
            verify_outputs=verify_outputs,
            timeout_s=self.config.frame_timeout_s,
            batch_size=self.config.extraction_batch_size,
            runner=self.runner,
        )

    def prepare(
        self,
        source_path: os.PathLike[str] | str,
        output_dir: os.PathLike[str] | str,
        *,
        calculate_hash: bool = True,
        overwrite: bool = False,
        verify_outputs: bool = True,
    ) -> PreparedVideo:
        metadata = self.probe(source_path, calculate_hash=calculate_hash)
        plan = self.plan(metadata)
        frames = self.extract(
            source_path,
            plan,
            output_dir,
            overwrite=overwrite,
            verify_outputs=verify_outputs,
        )
        return PreparedVideo(metadata, plan, frames)
