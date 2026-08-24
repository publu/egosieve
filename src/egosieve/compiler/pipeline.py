"""Small orchestration facade for the pure segment compiler and artifacts."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from egosieve.video.models import SamplingPlan, VideoMetadata

from .artifacts import export_segment_clips
from .manifest import write_manifest
from .segments import Segment, SegmentCompilerConfig, WindowScore, compile_segments


@dataclass(frozen=True)
class Compilation:
    """In-memory compiler result before or after artifact export."""

    metadata: VideoMetadata
    plan: SamplingPlan | None
    scores: tuple[WindowScore, ...]
    segments: tuple[Segment, ...]


class VideoCompiler:
    """Compile externally produced model scores without importing a model."""

    def __init__(self, config: SegmentCompilerConfig | None = None) -> None:
        self.config = config or SegmentCompilerConfig()

    def compile(
        self,
        metadata: VideoMetadata,
        scores: Iterable[WindowScore | Mapping[str, Any] | Sequence[float]],
        *,
        plan: SamplingPlan | None = None,
    ) -> Compilation:
        normalized: list[WindowScore] = []
        for fallback_index, value in enumerate(scores):
            if isinstance(value, WindowScore):
                score = value
            elif isinstance(value, Mapping):
                score = WindowScore.from_mapping(value, fallback_index=fallback_index)
            else:
                fields = tuple(value)
                if len(fields) not in {3, 4}:
                    raise ValueError("score tuples must be (start, end, score[, uncertainty])")
                score = WindowScore(
                    index=fallback_index,
                    start_s=float(fields[0]),
                    end_s=float(fields[1]),
                    score=float(fields[2]),
                    uncertainty=(
                        None if len(fields) == 3 or fields[3] is None else float(fields[3])
                    ),
                )
            if score.end_s > metadata.duration_s + 1e-6:
                raise ValueError(f"window {score.index} extends beyond the source duration")
            normalized.append(score)
        segments = compile_segments(normalized, config=self.config)
        return Compilation(metadata, plan, tuple(normalized), segments)

    def write_manifest(
        self,
        compilation: Compilation,
        path: os.PathLike[str] | str,
        **options: Any,
    ) -> Path:
        return write_manifest(
            path,
            compilation.metadata,
            plan=compilation.plan,
            scores=compilation.scores,
            segments=compilation.segments,
            **options,
        )

    def export_clips(
        self,
        compilation: Compilation,
        output_dir: os.PathLike[str] | str,
        **options: Any,
    ) -> dict[int, Path]:
        return export_segment_clips(
            compilation.metadata.source_path,
            compilation.segments,
            output_dir,
            **options,
        )
