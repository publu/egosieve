from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from egosieve.video import VideoProcessingConfig, VideoProcessor


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg tools are not installed",
)
def test_real_ffmpeg_probe_plan_and_extract(tmp_path: Path) -> None:
    source = tmp_path / "synthetic source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=orange:size=64x48:rate=12",
            "-t",
            "2.25",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            str(source),
        ],
        check=True,
        shell=False,
    )

    processor = VideoProcessor(
        VideoProcessingConfig(
            window_duration_s=1.0,
            stride_s=0.5,
            frames_per_window=3,
            output_size=(32, 32),
        )
    )
    prepared = processor.prepare(source, tmp_path / "frames")

    assert prepared.metadata.duration_s == pytest.approx(2.25, abs=0.1)
    assert prepared.metadata.source_sha256 is not None
    assert len(prepared.plan.windows) >= 3
    assert len(prepared.frames) == len(prepared.plan.samples)
    assert all(Path(frame.path).is_file() for frame in prepared.frames)
    assert len({frame.path for frame in prepared.frames}) == len(prepared.frames)
