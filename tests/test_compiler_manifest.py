from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from egosieve.compiler.manifest import (
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    ManifestError,
    build_manifest_records,
    read_manifest,
    write_manifest,
)
from egosieve.compiler.segments import KEEP, Segment, WindowScore
from egosieve.video.models import VideoMetadata
from egosieve.video.sampling import plan_frame_samples


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        source_path="input/source.mp4",
        source_sha256="f" * 64,
        source_size_bytes=1000,
        duration_s=8,
        start_time_s=1.5,
        width=1920,
        height=1080,
        display_width=1080,
        display_height=1920,
        rotation_degrees=90,
        fps=30,
        frame_count=240,
        codec_name="h264",
    )


def test_manifest_preserves_source_samples_scores_and_segment_timestamps() -> None:
    metadata = _metadata()
    plan = plan_frame_samples(
        metadata.duration_s,
        start_time_s=metadata.start_time_s,
        window_duration_s=4,
        stride_s=4,
        frames_per_window=2,
    )
    scores = [WindowScore(0, 0, 4, 0.8), WindowScore(1, 4, 8, 0.9)]
    segment = Segment(0, 8, KEEP, 0.85, 0.8, 0.9, (0, 1))
    records = build_manifest_records(
        metadata,
        plan=plan,
        scores=scores,
        segments=[segment],
        artifacts={0: {"clip": "clips/segment.mp4"}},
        created_at=datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc),
    )

    header = records[0]
    assert header["schema"] == MANIFEST_SCHEMA
    assert header["schema_version"] == MANIFEST_VERSION
    assert header["source"]["source_sha256"] == "f" * 64
    assert header["source"]["rotation_degrees"] == 90
    assert (header["source"]["width"], header["source"]["height"]) == (1920, 1080)
    first_window = records[1]
    assert len(first_window["timestamps_s"]) == 2
    assert first_window["source_timestamps_s"] == [2.5, 4.5]
    assert first_window["score"] == 0.8
    segment_record = records[-1]
    assert segment_record["source_start_s"] == 1.5
    assert segment_record["source_end_s"] == 9.5
    assert segment_record["artifacts"]["clip"] == "clips/segment.mp4"


def test_manifest_jsonl_round_trip_is_versioned_and_deterministic(tmp_path: Path) -> None:
    path = write_manifest(
        tmp_path / "nested" / "manifest.jsonl",
        _metadata(),
        created_at="2026-01-02T03:04:00Z",
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["schema_version"] == MANIFEST_VERSION
    assert read_manifest(path)[0]["record_type"] == "manifest"


def test_reader_rejects_unknown_manifest_version(tmp_path: Path) -> None:
    path = tmp_path / "future.jsonl"
    path.write_text(
        json.dumps(
            {
                "record_type": "manifest",
                "schema": MANIFEST_SCHEMA,
                "schema_version": MANIFEST_VERSION + 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="unsupported"):
        read_manifest(path)
