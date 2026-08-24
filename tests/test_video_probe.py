from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from egosieve.video.probe import (
    VideoProbeError,
    build_ffprobe_command,
    parse_ffprobe_json,
    parse_rate,
    probe_video,
)


def _probe_payload(**stream_overrides: object) -> dict[str, object]:
    stream = {
        "index": 2,
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": 1920,
        "height": 1080,
        "avg_frame_rate": "30000/1001",
        "r_frame_rate": "30/1",
        "nb_frames": "300",
        "duration": "10.01",
        "start_time": "1.25",
        "time_base": "1/90000",
        "tags": {"rotate": "90"},
    }
    stream.update(stream_overrides)
    return {
        "streams": [stream],
        "format": {
            "duration": "10.02",
            "start_time": "1.2",
            "size": "12345",
            "format_name": "mov,mp4",
        },
    }


def test_parse_ffprobe_preserves_identity_time_rotation_and_dimensions() -> None:
    metadata = parse_ffprobe_json(
        _probe_payload(),
        source_path="relative/source.mp4",
        source_sha256="a" * 64,
    )

    assert metadata.source_path == "relative/source.mp4"
    assert metadata.source_sha256 == "a" * 64
    assert metadata.duration_s == 10.01  # stream duration wins when available
    assert metadata.start_time_s == 1.25
    assert metadata.end_time_s == pytest.approx(11.26)
    assert metadata.encoded_dimensions == (1920, 1080)
    assert metadata.display_dimensions == (1080, 1920)
    assert metadata.rotation_degrees == 90
    assert metadata.fps == pytest.approx(30000 / 1001)
    assert metadata.source_size_bytes == 12345


def test_parse_ffprobe_uses_side_data_and_duration_fallbacks() -> None:
    payload = _probe_payload(
        duration="N/A",
        avg_frame_rate="0/0",
        r_frame_rate="25/1",
        nb_frames="250",
        tags={},
        side_data_list=[{"side_data_type": "Display Matrix", "rotation": -90}],
    )
    payload["format"]["duration"] = "N/A"  # type: ignore[index]
    metadata = parse_ffprobe_json(payload, source_path="clip.mov")

    assert metadata.duration_s == 10
    assert metadata.rotation_degrees == 270
    assert metadata.display_dimensions == (1080, 1920)


@pytest.mark.parametrize("value", [None, "N/A", "0/0", "garbage", 0, -2])
def test_parse_rate_rejects_invalid_rates(value: object) -> None:
    assert parse_rate(value) is None


def test_probe_errors_on_missing_video_stream() -> None:
    with pytest.raises(VideoProbeError, match="no video stream"):
        parse_ffprobe_json({"streams": []}, source_path="empty.mp4")


def test_ffprobe_command_is_an_argument_array_with_an_absolute_operand(tmp_path: Path) -> None:
    source = tmp_path / "-looks-like-an-option.mp4"
    command = build_ffprobe_command(source, ffprobe_bin="custom-ffprobe")

    assert isinstance(command, list)
    assert command[0] == "custom-ffprobe"
    assert command[-1] == str(source.resolve())
    assert command[-1].startswith("/")


def test_probe_runner_never_uses_a_shell_and_hashes_source(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic video placeholder")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append((command, kwargs))
        return CompletedProcess(command, 0, stdout=json.dumps(_probe_payload()), stderr="")

    metadata = probe_video(source, runner=runner)

    assert len(calls) == 1
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["timeout"] == 30.0
    assert metadata.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert metadata.source_path == str(source)


def test_probe_timeout_is_a_domain_error(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"placeholder")

    def runner(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    with pytest.raises(VideoProbeError, match="timeout"):
        probe_video(source, runner=runner, timeout_s=0.01)
