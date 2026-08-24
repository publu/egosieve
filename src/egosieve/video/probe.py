"""Safe ffprobe invocation and normalization of its JSON output."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping
from math import isfinite
from pathlib import Path
from typing import Any

from .models import VideoMetadata


class VideoProbeError(RuntimeError):
    """Raised when ffprobe fails or returns unusable metadata."""


def _local_path(path: os.PathLike[str] | str) -> str:
    """Return an absolute local path so names beginning with '-' stay operands."""

    return str(Path(path).expanduser().resolve(strict=False))


def build_ffprobe_command(
    source_path: os.PathLike[str] | str, *, ffprobe_bin: os.PathLike[str] | str = "ffprobe"
) -> list[str]:
    """Build an ffprobe argument array; no value is passed through a shell."""

    return [
        os.fspath(ffprobe_bin),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=index,codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate,"
            "codec_type,nb_frames,duration,start_time,time_base:stream_tags=rotate:"
            "stream_side_data=rotation:format=duration,start_time,size,format_name"
        ),
        "-of",
        "json",
        _local_path(source_path),
    ]


def sha256_file(path: os.PathLike[str] | str, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_rate(value: Any) -> float | None:
    """Parse ffprobe decimal or rational rates, including ``N/A`` and ``0/0``."""

    if value is None or isinstance(value, bool):
        return None
    try:
        text = str(value).strip()
        if not text or text.upper() == "N/A":
            return None
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            result = float(numerator) / float(denominator)
        else:
            result = float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if isfinite(result) and result > 0 else None


def _number(value: Any, *, positive: bool = False) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(result) or (positive and result <= 0):
        return None
    return result


def _integer(value: Any, *, positive: bool = False) -> int | None:
    number = _number(value, positive=positive)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _rotation(stream: Mapping[str, Any]) -> float:
    # A stream tag is the clearest representation when present. Newer ffprobe
    # versions commonly expose display-matrix rotation through side data only.
    tags = stream.get("tags")
    if isinstance(tags, Mapping):
        tagged = _number(tags.get("rotate"))
        if tagged is not None:
            return tagged % 360.0
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            if isinstance(item, Mapping):
                value = _number(item.get("rotation"))
                if value is not None:
                    return value % 360.0
    return 0.0


def parse_ffprobe_json(
    payload: str | bytes | Mapping[str, Any],
    *,
    source_path: os.PathLike[str] | str,
    source_sha256: str | None = None,
    source_size_bytes: int | None = None,
) -> VideoMetadata:
    """Normalize ffprobe JSON into :class:`VideoMetadata`.

    Duration and frame rate are selected from multiple ffprobe fields because
    containers and elementary streams do not populate them consistently.
    """

    if isinstance(payload, Mapping):
        document: Mapping[str, Any] = payload
    else:
        try:
            document = json.loads(payload)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
            raise VideoProbeError("ffprobe did not return valid JSON") from exc
    streams = document.get("streams")
    if not isinstance(streams, list) or not streams:
        raise VideoProbeError("ffprobe found no video stream")
    declared_video_streams = [
        item for item in streams if isinstance(item, Mapping) and item.get("codec_type") == "video"
    ]
    mapping_streams = [item for item in streams if isinstance(item, Mapping)]
    if declared_video_streams:
        stream = declared_video_streams[0]
    elif mapping_streams and not any(item.get("codec_type") for item in mapping_streams):
        # ``codec_type`` was not included by older/custom ffprobe invocations;
        # their already-selected first stream remains a valid fallback.
        stream = mapping_streams[0]
    else:
        raise VideoProbeError("ffprobe found no video stream")
    if not isinstance(stream, Mapping):  # defensive narrowing for type checkers
        raise VideoProbeError("ffprobe returned an invalid video stream")
    format_data = document.get("format")
    if not isinstance(format_data, Mapping):
        format_data = {}

    width = _integer(stream.get("width"), positive=True)
    height = _integer(stream.get("height"), positive=True)
    if width is None or height is None:
        raise VideoProbeError("video stream has invalid dimensions")

    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))
    frame_count = _integer(stream.get("nb_frames"), positive=True)
    duration = _number(stream.get("duration"), positive=True) or _number(
        format_data.get("duration"), positive=True
    )
    if duration is None and frame_count is not None and fps is not None:
        duration = frame_count / fps
    if duration is None:
        raise VideoProbeError("video duration is missing or invalid")

    start_time = _number(stream.get("start_time"))
    if start_time is None:
        start_time = _number(format_data.get("start_time"))
    if start_time is None:
        start_time = 0.0

    rotation = _rotation(stream)
    quarter_turn = int(round(rotation / 90.0)) % 4 if abs(rotation % 90.0) < 1e-6 else None
    if quarter_turn in {1, 3}:
        display_width, display_height = height, width
    else:
        display_width, display_height = width, height

    if source_size_bytes is None:
        source_size_bytes = _integer(format_data.get("size"))

    return VideoMetadata(
        source_path=os.fspath(source_path),
        source_sha256=source_sha256,
        source_size_bytes=source_size_bytes,
        duration_s=duration,
        start_time_s=start_time,
        width=width,
        height=height,
        display_width=display_width,
        display_height=display_height,
        rotation_degrees=rotation,
        fps=fps,
        frame_count=frame_count,
        codec_name=str(stream["codec_name"]) if stream.get("codec_name") else None,
        pixel_format=str(stream["pix_fmt"]) if stream.get("pix_fmt") else None,
        time_base=str(stream["time_base"]) if stream.get("time_base") else None,
        format_name=str(format_data["format_name"]) if format_data.get("format_name") else None,
        stream_index=_integer(stream.get("index")) or 0,
    )


def probe_video(
    source_path: os.PathLike[str] | str,
    *,
    ffprobe_bin: os.PathLike[str] | str = "ffprobe",
    calculate_hash: bool = True,
    timeout_s: float | None = 30.0,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> VideoMetadata:
    """Probe a local video and include its SHA-256 identity by default."""

    path = Path(source_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    if timeout_s is not None and (not isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError("timeout_s must be finite and positive, or None")
    command = build_ffprobe_command(path, ffprobe_bin=ffprobe_bin)
    try:
        completed = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoProbeError(f"ffprobe exceeded its {timeout_s:g}s timeout") from exc
    except OSError as exc:
        raise VideoProbeError(f"could not execute {os.fspath(ffprobe_bin)!r}: {exc}") from exc
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        detail = stderr[-1000:] if stderr else "unknown ffprobe error"
        raise VideoProbeError(f"ffprobe failed with exit code {completed.returncode}: {detail}")

    resolved = path.resolve()
    file_hash = sha256_file(resolved) if calculate_hash else None
    return parse_ffprobe_json(
        completed.stdout,
        source_path=os.fspath(source_path),
        source_sha256=file_hash,
        source_size_bytes=resolved.stat().st_size,
    )
