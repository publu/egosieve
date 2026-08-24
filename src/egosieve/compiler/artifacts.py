"""Optional clip and contact-sheet artifacts for compiled segments."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Iterable, Sequence
from math import ceil, isfinite
from pathlib import Path

from egosieve.video.frames import format_timestamp
from egosieve.video.models import ExtractedFrame

from .segments import KEEP, Segment


class ArtifactError(RuntimeError):
    """Raised when an optional media artifact cannot be produced."""


def _absolute(path: os.PathLike[str] | str) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def build_clip_command(
    source_path: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    *,
    start_s: float,
    end_s: float,
    ffmpeg_bin: os.PathLike[str] | str = "ffmpeg",
    reencode: bool = True,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    crf: int = 18,
    preset: str = "medium",
    overwrite: bool = False,
) -> list[str]:
    """Build a shell-free ffmpeg command for one exact segment clip."""

    start_s, end_s = float(start_s), float(end_s)
    if not isfinite(start_s) or not isfinite(end_s) or start_s < 0 or end_s <= start_s:
        raise ValueError("clip times must satisfy 0 <= start_s < end_s")
    if not 0 <= int(crf) <= 63:
        raise ValueError("crf must be between 0 and 63")
    duration = end_s - start_s
    command = [
        os.fspath(ffmpeg_bin),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
    ]
    if not reencode:
        command.extend(["-ss", format_timestamp(start_s)])
    command.extend(["-i", _absolute(source_path)])
    if reencode:
        command.extend(["-ss", format_timestamp(start_s)])
    command.extend(
        [
            "-t",
            format_timestamp(duration),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
        ]
    )
    if reencode:
        command.extend(
            [
                "-c:v",
                str(video_codec),
                "-preset",
                str(preset),
                "-crf",
                str(int(crf)),
                "-c:a",
                str(audio_codec),
                "-movflags",
                "+faststart",
            ]
        )
    else:
        command.extend(["-c", "copy", "-avoid_negative_ts", "make_zero"])
    command.extend(["-y" if overwrite else "-n", _absolute(output_path)])
    return command


def export_segment_clips(
    source_path: os.PathLike[str] | str,
    segments: Iterable[Segment],
    output_dir: os.PathLike[str] | str,
    *,
    routes: Sequence[str] = (KEEP,),
    extension: str = "mp4",
    ffmpeg_bin: os.PathLike[str] | str = "ffmpeg",
    reencode: bool = True,
    overwrite: bool = False,
    verify_outputs: bool = True,
    timeout_s: float | None = 300.0,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[int, Path]:
    """Export selected routes and map source segment indices to clip paths."""

    suffix = extension.lower().lstrip(".")
    if not suffix or not suffix.replace("_", "").isalnum():
        raise ValueError("extension must be a simple filename extension")
    if timeout_s is not None and (not isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError("timeout_s must be finite and positive, or None")
    route_set = frozenset(routes)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: dict[int, Path] = {}
    for index, segment in enumerate(segments):
        if segment.route not in route_set:
            continue
        output = destination / (
            f"segment_{index:04d}_{format_timestamp(segment.start_s).replace('.', '_')}_"
            f"{format_timestamp(segment.end_s).replace('.', '_')}.{suffix}"
        )
        command = build_clip_command(
            source_path,
            output,
            start_s=segment.start_s,
            end_s=segment.end_s,
            ffmpeg_bin=ffmpeg_bin,
            reencode=reencode,
            overwrite=overwrite,
        )
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
            raise ArtifactError(
                f"ffmpeg exceeded its {timeout_s:g}s timeout for segment {index}"
            ) from exc
        except OSError as exc:
            raise ArtifactError(f"could not execute {os.fspath(ffmpeg_bin)!r}: {exc}") from exc
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise ArtifactError(
                f"ffmpeg failed for segment {index}: {stderr[-1000:] or 'unknown error'}"
            )
        if verify_outputs and not output.is_file():
            raise ArtifactError(f"ffmpeg reported success but did not create {output}")
        outputs[index] = output
    return outputs


def contact_sheet_layout(item_count: int, *, columns: int = 4) -> tuple[int, int]:
    """Return ``(rows, columns)`` for a compact fixed-column contact sheet."""

    if isinstance(item_count, bool) or not isinstance(item_count, int) or item_count <= 0:
        raise ValueError("item_count must be a positive integer")
    if isinstance(columns, bool) or not isinstance(columns, int) or columns <= 0:
        raise ValueError("columns must be a positive integer")
    actual_columns = min(columns, item_count)
    return (ceil(item_count / actual_columns), actual_columns)


def write_contact_sheet(
    frames: Sequence[os.PathLike[str] | str | ExtractedFrame],
    output_path: os.PathLike[str] | str,
    *,
    columns: int = 4,
    thumbnail_size: tuple[int, int] = (320, 180),
    labels: Sequence[str] | None = None,
    background: tuple[int, int, int] = (16, 18, 22),
    overwrite: bool = False,
) -> Path:
    """Create a contact sheet from already sampled frames using Pillow lazily.

    Reusing extracted samples avoids decoding source frames a second time.
    ``thumbnail_size`` is ``(width, height)``.
    """

    if not frames:
        raise ValueError("frames must not be empty")
    if labels is not None and len(labels) != len(frames):
        raise ValueError("labels must have one value per frame")
    width, height = (int(thumbnail_size[0]), int(thumbnail_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("thumbnail_size values must be greater than zero")
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - depends on installation extras
        raise RuntimeError("contact sheet generation requires Pillow") from exc

    destination = Path(output_path).expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows, actual_columns = contact_sheet_layout(len(frames), columns=columns)
    label_height = 20 if labels is not None else 0
    sheet = Image.new("RGB", (actual_columns * width, rows * (height + label_height)), background)
    draw = ImageDraw.Draw(sheet)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    for index, item in enumerate(frames):
        path = Path(item.path if isinstance(item, ExtractedFrame) else item)
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            image.thumbnail((width, height), resampling)
            row, column = divmod(index, actual_columns)
            left = column * width + (width - image.width) // 2
            top = row * (height + label_height) + (height - image.height) // 2
            sheet.paste(image, (left, top))
        if labels is not None:
            draw.text(
                (column * width + 4, row * (height + label_height) + height + 2),
                str(labels[index]),
                fill=(235, 238, 242),
            )
    sheet.save(destination)
    return destination
