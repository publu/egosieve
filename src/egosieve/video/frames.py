"""Frame extraction command construction and lightweight tensor transforms."""

from __future__ import annotations

import io
import os
import subprocess
from collections.abc import Callable, Sequence
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import ExtractedFrame, SamplingPlan

if TYPE_CHECKING:  # NumPy remains a lightweight, lazy runtime dependency.
    import numpy as np


class FrameExtractionError(RuntimeError):
    """Raised when ffmpeg cannot materialize a requested frame."""


def format_timestamp(seconds: float) -> str:
    """Format a timestamp for ffmpeg without locale or exponent notation."""

    seconds = float(seconds)
    if seconds < 0:
        raise ValueError("timestamp must not be negative")
    value = f"{seconds:.9f}".rstrip("0").rstrip(".")
    return value or "0"


def _absolute(path: os.PathLike[str] | str) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def build_frame_extract_command(
    source_path: os.PathLike[str] | str,
    timestamp_s: float,
    output_path: os.PathLike[str] | str,
    *,
    ffmpeg_bin: os.PathLike[str] | str = "ffmpeg",
    output_size: tuple[int, int] | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Build a safe, accurate single-frame ffmpeg command.

    Seeking after opening the input is slower than input seeking but yields a
    more accurate frame around non-keyframe timestamps. FFmpeg's normal display
    matrix autorotation remains enabled. ``output_size`` is ``(width, height)``.
    """

    command = [
        os.fspath(ffmpeg_bin),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        _absolute(source_path),
        "-ss",
        format_timestamp(timestamp_s),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
    ]
    if output_size is not None:
        width, height = output_size
        if width <= 0 or height <= 0:
            raise ValueError("output_size values must be greater than zero")
        # force_original_aspect_ratio and even dimensions keep the transform
        # valid for common encoders while padding deterministically.
        filter_graph = (
            f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=decrease,"
            f"pad={int(width)}:{int(height)}:(ow-iw)/2:(oh-ih)/2"
        )
        command.extend(["-vf", filter_graph])
    command.extend(["-y" if overwrite else "-n", _absolute(output_path)])
    return command


def build_frame_batch_extract_command(
    source_path: os.PathLike[str] | str,
    requests: Sequence[tuple[float, os.PathLike[str] | str]],
    *,
    ffmpeg_bin: os.PathLike[str] | str = "ffmpeg",
    output_size: tuple[int, int] | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Build one ffmpeg invocation for several ordered frame requests.

    The input seeks to the first requested timestamp, then each output selects
    its relative presentation time. Chunking extraction this way avoids one
    process per frame while keeping command size and per-process work bounded.
    """

    if not requests:
        raise ValueError("requests must not be empty")
    normalized = [(float(timestamp), output) for timestamp, output in requests]
    if any(timestamp < 0 for timestamp, _ in normalized):
        raise ValueError("timestamps must not be negative")
    if any(
        current[0] < previous[0]
        for previous, current in zip(normalized, normalized[1:], strict=False)
    ):
        raise ValueError("requests must be ordered by timestamp")
    if output_size is not None:
        width, height = output_size
        if width <= 0 or height <= 0:
            raise ValueError("output_size values must be greater than zero")
        filter_graph = (
            f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=decrease,"
            f"pad={int(width)}:{int(height)}:(ow-iw)/2:(oh-ih)/2"
        )
    else:
        filter_graph = None

    origin = normalized[0][0]
    command = [
        os.fspath(ffmpeg_bin),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y" if overwrite else "-n",
    ]
    if origin > 0:
        command.extend(["-ss", format_timestamp(origin)])
    command.extend(["-i", _absolute(source_path)])
    for timestamp, output in normalized:
        command.extend(
            [
                "-ss",
                format_timestamp(timestamp - origin),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
            ]
        )
        if filter_graph is not None:
            command.extend(["-vf", filter_graph])
        command.append(_absolute(output))
    return command


def extract_plan_frames(
    source_path: os.PathLike[str] | str,
    plan: SamplingPlan,
    output_dir: os.PathLike[str] | str,
    *,
    ffmpeg_bin: os.PathLike[str] | str = "ffmpeg",
    image_format: str = "jpg",
    output_size: tuple[int, int] | None = None,
    overwrite: bool = False,
    verify_outputs: bool = True,
    timeout_s: float | None = 120.0,
    batch_size: int = 64,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[ExtractedFrame, ...]:
    """Extract every unique sample in ``plan`` exactly once.

    Overlapping windows only refer to the resulting frame records; they never
    trigger duplicate ffmpeg calls. The runner is injectable for unit tests.
    """

    suffix = image_format.lower().lstrip(".")
    if not suffix or not suffix.replace("_", "").isalnum():
        raise ValueError("image_format must be a simple filename extension")
    if timeout_s is not None and (not isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError("timeout_s must be finite and positive, or None")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    destination = Path(output_dir).expanduser().resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    frames: list[ExtractedFrame] = []
    for sample in plan.samples:
        name = f"frame_{sample.index:06d}_{format_timestamp(sample.timestamp_s).replace('.', '_')}.{suffix}"
        output_path = destination / name
        frames.append(ExtractedFrame(sample=sample, path=str(output_path)))

    for offset in range(0, len(frames), batch_size):
        chunk = frames[offset : offset + batch_size]
        command = build_frame_batch_extract_command(
            source_path,
            [(frame.sample.timestamp_s, frame.path) for frame in chunk],
            ffmpeg_bin=ffmpeg_bin,
            output_size=output_size,
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
            raise FrameExtractionError(
                f"ffmpeg exceeded its {timeout_s:g}s timeout for samples "
                f"{chunk[0].sample.index}-{chunk[-1].sample.index}"
            ) from exc
        except OSError as exc:
            raise FrameExtractionError(
                f"could not execute {os.fspath(ffmpeg_bin)!r}: {exc}"
            ) from exc
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            detail = stderr[-1000:] if stderr else "unknown ffmpeg error"
            raise FrameExtractionError(
                f"ffmpeg failed for samples {chunk[0].sample.index}-"
                f"{chunk[-1].sample.index}: {detail}"
            )
        if verify_outputs:
            missing = [frame.path for frame in chunk if not Path(frame.path).is_file()]
            if missing:
                raise FrameExtractionError(
                    f"ffmpeg reported success but did not create {len(missing)} frame(s); "
                    f"first missing: {missing[0]}"
                )
    return tuple(frames)


def _import_numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - depends on installation extras
        raise RuntimeError("frame tensor conversion requires NumPy") from exc
    return numpy


def _decode_image(frame: Any, np: Any) -> Any:
    if isinstance(frame, np.ndarray):
        return frame
    if isinstance(frame, (bytes, bytearray, memoryview)):
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - depends on installation extras
            raise RuntimeError("decoding encoded image bytes requires Pillow") from exc
        with Image.open(io.BytesIO(bytes(frame))) as image:
            return np.asarray(image.convert("RGB"))
    if isinstance(frame, (str, os.PathLike)):
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - depends on installation extras
            raise RuntimeError("loading an image path requires Pillow") from exc
        with Image.open(frame) as image:
            return np.asarray(image.convert("RGB"))
    # Pillow Image objects and other array-protocol values land here without a
    # top-level Pillow import.
    try:
        return np.asarray(frame)
    except Exception as exc:  # pragma: no cover - NumPy controls exact exception
        raise TypeError("frame must be an image path, encoded bytes, or array-like") from exc


def _as_hwc_rgb(array: Any, np: Any, input_layout: str) -> Any:
    if input_layout not in {"HWC", "CHW"}:
        raise ValueError("input_layout must be 'HWC' or 'CHW'")
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError("a frame must have two spatial dimensions and optional channels")
    if input_layout == "CHW":
        array = np.transpose(array, (1, 2, 0))
    channels = array.shape[2]
    if channels == 1:
        array = np.repeat(array, 3, axis=2)
    elif channels == 4:
        array = array[:, :, :3]
    elif channels != 3:
        raise ValueError("a frame must contain 1, 3, or 4 channels")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("a frame cannot have an empty spatial dimension")
    return array


def _resize_bilinear(array: Any, size: tuple[int, int], np: Any) -> Any:
    """Dependency-free bilinear resize for an HWC float array.

    ``size`` follows tensor convention and is ``(height, width)``.
    """

    out_h, out_w = (int(size[0]), int(size[1]))
    if out_h <= 0 or out_w <= 0:
        raise ValueError("size values must be greater than zero")
    in_h, in_w = array.shape[:2]
    if (out_h, out_w) == (in_h, in_w):
        return array

    ys = np.linspace(0.0, max(in_h - 1, 0), out_h)
    xs = np.linspace(0.0, max(in_w - 1, 0), out_w)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0).reshape(out_h, 1, 1)
    wx = (xs - x0).reshape(1, out_w, 1)
    top = array[y0[:, None], x0[None, :]] * (1.0 - wx) + array[y0[:, None], x1[None, :]] * wx
    bottom = array[y1[:, None], x0[None, :]] * (1.0 - wx) + array[y1[:, None], x1[None, :]] * wx
    return top * (1.0 - wy) + bottom * wy


def frame_to_tensor(
    frame: Any,
    *,
    size: tuple[int, int] | None = None,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    input_layout: str = "HWC",
    output_layout: str = "CHW",
    dtype: str = "float32",
) -> np.ndarray:
    """Convert one RGB-like frame to a NumPy tensor without model imports.

    Integer pixels are scaled by their dtype's maximum into ``[0, 1]``.
    Floating inputs are assumed already scaled. Optional normalization is
    channel-wise and requires both ``mean`` and ``std``.
    """

    np = _import_numpy()
    array = _as_hwc_rgb(_decode_image(frame, np), np, input_layout)
    target_dtype = np.dtype(dtype)
    if not np.issubdtype(target_dtype, np.floating):
        raise ValueError("dtype must be a floating NumPy dtype")
    original_dtype = array.dtype
    if np.issubdtype(original_dtype, np.integer):
        maximum = float(np.iinfo(original_dtype).max)
        array = array.astype(dtype, copy=False) / maximum
    elif np.issubdtype(original_dtype, np.bool_) or np.issubdtype(original_dtype, np.floating):
        array = array.astype(dtype, copy=False)
    else:
        raise TypeError(f"unsupported pixel dtype: {original_dtype}")

    if size is not None:
        array = _resize_bilinear(array, size, np).astype(dtype, copy=False)
    if (mean is None) != (std is None):
        raise ValueError("mean and std must either both be provided or both be omitted")
    if mean is not None and std is not None:
        mean_array = np.asarray(mean, dtype=dtype)
        std_array = np.asarray(std, dtype=dtype)
        if mean_array.shape != (3,) or std_array.shape != (3,):
            raise ValueError("mean and std must each contain three values")
        if not np.all(np.isfinite(mean_array)) or not np.all(np.isfinite(std_array)):
            raise ValueError("mean and std must be finite")
        if np.any(std_array == 0):
            raise ValueError("std values cannot be zero")
        array = (array - mean_array.reshape(1, 1, 3)) / std_array.reshape(1, 1, 3)

    if output_layout == "CHW":
        array = np.transpose(array, (2, 0, 1))
    elif output_layout != "HWC":
        raise ValueError("output_layout must be 'HWC' or 'CHW'")
    return np.ascontiguousarray(array, dtype=dtype)


def frames_to_tensor(frames: Sequence[Any], **kwargs: Any) -> np.ndarray:
    """Convert and stack frames into ``NCHW`` or ``NHWC`` model input."""

    if not frames:
        raise ValueError("frames must not be empty")
    np = _import_numpy()
    tensors = [frame_to_tensor(frame, **kwargs) for frame in frames]
    expected_shape = tensors[0].shape
    if any(tensor.shape != expected_shape for tensor in tensors[1:]):
        raise ValueError("all transformed frames must have the same shape; provide size to resize")
    return np.stack(tensors, axis=0)


# Backwards-friendly descriptive aliases.
to_tensor = frame_to_tensor
batch_frames_to_tensor = frames_to_tensor
