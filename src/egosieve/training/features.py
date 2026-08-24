"""Deterministic video-window expansion and DINOv2 feature caching."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..processing_egosieve import EgoSieveProcessor
from ..video.frames import build_frame_batch_extract_command
from ..video.sampling import sample_timestamps
from .data import TrainingRecord, TrainingWindow


class FeaturePreparationError(RuntimeError):
    """A training window could not be decoded or encoded."""


@dataclass(frozen=True)
class WindowExample:
    """One labeled window with a resolved local media path."""

    key: str
    record_id: str
    group_id: str
    source: str
    license: str
    video_path: Path
    window_index: int
    window: TrainingWindow
    timestamps_s: tuple[float, ...]


def _resolved_video(video: str, media_root: Path) -> Path:
    candidate = Path(video).expanduser()
    if not candidate.is_absolute():
        candidate = media_root / candidate
    candidate = candidate.resolve(strict=True)
    if not candidate.is_file():
        raise FileNotFoundError(f"training media is not a file: {candidate}")
    return candidate


def expand_training_windows(
    records: Iterable[TrainingRecord],
    *,
    media_root: str | Path,
    num_frames: int,
    allowed_licenses: Iterable[str] | None = None,
) -> tuple[WindowExample, ...]:
    """Flatten records while resolving media and enforcing an optional license gate."""

    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
        raise ValueError("num_frames must be a positive integer")
    root = Path(media_root).expanduser().resolve(strict=True)
    allowed = None if allowed_licenses is None else {item.strip() for item in allowed_licenses}
    if allowed is not None and (not allowed or "" in allowed):
        raise ValueError("allowed_licenses must contain non-empty license identifiers")

    examples: list[WindowExample] = []
    seen: set[str] = set()
    for record in records:
        if allowed is not None and record.license not in allowed:
            raise ValueError(
                f"record {record.id!r} declares license {record.license!r}, which is not "
                "in the explicit allowlist"
            )
        path = _resolved_video(record.video, root)
        source = str(record.extra.get("source") or record.extra.get("dataset") or record.license)
        for index, window in enumerate(record.windows):
            key = f"{record.id}:{index}"
            if key in seen:
                raise ValueError(f"duplicate expanded window key: {key}")
            seen.add(key)
            examples.append(
                WindowExample(
                    key=key,
                    record_id=record.id,
                    group_id=record.group_id,
                    source=source,
                    license=record.license,
                    video_path=path,
                    window_index=index,
                    window=window,
                    timestamps_s=sample_timestamps(
                        window.start_s,
                        window.end_s,
                        num_frames,
                        strategy="center",
                    ),
                )
            )
    if not examples:
        raise ValueError("the annotation file contains no training windows")
    return tuple(examples)


def artifact_fingerprint(path: str | Path) -> str:
    """Hash the exact files that define a local model/processor artifact."""

    root = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    names = (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "pytorch_model.bin",
    )
    found = False
    for name in names:
        candidate = root / name
        if not candidate.is_file():
            continue
        found = True
        digest.update(name.encode("utf-8") + b"\0")
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    if not found:
        raise FileNotFoundError(f"no model artifact files found in {root}")
    return digest.hexdigest()


def _source_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def feature_cache_key(example: WindowExample, artifact_sha256: str) -> str:
    """Return a label-independent key with source staleness protection.

    Multiple supervision rows can intentionally reference the same decoded
    window (for example, a matched corruption control and a readiness label).
    Their visual features are identical, so record ids and label indexes must
    not force redundant decoding and backbone execution.
    """

    payload = {
        "schema": "egosieve.feature-cache/v1",
        "artifact_sha256": artifact_sha256,
        "source": _source_identity(example.video_path),
        "timestamps_s": example.timestamps_s,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _extract_frames(
    example: WindowExample,
    directory: Path,
    *,
    ffmpeg_bin: str,
    timeout_s: float,
) -> list[Path]:
    outputs = [directory / f"frame-{index:04d}.png" for index in range(len(example.timestamps_s))]
    command = build_frame_batch_extract_command(
        example.video_path,
        list(zip(example.timestamps_s, outputs, strict=True)),
        ffmpeg_bin=ffmpeg_bin,
        overwrite=True,
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise FeaturePreparationError(
            f"ffmpeg timed out while decoding {example.key} after {timeout_s:g}s"
        ) from exc
    except OSError as exc:
        raise FeaturePreparationError(f"could not execute {ffmpeg_bin!r}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "unknown ffmpeg error").strip()[-1000:]
        raise FeaturePreparationError(f"ffmpeg failed for {example.key}: {detail}")
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise FeaturePreparationError(
            f"ffmpeg did not create {len(missing)} frame(s) for {example.key}"
        )
    return outputs


class FeatureCache:
    """Materialize immutable frame embeddings and reuse them across runs."""

    def __init__(
        self,
        root: str | Path,
        *,
        artifact_sha256: str,
        processor: EgoSieveProcessor,
        vision_model: torch.nn.Module,
        device: torch.device,
        ffmpeg_bin: str = "ffmpeg",
        timeout_s: float = 120.0,
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_sha256 = artifact_sha256
        self.processor = processor
        self.vision_model = vision_model.to(device).eval()
        self.device = device
        self.ffmpeg_bin = ffmpeg_bin
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = float(timeout_s)

    def path_for(self, example: WindowExample) -> Path:
        key = feature_cache_key(example, self.artifact_sha256)
        return self.root / key[:2] / f"{key}.npz"

    def materialize(self, example: WindowExample) -> Path:
        destination = self.path_for(example)
        if destination.is_file():
            self._validate(destination, example)
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="egosieve-frames-") as temp_name:
            frames = _extract_frames(
                example,
                Path(temp_name),
                ffmpeg_bin=self.ffmpeg_bin,
                timeout_s=self.timeout_s,
            )
            processed = self.processor(videos=frames, return_tensors="pt")
            pixels = processed["pixel_values"]
            frame_mask = processed["frame_mask"].to(dtype=torch.bool)
            if pixels.shape[1] != len(example.timestamps_s):
                raise FeaturePreparationError(
                    "processor frame count differs from the annotation sampling contract"
                )
            batch, frames_count, channels, height, width = pixels.shape
            with torch.inference_mode():
                output = self.vision_model(
                    pixel_values=pixels.reshape(batch * frames_count, channels, height, width).to(
                        self.device
                    ),
                    return_dict=True,
                )
                embeddings = output.last_hidden_state[:, 0].reshape(batch, frames_count, -1)[0]
            arrays = {
                "frame_embeddings": embeddings.detach().cpu().float().numpy(),
                "frame_mask": frame_mask[0].cpu().numpy().astype(np.bool_),
                "timestamps_s": np.asarray(example.timestamps_s, dtype=np.float64),
                "artifact_sha256": np.asarray(self.artifact_sha256),
            }
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npz")
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, destination)
        self._validate(destination, example)
        return destination

    def _validate(self, path: Path, example: WindowExample) -> None:
        try:
            with np.load(path, allow_pickle=False) as data:
                features = data["frame_embeddings"]
                timestamps = data["timestamps_s"]
                artifact = str(data["artifact_sha256"].item())
        except Exception as exc:
            raise FeaturePreparationError(f"invalid feature cache entry {path}: {exc}") from exc
        if features.ndim != 2 or features.shape[0] != len(example.timestamps_s):
            raise FeaturePreparationError(f"feature cache entry has invalid shape: {path}")
        if not np.isfinite(features).all():
            raise FeaturePreparationError(f"feature cache entry contains non-finite values: {path}")
        if not np.array_equal(timestamps, np.asarray(example.timestamps_s, dtype=np.float64)):
            raise FeaturePreparationError(f"feature cache timestamps do not match: {path}")
        if artifact != self.artifact_sha256:
            raise FeaturePreparationError(f"feature cache identity does not match: {path}")

    def load(self, example: WindowExample) -> np.ndarray:
        path = self.materialize(example)
        with np.load(path, allow_pickle=False) as data:
            return data["frame_embeddings"].astype(np.float32, copy=True)

    def prepare(self, examples: Sequence[WindowExample]) -> dict[str, int]:
        hits = 0
        created = 0
        for example in examples:
            existed = self.path_for(example).is_file()
            self.materialize(example)
            hits += int(existed)
            created += int(not existed)
        return {"examples": len(examples), "cache_hits": hits, "created": created}


__all__ = [
    "FeatureCache",
    "FeaturePreparationError",
    "WindowExample",
    "artifact_fingerprint",
    "expand_training_windows",
    "feature_cache_key",
]
