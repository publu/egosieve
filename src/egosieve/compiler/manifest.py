"""Versioned JSON Lines manifests for reproducible video compilations."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from egosieve.video.models import SamplingPlan, VideoMetadata

from .segments import Segment, WindowScore

MANIFEST_SCHEMA = "egosieve.video-compilation"
MANIFEST_VERSION = 1


class ManifestError(ValueError):
    """Raised when a manifest is malformed or uses an unsupported version."""


def _json_value(value: Any) -> Any:
    """Normalize common path/container values without hiding bad numerics."""

    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _timestamp(created_at: datetime | str | None) -> str:
    if created_at is None:
        created_at = datetime.now(timezone.utc)
    if isinstance(created_at, str):
        if not created_at.strip():
            raise ValueError("created_at cannot be empty")
        return created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return created_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _score_mapping(
    scores: Iterable[WindowScore | Mapping[str, Any]] | None,
) -> dict[int, WindowScore]:
    result: dict[int, WindowScore] = {}
    for fallback_index, value in enumerate(scores or ()):
        score = (
            value
            if isinstance(value, WindowScore)
            else WindowScore.from_mapping(value, fallback_index=fallback_index)
        )
        if score.index in result:
            raise ValueError(f"duplicate score for window {score.index}")
        result[score.index] = score
    return result


def build_manifest_records(
    metadata: VideoMetadata,
    *,
    plan: SamplingPlan | None = None,
    scores: Iterable[WindowScore | Mapping[str, Any]] | None = None,
    segments: Iterable[Segment] = (),
    artifacts: Mapping[int, Mapping[str, Any]] | None = None,
    created_at: datetime | str | None = None,
    generator: str = "egosieve",
) -> tuple[dict[str, Any], ...]:
    """Build deterministic, JSON-compatible manifest records.

    The header contains complete source identity and geometry. Window and
    segment records preserve both source-relative and stream timestamps, so a
    non-zero container start time is never silently discarded.
    """

    score_by_index = _score_mapping(scores)
    segment_items = tuple(segments)
    header: dict[str, Any] = {
        "record_type": "manifest",
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_VERSION,
        "created_at": _timestamp(created_at),
        "generator": str(generator),
        "source": metadata.to_dict(),
        "counts": {
            "windows": len(plan.windows) if plan else len(score_by_index),
            "unique_samples": len(plan.samples) if plan else 0,
            "segments": len(segment_items),
        },
    }
    if plan is not None:
        header["sampling"] = {
            "window_duration_s": plan.window_duration_s,
            "stride_s": plan.stride_s,
            "frames_per_window": plan.frames_per_window,
            "window_count": len(plan.windows),
            "unique_sample_count": len(plan.samples),
        }

    records: list[dict[str, Any]] = [header]
    if plan is not None:
        for window in plan.windows:
            samples = plan.samples_for_window(window)
            record: dict[str, Any] = {
                "record_type": "window",
                "schema_version": MANIFEST_VERSION,
                "window_index": window.index,
                "start_s": window.start_s,
                "end_s": window.end_s,
                "source_start_s": window.source_start_s,
                "source_end_s": window.source_end_s,
                "sample_indices": list(window.sample_indices),
                "timestamps_s": [sample.timestamp_s for sample in samples],
                "source_timestamps_s": [sample.source_timestamp_s for sample in samples],
            }
            score = score_by_index.get(window.index)
            if score is not None:
                record["score"] = score.score
                record["uncertainty"] = score.uncertainty
            records.append(record)
    else:
        for score in sorted(score_by_index.values(), key=lambda item: item.index):
            records.append(
                {
                    "record_type": "window",
                    "schema_version": MANIFEST_VERSION,
                    "window_index": score.index,
                    "start_s": score.start_s,
                    "end_s": score.end_s,
                    "source_start_s": metadata.start_time_s + score.start_s,
                    "source_end_s": metadata.start_time_s + score.end_s,
                    "score": score.score,
                    "uncertainty": score.uncertainty,
                }
            )

    artifact_map = artifacts or {}
    for segment_index, segment in enumerate(segment_items):
        record = {
            "record_type": "segment",
            "schema_version": MANIFEST_VERSION,
            "segment_index": segment_index,
            **segment.to_dict(),
            "source_start_s": metadata.start_time_s + segment.start_s,
            "source_end_s": metadata.start_time_s + segment.end_s,
        }
        if segment_index in artifact_map:
            record["artifacts"] = _json_value(artifact_map[segment_index])
        records.append(record)
    return tuple(records)


def write_jsonl(
    path: os.PathLike[str] | str,
    records: Iterable[Mapping[str, Any]],
    *,
    atomic: bool = True,
) -> Path:
    """Write JSONL with deterministic keys and reject NaN/Infinity values."""

    destination = Path(path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(dict(record), sort_keys=True, separators=(",", ":"), allow_nan=False)
        for record in records
    ]
    payload = "".join(f"{line}\n" for line in lines)
    if not atomic:
        destination.write_text(payload, encoding="utf-8")
        return destination

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                Path(temporary_name).unlink()
    return destination


def write_manifest(
    path: os.PathLike[str] | str,
    metadata: VideoMetadata,
    **record_options: Any,
) -> Path:
    """Build and atomically write a versioned compilation manifest."""

    return write_jsonl(path, build_manifest_records(metadata, **record_options))


def read_manifest(
    path: os.PathLike[str] | str,
    *,
    expected_version: int = MANIFEST_VERSION,
) -> tuple[dict[str, Any], ...]:
    """Read and minimally validate an EgoSieve manifest."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"invalid JSON on line {line_number}") from exc
            if not isinstance(record, dict):
                raise ManifestError(f"manifest line {line_number} is not an object")
            records.append(record)
    if not records:
        raise ManifestError("manifest is empty")
    header = records[0]
    if header.get("record_type") != "manifest" or header.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError("manifest header has an unknown schema")
    if header.get("schema_version") != expected_version:
        raise ManifestError(
            f"unsupported manifest version {header.get('schema_version')!r}; "
            f"expected {expected_version}"
        )
    for line_number, record in enumerate(records[1:], 2):
        if record.get("schema_version") != expected_version:
            raise ManifestError(f"version mismatch on line {line_number}")
    return tuple(records)


# Explicit name for discoverability in callers that deal with several JSONL files.
write_jsonl_manifest = write_manifest
