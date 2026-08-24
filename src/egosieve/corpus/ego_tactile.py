"""Deterministic Ego-Tactile Manipulation adapter.

The publisher describes these action spans as contact/grip-force-derived and
explicitly states that they are not human ground truth.  Consequently this
adapter emits proxy temporal-boundary targets only.  It never fabricates a
KEEP/REVIEW/REJECT target or a visual-quality issue target.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .acquisition import (
    CORPUS_MANIFEST_NAME,
    EGO_TACTILE,
    CorpusAcquisitionError,
    validate_repository_path,
    validate_revision,
    verify_corpus,
)

TRAINING_SCHEMA = "egosieve.training/v1"
VIDEO_KEY = "observation.images.ego"
_EPISODE_METADATA_RE = re.compile(r"^meta/episodes/chunk-\d+/file-\d+\.parquet$")
_ANNOTATION_COLUMNS = (
    "annotation.l0_action",
    "annotation.l1_subaction",
    "annotation.anchor",
    "annotation.grasp_type",
    "annotation.hand",
    "annotation.target_object",
)
_FRAME_COLUMNS = (
    "timestamp",
    "frame_index",
    "episode_index",
    "task_index",
    *_ANNOTATION_COLUMNS,
)
_EPISODE_COLUMNS = (
    "episode_index",
    "length",
    "data/chunk_index",
    "data/file_index",
    f"videos/{VIDEO_KEY}/chunk_index",
    f"videos/{VIDEO_KEY}/file_index",
    f"videos/{VIDEO_KEY}/from_timestamp",
    f"videos/{VIDEO_KEY}/to_timestamp",
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CorpusAcquisitionError(f"{path} must be an object")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise CorpusAcquisitionError(f"{path} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise CorpusAcquisitionError(f"{path} must be an integer") from error
    if result != value or result < minimum:
        raise CorpusAcquisitionError(f"{path} must be an integer >= {minimum}")
    return result


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise CorpusAcquisitionError(f"{path} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise CorpusAcquisitionError(f"{path} must be a finite number") from error
    if not math.isfinite(result):
        raise CorpusAcquisitionError(f"{path} must be a finite number")
    return result


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CorpusAcquisitionError(f"{path} must be a non-empty string")
    return value


def _optional_label(value: Any, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CorpusAcquisitionError(f"{path} must be a string or null")
    stripped = value.strip()
    return stripped or None


def _timestamp(value: float) -> float:
    """Use microsecond precision for byte-stable JSON across readers."""

    return round(float(value), 6)


def _validate_info(info_value: Mapping[str, Any]) -> tuple[Mapping[str, Any], float]:
    info = _mapping(info_value, "meta/info.json")
    if info.get("codebase_version") != "v3.0":
        raise CorpusAcquisitionError(
            "Ego-Tactile adapter requires the inspected LeRobot v3.0 schema"
        )
    fps = _number(info.get("fps"), "meta/info.json.fps")
    if fps <= 0:
        raise CorpusAcquisitionError("meta/info.json.fps must be positive")
    features = _mapping(info.get("features"), "meta/info.json.features")
    required_features = {
        VIDEO_KEY,
        *_ANNOTATION_COLUMNS,
        "timestamp",
        "frame_index",
        "episode_index",
    }
    missing = sorted(required_features - set(features))
    if missing:
        raise CorpusAcquisitionError(
            "meta/info.json is missing inspected Ego-Tactile features: " + ", ".join(missing)
        )
    _string(info.get("data_path"), "meta/info.json.data_path")
    _string(info.get("video_path"), "meta/info.json.video_path")
    return info, fps


def _format_path(template: str, **values: Any) -> str:
    try:
        path = template.format(**values)
    except (KeyError, ValueError) as error:
        raise CorpusAcquisitionError(f"unsupported LeRobot path template {template!r}") from error
    return validate_repository_path(path)


def episode_repository_paths(
    info_value: Mapping[str, Any], episode_value: Mapping[str, Any]
) -> tuple[str, str]:
    """Return the exact data and video repository paths for one episode row."""

    info, _ = _validate_info(info_value)
    episode = _mapping(episode_value, "episode")
    data_path = _format_path(
        _string(info["data_path"], "meta/info.json.data_path"),
        chunk_index=_integer(episode.get("data/chunk_index"), "episode.data/chunk_index"),
        file_index=_integer(episode.get("data/file_index"), "episode.data/file_index"),
    )
    video_path = _format_path(
        _string(info["video_path"], "meta/info.json.video_path"),
        video_key=VIDEO_KEY,
        chunk_index=_integer(
            episode.get(f"videos/{VIDEO_KEY}/chunk_index"),
            f"episode.videos/{VIDEO_KEY}/chunk_index",
        ),
        file_index=_integer(
            episode.get(f"videos/{VIDEO_KEY}/file_index"),
            f"episode.videos/{VIDEO_KEY}/file_index",
        ),
    )
    return data_path, video_path


def _source_metadata(source_value: Mapping[str, Any]) -> tuple[str, str, str, str]:
    source = _mapping(source_value, "manifest.source")
    if source.get("profile") != EGO_TACTILE.key:
        raise CorpusAcquisitionError("corpus manifest is not an Ego-Tactile acquisition")
    repository = _mapping(source.get("repository"), "manifest.source.repository")
    if repository.get("id") != EGO_TACTILE.repository_id:
        raise CorpusAcquisitionError("corpus manifest has an unexpected repository id")
    revision = validate_revision(
        _string(repository.get("revision"), "manifest.source.repository.revision")
    )
    license_info = _mapping(source.get("license"), "manifest.source.license")
    if license_info.get("id") != EGO_TACTILE.license_id:
        raise CorpusAcquisitionError("corpus manifest has an unexpected Ego-Tactile license id")
    license_url = _string(license_info.get("url"), "manifest.source.license.url")
    attribution = _string(license_info.get("attribution"), "manifest.source.license.attribution")
    source_url = _string(source.get("source_url"), "manifest.source.source_url")
    if (
        license_url != EGO_TACTILE.license_url
        or attribution != EGO_TACTILE.attribution
        or source_url != EGO_TACTILE.source_url
    ):
        raise CorpusAcquisitionError(
            "corpus manifest source, license, or attribution does not match the reviewed profile"
        )
    return revision, license_url, attribution, source_url


def _span_key(row: Mapping[str, Any], row_path: str) -> tuple[Any, ...]:
    task_index = _integer(row.get("task_index"), f"{row_path}.task_index")
    labels = tuple(
        _optional_label(row.get(column), f"{row_path}.{column}") for column in _ANNOTATION_COLUMNS
    )
    return (task_index, *labels)


def _proxy_window(
    frames: Sequence[Mapping[str, Any]],
    *,
    first: int,
    stop: int,
    video_from_s: float,
    video_to_s: float,
    fps: float,
) -> dict[str, Any]:
    first_row = frames[first]
    last_row = frames[stop - 1]
    local_start = _number(first_row.get("timestamp"), f"frames[{first}].timestamp")
    if stop < len(frames):
        local_end = _number(frames[stop].get("timestamp"), f"frames[{stop}].timestamp")
    else:
        local_end = video_to_s - video_from_s
    if local_end <= local_start:
        local_end = local_start + (1.0 / fps)
    start_s = _timestamp(video_from_s + local_start)
    end_s = _timestamp(min(video_to_s, video_from_s + local_end))
    if end_s <= start_s:
        raise CorpusAcquisitionError("proxy span collapses after timestamp normalization")

    annotation = {
        "task_index": _integer(first_row.get("task_index"), "span.task_index"),
        "l0_action": _optional_label(first_row.get("annotation.l0_action"), "span.l0_action"),
        "l1_subaction": _optional_label(
            first_row.get("annotation.l1_subaction"), "span.l1_subaction"
        ),
        "anchor": _optional_label(first_row.get("annotation.anchor"), "span.anchor"),
        "grasp_type": _optional_label(first_row.get("annotation.grasp_type"), "span.grasp_type"),
        "hand": _optional_label(first_row.get("annotation.hand"), "span.hand"),
        "target_object": _optional_label(
            first_row.get("annotation.target_object"), "span.target_object"
        ),
        "frame_start": _integer(first_row.get("frame_index"), "span.frame_start"),
        "frame_end_inclusive": _integer(last_row.get("frame_index"), "span.frame_end"),
    }
    return {
        "start_s": start_s,
        "end_s": end_s,
        "readiness_valid": False,
        "boundaries_s": {"start": start_s, "end": end_s},
        "boundary_valid": {"start": True, "end": True},
        "label_source": {
            "kind": "proxy",
            "human_reviewed": False,
            "method": "publisher-provided contact/grip-force action segmentation",
        },
        "source_annotation": annotation,
    }


def build_ego_tactile_records(
    info_value: Mapping[str, Any],
    episode_rows: Iterable[Mapping[str, Any]],
    frame_rows: Iterable[Mapping[str, Any]],
    *,
    episode_indexes: Sequence[int],
    manifest_source: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build deterministic ``egosieve.training/v1`` proxy-boundary records."""

    info, fps = _validate_info(info_value)
    revision, license_url, attribution, source_url = _source_metadata(manifest_source)
    if isinstance(episode_indexes, str | bytes) or not episode_indexes:
        raise CorpusAcquisitionError("select at least one episode index")
    selected = [_integer(index, "episode index") for index in episode_indexes]
    if len(selected) != len(set(selected)):
        raise CorpusAcquisitionError("episode indexes must not contain duplicates")
    selected = sorted(selected)

    episodes: dict[int, Mapping[str, Any]] = {}
    for row_number, row_value in enumerate(episode_rows):
        row = _mapping(row_value, f"episodes[{row_number}]")
        index = _integer(row.get("episode_index"), f"episodes[{row_number}].episode_index")
        if index in episodes:
            raise CorpusAcquisitionError(f"duplicate episode metadata for index {index}")
        episodes[index] = row
    missing_episodes = [index for index in selected if index not in episodes]
    if missing_episodes:
        raise CorpusAcquisitionError(
            "selected episode metadata is missing for: "
            + ", ".join(str(index) for index in missing_episodes)
        )

    frames_by_episode: dict[int, list[Mapping[str, Any]]] = {index: [] for index in selected}
    for row_number, row_value in enumerate(frame_rows):
        row = _mapping(row_value, f"frames[{row_number}]")
        episode_index = _integer(row.get("episode_index"), f"frames[{row_number}].episode_index")
        if episode_index in frames_by_episode:
            frames_by_episode[episode_index].append(row)

    records: list[dict[str, Any]] = []
    for episode_index in selected:
        episode = episodes[episode_index]
        frames = frames_by_episode[episode_index]
        frames.sort(key=lambda row: _integer(row.get("frame_index"), "frame.frame_index"))
        expected_length = _integer(
            episode.get("length"), f"episode[{episode_index}].length", minimum=1
        )
        if len(frames) != expected_length:
            raise CorpusAcquisitionError(
                f"episode {episode_index} expected {expected_length} frames, found {len(frames)}"
            )
        frame_indexes = [
            _integer(row.get("frame_index"), f"episode[{episode_index}].frame_index")
            for row in frames
        ]
        if frame_indexes != list(range(expected_length)):
            raise CorpusAcquisitionError(
                f"episode {episode_index} frame indexes must be contiguous from zero"
            )
        timestamps = [
            _number(row.get("timestamp"), f"episode[{episode_index}].timestamp") for row in frames
        ]
        if any(
            current <= previous
            for previous, current in zip(timestamps, timestamps[1:], strict=False)
        ):
            raise CorpusAcquisitionError(
                f"episode {episode_index} timestamps must be strictly increasing"
            )

        video_from_s = _number(
            episode.get(f"videos/{VIDEO_KEY}/from_timestamp"),
            f"episode[{episode_index}].video_from_timestamp",
        )
        video_to_s = _number(
            episode.get(f"videos/{VIDEO_KEY}/to_timestamp"),
            f"episode[{episode_index}].video_to_timestamp",
        )
        if video_from_s < 0 or video_to_s <= video_from_s:
            raise CorpusAcquisitionError(
                f"episode {episode_index} has an invalid packed-video timestamp range"
            )
        if timestamps and (
            timestamps[0] < -(1.0 / fps) or timestamps[-1] > video_to_s - video_from_s
        ):
            raise CorpusAcquisitionError(
                f"episode {episode_index} frame timestamps exceed its packed-video range"
            )

        data_path, video_path = episode_repository_paths(info, episode)
        starts = [0]
        previous_key = _span_key(frames[0], f"episode[{episode_index}].frames[0]")
        for frame_position in range(1, len(frames)):
            key = _span_key(
                frames[frame_position],
                f"episode[{episode_index}].frames[{frame_position}]",
            )
            if key != previous_key:
                starts.append(frame_position)
                previous_key = key
        stops = starts[1:] + [len(frames)]
        windows = [
            _proxy_window(
                frames,
                first=first,
                stop=stop,
                video_from_s=video_from_s,
                video_to_s=video_to_s,
                fps=fps,
            )
            for first, stop in zip(starts, stops, strict=True)
        ]
        records.append(
            {
                "schema": TRAINING_SCHEMA,
                "id": f"ego-tactile-episode-{episode_index:06d}",
                # Several episodes may share one packed MP4.  Grouping by that
                # file prevents adjacent footage from crossing data splits.
                "group_id": f"ego-tactile:{video_path}",
                "video": (PurePosixPath("files") / video_path).as_posix(),
                "license": EGO_TACTILE.license_id,
                "provenance": {
                    "corpus_manifest": CORPUS_MANIFEST_NAME,
                    "dataset": EGO_TACTILE.repository_id,
                    "source_url": source_url,
                    "revision": revision,
                    "license_url": license_url,
                    "attribution": attribution,
                    "episode_index": episode_index,
                    "source_files": {"frames": data_path, "video": video_path},
                },
                "label_policy": {
                    "kind": "proxy",
                    "human_reviewed": False,
                    "proxy_boundary_targets": True,
                    "readiness_targets": False,
                    "issue_targets": False,
                    "publisher_method": "physical contact and grip-force action segmentation",
                },
                "windows": windows,
            }
        )
    return records


def write_training_jsonl(records: Sequence[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Write byte-stable JSONL without overwriting an existing annotation file."""

    output = Path(output_path)
    if output.exists():
        raise CorpusAcquisitionError(f"annotation output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n" for record in records
    )
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return output


def _read_parquet_rows(path: Path, columns: Sequence[str]) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise CorpusAcquisitionError(
            "adapting LeRobot Parquet requires pyarrow; install egosieve[hub]"
        ) from error
    parquet = pq.ParquetFile(path)
    missing = sorted(set(columns) - set(parquet.schema_arrow.names))
    if missing:
        raise CorpusAcquisitionError(
            f"{path} is missing inspected Ego-Tactile columns: {', '.join(missing)}"
        )
    return parquet.read(columns=list(columns)).to_pylist()


def adapt_acquired_ego_tactile(
    corpus_dir: str | Path,
    *,
    episode_indexes: Sequence[int],
    output_path: str | Path,
) -> dict[str, Any]:
    """Verify an acquisition and adapt selected episodes to training JSONL."""

    output = Path(output_path)
    if output.exists():
        raise CorpusAcquisitionError(f"annotation output already exists: {output}")
    verification = verify_corpus(corpus_dir)
    manifest = _mapping(verification["manifest"], "manifest")
    source = _mapping(manifest.get("source"), "manifest.source")
    _source_metadata(source)
    root = Path(corpus_dir)
    acquired = {
        _string(entry.get("repository_path"), "manifest file repository_path"): root
        / _string(entry.get("local_path"), "manifest file local_path")
        for entry in verification["files"]
    }

    info_path = acquired.get("meta/info.json")
    if info_path is None:
        raise CorpusAcquisitionError("adapter requires explicitly acquired meta/info.json")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CorpusAcquisitionError(f"invalid JSON in {info_path}: {error.msg}") from error
    _validate_info(info)

    metadata_paths = sorted(
        path
        for repository_path, path in acquired.items()
        if _EPISODE_METADATA_RE.fullmatch(repository_path)
    )
    if not metadata_paths:
        raise CorpusAcquisitionError(
            "adapter requires an explicitly acquired meta/episodes/chunk-*/file-*.parquet"
        )
    episode_rows = [
        row for path in metadata_paths for row in _read_parquet_rows(path, _EPISODE_COLUMNS)
    ]
    selected = [_integer(index, "episode index") for index in episode_indexes]
    selected_set = set(selected)
    selected_episode_rows = [
        row
        for row in episode_rows
        if _integer(row.get("episode_index"), "episode_index") in selected_set
    ]
    data_paths: set[str] = set()
    video_paths: set[str] = set()
    for episode in selected_episode_rows:
        data_path, video_path = episode_repository_paths(info, episode)
        data_paths.add(data_path)
        video_paths.add(video_path)
    missing_data = sorted(path for path in data_paths if path not in acquired)
    missing_video = sorted(path for path in video_paths if path not in acquired)
    if missing_data or missing_video:
        details = []
        if missing_data:
            details.append("data: " + ", ".join(missing_data))
        if missing_video:
            details.append("video: " + ", ".join(missing_video))
        raise CorpusAcquisitionError(
            "selected episodes require files that were not explicitly acquired ("
            + "; ".join(details)
            + ")"
        )
    frame_rows = [
        row
        for repository_path in sorted(data_paths)
        for row in _read_parquet_rows(acquired[repository_path], _FRAME_COLUMNS)
    ]
    records = build_ego_tactile_records(
        info,
        episode_rows,
        frame_rows,
        episode_indexes=selected,
        manifest_source=source,
    )
    write_training_jsonl(records, output)
    return {
        "schema": TRAINING_SCHEMA,
        "output": str(output),
        "records": len(records),
        "windows": sum(len(record["windows"]) for record in records),
        "episodes": sorted(selected),
        "label_source": "proxy",
        "human_reviewed": False,
        "readiness_targets": 0,
        "issue_targets": 0,
    }


__all__ = [
    "TRAINING_SCHEMA",
    "VIDEO_KEY",
    "adapt_acquired_ego_tactile",
    "build_ego_tactile_records",
    "episode_repository_paths",
    "write_training_jsonl",
]
