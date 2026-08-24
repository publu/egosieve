"""Deterministic HoloAssist readiness-proxy adapter.

HoloAssist's fine-action intervals were created by professional annotators,
self-reviewed, and independently audited.  They are not EgoSieve readiness
judgments.  This adapter therefore keeps the publisher intervals and their
correctness attributes intact while applying a transparent fixed-window
occupancy rubric.  It never emits technical-issue labels.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .acquisition import (
    CORPUS_MANIFEST_NAME,
    HOLOASSIST,
    HOLOASSIST_ANNOTATION_PATH,
    HOLOASSIST_ANNOTATION_RELEASE,
    HOLOASSIST_ANNOTATION_URL,
    HOLOASSIST_AUDIT_URL,
    HOLOASSIST_MIRROR_REVISION,
    HOLOASSIST_MIRROR_URL,
    HOLOASSIST_SCHEMA_URL,
    CorpusAcquisitionError,
    validate_revision,
    verify_corpus,
)
from .ego_tactile import write_training_jsonl

TRAINING_SCHEMA = "egosieve.training/v1"
READINESS_RUBRIC_VERSION = "egosieve.holoassist-readiness-proxy/v1"
SOURCE_RUBRIC_VERSION = "HoloAssist fine-action annotations v1_1"
PUBLISHED_REVIEW_COUNT = 2
DEFAULT_WINDOW_S = 6.0
DEFAULT_STRIDE_S = 3.0
DEFAULT_KEEP_OCCUPANCY_THRESHOLD = 0.5
PUBLISHER_DURATION_TOLERANCE_S = 0.5
DEFAULT_MAX_KEEP_PER_VIDEO = 64
DEFAULT_MAX_REVIEW_PER_VIDEO = 64
DEFAULT_MAX_REJECT_PER_VIDEO = 64

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DURATION_RE = re.compile(r"^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$")
_FINE_ACTION_LABEL = "Fine grained action"
_CLASS_ORDER = {"KEEP": 0, "REVIEW": 1, "REJECT": 2}


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CorpusAcquisitionError(f"{path} must be an object")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusAcquisitionError(f"{path} must be a non-empty string")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CorpusAcquisitionError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CorpusAcquisitionError(f"{path} must be a finite number")
    return result


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CorpusAcquisitionError(f"{path} must be an integer >= {minimum}")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise CorpusAcquisitionError(f"{path} must be an array")
    return value


def _video_id(value: Any, path: str) -> str:
    result = _string(value, path)
    if result.endswith(".mp4") or not _VIDEO_ID_RE.fullmatch(result):
        raise CorpusAcquisitionError(
            f"{path} must be an exact HoloAssist video_name without an .mp4 suffix"
        )
    return result


def _positive(value: Any, path: str) -> float:
    result = _number(value, path)
    if result <= 0:
        raise CorpusAcquisitionError(f"{path} must be positive")
    return result


def _duration_seconds(value: Mapping[str, Any], path: str) -> float:
    metadata = _mapping(value, path)
    whole_seconds = _positive(metadata.get("seconds"), f"{path}.seconds")
    raw = _string(metadata.get("raw"), f"{path}.raw")
    match = _DURATION_RE.fullmatch(raw)
    if match is None:
        raise CorpusAcquisitionError(f"{path}.raw must use HH:MM:SS[.fraction] format")
    hours, minutes, seconds = match.groups()
    if int(minutes) >= 60 or float(seconds) >= 60:
        raise CorpusAcquisitionError(f"{path}.raw contains an invalid clock time")
    precise_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if precise_seconds <= 0 or abs(math.floor(precise_seconds) - whole_seconds) > 1:
        raise CorpusAcquisitionError(f"{path}.raw and {path}.seconds are inconsistent")
    # The released annotations expose this integer field directly and the
    # adapter's 6 s / 3 s grid audit uses it.  The subsecond raw value is
    # validated and retained in provenance, but not used to add a tail window.
    return whole_seconds


def _cap(value: int | None, path: str) -> int | None:
    if value is None:
        return None
    return _integer(value, path, minimum=1)


def _source_metadata(source_value: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(source_value, "manifest.source")
    if source.get("profile") != HOLOASSIST.key:
        raise CorpusAcquisitionError("corpus manifest is not a HoloAssist acquisition")
    if source.get("name") != HOLOASSIST.name or source.get("source_url") != HOLOASSIST.source_url:
        raise CorpusAcquisitionError("manifest has unexpected HoloAssist publisher metadata")

    repository = _mapping(source.get("repository"), "manifest.source.repository")
    if (
        repository.get("type") != "huggingface_dataset_mirror"
        or repository.get("id") != HOLOASSIST.repository_id
        or repository.get("source_url") != HOLOASSIST_MIRROR_URL
        or repository.get("role") != "media-transport-only"
        or repository.get("license_declaration_url") is not None
        or repository.get("publisher_byte_equivalence_verified") is not False
    ):
        raise CorpusAcquisitionError("manifest has unexpected HoloAssist mirror metadata")
    revision = validate_revision(
        _string(repository.get("revision"), "manifest.source.repository.revision")
    )
    if revision != HOLOASSIST_MIRROR_REVISION:
        raise CorpusAcquisitionError(
            "manifest does not use the reviewed exact HoloAssist mirror revision"
        )

    annotation = _mapping(source.get("annotation_release"), "manifest.source.annotation_release")
    expected_annotation = {
        "type": "publisher_direct_file",
        "version": HOLOASSIST_ANNOTATION_RELEASE,
        "filename": HOLOASSIST_ANNOTATION_PATH,
        "download_url": HOLOASSIST_ANNOTATION_URL,
        "schema_url": HOLOASSIST_SCHEMA_URL,
        "audit_url": HOLOASSIST_AUDIT_URL,
    }
    if dict(annotation) != expected_annotation:
        raise CorpusAcquisitionError("manifest has unexpected HoloAssist annotation provenance")

    license_info = _mapping(source.get("license"), "manifest.source.license")
    expected_license = {
        "id": HOLOASSIST.license_id,
        "url": HOLOASSIST.license_url,
        "declaration_url": HOLOASSIST.license_declaration_url,
        "attribution": HOLOASSIST.attribution,
    }
    if dict(license_info) != expected_license:
        raise CorpusAcquisitionError("manifest has unexpected HoloAssist license metadata")
    return {
        "revision": revision,
        "repository": dict(repository),
        "annotation": dict(annotation),
        "license": dict(license_info),
        "source_url": HOLOASSIST.source_url,
    }


def _source_annotation(event: Mapping[str, Any], *, event_path: str) -> dict[str, Any]:
    event_id = _integer(event.get("id"), f"{event_path}.id")
    if event.get("type") != "range":
        raise CorpusAcquisitionError(f"{event_path}.type must be 'range'")
    start_s = _number(event.get("start"), f"{event_path}.start")
    end_s = _number(event.get("end"), f"{event_path}.end")
    if start_s < 0 or end_s <= start_s:
        raise CorpusAcquisitionError(f"{event_path} has an invalid official action interval")
    attributes = _mapping(event.get("attributes"), f"{event_path}.attributes")
    correctness = _string(
        attributes.get("Action Correctness"),
        f"{event_path}.attributes.Action Correctness",
    )
    return {
        "id": event_id,
        "label": _FINE_ACTION_LABEL,
        "start_s": start_s,
        "end_s": end_s,
        "action_correctness": correctness,
        "attributes": dict(attributes),
    }


def _label_source(rule: str) -> dict[str, Any]:
    return {
        "kind": "programmatic_readiness_proxy",
        "human_reviewed": False,
        "rule": rule,
        "source_intervals_human_annotated": True,
        "source_interval_review_count": PUBLISHED_REVIEW_COUNT,
        "source_interval_review_basis": (
            "publisher-described original professional annotator self-review plus "
            "independent reviewer audit"
        ),
        "publisher_additional_review": (
            "targeted review of open-ended text fields; reviewer count not published"
        ),
        "audit_url": HOLOASSIST_AUDIT_URL,
        "source_rubric_version": SOURCE_RUBRIC_VERSION,
        "source_rubric_url": HOLOASSIST_SCHEMA_URL,
    }


def _base_window(
    *,
    start_s: float,
    end_s: float,
    readiness: str,
    rule: str,
) -> dict[str, Any]:
    return {
        "start_s": start_s,
        "end_s": end_s,
        "readiness": readiness,
        "readiness_valid": True,
        # This count describes review of the source action intervals only.  It
        # must not be interpreted as two reviews of the derived readiness.
        "review_count": PUBLISHED_REVIEW_COUNT,
        "review_count_scope": "source_action_intervals",
        "rubric_version": READINESS_RUBRIC_VERSION,
        "label_source": _label_source(rule),
    }


def _merge_intervals(actions: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for action in sorted(actions, key=lambda item: (item["start_s"], item["end_s"], item["id"])):
        start_s = float(action["start_s"])
        end_s = float(action["end_s"])
        if not merged or start_s > merged[-1][1]:
            merged.append((start_s, end_s))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end_s))
    return merged


def _window_occupancy(
    actions: Sequence[Mapping[str, Any]],
    *,
    start_s: float,
    end_s: float,
) -> tuple[float, list[dict[str, Any]]]:
    overlapping = [
        dict(action)
        for action in actions
        if float(action["start_s"]) < end_s and float(action["end_s"]) > start_s
    ]
    clipped = [
        {
            "id": action["id"],
            "start_s": max(start_s, float(action["start_s"])),
            "end_s": min(end_s, float(action["end_s"])),
        }
        for action in overlapping
    ]
    occupied = sum(end - start for start, end in _merge_intervals(clipped))
    return occupied / (end_s - start_s), overlapping


def _grid_windows(
    actions: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    window_s: float,
    stride_s: float,
    keep_occupancy_threshold: float,
) -> list[dict[str, Any]]:
    # Reuse the scanner's exact window planner, including its one end-anchored
    # tail window, so corpus targets align with inference windows.
    from ..video.sampling import sliding_windows

    intervals = sliding_windows(duration_s, window_s, stride_s, include_tail=True)
    windows: list[dict[str, Any]] = []
    for grid_index, (start_s, end_s) in enumerate(intervals):
        occupancy, overlapping = _window_occupancy(actions, start_s=start_s, end_s=end_s)
        if occupancy >= keep_occupancy_threshold:
            readiness = "KEEP"
            rule = "union fine-action occupancy is at or above the KEEP threshold"
        elif occupancy > 0:
            readiness = "REVIEW"
            rule = "union fine-action occupancy is positive but below the KEEP threshold"
        else:
            readiness = "REJECT"
            rule = "union fine-action occupancy is zero"

        starts_inside = sorted(
            float(action["start_s"])
            for action in actions
            if start_s <= float(action["start_s"]) <= end_s
        )
        ends_inside = sorted(
            float(action["end_s"])
            for action in actions
            if start_s <= float(action["end_s"]) <= end_s
        )
        boundaries: dict[str, float] = {}
        boundary_valid = {"start": bool(starts_inside), "end": bool(ends_inside)}
        if starts_inside:
            boundaries["start"] = starts_inside[0]
        if ends_inside:
            boundaries["end"] = ends_inside[-1]
        boundary_pair_suppressed = (
            "start" in boundaries
            and "end" in boundaries
            and boundaries["end"] < boundaries["start"]
        )
        if boundary_pair_suppressed:
            # The v1 training contract represents one ordered start/end pair.
            # Preserve every official boundary in source_annotations, but mask
            # a cross-action pair that would invert that contract.
            boundaries = {}
            boundary_valid = {"start": False, "end": False}

        window = _base_window(
            start_s=start_s,
            end_s=end_s,
            readiness=readiness,
            rule=rule,
        )
        window.update(
            {
                "boundaries_s": boundaries,
                "boundary_valid": boundary_valid,
                "fine_action_occupancy": round(occupancy, 8),
                "keep_occupancy_threshold": keep_occupancy_threshold,
                "source_annotations": overlapping,
                "source_boundary_candidates_s": {
                    "starts": starts_inside,
                    "ends": ends_inside,
                },
                "boundary_pair_suppressed": boundary_pair_suppressed,
                "derived_from": {
                    "kind": "scanner_window_fine_action_union_occupancy",
                    "grid_index": grid_index,
                    "window_s": window_s,
                    "stride_s": stride_s,
                    "include_tail": True,
                },
                "proxy_id": f"grid-{grid_index:06d}",
            }
        )
        windows.append(window)
    return windows


def _window_sort_key(window: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        float(window["start_s"]),
        float(window["end_s"]),
        _CLASS_ORDER[_string(window.get("readiness"), "window.readiness")],
        _string(window.get("proxy_id"), "window.proxy_id"),
    )


def _evenly_cap(windows: Sequence[dict[str, Any]], cap: int | None) -> list[dict[str, Any]]:
    ordered = sorted(windows, key=_window_sort_key)
    if cap is None or len(ordered) <= cap:
        return ordered
    if cap == 1:
        return [ordered[(len(ordered) - 1) // 2]]
    indexes = [(index * (len(ordered) - 1)) // (cap - 1) for index in range(cap)]
    return [ordered[index] for index in indexes]


def _record_actions(record: Mapping[str, Any], *, record_path: str) -> list[dict[str, Any]]:
    events = _sequence(record.get("events"), f"{record_path}.events")
    actions: list[dict[str, Any]] = []
    for event_index, event_value in enumerate(events):
        event = _mapping(event_value, f"{record_path}.events[{event_index}]")
        if event.get("label") != _FINE_ACTION_LABEL:
            continue
        actions.append(_source_annotation(event, event_path=f"{record_path}.events[{event_index}]"))
    actions.sort(key=lambda item: (item["start_s"], item["end_s"], item["id"]))
    if not actions:
        raise CorpusAcquisitionError(f"{record_path} contains no fine-grained actions")
    return actions


def build_holoassist_records(
    annotation_value: Sequence[Mapping[str, Any]],
    *,
    video_ids: Sequence[str],
    manifest_source: Mapping[str, Any],
    window_s: float = DEFAULT_WINDOW_S,
    stride_s: float = DEFAULT_STRIDE_S,
    keep_occupancy_threshold: float = DEFAULT_KEEP_OCCUPANCY_THRESHOLD,
    max_keep_per_video: int | None = DEFAULT_MAX_KEEP_PER_VIDEO,
    max_review_per_video: int | None = DEFAULT_MAX_REVIEW_PER_VIDEO,
    max_reject_per_video: int | None = DEFAULT_MAX_REJECT_PER_VIDEO,
) -> list[dict[str, Any]]:
    """Build deterministic per-video HoloAssist readiness-proxy records."""

    annotations = _sequence(annotation_value, "HoloAssist annotations")
    source = _source_metadata(manifest_source)
    window_s = _positive(window_s, "window_s")
    stride_s = _positive(stride_s, "stride_s")
    keep_occupancy_threshold = _number(keep_occupancy_threshold, "keep_occupancy_threshold")
    if not 0 < keep_occupancy_threshold <= 1:
        raise CorpusAcquisitionError("keep_occupancy_threshold must be in (0, 1]")
    caps = {
        "KEEP": _cap(max_keep_per_video, "max_keep_per_video"),
        "REVIEW": _cap(max_review_per_video, "max_review_per_video"),
        "REJECT": _cap(max_reject_per_video, "max_reject_per_video"),
    }
    if isinstance(video_ids, str | bytes) or not video_ids:
        raise CorpusAcquisitionError("select at least one exact HoloAssist video_name")
    selected = [_video_id(value, "video_id") for value in video_ids]
    if len(selected) != len(set(selected)):
        raise CorpusAcquisitionError("HoloAssist video selections must not contain duplicates")
    selected = sorted(selected)
    selected_set = set(selected)

    by_video: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for record_index, record_value in enumerate(annotations):
        record = _mapping(record_value, f"annotations[{record_index}]")
        name_value = record.get("video_name")
        if not isinstance(name_value, str) or name_value not in selected_set:
            continue
        name = _video_id(name_value, f"annotations[{record_index}].video_name")
        if name in by_video:
            raise CorpusAcquisitionError(f"duplicate HoloAssist annotation record for {name!r}")
        by_video[name] = (record_index, record)
    missing = sorted(selected_set - set(by_video))
    if missing:
        raise CorpusAcquisitionError(
            "selected HoloAssist annotations are missing for: " + ", ".join(missing)
        )

    records: list[dict[str, Any]] = []
    for video_id in selected:
        record_index, source_record = by_video[video_id]
        record_path = f"annotations[{record_index}]"
        video_metadata = _mapping(
            source_record.get("videoMetadata"), f"{record_path}.videoMetadata"
        )
        duration_metadata = _mapping(
            video_metadata.get("duration"), f"{record_path}.videoMetadata.duration"
        )
        duration_s = _duration_seconds(
            duration_metadata,
            f"{record_path}.videoMetadata.duration",
        )
        actions = _record_actions(source_record, record_path=record_path)
        if any(
            float(action["end_s"]) > duration_s + PUBLISHER_DURATION_TOLERANCE_S
            for action in actions
        ):
            raise CorpusAcquisitionError(
                f"{record_path} contains a fine-action boundary after the declared video duration"
            )

        candidates = _grid_windows(
            actions,
            duration_s=duration_s,
            window_s=window_s,
            stride_s=stride_s,
            keep_occupancy_threshold=keep_occupancy_threshold,
        )
        selected_windows = {
            readiness: _evenly_cap(
                [window for window in candidates if window["readiness"] == readiness],
                caps[readiness],
            )
            for readiness in ("KEEP", "REVIEW", "REJECT")
        }
        windows = sorted(
            [
                window
                for readiness in ("KEEP", "REVIEW", "REJECT")
                for window in selected_windows[readiness]
            ],
            key=_window_sort_key,
        )
        mirror_path = f"HoloAssist/video/{video_id}.mp4"
        records.append(
            {
                "schema": TRAINING_SCHEMA,
                "id": f"holoassist-{video_id}",
                "source": "HoloAssist",
                # The original video is the largest shared-content unit.
                "group_id": f"holoassist:{video_id}",
                "video": (PurePosixPath("files") / mirror_path).as_posix(),
                "license": HOLOASSIST.license_id,
                "provenance": {
                    "corpus_manifest": CORPUS_MANIFEST_NAME,
                    "dataset": "HoloAssist",
                    "source_url": source["source_url"],
                    "annotation_release": source["annotation"]["version"],
                    "annotation_url": source["annotation"]["download_url"],
                    "annotation_schema_url": source["annotation"]["schema_url"],
                    "annotation_audit_url": source["annotation"]["audit_url"],
                    "license_url": source["license"]["url"],
                    "attribution": source["license"]["attribution"],
                    "video_name": video_id,
                    "task_id": _string(source_record.get("taskId"), f"{record_path}.taskId"),
                    "task_type": _string(source_record.get("taskType"), f"{record_path}.taskType"),
                    "batch": _string(source_record.get("batch"), f"{record_path}.batch"),
                    "duration_s": duration_s,
                    "duration_source": "publisher_videoMetadata.duration.seconds",
                    "publisher_duration_raw": _string(
                        duration_metadata.get("raw"),
                        f"{record_path}.videoMetadata.duration.raw",
                    ),
                    "publisher_duration_tolerance_s": PUBLISHER_DURATION_TOLERANCE_S,
                    "media_mirror": {
                        "repository_id": source["repository"]["id"],
                        "repository_url": source["repository"]["source_url"],
                        "revision": source["revision"],
                        "path": mirror_path,
                        "role": source["repository"]["role"],
                        "license_declaration_url": None,
                        "publisher_byte_equivalence_verified": False,
                    },
                },
                "label_policy": {
                    "kind": "programmatic_readiness_proxy",
                    "human_reviewed": False,
                    "source_intervals_human_annotated": True,
                    "source_interval_review_count": PUBLISHED_REVIEW_COUNT,
                    "source_interval_review_count_basis": (
                        "original professional annotator after self-review plus one "
                        "independent audit reviewer"
                    ),
                    "publisher_additional_review": (
                        "targeted review of open-ended text fields; reviewer count not published"
                    ),
                    "rubric_version": READINESS_RUBRIC_VERSION,
                    "source_rubric_version": SOURCE_RUBRIC_VERSION,
                    "source_rubric_url": HOLOASSIST_SCHEMA_URL,
                    "audit_url": HOLOASSIST_AUDIT_URL,
                    "readiness_rules": {
                        "KEEP": "union fine-action occupancy >= threshold",
                        "REVIEW": "0 < union fine-action occupancy < threshold",
                        "REJECT": "union fine-action occupancy = 0",
                    },
                    "action_correctness_used_for_readiness": False,
                    "issue_targets": False,
                    "window_s": window_s,
                    "stride_s": stride_s,
                    "keep_occupancy_threshold": keep_occupancy_threshold,
                    "occupancy_measure": "temporal union of all overlapping fine-action intervals",
                    "per_video_caps": dict(caps),
                },
                "windows": windows,
            }
        )
    return records


def adapt_acquired_holoassist(
    corpus_dir: str | Path,
    *,
    video_ids: Sequence[str],
    output_path: str | Path,
    window_s: float = DEFAULT_WINDOW_S,
    stride_s: float = DEFAULT_STRIDE_S,
    keep_occupancy_threshold: float = DEFAULT_KEEP_OCCUPANCY_THRESHOLD,
    max_keep_per_video: int | None = DEFAULT_MAX_KEEP_PER_VIDEO,
    max_review_per_video: int | None = DEFAULT_MAX_REVIEW_PER_VIDEO,
    max_reject_per_video: int | None = DEFAULT_MAX_REJECT_PER_VIDEO,
) -> dict[str, Any]:
    """Verify an acquisition and adapt explicitly selected HoloAssist videos."""

    output = Path(output_path)
    if output.exists():
        raise CorpusAcquisitionError(f"annotation output already exists: {output}")
    verification = verify_corpus(corpus_dir)
    manifest = _mapping(verification["manifest"], "manifest")
    source = _mapping(manifest.get("source"), "manifest.source")
    _source_metadata(source)
    root = Path(corpus_dir)
    acquired = {
        _string(entry.get("selection_path"), "manifest file selection_path"): root
        / _string(entry.get("local_path"), "manifest file local_path")
        for entry in verification["files"]
    }
    annotation_path = acquired.get(HOLOASSIST_ANNOTATION_PATH)
    if annotation_path is None:
        raise CorpusAcquisitionError(
            f"adapter requires explicitly acquired {HOLOASSIST_ANNOTATION_PATH}"
        )
    selected = [_video_id(value, "video_id") for value in video_ids]
    required_videos = {video_id: f"HoloAssist/video/{video_id}.mp4" for video_id in selected}
    missing_videos = sorted(path for path in required_videos.values() if path not in acquired)
    if missing_videos:
        raise CorpusAcquisitionError(
            "selected videos were not explicitly acquired: " + ", ".join(missing_videos)
        )
    try:
        annotation_value = json.loads(annotation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CorpusAcquisitionError(f"invalid JSON in {annotation_path}: {error.msg}") from error
    records = build_holoassist_records(
        annotation_value,
        video_ids=selected,
        manifest_source=source,
        window_s=window_s,
        stride_s=stride_s,
        keep_occupancy_threshold=keep_occupancy_threshold,
        max_keep_per_video=max_keep_per_video,
        max_review_per_video=max_review_per_video,
        max_reject_per_video=max_reject_per_video,
    )
    write_training_jsonl(records, output)
    class_counts = {
        readiness: sum(
            window["readiness"] == readiness for record in records for window in record["windows"]
        )
        for readiness in ("KEEP", "REVIEW", "REJECT")
    }
    return {
        "schema": TRAINING_SCHEMA,
        "output": str(output),
        "records": len(records),
        "windows": sum(class_counts.values()),
        "videos": sorted(selected),
        "readiness_targets": class_counts,
        "label_source": "programmatic_readiness_proxy",
        "human_reviewed": False,
        "source_interval_review_count": PUBLISHED_REVIEW_COUNT,
        "issue_targets": 0,
        "rubric_version": READINESS_RUBRIC_VERSION,
    }


__all__ = [
    "DEFAULT_KEEP_OCCUPANCY_THRESHOLD",
    "DEFAULT_MAX_KEEP_PER_VIDEO",
    "DEFAULT_MAX_REJECT_PER_VIDEO",
    "DEFAULT_MAX_REVIEW_PER_VIDEO",
    "DEFAULT_STRIDE_S",
    "DEFAULT_WINDOW_S",
    "PUBLISHED_REVIEW_COUNT",
    "PUBLISHER_DURATION_TOLERANCE_S",
    "READINESS_RUBRIC_VERSION",
    "SOURCE_RUBRIC_VERSION",
    "adapt_acquired_holoassist",
    "build_holoassist_records",
]
