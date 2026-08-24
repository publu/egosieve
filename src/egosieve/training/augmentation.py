"""Deterministic, license-preserving observable-issue corpus construction.

The builder emits six positive programmatic corruptions. The synthetic
``hand_occlusion`` overlay maps to ``acting_hand_not_visible`` only as a
controlled-corruption proxy; it is not inherited human evidence or a naturally
observed visibility failure. Separately, the transform literally named
``acting_hand_not_visible`` only re-encodes a source window that already has a
valid, human-grounded positive for that issue. It does not draw an occluder or
manufacture a new label. ``low_hand_activity`` follows the same faithful
re-encode policy. Every derived window activates exactly one issue target;
readiness, temporal boundaries, and every other issue remain masked. This is a
deliberate safety property: an unmentioned issue in a source annotation is
unknown, never a negative example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from egosieve import __version__
from egosieve.video.probe import (
    VideoProbeError,
    build_ffprobe_command,
    probe_video,
    sha256_file,
)

from .data import SCHEMA_VERSION, TrainingRecord, TrainingWindow, load_jsonl

AUGMENTATION_SCHEMA = "egosieve.augmentation/v1"
PROVENANCE_SCHEMA = "egosieve.augmentation-provenance/v1"
MANIFEST_SCHEMA = "egosieve.augmentation-manifest/v1"
SKIP_SCHEMA = "egosieve.augmentation-skip/v1"

TRANSFORM_TO_ISSUE: dict[str, str] = {
    "blur": "blur",
    "exposure": "exposure",
    "camera_instability": "camera_instability",
    "freeze": "duplicate_frames",
    "scene_cut": "scene_cut",
    "hand_occlusion": "acting_hand_not_visible",
    "acting_hand_not_visible": "acting_hand_not_visible",
    "low_hand_activity": "low_hand_activity",
}
TRANSFORM_NAMES = tuple(TRANSFORM_TO_ISSUE)
PROGRAMMATIC_TRANSFORMS = tuple(
    name for name in TRANSFORM_NAMES if name not in {"acting_hand_not_visible", "low_hand_activity"}
)
SOURCE_INHERITED_TRANSFORMS = ("acting_hand_not_visible", "low_hand_activity")
CONTROLLED_CORRUPTION_KIND = "programmatic-controlled-corruption"
HUMAN_DERIVED_KIND = "human-derived"
HUMAN_GROUNDED_KINDS = frozenset({"human", HUMAN_DERIVED_KIND})
UNLABELED_KIND = "unlabeled"
LABEL_TASKS = ("readiness", "issues", "boundaries")
_COPY_EXTRA_FIELDS = (
    "split",
    "dataset",
    "dataset_revision",
    "source",
    "source_revision",
    "license_url",
    "license_text",
    "rights",
    "attribution",
    "attribution_url",
    "creator",
    "authors",
    "citation",
    "homepage",
    "source_url",
)


class AugmentationError(RuntimeError):
    """Raised when a licensed corpus cannot be built safely."""


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = f" and at least {minimum:g}" if minimum is not None else ""
        raise ValueError(f"{name} must be finite{qualifier}")
    return result


def _number(value: float) -> str:
    result = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return result or "0"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(_canonical_json(dict(row)) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalize_transforms(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return TRANSFORM_NAMES
    materialized = (str(values),) if isinstance(values, (str, bytes)) else tuple(values)
    unknown = sorted(set(materialized) - set(TRANSFORM_NAMES))
    if unknown:
        raise ValueError(
            f"unknown transformations {unknown!r}; expected values from {TRANSFORM_NAMES!r}"
        )
    selected = set(materialized)
    if not selected:
        raise ValueError("at least one transformation must be selected")
    return tuple(name for name in TRANSFORM_NAMES if name in selected)


@dataclass(frozen=True)
class AugmentationConfig:
    """Reproducible transformation and encoding parameters."""

    transforms: tuple[str, ...] = TRANSFORM_NAMES
    output_fps: float = 12.0
    video_codec: str = "libx264"
    preset: str = "medium"
    crf: int = 18
    blur_sigma: float = 8.0
    blur_steps: int = 3
    exposure_brightness: float = -0.45
    exposure_contrast: float = 0.70
    exposure_saturation: float = 0.85
    instability_crop_fraction: float = 0.84
    instability_x_frequency: float = 17.0
    instability_y_frequency: float = 13.0
    freeze_start_fraction: float = 0.35
    freeze_end_fraction: float = 0.68
    scene_gap_fraction: float = 0.16
    occlusion_margin_fraction: float = 0.06
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "transforms", _normalize_transforms(self.transforms))
        object.__setattr__(self, "output_fps", _finite(self.output_fps, "output_fps", minimum=1.0))
        if not isinstance(self.video_codec, str) or not self.video_codec.strip():
            raise ValueError("video_codec must be a non-empty string")
        if not isinstance(self.preset, str) or not self.preset.strip():
            raise ValueError("preset must be a non-empty string")
        if isinstance(self.crf, bool) or not isinstance(self.crf, int) or not 0 <= self.crf <= 51:
            raise ValueError("crf must be an integer between 0 and 51")
        object.__setattr__(self, "blur_sigma", _finite(self.blur_sigma, "blur_sigma", minimum=0.1))
        if (
            isinstance(self.blur_steps, bool)
            or not isinstance(self.blur_steps, int)
            or self.blur_steps < 1
        ):
            raise ValueError("blur_steps must be a positive integer")
        for name in (
            "exposure_brightness",
            "exposure_contrast",
            "exposure_saturation",
            "instability_x_frequency",
            "instability_y_frequency",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if not -1.0 <= self.exposure_brightness <= 1.0:
            raise ValueError("exposure_brightness must be between -1 and 1")
        if not 0.0 <= self.exposure_contrast <= 2.0:
            raise ValueError("exposure_contrast must be between 0 and 2")
        if not 0.0 <= self.exposure_saturation <= 3.0:
            raise ValueError("exposure_saturation must be between 0 and 3")
        crop = _finite(
            self.instability_crop_fraction,
            "instability_crop_fraction",
            minimum=0.5,
        )
        if crop >= 1.0:
            raise ValueError("instability_crop_fraction must be less than 1")
        object.__setattr__(self, "instability_crop_fraction", crop)
        freeze_start = _finite(self.freeze_start_fraction, "freeze_start_fraction", minimum=0.0)
        freeze_end = _finite(self.freeze_end_fraction, "freeze_end_fraction", minimum=0.0)
        if not 0.0 < freeze_start < freeze_end < 1.0:
            raise ValueError("freeze fractions must satisfy 0 < start < end < 1")
        gap = _finite(self.scene_gap_fraction, "scene_gap_fraction", minimum=0.01)
        if gap >= 0.8:
            raise ValueError("scene_gap_fraction must be less than 0.8")
        object.__setattr__(self, "scene_gap_fraction", gap)
        margin = _finite(
            self.occlusion_margin_fraction,
            "occlusion_margin_fraction",
            minimum=0.0,
        )
        if margin >= 0.5:
            raise ValueError("occlusion_margin_fraction must be less than 0.5")
        object.__setattr__(self, "occlusion_margin_fraction", margin)
        object.__setattr__(self, "timeout_s", _finite(self.timeout_s, "timeout_s", minimum=0.1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "transforms": list(self.transforms),
            "output_fps": self.output_fps,
            "video_codec": self.video_codec,
            "preset": self.preset,
            "crf": self.crf,
            "blur_sigma": self.blur_sigma,
            "blur_steps": self.blur_steps,
            "exposure_brightness": self.exposure_brightness,
            "exposure_contrast": self.exposure_contrast,
            "exposure_saturation": self.exposure_saturation,
            "instability_crop_fraction": self.instability_crop_fraction,
            "instability_x_frequency": self.instability_x_frequency,
            "instability_y_frequency": self.instability_y_frequency,
            "freeze_start_fraction": self.freeze_start_fraction,
            "freeze_end_fraction": self.freeze_end_fraction,
            "scene_gap_fraction": self.scene_gap_fraction,
            "occlusion_margin_fraction": self.occlusion_margin_fraction,
            "timeout_s": self.timeout_s,
        }


@dataclass(frozen=True)
class ToolInfo:
    requested: str
    resolved_path: str
    binary_sha256: str | None
    version_line: str
    version_output_sha256: str
    version_output: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "resolved_path": self.resolved_path,
            "binary_sha256": self.binary_sha256,
            "version_line": self.version_line,
            "version_output_sha256": self.version_output_sha256,
            "version_output": self.version_output,
        }


def inspect_tool(executable: os.PathLike[str] | str, *, timeout_s: float = 15.0) -> ToolInfo:
    """Return exact executable identity and ``-version`` output."""

    requested = os.fspath(executable)
    located = shutil.which(requested)
    if located is None:
        candidate = Path(requested).expanduser()
        if not candidate.is_file():
            raise AugmentationError(f"could not find executable {requested!r}")
        located = str(candidate)
    resolved = Path(located).resolve()
    try:
        completed = subprocess.run(
            [str(resolved), "-version"],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AugmentationError(f"could not inspect executable {requested!r}: {exc}") from exc
    if completed.returncode != 0:
        raise AugmentationError(f"{requested!r} -version exited with status {completed.returncode}")
    version_output = (completed.stdout or completed.stderr or "").strip()
    if not version_output:
        raise AugmentationError(f"{requested!r} returned an empty version string")
    return ToolInfo(
        requested=requested,
        resolved_path=str(resolved),
        binary_sha256=sha256_file(resolved) if resolved.is_file() else None,
        version_line=version_output.splitlines()[0],
        version_output_sha256=hashlib.sha256(version_output.encode("utf-8")).hexdigest(),
        version_output=version_output,
    )


@dataclass(frozen=True)
class _SourceWindow:
    record: TrainingRecord
    window: TrainingWindow
    window_index: int
    path: Path
    source_sha256: str
    source_size_bytes: int
    duration_s: float
    coded_width: int
    coded_height: int
    split: str | None
    record_sha256: str

    @property
    def window_duration_s(self) -> float:
        return self.window.end_s - self.window.start_s

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "record_id": self.record.id,
            "group_id": self.record.group_id,
            "window_index": self.window_index,
            "source_sha256": self.source_sha256,
            "window_start_s": self.window.start_s,
            "window_end_s": self.window.end_s,
        }


@dataclass(frozen=True)
class TransformPlan:
    """A shell-free ffmpeg invocation plus the target it supports."""

    name: str
    issue: str
    filter_complex: str
    parameters: dict[str, Any]
    output_duration_estimate_s: float
    label_origin: str = CONTROLLED_CORRUPTION_KIND


@dataclass(frozen=True)
class AugmentedCorpusResult:
    output_dir: Path
    annotations_path: Path
    provenance_path: Path
    skipped_path: Path
    manifest_path: Path
    generated_count: int
    skipped_count: int
    issue_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "annotations": str(self.annotations_path),
            "provenance": str(self.provenance_path),
            "skipped": str(self.skipped_path),
            "manifest": str(self.manifest_path),
            "generated_count": self.generated_count,
            "skipped_count": self.skipped_count,
            "issue_counts": dict(sorted(self.issue_counts.items())),
        }


def _source_split(record: TrainingRecord) -> str | None:
    value = record.extra.get("split")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AugmentationError(f"record {record.id!r} has a non-string or empty split")
    return value


def _resolve_video(record: TrainingRecord, base: Path) -> Path:
    declared = Path(record.video).expanduser()
    if declared.is_absolute():
        candidate = declared.resolve(strict=False)
    else:
        candidate = (base / declared).resolve(strict=False)
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise AugmentationError(
                f"record {record.id!r} video escapes the declared media root: {record.video!r}"
            ) from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _declared_task_label_kind(window: TrainingWindow, task: str) -> str:
    if task not in LABEL_TASKS:
        raise ValueError(f"unknown label task {task!r}")
    provenance = window.extra.get("label_provenance")
    if isinstance(provenance, Mapping):
        task_provenance = provenance.get(task)
        if isinstance(task_provenance, Mapping):
            value = task_provenance.get("kind")
            if isinstance(value, str) and value.strip():
                return value
    return UNLABELED_KIND


def _legacy_label_kind(window: TrainingWindow) -> str | None:
    provenance = window.extra.get("label_provenance")
    if isinstance(provenance, Mapping):
        value = provenance.get("kind")
        if isinstance(value, str) and value.strip():
            return value
    return None


def _source_review_metadata(window: TrainingWindow) -> dict[str, Any]:
    """Return explicitly declared release-review fields without inferring them."""

    result: dict[str, Any] = {}
    provenance = window.extra.get("label_provenance")
    for name in ("review_count", "rubric_version"):
        if name in window.extra:
            result[name] = window.extra[name]
        elif isinstance(provenance, Mapping) and name in provenance:
            result[name] = provenance[name]
    return result


def _has_release_review_metadata(window: TrainingWindow) -> bool:
    review = _source_review_metadata(window)
    count = review.get("review_count")
    rubric = review.get("rubric_version")
    return (
        isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 2
        and isinstance(rubric, str)
        and bool(rubric.strip())
    )


def _source_label_summary(window: TrainingWindow) -> dict[str, Any]:
    summary = {
        "sha256": _json_sha256(window.to_dict()),
        "task_kinds": {task: _declared_task_label_kind(window, task) for task in LABEL_TASKS},
        "annotator": window.annotator,
        "readiness": window.readiness,
        "readiness_valid": window.readiness_valid,
        "issues": dict(window.issues),
        "issue_valid": dict(window.issue_valid),
        "boundaries_s": dict(window.boundaries_s),
        "boundary_valid": (
            dict(window.boundary_valid)
            if isinstance(window.boundary_valid, dict)
            else window.boundary_valid
        ),
    }
    review = _source_review_metadata(window)
    if review:
        summary["review"] = review
    legacy_kind = _legacy_label_kind(window)
    if legacy_kind is not None:
        summary["legacy_kind"] = legacy_kind
    return summary


def _hand_union(window: TrainingWindow, margin: float) -> tuple[float, float, float, float] | None:
    """Return a clamped union of normalized coded-frame hand regions.

    The optional source field is ``hand_regions``: a list of ``[x, y, w, h]``
    boxes normalized to coded-frame width and height.  We deliberately refuse
    to create the controlled ``acting_hand_not_visible`` proxy with the
    synthetic ``hand_occlusion`` transform without these spatial annotations.
    This path is distinct from the faithful ``acting_hand_not_visible``
    re-encode, which requires an existing human-grounded positive label.
    """

    raw = window.extra.get("hand_regions")
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise AugmentationError("hand_regions must be a non-empty list of [x, y, w, h] boxes")
    boxes: list[tuple[float, float, float, float]] = []
    for index, box in enumerate(raw):
        if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
            raise AugmentationError(f"hand_regions[{index}] must contain [x, y, w, h]")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in box):
            raise AugmentationError(f"hand_regions[{index}] must be numeric")
        x, y, width, height = (float(item) for item in box)
        if not all(math.isfinite(item) for item in (x, y, width, height)):
            raise AugmentationError(f"hand_regions[{index}] must be finite")
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
            raise AugmentationError(
                f"hand_regions[{index}] must be a positive normalized box inside [0, 1]"
            )
        boxes.append((x, y, width, height))
    left = max(0.0, min(box[0] for box in boxes) - margin)
    top = max(0.0, min(box[1] for box in boxes) - margin)
    right = min(1.0, max(box[0] + box[2] for box in boxes) + margin)
    bottom = min(1.0, max(box[1] + box[3] for box in boxes) + margin)
    return left, top, right - left, bottom - top


def _common_filter(item: _SourceWindow, config: AugmentationConfig) -> str:
    return (
        f"trim=start={_number(item.window.start_s)}:duration={_number(item.window_duration_s)},"
        f"setpts=PTS-STARTPTS,fps={_number(config.output_fps)},"
        "scale=w='trunc(iw/2)*2':h='trunc(ih/2)*2',setsar=1"
    )


def _has_valid_positive(window: TrainingWindow, issue: str) -> bool:
    return window.issue_valid.get(issue) is True and window.issues.get(issue) is True


def build_transform_plan(
    item: _SourceWindow,
    transform: str,
    config: AugmentationConfig,
) -> TransformPlan | tuple[str, str]:
    """Build a transform plan or return ``(skip_code, explanation)``."""

    if transform not in TRANSFORM_TO_ISSUE:
        raise ValueError(f"unknown transformation {transform!r}")
    duration = item.window_duration_s
    common = _common_filter(item, config)
    issue = TRANSFORM_TO_ISSUE[transform]

    if transform in SOURCE_INHERITED_TRANSFORMS:
        if not _has_valid_positive(item.window, issue):
            return (
                "source_positive_required",
                f"{issue} requires an existing valid positive source label; it is not synthesized",
            )
        source_kind = _declared_task_label_kind(item.window, "issues")
        if source_kind not in HUMAN_GROUNDED_KINDS:
            return (
                "human_issue_provenance_required",
                f"{issue} faithful re-encoding requires explicit human or human-derived "
                "source issue provenance",
            )
        graph = f"[0:v]{common}[outv]"
        parameters = {
            "method": "faithful_window_reencode",
            "source_issue": issue,
            "source_issue_value": True,
            "source_issue_valid": True,
            "source_annotation_kind": source_kind,
        }
        return TransformPlan(
            name=transform,
            issue=issue,
            filter_complex=graph,
            parameters=parameters,
            output_duration_estimate_s=duration,
            label_origin=HUMAN_DERIVED_KIND,
        )
    if transform == "blur":
        graph = (
            f"[0:v]{common},gblur=sigma={_number(config.blur_sigma)}:steps={config.blur_steps}"
            "[outv]"
        )
        parameters = {"sigma": config.blur_sigma, "steps": config.blur_steps}
    elif transform == "exposure":
        graph = (
            f"[0:v]{common},eq=brightness={_number(config.exposure_brightness)}:"
            f"contrast={_number(config.exposure_contrast)}:"
            f"saturation={_number(config.exposure_saturation)}[outv]"
        )
        parameters = {
            "mode": "underexposure",
            "brightness": config.exposure_brightness,
            "contrast": config.exposure_contrast,
            "saturation": config.exposure_saturation,
        }
    elif transform == "camera_instability":
        if duration < 0.5:
            return "window_too_short", "camera instability requires at least 0.5 seconds"
        crop = _number(config.instability_crop_fraction)
        output_width = max(2, item.coded_width // 2 * 2)
        output_height = max(2, item.coded_height // 2 * 2)
        graph = (
            f"[0:v]{common},"
            f"crop=w='trunc(iw*{crop}/2)*2':h='trunc(ih*{crop}/2)*2':"
            f"x='(iw-ow)/2*(1+sin({_number(config.instability_x_frequency)}*t))':"
            f"y='(ih-oh)/2*(1+sin({_number(config.instability_y_frequency)}*t))',"
            f"scale=w={output_width}:h={output_height}[outv]"
        )
        parameters = {
            "method": "time_varying_translation_crop",
            "crop_fraction": config.instability_crop_fraction,
            "x_frequency": config.instability_x_frequency,
            "y_frequency": config.instability_y_frequency,
        }
    elif transform == "freeze":
        frame_count = max(1, int(math.floor(duration * config.output_fps + 1e-9)))
        first = max(1, int(math.floor(frame_count * config.freeze_start_fraction)))
        last = min(frame_count - 1, int(math.ceil(frame_count * config.freeze_end_fraction)) - 1)
        if frame_count < 6 or last <= first:
            return "window_too_short", "freeze requires at least six output frames"
        replace = first - 1
        graph = (
            f"[0:v]{common},split=2[main][reference];"
            f"[main][reference]freezeframes=first={first}:last={last}:replace={replace}[outv]"
        )
        parameters = {
            "method": "frame_replacement",
            "first_frame": first,
            "last_frame": last,
            "replacement_frame": replace,
            "freeze_start_s": first / config.output_fps,
            "freeze_end_s": (last + 1) / config.output_fps,
        }
    elif transform == "scene_cut":
        total_frames = max(1, int(math.floor(duration * config.output_fps + 1e-9)))
        gap_frames = max(2, int(round(total_frames * config.scene_gap_fraction)))
        remaining_frames = total_frames - gap_frames
        first_frames = remaining_frames // 2
        second_frames = remaining_frames - first_frames
        if total_frames < 10 or first_frames < 3 or second_frames < 3:
            return "window_too_short", "scene cut requires at least ten output frames"
        first_duration = first_frames / config.output_fps
        gap_duration = gap_frames / config.output_fps
        second_duration = second_frames / config.output_fps
        second_start = item.window.start_s + first_duration + gap_duration
        normalize = (
            f"fps={_number(config.output_fps)},scale=w='trunc(iw/2)*2':h='trunc(ih/2)*2',setsar=1"
        )
        graph = (
            "[0:v]split=2[first_source][second_source];"
            f"[first_source]trim=start={_number(item.window.start_s)}:"
            f"duration={_number(first_duration)},setpts=PTS-STARTPTS,{normalize}[first];"
            f"[second_source]trim=start={_number(second_start)}:"
            f"duration={_number(second_duration)},setpts=PTS-STARTPTS,{normalize}[second];"
            "[first][second]concat=n=2:v=1:a=0[outv]"
        )
        parameters = {
            "method": "hard_temporal_splice",
            "output_cut_s": first_duration,
            "removed_source_start_s": item.window.start_s + first_duration,
            "removed_source_end_s": second_start,
            "removed_frame_count": gap_frames,
        }
        duration = first_duration + second_duration
    else:
        hand_box = _hand_union(item.window, config.occlusion_margin_fraction)
        if hand_box is None:
            return (
                "hand_regions_missing",
                "hand occlusion is not defensible without normalized hand_regions",
            )
        x, y, width, height = hand_box
        graph = (
            f"[0:v]{common},drawbox=x='iw*{_number(x)}':y='ih*{_number(y)}':"
            f"w='iw*{_number(width)}':h='ih*{_number(height)}':color=black@1:t=fill[outv]"
        )
        parameters = {
            "method": "opaque_hand_region_overlay",
            "normalized_box": [x, y, width, height],
            "source_region_field": "hand_regions",
        }

    return TransformPlan(
        name=transform,
        issue=issue,
        filter_complex=graph,
        parameters=parameters,
        output_duration_estimate_s=duration,
    )


def build_transform_command(
    source_path: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    plan: TransformPlan,
    config: AugmentationConfig,
    *,
    ffmpeg_bin: os.PathLike[str] | str = "ffmpeg",
) -> list[str]:
    """Return the exact shell-free ffmpeg command for one derived clip."""

    return [
        os.fspath(ffmpeg_bin),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-noautorotate",
        "-i",
        str(Path(source_path).expanduser().resolve(strict=False)),
        "-filter_complex",
        plan.filter_complex,
        "-map",
        "[outv]",
        "-an",
        "-c:v",
        config.video_codec,
        "-preset",
        config.preset,
        "-crf",
        str(config.crf),
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        "-metadata",
        "creation_time=1970-01-01T00:00:00Z",
        "-movflags",
        "+faststart",
        "-n",
        str(Path(output_path).expanduser().resolve(strict=False)),
    ]


def _run_transform(command: Sequence[str], output: Path, *, timeout_s: float) -> None:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise AugmentationError(f"ffmpeg exceeded its {timeout_s:g}s timeout") from exc
    except OSError as exc:
        raise AugmentationError(f"could not execute ffmpeg: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "unknown ffmpeg error").strip()[-2000:]
        raise AugmentationError(f"ffmpeg failed with exit code {completed.returncode}: {detail}")
    if not output.is_file() or output.stat().st_size == 0:
        raise AugmentationError(f"ffmpeg reported success but did not create {output}")


def _derived_id(
    item: _SourceWindow,
    plan: TransformPlan,
    config: AugmentationConfig,
    toolchain_sha256: str,
) -> str:
    payload = {
        "schema": AUGMENTATION_SCHEMA,
        "source": item.identity,
        "source_record_sha256": item.record_sha256,
        "transform": plan.name,
        "issue": plan.issue,
        "label_origin": plan.label_origin,
        "parameters": plan.parameters,
        "config": config.to_dict(),
        "toolchain_sha256": toolchain_sha256,
    }
    return "aug-" + _json_sha256(payload)[:32]


def _copied_record_metadata(record: TrainingRecord) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in _COPY_EXTRA_FIELDS:
        if name in record.extra:
            result[name] = record.extra[name]
    return result


def _derived_annotation(
    item: _SourceWindow,
    plan: TransformPlan,
    *,
    record_id: str,
    relative_video: str,
    duration_s: float,
    provenance_id: str,
) -> dict[str, Any]:
    extra = _copied_record_metadata(item.record)
    extra.update(
        {
            "derived": True,
            "source_record_id": item.record.id,
            "source_group_id": item.record.group_id,
            "source_window_index": item.window_index,
            "augmentation": {
                "schema": AUGMENTATION_SCHEMA,
                "transform": plan.name,
                "provenance_id": provenance_id,
            },
        }
    )
    source_issue_kind = _declared_task_label_kind(item.window, "issues")
    source_readiness_kind = _declared_task_label_kind(item.window, "readiness")
    faithful_human_derivative = (
        plan.name in SOURCE_INHERITED_TRANSFORMS
        and plan.label_origin == HUMAN_DERIVED_KIND
        and source_issue_kind in HUMAN_GROUNDED_KINDS
    )
    source_review = _source_review_metadata(item.window)
    inherit_readiness = (
        faithful_human_derivative
        and source_readiness_kind in HUMAN_GROUNDED_KINDS
        and item.window.readiness_valid
        and item.window.readiness is not None
        and _has_release_review_metadata(item.window)
    )
    source_window_sha256 = _json_sha256(item.window.to_dict())
    if faithful_human_derivative:
        issue_provenance = {
            "kind": HUMAN_DERIVED_KIND,
            "inheritance": "source_window",
            "derived_reencode": True,
            "positive_only": True,
            "source_kind": source_issue_kind,
            "source_annotator": item.window.annotator,
            "generator": AUGMENTATION_SCHEMA,
            "transform": plan.name,
            "issue": plan.issue,
            "source_window_sha256": source_window_sha256,
        }
        annotator = f"human-derived:{AUGMENTATION_SCHEMA}"
    else:
        issue_provenance = {
            "kind": CONTROLLED_CORRUPTION_KIND,
            "positive_only": True,
            "generator": AUGMENTATION_SCHEMA,
            "transform": plan.name,
            "issue": plan.issue,
            "source_kind": source_issue_kind,
            "source_window_sha256": source_window_sha256,
        }
        annotator = f"controlled-corruption:{AUGMENTATION_SCHEMA}"
    readiness_provenance: dict[str, Any] = {"kind": UNLABELED_KIND}
    if inherit_readiness:
        readiness_provenance = {
            "kind": HUMAN_DERIVED_KIND,
            "inheritance": "source_window",
            "derived_reencode": True,
            "source_kind": source_readiness_kind,
            "source_annotator": item.window.annotator,
            "source_window_sha256": source_window_sha256,
        }
    label_provenance: dict[str, Any] = {
        "readiness": readiness_provenance,
        "issues": issue_provenance,
        "boundaries": {"kind": UNLABELED_KIND},
    }
    if source_review:
        label_provenance["source_review"] = source_review

    window: dict[str, Any] = {
        "start_s": 0.0,
        "end_s": duration_s,
        "readiness_valid": inherit_readiness,
        "issues": {plan.issue: True},
        "issue_valid": {plan.issue: True},
        "boundary_valid": False,
        "label_provenance": label_provenance,
    }
    if annotator is not None:
        window["annotator"] = annotator
    if inherit_readiness:
        window["readiness"] = item.window.readiness
    if faithful_human_derivative and _has_release_review_metadata(item.window):
        window.update(source_review)
    return {
        "schema": SCHEMA_VERSION,
        "id": record_id,
        "group_id": item.record.group_id,
        "video": relative_video,
        "license": item.record.license,
        "windows": [window],
        **extra,
    }


def _prepare_sources(
    records: Sequence[TrainingRecord],
    *,
    media_root: Path,
    allowed_licenses: set[str],
    ffprobe_bin: str,
    timeout_s: float,
) -> list[_SourceWindow]:
    group_splits: dict[str, str | None] = {}
    metadata_cache: dict[Path, Any] = {}
    prepared: list[_SourceWindow] = []
    for record in sorted(records, key=lambda value: value.id):
        if record.license not in allowed_licenses:
            raise AugmentationError(
                f"record {record.id!r} declares license {record.license!r}, which is not in "
                "the explicit allowed-license set"
            )
        split = _source_split(record)
        if record.group_id in group_splits and group_splits[record.group_id] != split:
            raise AugmentationError(
                f"group {record.group_id!r} crosses source splits "
                f"{group_splits[record.group_id]!r} and {split!r}"
            )
        group_splits[record.group_id] = split
        path = _resolve_video(record, media_root)
        metadata = metadata_cache.get(path)
        if metadata is None:
            try:
                metadata = probe_video(
                    path,
                    ffprobe_bin=ffprobe_bin,
                    calculate_hash=True,
                    timeout_s=timeout_s,
                )
            except VideoProbeError as exc:
                raise AugmentationError(f"could not probe record {record.id!r}: {exc}") from exc
            metadata_cache[path] = metadata
        if metadata.source_sha256 is None or metadata.source_size_bytes is None:
            raise AugmentationError(f"record {record.id!r} source identity is incomplete")
        record_sha256 = _json_sha256(record.to_dict())
        for window_index, window in enumerate(record.windows):
            if window.end_s > metadata.duration_s + 0.05:
                raise AugmentationError(
                    f"record {record.id!r} window {window_index} ends at {window.end_s:g}s, "
                    f"past source duration {metadata.duration_s:g}s"
                )
            prepared.append(
                _SourceWindow(
                    record=record,
                    window=window,
                    window_index=window_index,
                    path=path,
                    source_sha256=metadata.source_sha256,
                    source_size_bytes=metadata.source_size_bytes,
                    duration_s=metadata.duration_s,
                    coded_width=metadata.width,
                    coded_height=metadata.height,
                    split=split,
                    record_sha256=record_sha256,
                )
            )
    return prepared


def _staging_path(output: Path) -> Path:
    return output.with_name(output.name + ".building")


def build_augmented_corpus(
    annotations: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    *,
    allowed_licenses: Iterable[str],
    media_root: os.PathLike[str] | str | None = None,
    config: AugmentationConfig | None = None,
    ffmpeg_bin: os.PathLike[str] | str = "ffmpeg",
    ffprobe_bin: os.PathLike[str] | str = "ffprobe",
) -> AugmentedCorpusResult:
    """Build a deterministic positive-only issue corpus from licensed sources.

    ``allowed_licenses`` is mandatory and exact-match.  The builder does not
    interpret legal terms; it makes the caller state which declared licenses
    have been reviewed for this use, then preserves each declaration without
    substitution.  The destination and its ``.building`` sibling must not
    already exist, preventing accidental overwrite of source or prior data.
    """

    source_annotations_path = Path(annotations).expanduser().resolve()
    if not source_annotations_path.is_file():
        raise FileNotFoundError(source_annotations_path)
    source_annotations_sha256 = sha256_file(source_annotations_path)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    staging = _staging_path(destination)
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    if staging.exists():
        raise FileExistsError(f"staging destination already exists: {staging}")
    license_values = (
        (allowed_licenses,) if isinstance(allowed_licenses, str) else tuple(allowed_licenses)
    )
    if not license_values:
        raise ValueError("allowed_licenses must contain at least one exact license identifier")
    for index, value in enumerate(license_values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"allowed_licenses[{index}] must be a non-empty string")
    license_set = set(license_values)

    run_config = config or AugmentationConfig()
    records = load_jsonl(source_annotations_path)
    if not records:
        raise AugmentationError("the source annotation file contains no records")
    base = (
        Path(media_root).expanduser().resolve()
        if media_root is not None
        else source_annotations_path.parent.resolve()
    )
    if not base.is_dir():
        raise FileNotFoundError(base)

    ffmpeg = inspect_tool(ffmpeg_bin)
    ffprobe = inspect_tool(ffprobe_bin)
    toolchain_sha256 = _json_sha256(
        {
            "ffmpeg_binary": ffmpeg.binary_sha256,
            "ffmpeg_version": ffmpeg.version_output_sha256,
            "ffprobe_binary": ffprobe.binary_sha256,
            "ffprobe_version": ffprobe.version_output_sha256,
        }
    )
    sources = _prepare_sources(
        records,
        media_root=base,
        allowed_licenses=license_set,
        ffprobe_bin=ffprobe.resolved_path,
        timeout_s=run_config.timeout_s,
    )
    if not sources:
        raise AugmentationError("the source annotation file contains no windows")

    staging.mkdir(parents=True)
    (staging / "media").mkdir()
    annotations_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    generated_ids: set[str] = set()

    for item in sources:
        for transform in run_config.transforms:
            planned = build_transform_plan(item, transform, run_config)
            if isinstance(planned, tuple):
                code, explanation = planned
                skipped_rows.append(
                    {
                        "schema": SKIP_SCHEMA,
                        "source": item.identity,
                        "transform": transform,
                        "issue": TRANSFORM_TO_ISSUE[transform],
                        "code": code,
                        "explanation": explanation,
                    }
                )
                continue
            plan = planned
            record_id = _derived_id(item, plan, run_config, toolchain_sha256)
            if record_id in generated_ids:
                raise AugmentationError(f"derived record id collision: {record_id}")
            generated_ids.add(record_id)
            relative_video = Path("media") / plan.name / f"{record_id}.mp4"
            final_in_staging = staging / relative_video
            final_in_staging.parent.mkdir(parents=True, exist_ok=True)
            command = build_transform_command(
                item.path,
                final_in_staging,
                plan,
                run_config,
                ffmpeg_bin=ffmpeg.resolved_path,
            )
            _run_transform(command, final_in_staging, timeout_s=run_config.timeout_s)
            try:
                output_metadata = probe_video(
                    final_in_staging,
                    ffprobe_bin=ffprobe.resolved_path,
                    calculate_hash=True,
                    timeout_s=run_config.timeout_s,
                )
            except VideoProbeError as exc:
                raise AugmentationError(
                    f"could not probe generated record {record_id}: {exc}"
                ) from exc
            if output_metadata.source_sha256 is None or output_metadata.source_size_bytes is None:
                raise AugmentationError(f"generated record {record_id} has no output identity")
            if output_metadata.duration_s <= 0:
                raise AugmentationError(f"generated record {record_id} has invalid duration")
            provenance_id = "prov-" + record_id.removeprefix("aug-")
            annotations_rows.append(
                _derived_annotation(
                    item,
                    plan,
                    record_id=record_id,
                    relative_video=relative_video.as_posix(),
                    duration_s=output_metadata.duration_s,
                    provenance_id=provenance_id,
                )
            )
            provenance_rows.append(
                {
                    "schema": PROVENANCE_SCHEMA,
                    "id": provenance_id,
                    "derived_record_id": record_id,
                    "generator": {"name": "egosieve", "version": __version__},
                    "source": {
                        **item.identity,
                        "path": str(item.path),
                        "size_bytes": item.source_size_bytes,
                        "record_sha256": item.record_sha256,
                        "license": item.record.license,
                        "split": item.split,
                        "label_summary": _source_label_summary(item.window),
                        "ffprobe_argv": build_ffprobe_command(
                            item.path, ffprobe_bin=ffprobe.resolved_path
                        ),
                    },
                    "transformation": {
                        "name": plan.name,
                        "target_issue": plan.issue,
                        "label_kind": plan.label_origin,
                        "positive_only": True,
                        "parameters": plan.parameters,
                        "filter_complex": plan.filter_complex,
                    },
                    "command": {
                        "argv": command,
                        "display": shlex.join(command),
                        "shell": False,
                    },
                    "output": {
                        "path": relative_video.as_posix(),
                        "sha256": output_metadata.source_sha256,
                        "size_bytes": output_metadata.source_size_bytes,
                        "duration_s": output_metadata.duration_s,
                        "width": output_metadata.display_width,
                        "height": output_metadata.display_height,
                        "fps": output_metadata.fps,
                        "codec": output_metadata.codec_name,
                        "ffprobe_argv": build_ffprobe_command(
                            final_in_staging, ffprobe_bin=ffprobe.resolved_path
                        ),
                    },
                }
            )

    if not annotations_rows:
        shutil.rmtree(staging)
        raise AugmentationError("all requested transformations were skipped; no corpus was created")

    # Hash again after every encode.  A source that changes during the build
    # would make the recorded command truthful but its input identity false.
    verified_source_paths: set[Path] = set()
    for item in sources:
        if item.path in verified_source_paths:
            continue
        verified_source_paths.add(item.path)
        if item.path.stat().st_size != item.source_size_bytes:
            raise AugmentationError(f"source changed size during build: {item.path}")
        if sha256_file(item.path) != item.source_sha256:
            raise AugmentationError(f"source changed content during build: {item.path}")
    if sha256_file(source_annotations_path) != source_annotations_sha256:
        raise AugmentationError(
            f"source annotations changed content during build: {source_annotations_path}"
        )

    annotations_path = staging / "annotations.jsonl"
    provenance_path = staging / "provenance.jsonl"
    skipped_path = staging / "skipped.jsonl"
    _write_jsonl(annotations_path, annotations_rows)
    _write_jsonl(provenance_path, provenance_rows)
    _write_jsonl(skipped_path, skipped_rows)

    issue_counts = Counter(
        next(iter(record["windows"][0]["issues"])) for record in annotations_rows
    )
    transform_counts = Counter(record["augmentation"]["transform"] for record in annotations_rows)
    split_counts = Counter(record.get("split", "<unassigned>") for record in annotations_rows)
    license_counts = Counter(record["license"] for record in annotations_rows)
    skip_counts = Counter(row["code"] for row in skipped_rows)
    label_origin_counts = Counter(row["transformation"]["label_kind"] for row in provenance_rows)
    inherited_readiness_count = sum(
        bool(record["windows"][0].get("readiness_valid")) for record in annotations_rows
    )
    source_fingerprints = sorted(
        {(item.source_sha256, item.source_size_bytes, str(item.path)) for item in sources}
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generator": {"name": "egosieve", "version": __version__},
        "build_fingerprint": _json_sha256(
            {
                "source_annotations_sha256": source_annotations_sha256,
                "source_files": [
                    {"sha256": sha, "size_bytes": size} for sha, size, _ in source_fingerprints
                ],
                "config": run_config.to_dict(),
                "allowed_licenses": sorted(license_set),
                "toolchain_sha256": toolchain_sha256,
            }
        ),
        "source_annotations": {
            "path": str(source_annotations_path),
            "sha256": source_annotations_sha256,
            "record_count": len(records),
            "window_count": len(sources),
        },
        "source_files": [
            {"path": path, "sha256": sha, "size_bytes": size}
            for sha, size, path in source_fingerprints
        ],
        "licensing": {
            "policy": "exact-match allowlist; each derivative retains its source license",
            "allowed_licenses": sorted(license_set),
            "derived_records_by_license": dict(sorted(license_counts.items())),
        },
        "label_policy": {
            "kinds": dict(sorted(label_origin_counts.items())),
            "positive_only": True,
            "unknown_issues_are_negative": False,
            "programmatic_readiness_valid": False,
            "source_human_readiness_inherited_when_valid": True,
            "boundary_valid": False,
            "hand_occlusion_transform_requires": "normalized coded-frame hand_regions",
            "source_inherited_issues": list(SOURCE_INHERITED_TRANSFORMS),
            "source_inherited_requires": (
                "an existing valid positive source label with explicit human or "
                "human-derived issue provenance"
            ),
            "human_review_fields_inherited_only_for": HUMAN_DERIVED_KIND,
        },
        "split_policy": {
            "group_id_preserved": True,
            "source_split_preserved_when_present": True,
            "cross_group_scene_splices": False,
            "derived_records_by_split": dict(sorted(split_counts.items())),
        },
        "configuration": run_config.to_dict(),
        "tools": {
            "ffmpeg": ffmpeg.to_dict(),
            "ffprobe": ffprobe.to_dict(),
            "toolchain_sha256": toolchain_sha256,
        },
        "artifacts": {
            "annotations": {
                "path": "annotations.jsonl",
                "sha256": sha256_file(annotations_path),
            },
            "provenance": {
                "path": "provenance.jsonl",
                "sha256": sha256_file(provenance_path),
            },
            "skipped": {"path": "skipped.jsonl", "sha256": sha256_file(skipped_path)},
        },
        "counts": {
            "generated": len(annotations_rows),
            "skipped": len(skipped_rows),
            "by_transform": dict(sorted(transform_counts.items())),
            "by_issue": dict(sorted(issue_counts.items())),
            "by_label_origin": dict(sorted(label_origin_counts.items())),
            "inherited_readiness": inherited_readiness_count,
            "skipped_by_code": dict(sorted(skip_counts.items())),
        },
    }
    manifest_path = staging / "manifest.json"
    _write_json(manifest_path, manifest)
    os.replace(staging, destination)

    return AugmentedCorpusResult(
        output_dir=destination,
        annotations_path=destination / "annotations.jsonl",
        provenance_path=destination / "provenance.jsonl",
        skipped_path=destination / "skipped.jsonl",
        manifest_path=destination / "manifest.json",
        generated_count=len(annotations_rows),
        skipped_count=len(skipped_rows),
        issue_counts=dict(issue_counts),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="egosieve-build-corpus",
        description="Build deterministic positive issue examples from explicitly licensed videos.",
    )
    parser.add_argument("annotations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument(
        "--allowed-license",
        action="append",
        required=True,
        help="exact declared license reviewed for this use; repeat for multiple licenses",
    )
    parser.add_argument(
        "--transform",
        action="append",
        choices=TRANSFORM_NAMES,
        help="transformation to include; repeat as needed (default: all)",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = build_augmented_corpus(
            args.annotations,
            args.output,
            media_root=args.media_root,
            allowed_licenses=args.allowed_license,
            config=AugmentationConfig(
                transforms=_normalize_transforms(args.transform),
                output_fps=args.fps,
                crf=args.crf,
                preset=args.preset,
                timeout_s=args.timeout,
            ),
            ffmpeg_bin=args.ffmpeg,
            ffprobe_bin=args.ffprobe,
        )
    except (AugmentationError, FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"egosieve-build-corpus: error: {exc}\n")
    json.dump(result.to_dict(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


__all__ = [
    "AUGMENTATION_SCHEMA",
    "MANIFEST_SCHEMA",
    "PROGRAMMATIC_TRANSFORMS",
    "PROVENANCE_SCHEMA",
    "SKIP_SCHEMA",
    "SOURCE_INHERITED_TRANSFORMS",
    "TRANSFORM_NAMES",
    "TRANSFORM_TO_ISSUE",
    "AugmentationConfig",
    "AugmentationError",
    "AugmentedCorpusResult",
    "TransformPlan",
    "build_augmented_corpus",
    "build_transform_command",
    "build_transform_plan",
    "inspect_tool",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
