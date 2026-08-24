#!/usr/bin/env python3
"""Build the local mixed-evidence corpus used for EgoSieve-S v0.1.

The input is the uncapped output of ``egosieve corpus adapt-holoassist``.
This recipe keeps the source media outside the repository, creates a balanced
readiness subset, derives two conservative annotation proxies, and generates
matched controlled corruptions for five purely visual issue heads.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from egosieve.training import ISSUE_LABELS, group_assignments, load_jsonl
from egosieve.training.augmentation import AugmentationConfig, build_augmented_corpus

SCHEMA = "egosieve.training/v1"
RECIPE_SCHEMA = "egosieve.v01-corpus-recipe/v1"
HUMAN_DERIVED_KIND = "human-derived"
CONTROLLED_KIND = "programmatic-controlled-corruption"
UNLABELED_KIND = "unlabeled"
HUMAN_GROUNDED_KINDS = frozenset({"human", HUMAN_DERIVED_KIND})
ISSUE_EVIDENCE_KINDS = frozenset({"human", HUMAN_DERIVED_KIND, CONTROLLED_KIND})
UNLABELED = {"kind": UNLABELED_KIND}
CONTROLLED_TRANSFORMS = (
    "blur",
    "exposure",
    "camera_instability",
    "freeze",
    "scene_cut",
)
CONTROLLED_ISSUES = (
    "blur",
    "exposure",
    "camera_instability",
    "duplicate_frames",
    "scene_cut",
)
READINESS_ORDER = {"KEEP": 0, "REVIEW": 1, "REJECT": 2}
VISIBILITY_POSITIVE = "hand not visible"
VISIBILITY_NEGATIVE = {"left hand", "right hand"}
VISIBILITY_ISSUE = "acting_hand_not_visible"
MIN_SPLIT_EVIDENCE = 3


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(_canonical(dict(row)) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no records")
    return rows


def _window_key(window: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        float(window["start_s"]),
        float(window["end_s"]),
        READINESS_ORDER.get(str(window.get("readiness")), 99),
        str(window.get("proxy_id", "")),
        _canonical(window),
    )


def _evenly_select(
    windows: Sequence[Mapping[str, Any]],
    cap: int,
) -> list[dict[str, Any]]:
    ordered = sorted((copy.deepcopy(dict(window)) for window in windows), key=_window_key)
    if len(ordered) <= cap:
        return ordered
    if cap == 1:
        return [ordered[(len(ordered) - 1) // 2]]
    indexes = [(index * (len(ordered) - 1)) // (cap - 1) for index in range(cap)]
    return [ordered[index] for index in indexes]


def _resolve_video(record: Mapping[str, Any], acquired_root: Path) -> Path:
    root = acquired_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    declared_value = record.get("video")
    if not isinstance(declared_value, str) or not declared_value.strip():
        raise ValueError(f"record {record.get('id')!r} has no non-empty video path")
    declared = Path(declared_value)
    if declared.is_absolute() or ".." in declared.parts:
        raise ValueError(f"record {record.get('id')!r} video must be relative to the acquired root")
    result = (root / declared).resolve(strict=True)
    try:
        result.relative_to(root)
    except ValueError as error:
        raise ValueError(f"record {record.get('id')!r} video escapes the acquired root") from error
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _task_provenance(
    *,
    readiness: Mapping[str, Any] = UNLABELED,
    issues: Mapping[str, Any] = UNLABELED,
    boundaries: Mapping[str, Any] = UNLABELED,
) -> dict[str, dict[str, Any]]:
    return {
        "readiness": dict(readiness),
        "issues": dict(issues),
        "boundaries": dict(boundaries),
    }


def _has_valid_boundary(window: Mapping[str, Any]) -> bool:
    value = window.get("boundary_valid", False)
    return any(value.values()) if isinstance(value, dict) else bool(value)


def _source_proxy_metadata(window: Mapping[str, Any]) -> Mapping[str, Any]:
    label_source = window.get("label_source")
    if not isinstance(label_source, Mapping):
        raise ValueError("readiness source window lacks label_source metadata")
    if label_source.get("kind") != "programmatic_readiness_proxy":
        raise ValueError("readiness source window is not a HoloAssist readiness proxy")
    if label_source.get("human_reviewed") is not False:
        raise ValueError("derived readiness must explicitly declare human_reviewed=false")
    if window.get("review_count_scope") != "source_action_intervals":
        raise ValueError("HoloAssist review_count must be scoped to source_action_intervals")
    return label_source


def _validate_source_contract(rows: Sequence[Mapping[str, Any]]) -> None:
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    media_groups: dict[str, str] = {}
    for index, record in enumerate(rows):
        location = f"source record {index}"
        record_id = record.get("id")
        group_id = record.get("group_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"{location} has no non-empty id")
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(f"{location} has no non-empty group_id")
        if record_id in seen_ids:
            raise ValueError(f"duplicate source record id {record_id!r}")
        if group_id in seen_groups:
            raise ValueError(f"duplicate HoloAssist source group {group_id!r}")
        seen_ids.add(record_id)
        seen_groups.add(group_id)

        if record.get("schema") != SCHEMA:
            raise ValueError(f"record {record_id!r} must use schema {SCHEMA!r}")
        if record.get("source") != "HoloAssist":
            raise ValueError(f"record {record_id!r} is not a HoloAssist adapter record")
        if record.get("license") != "CDLA-Permissive-2.0":
            raise ValueError(f"record {record_id!r} has an unexpected license identifier")

        label_policy = record.get("label_policy")
        if not isinstance(label_policy, Mapping):
            raise ValueError(f"record {record_id!r} lacks adapter label_policy metadata")
        if label_policy.get("kind") != "programmatic_readiness_proxy":
            raise ValueError(f"record {record_id!r} has an unexpected label policy")
        if label_policy.get("human_reviewed") is not False:
            raise ValueError(f"record {record_id!r} must declare human_reviewed=false")
        if label_policy.get("issue_targets") is not False:
            raise ValueError(f"record {record_id!r} adapter must not supply issue targets")

        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"record {record_id!r} lacks source provenance")
        for name in ("annotation_release", "task_id", "task_type", "batch"):
            if not isinstance(provenance.get(name), str) or not str(provenance[name]).strip():
                raise ValueError(f"record {record_id!r} provenance.{name} must be non-empty")
        mirror = provenance.get("media_mirror")
        if not isinstance(mirror, Mapping):
            raise ValueError(f"record {record_id!r} lacks media_mirror provenance")
        mirror_path = mirror.get("path")
        if not isinstance(mirror_path, str) or not mirror_path.strip():
            raise ValueError(f"record {record_id!r} media_mirror.path must be non-empty")
        expected_video = (PurePosixPath("files") / PurePosixPath(mirror_path)).as_posix()
        if record.get("video") != expected_video:
            raise ValueError(
                f"record {record_id!r} video does not match its media_mirror provenance"
            )
        previous_group = media_groups.setdefault(expected_video, group_id)
        if previous_group != group_id:
            raise ValueError(f"source media {expected_video!r} appears in multiple leakage groups")

        windows = record.get("windows")
        if not isinstance(windows, list) or not windows:
            raise ValueError(f"record {record_id!r} has no source windows")
        seen_proxy_ids: set[str] = set()
        for window_index, window in enumerate(windows):
            if not isinstance(window, Mapping):
                raise ValueError(f"record {record_id!r} window {window_index} must be an object")
            _source_proxy_metadata(window)
            proxy_id = window.get("proxy_id")
            if not isinstance(proxy_id, str) or not proxy_id.strip():
                raise ValueError(f"record {record_id!r} window {window_index} lacks a proxy_id")
            if proxy_id in seen_proxy_ids:
                raise ValueError(f"record {record_id!r} repeats proxy_id {proxy_id!r}")
            seen_proxy_ids.add(proxy_id)


def _decorate_readiness_window(window: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(window))
    source_metadata = _source_proxy_metadata(result)
    readiness = str(result.get("readiness"))
    if readiness not in READINESS_ORDER or result.get("readiness_valid") is not True:
        raise ValueError("readiness source window lacks a valid KEEP/REVIEW/REJECT target")

    issues: dict[str, bool] = {}
    issue_valid: dict[str, bool] = {}
    if readiness == "KEEP":
        issues["low_hand_activity"] = False
        issue_valid["low_hand_activity"] = True
    elif readiness == "REJECT" and float(result.get("fine_action_occupancy", -1)) == 0.0:
        issues["low_hand_activity"] = True
        issue_valid["low_hand_activity"] = True
    if issues:
        result["issues"] = issues
        result["issue_valid"] = issue_valid
    else:
        result.pop("issues", None)
        result.pop("issue_valid", None)

    proxy_provenance = {
        "kind": HUMAN_DERIVED_KIND,
        "direct_egosieve_reviewed": False,
        "source_kind": "programmatic_readiness_proxy",
        "source_review_scope": "source_action_intervals",
        "source_interval_review_count": source_metadata.get("source_interval_review_count"),
    }
    boundary_provenance = {
        "kind": HUMAN_DERIVED_KIND,
        "direct_egosieve_reviewed": False,
        "source_kind": "publisher_fine_action_interval_boundaries",
        "source_review_scope": "source_action_intervals",
    }
    result["annotator"] = "derived-proxy:HoloAssist-v1_1"
    result["label_provenance"] = _task_provenance(
        readiness=proxy_provenance,
        issues=(
            {
                **proxy_provenance,
                "source_kind": "programmatic_fine_action_occupancy_proxy",
                "target_issue": "low_hand_activity",
            }
            if issues
            else UNLABELED
        ),
        boundaries=boundary_provenance if _has_valid_boundary(result) else UNLABELED,
    )
    if issues:
        result["issue_proxy_basis"] = (
            "zero publisher fine-action occupancy"
            if readiness == "REJECT"
            else "publisher fine-action occupancy at or above the readiness threshold"
        )
        result["issue_proxy_human_reviewed"] = False
    return result


def _record_shell(record: Mapping[str, Any], *, record_id: str, video: Path) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "id": record_id,
        "group_id": str(record["group_id"]),
        "video": str(video),
        "license": str(record["license"]),
        "source": "HoloAssist",
        "dataset_revision": record.get("provenance", {}).get("annotation_release", "v1_1"),
        "source_record_id": str(record["id"]),
        "source_provenance": copy.deepcopy(record.get("provenance", {})),
    }


def _readiness_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    acquired_root: Path,
    class_caps: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    records: list[dict[str, Any]] = []
    selected_by_group: dict[str, list[dict[str, Any]]] = {}
    for source in sorted(rows, key=lambda row: str(row["id"])):
        video = _resolve_video(source, acquired_root)
        source_windows = source.get("windows")
        if not isinstance(source_windows, list):
            raise ValueError(f"record {source.get('id')!r} has no windows array")
        selected: list[dict[str, Any]] = []
        for readiness in ("KEEP", "REVIEW", "REJECT"):
            candidates = [
                window for window in source_windows if window.get("readiness") == readiness
            ]
            selected.extend(_evenly_select(candidates, class_caps[readiness]))
        selected = sorted(selected, key=_window_key)
        if {str(window.get("readiness")) for window in selected} != set(READINESS_ORDER):
            raise ValueError(f"record {source['id']!r} does not contribute all readiness classes")
        decorated = [_decorate_readiness_window(window) for window in selected]
        result = _record_shell(source, record_id=str(source["id"]), video=video)
        result["windows"] = decorated
        records.append(result)
        selected_by_group[str(source["group_id"])] = decorated
    return records, selected_by_group


def _deduplicated_actions(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions: dict[int, dict[str, Any]] = {}
    windows = record.get("windows")
    if not isinstance(windows, list):
        return []
    for window in windows:
        if not isinstance(window, Mapping):
            continue
        annotations = window.get("source_annotations", [])
        if not isinstance(annotations, list):
            continue
        for value in annotations:
            if (
                not isinstance(value, Mapping)
                or isinstance(value.get("id"), bool)
                or not isinstance(value.get("id"), int)
            ):
                continue
            event_id = int(value["id"])
            candidate = copy.deepcopy(dict(value))
            existing = actions.get(event_id)
            if existing is not None and _canonical(existing) != _canonical(candidate):
                raise ValueError(
                    f"record {record.get('id')!r} has conflicting copies of source event {event_id}"
                )
            actions[event_id] = candidate
    return sorted(
        actions.values(),
        key=lambda action: (float(action["start_s"]), float(action["end_s"]), int(action["id"])),
    )


def _visibility_window(action: Mapping[str, Any], *, present: bool) -> dict[str, Any]:
    source_modifier = str(action.get("attributes", {}).get("adverbial", ""))
    return {
        "start_s": float(action["start_s"]),
        "end_s": float(action["end_s"]),
        "readiness_valid": False,
        "issues": {VISIBILITY_ISSUE: not present},
        "issue_valid": {VISIBILITY_ISSUE: True},
        "boundary_valid": False,
        "annotator": "derived-proxy:HoloAssist-v1_1",
        "label_provenance": _task_provenance(
            issues={
                "kind": HUMAN_DERIVED_KIND,
                "direct_egosieve_reviewed": False,
                "source_kind": "publisher_acting_hand_adverbial_proxy",
                "source_review_scope": "source_action_intervals",
                "target_issue": VISIBILITY_ISSUE,
            }
        ),
        "source_annotation": copy.deepcopy(dict(action)),
        "visibility_proxy_scope": {
            "field": "Fine grained action.attributes.adverbial",
            "source_value": source_modifier,
            "positive_rule": "exact value 'hand not visible'",
            "negative_rule": "exact value 'left hand' or 'right hand'",
            "scope": "acting hand only; never a claim that every hand is absent",
            "human_reviewed_as_egosieve_issue": False,
        },
    }


def _visibility_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    acquired_root: Path,
    per_class_cap: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in sorted(rows, key=lambda row: str(row["id"])):
        positive: list[dict[str, Any]] = []
        negative: list[dict[str, Any]] = []
        for action in _deduplicated_actions(source):
            attributes = action.get("attributes", {})
            adverbial = str(attributes.get("adverbial", "")).strip().casefold()
            if adverbial == VISIBILITY_POSITIVE:
                positive.append(_visibility_window(action, present=False))
            elif adverbial in VISIBILITY_NEGATIVE:
                negative.append(_visibility_window(action, present=True))
        windows = _evenly_select(positive, per_class_cap) + _evenly_select(negative, per_class_cap)
        if not windows:
            continue
        result = _record_shell(
            source,
            record_id=f"{source['id']}-hand-visibility",
            video=_resolve_video(source, acquired_root),
        )
        result["windows"] = sorted(windows, key=_window_key)
        records.append(result)
    return records


def _control_window(window: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "start_s": float(window["start_s"]),
        "end_s": float(window["end_s"]),
        "readiness_valid": False,
        "issues": {issue: False for issue in CONTROLLED_ISSUES},
        "issue_valid": {issue: True for issue in CONTROLLED_ISSUES},
        "boundary_valid": False,
        "annotator": "controlled-reference:egosieve-v0.1",
        "label_provenance": _task_provenance(
            issues={
                "kind": CONTROLLED_KIND,
                "role": "matched_unmodified_reference",
                "natural_issue_absence_human_reviewed": False,
                "metric_scope": "injected-corruption-vs-unmodified discrimination",
            }
        ),
        "controlled_reference": {
            "role": "unmodified source for paired deterministic corruption",
            "source_proxy_id": window.get("proxy_id"),
            "natural_issue_absence_human_reviewed": False,
            "metric_scope": "injected-corruption-vs-unmodified discrimination",
        },
    }


def _control_records(
    rows: Sequence[Mapping[str, Any]],
    selected_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    acquired_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in sorted(rows, key=lambda row: str(row["id"])):
        group = str(source["group_id"])
        keep = [window for window in selected_by_group[group] if window.get("readiness") == "KEEP"]
        if not keep:
            raise ValueError(f"group {group!r} has no KEEP control candidate")
        chosen = sorted(
            keep,
            key=lambda window: (
                -float(window.get("fine_action_occupancy", 0.0)),
                float(window["start_s"]),
            ),
        )[0]
        result = _record_shell(
            source,
            record_id=f"{source['id']}-controlled-reference",
            video=_resolve_video(source, acquired_root),
        )
        result["source"] = "HoloAssist controlled corruptions"
        result["evidence_scope"] = "injected-corruption-vs-unmodified discrimination"
        result["windows"] = [_control_window(chosen)]
        records.append(result)
    return records


def _absolute_derived_records(
    annotations_path: Path,
    *,
    final_controlled_root: Path,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(annotations_path)
    root = final_controlled_root.resolve(strict=True)
    for row in rows:
        relative = Path(str(row["video"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unexpected generated media path: {relative}")
        resolved = (root / relative).resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"generated media escapes controlled root: {relative}") from error
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        row["video"] = str(resolved)
        row["source"] = "HoloAssist controlled corruptions"
        row["evidence_scope"] = "injected-corruption-vs-unmodified discrimination"
    return rows


def _valid_boundary(window: Mapping[str, Any], name: str) -> bool:
    declared = window.get("boundary_valid", False)
    if isinstance(declared, Mapping):
        return declared.get(name) is True
    return bool(declared) and window.get("boundaries_s", {}).get(name) is not None


def _provenance_kind(window: Mapping[str, Any], task: str) -> str:
    provenance = window.get("label_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("a selected window lacks task-level label_provenance")
    task_provenance = provenance.get(task)
    if not isinstance(task_provenance, Mapping):
        raise ValueError(f"a selected window lacks label_provenance.{task}")
    kind = task_provenance.get("kind")
    if not isinstance(kind, str):
        raise ValueError(f"a selected window lacks label_provenance.{task}.kind")
    return kind


def _split_support(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    fractions: tuple[float, float, float],
) -> dict[str, Any]:
    assignments = group_assignments(
        rows,
        train_fraction=fractions[0],
        validation_fraction=fractions[1],
        test_fraction=fractions[2],
        seed=seed,
    )
    support: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        readiness = Counter()
        readiness_provenance = Counter()
        issue_positive = Counter()
        issue_negative = Counter()
        issue_provenance = Counter()
        boundaries = Counter()
        groups = set()
        examples = 0
        for record in rows:
            if assignments[str(record["group_id"])] != split:
                continue
            groups.add(str(record["group_id"]))
            for window in record["windows"]:
                examples += 1
                if window.get("readiness_valid") is True:
                    kind = _provenance_kind(window, "readiness")
                    if kind not in HUMAN_GROUNDED_KINDS:
                        raise ValueError("a valid readiness target lacks human-derived provenance")
                    readiness[str(window["readiness"])] += 1
                    readiness_provenance[kind] += 1
                valid_issue_kind: str | None = None
                for issue, valid in window.get("issue_valid", {}).items():
                    if valid is not True:
                        continue
                    target = window.get("issues", {}).get(issue)
                    if not isinstance(target, bool):
                        raise ValueError(f"valid issue target {issue!r} must be boolean")
                    kind = _provenance_kind(window, "issues")
                    if kind not in ISSUE_EVIDENCE_KINDS:
                        raise ValueError(f"valid issue target {issue!r} has invalid provenance")
                    if valid_issue_kind is not None and kind != valid_issue_kind:
                        raise ValueError("one issue row cannot declare multiple provenance kinds")
                    valid_issue_kind = kind
                    (issue_positive if target is True else issue_negative)[issue] += 1
                if valid_issue_kind is not None:
                    issue_provenance[valid_issue_kind] += 1
                for boundary in ("start", "end"):
                    if _valid_boundary(window, boundary):
                        if _provenance_kind(window, "boundaries") not in HUMAN_GROUNDED_KINDS:
                            raise ValueError(
                                f"valid {boundary} boundary lacks human-derived provenance"
                            )
                        boundaries[boundary] += 1
        support[split] = {
            "groups": len(groups),
            "examples": examples,
            "readiness": dict(sorted(readiness.items())),
            "readiness_by_provenance": dict(sorted(readiness_provenance.items())),
            "issue_positive": dict(sorted(issue_positive.items())),
            "issue_negative": dict(sorted(issue_negative.items())),
            "issue_by_provenance": dict(sorted(issue_provenance.items())),
            "boundaries": dict(sorted(boundaries.items())),
        }
    return {"assignments": assignments, "support": support}


def _supports_release(
    split: Mapping[str, Any],
    *,
    minimum: int = MIN_SPLIT_EVIDENCE,
) -> bool:
    readiness = split["readiness"]
    positives = split["issue_positive"]
    negatives = split["issue_negative"]
    boundaries = split["boundaries"]
    issue_provenance = split["issue_by_provenance"]
    return (
        split.get("groups", 0) >= minimum
        and all(readiness.get(label, 0) >= minimum for label in READINESS_ORDER)
        and all(positives.get(issue, 0) >= minimum for issue in ISSUE_LABELS)
        and all(negatives.get(issue, 0) >= minimum for issue in ISSUE_LABELS)
        and all(boundaries.get(name, 0) >= minimum for name in ("start", "end"))
        and issue_provenance.get(CONTROLLED_KIND, 0) >= minimum
    )


def _find_seed(
    rows: Sequence[Mapping[str, Any]],
    *,
    fractions: tuple[float, float, float],
    maximum: int = 100_000,
) -> tuple[int, dict[str, Any]]:
    for seed in range(maximum):
        report = _split_support(rows, seed=seed, fractions=fractions)
        if all(
            _supports_release(report["support"][split]) for split in ("train", "validation", "test")
        ):
            return seed, report
    raise ValueError(f"no fully supported grouped split seed found below {maximum}")


def build_corpus(
    adapted_path: Path,
    acquired_root: Path,
    output: Path,
    *,
    keep_cap: int = 12,
    review_cap: int = 12,
    reject_cap: int = 6,
    visibility_cap: int = 6,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, Any]:
    caps = {
        "keep_cap": keep_cap,
        "review_cap": review_cap,
        "reject_cap": reject_cap,
        "visibility_cap": visibility_cap,
    }
    for name, value in caps.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError("train, validation, and test fractions must all be positive")
    required_issues = set(CONTROLLED_ISSUES) | {"low_hand_activity", VISIBILITY_ISSUE}
    missing_issues = sorted(required_issues - set(ISSUE_LABELS))
    if missing_issues:
        raise RuntimeError(f"recipe issue targets are absent from ISSUE_LABELS: {missing_issues}")

    adapted_path = adapted_path.resolve(strict=True)
    acquired_root = acquired_root.resolve(strict=True)
    output = output.resolve(strict=False)
    if output.exists():
        raise FileExistsError(output)

    rows = _read_jsonl(adapted_path)
    _validate_source_contract(rows)
    # Validate the fraction tuple before any output path is created.
    group_assignments(
        rows,
        train_fraction=fractions[0],
        validation_fraction=fractions[1],
        test_fraction=fractions[2],
        seed=0,
    )

    readiness, selected = _readiness_records(
        rows,
        acquired_root=acquired_root,
        class_caps={"KEEP": keep_cap, "REVIEW": review_cap, "REJECT": reject_cap},
    )
    visibility = _visibility_records(
        rows,
        acquired_root=acquired_root,
        per_class_cap=visibility_cap,
    )
    controls = _control_records(rows, selected, acquired_root=acquired_root)
    output.mkdir(parents=True)
    control_sources_path = output / "controlled-sources.jsonl"
    _write_jsonl(control_sources_path, controls)
    load_jsonl(control_sources_path)

    controlled_root = output / "controlled"
    augmentation = build_augmented_corpus(
        control_sources_path,
        controlled_root,
        allowed_licenses=["CDLA-Permissive-2.0"],
        config=AugmentationConfig(transforms=CONTROLLED_TRANSFORMS),
    )
    derived = _absolute_derived_records(
        augmentation.annotations_path,
        final_controlled_root=controlled_root,
    )
    combined = readiness + visibility + controls + derived
    combined.sort(key=lambda row: str(row["id"]))
    record_ids = [str(row["id"]) for row in combined]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError("combined corpus contains duplicate record ids")
    annotations_path = output / "annotations.jsonl"
    _write_jsonl(annotations_path, combined)
    validated = load_jsonl(annotations_path)
    if len(validated) != len(combined):
        raise RuntimeError("combined corpus did not round-trip through the training parser")

    seed, split_report = _find_seed(combined, fractions=fractions)
    counts = {
        "records": len(combined),
        "windows": sum(len(record["windows"]) for record in combined),
        "readiness_records": len(readiness),
        "visibility_records": len(visibility),
        "controlled_reference_records": len(controls),
        "controlled_derived_records": len(derived),
    }
    manifest = {
        "schema": RECIPE_SCHEMA,
        "source": {
            "adapted_annotations": str(adapted_path),
            "adapted_annotations_sha256": _sha256(adapted_path),
            "acquired_root": str(acquired_root),
            "dataset": "HoloAssist",
            "license": "CDLA-Permissive-2.0",
        },
        "selection": {
            "readiness_caps_per_video": {
                "KEEP": keep_cap,
                "REVIEW": review_cap,
                "REJECT": reject_cap,
            },
            "visibility_cap_per_polarity_per_video": visibility_cap,
            "visibility_proxy": (
                "HoloAssist acting-hand adverbial: 'hand not visible' versus explicit "
                "left/right hand, supervising only acting_hand_not_visible; not a claim "
                "about every hand in the frame"
            ),
            "visibility_proxy_human_reviewed_as_egosieve_issue": False,
            "low_hand_activity_proxy": ("zero fine-action occupancy versus at least 0.5 occupancy"),
            "controlled_transforms": list(CONTROLLED_TRANSFORMS),
            "controlled_reference_policy": (
                "highest-occupancy selected KEEP window per source video; unmodified "
                "references are not human audits of natural issue absence"
            ),
            "controlled_metric_scope": "injected-corruption-vs-unmodified discrimination",
        },
        "split": {
            "fractions": {
                "train": fractions[0],
                "validation": fractions[1],
                "test": fractions[2],
            },
            "minimum_evidence_per_class_or_issue_polarity": MIN_SPLIT_EVIDENCE,
            "support_required_in": ["train", "validation", "test"],
            "recommended_seed": seed,
            **split_report,
        },
        "counts": counts,
        "artifacts": {
            "annotations": {
                "path": "annotations.jsonl",
                "sha256": _sha256(annotations_path),
            },
            "controlled_sources": {
                "path": "controlled-sources.jsonl",
                "sha256": _sha256(control_sources_path),
            },
            "augmentation_manifest": {
                "path": "controlled/manifest.json",
                "sha256": _sha256(augmentation.manifest_path),
            },
            "augmentation_provenance": {
                "path": "controlled/provenance.jsonl",
                "sha256": _sha256(augmentation.provenance_path),
            },
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"output": str(output), "manifest": str(manifest_path), **counts, "seed": seed}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapted", type=Path)
    parser.add_argument("--acquired-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--keep-cap", type=int, default=12)
    parser.add_argument("--review-cap", type=int, default=12)
    parser.add_argument("--reject-cap", type=int, default=6)
    parser.add_argument("--visibility-cap", type=int, default=6)
    return parser


def main() -> int:
    args = _parser().parse_args()
    for name in ("keep_cap", "review_cap", "reject_cap", "visibility_cap"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"{name.replace('_', '-')} must be positive")
    result = build_corpus(
        args.adapted,
        args.acquired_root,
        args.output,
        keep_cap=args.keep_cap,
        review_cap=args.review_cap,
        reject_cap=args.reject_cap,
        visibility_cap=args.visibility_cap,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
