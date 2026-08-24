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
from pathlib import Path
from typing import Any

from egosieve.training import ISSUE_LABELS, group_assignments, load_jsonl
from egosieve.training.augmentation import AugmentationConfig, build_augmented_corpus

SCHEMA = "egosieve.training/v1"
RECIPE_SCHEMA = "egosieve.v01-corpus-recipe/v1"
HUMAN_DERIVED = {"kind": "human-derived"}
UNLABELED = {"kind": "unlabeled"}
CONTROLLED = {"kind": "programmatic-controlled-corruption"}
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
    declared = Path(str(record["video"])).expanduser()
    path = declared if declared.is_absolute() else acquired_root / declared
    result = path.resolve(strict=True)
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _task_provenance(
    *,
    readiness: Mapping[str, str] = UNLABELED,
    issues: Mapping[str, str] = UNLABELED,
    boundaries: Mapping[str, str] = UNLABELED,
) -> dict[str, dict[str, str]]:
    return {
        "readiness": dict(readiness),
        "issues": dict(issues),
        "boundaries": dict(boundaries),
    }


def _has_valid_boundary(window: Mapping[str, Any]) -> bool:
    value = window.get("boundary_valid", False)
    return any(value.values()) if isinstance(value, dict) else bool(value)


def _decorate_readiness_window(window: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(window))
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

    result["annotator"] = "human-derived:HoloAssist-v1_1"
    result["label_provenance"] = _task_provenance(
        readiness=HUMAN_DERIVED,
        issues=HUMAN_DERIVED if issues else UNLABELED,
        boundaries=HUMAN_DERIVED if _has_valid_boundary(result) else UNLABELED,
    )
    result["issue_proxy_basis"] = (
        "zero publisher fine-action occupancy"
        if readiness == "REJECT"
        else "at least 0.5 publisher fine-action occupancy"
        if readiness == "KEEP"
        else None
    )
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
            if not isinstance(value, Mapping) or not isinstance(value.get("id"), int):
                continue
            actions[int(value["id"])] = copy.deepcopy(dict(value))
    return sorted(
        actions.values(),
        key=lambda action: (float(action["start_s"]), float(action["end_s"]), int(action["id"])),
    )


def _visibility_window(action: Mapping[str, Any], *, present: bool) -> dict[str, Any]:
    return {
        "start_s": float(action["start_s"]),
        "end_s": float(action["end_s"]),
        "readiness_valid": False,
        "issues": {"no_hands": not present, "hand_occlusion": not present},
        "issue_valid": {"no_hands": True, "hand_occlusion": True},
        "boundary_valid": False,
        "annotator": "human-derived:HoloAssist-v1_1",
        "label_provenance": _task_provenance(issues=HUMAN_DERIVED),
        "source_annotation": copy.deepcopy(dict(action)),
        "visibility_proxy_scope": (
            "publisher modifier for the acting hand; it does not prove every hand is absent"
        ),
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
        "label_provenance": _task_provenance(issues=CONTROLLED),
        "controlled_reference": {
            "role": "unmodified source for paired deterministic corruption",
            "source_proxy_id": window.get("proxy_id"),
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
        result["windows"] = [_control_window(chosen)]
        records.append(result)
    return records


def _absolute_derived_records(
    annotations_path: Path,
    *,
    final_controlled_root: Path,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(annotations_path)
    for row in rows:
        relative = Path(str(row["video"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unexpected generated media path: {relative}")
        row["video"] = str((final_controlled_root / relative).resolve(strict=False))
        row["source"] = "HoloAssist controlled corruptions"
    return rows


def _valid_boundary(window: Mapping[str, Any], name: str) -> bool:
    declared = window.get("boundary_valid", False)
    if isinstance(declared, Mapping):
        return declared.get(name) is True
    return bool(declared) and window.get("boundaries_s", {}).get(name) is not None


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
        issue_positive = Counter()
        issue_negative = Counter()
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
                    readiness[str(window["readiness"])] += 1
                for issue, valid in window.get("issue_valid", {}).items():
                    if valid is not True:
                        continue
                    target = window.get("issues", {}).get(issue)
                    (issue_positive if target is True else issue_negative)[issue] += 1
                for boundary in ("start", "end"):
                    if _valid_boundary(window, boundary):
                        boundaries[boundary] += 1
        support[split] = {
            "groups": len(groups),
            "examples": examples,
            "readiness": dict(sorted(readiness.items())),
            "issue_positive": dict(sorted(issue_positive.items())),
            "issue_negative": dict(sorted(issue_negative.items())),
            "boundaries": dict(sorted(boundaries.items())),
        }
    return {"assignments": assignments, "support": support}


def _supports_release(split: Mapping[str, Any]) -> bool:
    readiness = split["readiness"]
    positives = split["issue_positive"]
    negatives = split["issue_negative"]
    boundaries = split["boundaries"]
    return (
        all(readiness.get(label, 0) > 0 for label in READINESS_ORDER)
        and all(positives.get(issue, 0) > 0 for issue in ISSUE_LABELS)
        and all(negatives.get(issue, 0) > 0 for issue in ISSUE_LABELS)
        and all(boundaries.get(name, 0) > 0 for name in ("start", "end"))
    )


def _find_seed(
    rows: Sequence[Mapping[str, Any]],
    *,
    fractions: tuple[float, float, float],
    maximum: int = 100_000,
) -> tuple[int, dict[str, Any]]:
    for seed in range(maximum):
        report = _split_support(rows, seed=seed, fractions=fractions)
        if _supports_release(report["support"]["validation"]) and _supports_release(
            report["support"]["test"]
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
    adapted_path = adapted_path.resolve(strict=True)
    acquired_root = acquired_root.resolve(strict=True)
    output = output.resolve(strict=False)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    rows = _read_jsonl(adapted_path)
    if any(row.get("source") != "HoloAssist" for row in rows):
        raise ValueError("the recipe accepts only HoloAssist adapter records")
    if any(row.get("license") != "CDLA-Permissive-2.0" for row in rows):
        raise ValueError("the recipe requires the declared HoloAssist license identifier")

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
                "left/right hand; not a claim about every hand in the frame"
            ),
            "low_hand_activity_proxy": ("zero fine-action occupancy versus at least 0.5 occupancy"),
            "controlled_transforms": list(CONTROLLED_TRANSFORMS),
            "controlled_reference_policy": (
                "highest-occupancy selected KEEP window per source video"
            ),
        },
        "split": {
            "fractions": {
                "train": fractions[0],
                "validation": fractions[1],
                "test": fractions[2],
            },
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
