#!/usr/bin/env python3
"""Build the metadata-only EgoSieve-Eval Hugging Face dataset release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from egosieve.release import validate_release
from egosieve.training import ISSUE_LABELS, load_jsonl

ROW_SCHEMA = "egosieve.eval-row/v1"
MANIFEST_SCHEMA = "egosieve.eval-dataset-release/v1"
SPLITS = ("train", "validation", "test")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    materialized = [dict(row) for row in rows]
    path.write_text("".join(_canonical(row) + "\n" for row in materialized), encoding="utf-8")
    return len(materialized)


def _split_index(model_dir: Path) -> dict[str, str]:
    document = _load_object(model_dir / "splits.json")
    result: dict[str, str] = {}
    for row in document.get("examples", []):
        if not isinstance(row, dict):
            raise ValueError("splits.json contains a non-object row")
        example_id = str(row.get("id", ""))
        split = str(row.get("split", ""))
        if not example_id or split not in SPLITS or example_id in result:
            raise ValueError("splits.json contains an invalid or duplicate example")
        result[example_id] = split
    if not result:
        raise ValueError("splits.json contains no examples")
    return result


def _source_video_id(record: Mapping[str, Any]) -> str | None:
    provenance = record.get("source_provenance")
    if isinstance(provenance, Mapping):
        value = provenance.get("video_name")
        if isinstance(value, str) and value.strip():
            return value.strip()
    record_id = record.get("source_record_id")
    if isinstance(record_id, str) and record_id.startswith("holoassist-"):
        return record_id.removeprefix("holoassist-")
    return None


def _media_reference(record: Mapping[str, Any]) -> dict[str, Any]:
    video_id = _source_video_id(record)
    derived = bool(record.get("derived"))
    return {
        "included": False,
        "kind": "controlled-derivative-withheld" if derived else "publisher-media-withheld",
        "source_video_id": video_id,
        "access": "https://holoassist.github.io/#download",
    }


def _boundary_valid(value: Any) -> dict[str, bool]:
    if isinstance(value, bool):
        return {"start": value, "end": value}
    if isinstance(value, Mapping):
        return {name: value.get(name) is True for name in ("start", "end")}
    return {"start": False, "end": False}


def _row(
    record: Mapping[str, Any],
    window: Mapping[str, Any],
    *,
    example_id: str,
    split: str,
) -> dict[str, Any]:
    readiness_valid = window.get("readiness_valid") is True
    issue_valid = {name: window.get("issue_valid", {}).get(name) is True for name in ISSUE_LABELS}
    issues = {
        name: window.get("issues", {}).get(name) if issue_valid[name] else None
        for name in ISSUE_LABELS
    }
    boundary_valid = _boundary_valid(window.get("boundary_valid", False))
    boundaries = window.get("boundaries_s", {})
    annotation_metadata = {
        name: window[name]
        for name in (
            "review_count",
            "review_count_scope",
            "rubric_version",
            "fine_action_occupancy",
            "keep_occupancy_threshold",
            "issue_proxy_basis",
            "visibility_proxy_scope",
            "controlled_reference",
            "source_annotation",
            "source_annotations",
            "derived_from",
            "augmentation",
        )
        if name in window
    }
    return {
        "schema": ROW_SCHEMA,
        "id": example_id,
        "split": split,
        "group_id": str(record["group_id"]),
        "source": str(record.get("source", "HoloAssist")),
        "source_record_id": str(record.get("source_record_id", record["id"])),
        "media": _media_reference(record),
        "license": str(record["license"]),
        "start_s": float(window["start_s"]),
        "end_s": float(window["end_s"]),
        "readiness": window.get("readiness") if readiness_valid else None,
        "readiness_valid": readiness_valid,
        "issues": issues,
        "issue_valid": issue_valid,
        "boundaries_s": {
            name: boundaries.get(name) if boundary_valid[name] else None
            for name in ("start", "end")
        },
        "boundary_valid": boundary_valid,
        "label_provenance": window.get("label_provenance"),
        "annotation_metadata": annotation_metadata,
    }


def _flatten(
    annotations_path: Path,
    splits: Mapping[str, str],
) -> dict[str, list[dict[str, Any]]]:
    raw_rows = [
        json.loads(line)
        for line in annotations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    load_jsonl(annotations_path)
    output: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    seen: set[str] = set()
    for record in raw_rows:
        if not isinstance(record, dict) or not isinstance(record.get("windows"), list):
            raise ValueError("training annotations contain an invalid record")
        for index, window in enumerate(record["windows"]):
            example_id = f"{record['id']}:{index}"
            split = splits.get(example_id)
            if split is None:
                raise ValueError(f"annotation example {example_id!r} is absent from splits.json")
            output[split].append(_row(record, window, example_id=example_id, split=split))
            seen.add(example_id)
    if seen != set(splits):
        missing = sorted(set(splits) - seen)
        raise ValueError(f"splits.json contains examples absent from annotations: {missing[:5]}")
    for rows in output.values():
        rows.sort(key=lambda row: row["id"])
    return output


def _card(metrics: Mapping[str, Any], counts: Mapping[str, int], source_commit: str) -> str:
    evaluation = metrics["evaluation"]
    readiness = evaluation["readiness_examples_by_provenance"]
    issue = evaluation["issue_examples_by_provenance"]
    return f"""---
license: cdla-permissive-2.0
task_categories:
  - video-classification
tags:
  - egocentric-video
  - robotics
  - video-quality
  - temporal-segmentation
  - metadata-only
pretty_name: EgoSieve-Eval
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: validation
        path: data/validation.jsonl
      - split: test
        path: data/test.jsonl
---

# EgoSieve-Eval

EgoSieve-Eval is the metadata-only, source-grouped evidence index used for
EgoSieve-S v0.1. It contains {sum(counts.values())} labeled window rows across
{counts["train"]} train, {counts["validation"]} validation, and
{counts["test"]} test examples. Source and generated videos are deliberately
not redistributed.

## What the labels mean

Readiness and boundary targets are derived from HoloAssist v1_1 fine-action
intervals using a published fixed-grid occupancy rule. The test set contains
{readiness["human"]} direct-human and {readiness["human-derived"]} human-derived
readiness rows. Human-derived rows are not independent judgments under the
EgoSieve rubric; the recorded review count applies to HoloAssist source
interval review.

`low_hand_activity` is an occupancy proxy. `acting_hand_not_visible` follows
HoloAssist's acting-hand visibility modifier; it does not assert that every
hand is absent from a frame. Blur, exposure, camera instability, duplicate
frames, and scene-cut labels are paired controlled corruptions versus
unmodified references. Their metrics measure that discrimination setting, not
human-audited natural prevalence.

Held-out issue rows by provenance: {issue["human"]} human,
{issue["human-derived"]} human-derived, and
{issue["programmatic-controlled-corruption"]} controlled-corruption.
Unknown task targets remain null and masked.

## Splits and leakage

Splits are assigned by original HoloAssist source video, so a source window,
its unmodified reference, and every generated variant stay in one split. The
exact membership is in `evidence/splits.json`; raw held-out predictions and
recomputed metrics are included alongside it.

## Media, license, and privacy

This repository contains labels, source identifiers, timestamps, and model
outputs only. Obtain source media from the
[HoloAssist project](https://holoassist.github.io/#download) under its terms.
HoloAssist declares CDLA-Permissive-2.0. First-person video can contain faces,
screens, homes, and bystanders; the absence of media here is intentional and
does not replace the source dataset's consent and privacy documentation.

The transport mirror used for the local build was pinned for byte stability
but did not declare its own license or establish publisher byte equivalence;
it is not presented here as the rights source.

## Reproducibility and limitations

The build recipe is in
[`scripts/build_v01_corpus.py`](https://github.com/publu/egosieve/blob/{source_commit}/scripts/build_v01_corpus.py).
Every label row carries task-level validity and provenance. These weak-label
metrics should not be read as performance on a direct, independently annotated
in-the-wild benchmark, and the model must not be used for robot control.

## Citation

```bibtex
@inproceedings{{wang2023holoassist,
  title={{HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World}},
  author={{Wang, et al.}},
  booktitle={{ICCV}},
  year={{2023}}
}}
```
"""


def build_release(
    annotations: Path,
    model_dir: Path,
    output: Path,
) -> dict[str, Any]:
    annotations = annotations.resolve(strict=True)
    model_dir = model_dir.resolve(strict=True)
    output = output.resolve(strict=False)
    if output.exists():
        raise FileExistsError(output)
    validation = validate_release(model_dir)
    splits = _split_index(model_dir)
    rows = _flatten(annotations, splits)
    output.mkdir(parents=True)
    data_dir = output / "data"
    evidence_dir = output / "evidence"
    data_dir.mkdir()
    evidence_dir.mkdir()
    counts = {split: _write_jsonl(data_dir / f"{split}.jsonl", rows[split]) for split in SPLITS}
    for name in ("splits.json", "test_predictions.json", "metrics.json", "training_report.json"):
        shutil.copy2(model_dir / name, evidence_dir / name)
    metrics = _load_object(model_dir / "metrics.json")
    report = _load_object(model_dir / "training_report.json")
    source_commit = str(report["source_commit"])
    (output / "README.md").write_text(
        _card(metrics, counts, source_commit),
        encoding="utf-8",
    )
    artifacts = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "media_included": False,
        "model_release_validated": bool(validation),
        "source_commit": source_commit,
        "counts": counts,
        "artifacts": artifacts,
        "label_names": {
            "readiness": ["KEEP", "REVIEW", "REJECT"],
            "issues": list(ISSUE_LABELS),
            "boundaries": ["start", "end"],
        },
    }
    _write_json(output / "manifest.json", manifest)
    return {"output": str(output), "counts": counts, "artifacts": len(artifacts) + 1}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    print(json.dumps(build_release(args.annotations, args.model_dir, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
