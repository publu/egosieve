from __future__ import annotations

import json

import numpy as np
import pytest

from egosieve import ISSUE_LABELS as MODEL_ISSUE_LABELS
from egosieve.training.data import (
    ISSUE_LABELS,
    SCHEMA_VERSION,
    TrainingDataError,
    encode_records,
    loads_jsonl,
    parse_record,
)
from egosieve.training.splits import grouped_split
from egosieve.training.targets import build_sampled_targets


def _record(record_id: str = "episode-1", group_id: str = "session-1") -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "id": record_id,
        "group_id": group_id,
        "video": f"videos/{record_id}.mp4",
        "license": "dataset-owner-declared",
        "windows": [],
    }


def test_training_and_model_issue_vocabularies_are_identical() -> None:
    assert ISSUE_LABELS == MODEL_ISSUE_LABELS


def test_sparse_labels_become_values_with_independent_validity_masks() -> None:
    raw = _record()
    raw["windows"] = [
        {"start_s": 0.0, "end_s": 2.0},
        {
            "start_s": 3.0,
            "end_s": 6.0,
            "readiness": "REVIEW",
            "readiness_valid": True,
            "issues": {"blur": False},
            "issue_valid": {"blur": True},
            "boundaries_s": {"start": 3.2},
            "boundary_valid": {"start": True},
        },
    ]

    targets = encode_records([parse_record(raw)])

    assert targets.readiness_labels.tolist() == [-100, 1]
    assert targets.readiness_label_mask.tolist() == [False, True]
    blur = ISSUE_LABELS.index("blur")
    assert targets.issue_labels[:, blur].tolist() == [0.0, 0.0]
    assert targets.issue_label_mask[:, blur].tolist() == [False, True]
    assert not targets.issue_label_mask[:, ISSUE_LABELS.index("acting_hand_not_visible")].any()
    np.testing.assert_allclose(targets.boundary_times_s[1], [3.2, 0.0])
    assert targets.boundary_time_mask.tolist() == [[False, False], [True, False]]


def test_jsonl_reports_version_and_duplicate_id_errors() -> None:
    wrong = _record()
    wrong["schema"] = "egosieve.training/v2"
    with pytest.raises(TrainingDataError, match="unsupported schema"):
        loads_jsonl(json.dumps(wrong))

    line = json.dumps(_record())
    with pytest.raises(TrainingDataError, match="duplicate record id"):
        loads_jsonl(f"{line}\n{line}\n")


def test_issue_typo_is_rejected_unless_vocabulary_is_explicit() -> None:
    raw = _record()
    raw["windows"] = [
        {
            "start_s": 0.0,
            "end_s": 1.0,
            "issues": {"blru": True},
            "issue_valid": {"blru": True},
        }
    ]
    with pytest.raises(TrainingDataError, match="unknown issue"):
        parse_record(raw)

    custom = parse_record(raw, issue_names=("blru",))
    targets = encode_records([custom], issue_names=("blru",))
    assert targets.issue_labels.tolist() == [[1.0]]
    assert targets.issue_label_mask.tolist() == [[True]]


@pytest.mark.parametrize("legacy_issue", ["no_hands", "hand_occlusion"])
def test_legacy_issue_keys_are_rejected(legacy_issue: str) -> None:
    raw = _record()
    raw["windows"] = [
        {
            "start_s": 0.0,
            "end_s": 1.0,
            "issues": {legacy_issue: True},
            "issue_valid": {legacy_issue: True},
        }
    ]

    with pytest.raises(TrainingDataError, match="unknown issue"):
        parse_record(raw)


def test_boundary_timestamps_are_rasterized_with_nearest_frame_and_tolerance() -> None:
    raw = _record()
    raw["windows"] = [
        {
            "start_s": 0.0,
            "end_s": 4.0,
            "boundaries_s": {"start": 1.5, "end": 3.9},
            "boundary_valid": True,
        }
    ]
    window = parse_record(raw).windows[0]
    targets = build_sampled_targets(
        [window],
        [[1.0, 2.0, 3.0]],
        boundary_tolerance_s=0.5,
    )

    # The start is equidistant from frames 0 and 1, so the earlier frame wins.
    assert targets.boundary_labels[0, :, 0].tolist() == [1.0, 0.0, 0.0]
    assert targets.boundary_label_mask[0, :, 0].tolist() == [True, True, True]
    # The end annotation is too far away; it cannot become all-negative data.
    assert targets.boundary_labels[0, :, 1].tolist() == [0.0, 0.0, 0.0]
    assert targets.boundary_label_mask[0, :, 1].tolist() == [False, False, False]
    assert targets.boundary_matched_mask.tolist() == [[True, False]]


def test_grouped_split_is_isolated_deterministic_and_input_order_independent() -> None:
    records = [_record(f"episode-{index}", f"session-{index // 2}") for index in range(12)]
    first = grouped_split(records, fractions=(0.5, 0.25, 0.25), seed=123)
    second = grouped_split(reversed(records), fractions=(0.5, 0.25, 0.25), seed=123)

    membership = {
        record["id"]: split_name
        for split_name, split_records in first.items()
        for record in split_records
    }
    reversed_membership = {
        record["id"]: split_name
        for split_name, split_records in second.items()
        for record in split_records
    }
    assert membership == reversed_membership

    group_splits: dict[str, set[str]] = {}
    for split_name, split_records in first.items():
        for record in split_records:
            group_splits.setdefault(record["group_id"], set()).add(split_name)
    assert all(len(split_names) == 1 for split_names in group_splits.values())
    assert all(first[name] for name in ("train", "validation", "test"))


def test_grouped_split_rejects_invalid_fractions_and_missing_groups() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        grouped_split([_record()], fractions=(0.8, 0.3, 0.0))
    missing = _record()
    del missing["group_id"]
    with pytest.raises(ValueError, match="group"):
        grouped_split([missing])
