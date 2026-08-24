from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_v01_corpus.py"
SPEC = importlib.util.spec_from_file_location("build_v01_corpus", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe)


def _action(event_id: int = 7, *, adverbial: str = "hand not visible") -> dict:
    return {
        "id": event_id,
        "start_s": 1.0,
        "end_s": 2.0,
        "attributes": {"adverbial": adverbial},
    }


def _source_window(proxy_id: str = "grid-000001") -> dict:
    return {
        "proxy_id": proxy_id,
        "start_s": 0.0,
        "end_s": 6.0,
        "readiness": "KEEP",
        "readiness_valid": True,
        "fine_action_occupancy": 0.75,
        "boundary_valid": False,
        "review_count": 2,
        "review_count_scope": "source_action_intervals",
        "label_source": {
            "kind": "programmatic_readiness_proxy",
            "human_reviewed": False,
            "source_interval_review_count": 2,
        },
        "source_annotations": [_action()],
    }


def _source_record() -> dict:
    return {
        "schema": recipe.SCHEMA,
        "id": "holoassist-video-1",
        "group_id": "holoassist:video-1",
        "video": "files/HoloAssist/video/video-1.mp4",
        "license": "CDLA-Permissive-2.0",
        "source": "HoloAssist",
        "label_policy": {
            "kind": "programmatic_readiness_proxy",
            "human_reviewed": False,
            "issue_targets": False,
        },
        "provenance": {
            "annotation_release": "v1_1",
            "task_id": "task-1",
            "task_type": "assembly",
            "batch": "batch-1",
            "media_mirror": {"path": "HoloAssist/video/video-1.mp4"},
        },
        "windows": [_source_window()],
    }


def _supported_split(count: int = recipe.MIN_SPLIT_EVIDENCE) -> dict:
    return {
        "groups": count,
        "examples": count,
        "readiness": {name: count for name in recipe.READINESS_ORDER},
        "readiness_by_provenance": {recipe.HUMAN_DERIVED_KIND: count * 3},
        "issue_positive": {name: count for name in recipe.ISSUE_LABELS},
        "issue_negative": {name: count for name in recipe.ISSUE_LABELS},
        "issue_by_provenance": {recipe.CONTROLLED_KIND: count},
        "boundaries": {"start": count, "end": count},
    }


def test_selection_is_input_order_independent_even_for_tied_windows() -> None:
    windows = [
        {"start_s": 0, "end_s": 6, "proxy_id": "same", "marker": marker}
        for marker in ("c", "a", "b")
    ]
    assert recipe._evenly_select(windows, 3) == recipe._evenly_select(list(reversed(windows)), 3)


def test_event_deduplication_rejects_conflicting_copies() -> None:
    action = _action()
    record = {"id": "source", "windows": [{"source_annotations": [action]}] * 2}
    assert recipe._deduplicated_actions(record) == [action]

    conflict = copy.deepcopy(action)
    conflict["end_s"] = 3.0
    record["windows"][1] = {"source_annotations": [conflict]}
    with pytest.raises(ValueError, match="conflicting copies"):
        recipe._deduplicated_actions(record)


def test_media_resolution_is_confined_to_acquired_root(tmp_path: Path) -> None:
    root = tmp_path / "acquired"
    video = root / "files" / "HoloAssist" / "video" / "video-1.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    record = _source_record()
    assert recipe._resolve_video(record, root) == video.resolve()

    record["video"] = str(video)
    with pytest.raises(ValueError, match="relative to the acquired root"):
        recipe._resolve_video(record, root)
    record["video"] = "../video-1.mp4"
    with pytest.raises(ValueError, match="relative to the acquired root"):
        recipe._resolve_video(record, root)


def test_source_contract_binds_media_and_task_provenance() -> None:
    record = _source_record()
    recipe._validate_source_contract([record])

    mismatched = copy.deepcopy(record)
    mismatched["video"] = "files/HoloAssist/video/other.mp4"
    with pytest.raises(ValueError, match="media_mirror provenance"):
        recipe._validate_source_contract([mismatched])

    missing_task = copy.deepcopy(record)
    missing_task["provenance"]["task_id"] = ""
    with pytest.raises(ValueError, match="provenance.task_id"):
        recipe._validate_source_contract([missing_task])


def test_proxy_provenance_and_acting_hand_scope_are_explicit() -> None:
    decorated = recipe._decorate_readiness_window(_source_window())
    readiness = decorated["label_provenance"]["readiness"]
    assert readiness["kind"] == "human-derived"
    assert readiness["direct_egosieve_reviewed"] is False
    assert readiness["source_review_scope"] == "source_action_intervals"
    assert decorated["annotator"].startswith("derived-proxy:")

    positive = recipe._visibility_window(_action(), present=False)
    negative = recipe._visibility_window(_action(adverbial="left hand"), present=True)
    assert positive["issues"] == {"acting_hand_not_visible": True}
    assert negative["issues"] == {"acting_hand_not_visible": False}
    assert positive["visibility_proxy_scope"]["scope"] == (
        "acting hand only; never a claim that every hand is absent"
    )
    assert "no_hands" not in positive["issues"]


def test_control_rows_disclose_their_metric_scope() -> None:
    window = recipe._control_window(_source_window())
    provenance = window["label_provenance"]["issues"]
    assert provenance["kind"] == recipe.CONTROLLED_KIND
    assert provenance["natural_issue_absence_human_reviewed"] is False
    assert provenance["metric_scope"] == "injected-corruption-vs-unmodified discrimination"


def test_seed_search_requires_minimum_support_in_every_split(monkeypatch) -> None:
    def fake_support(_rows, *, seed, fractions):
        del fractions
        support = {name: _supported_split() for name in ("train", "validation", "test")}
        if seed == 0:
            support["train"]["issue_positive"] = dict(
                support["train"]["issue_positive"], acting_hand_not_visible=2
            )
        return {"assignments": {}, "support": support}

    monkeypatch.setattr(recipe, "_split_support", fake_support)
    seed, _ = recipe._find_seed([], fractions=(0.7, 0.15, 0.15), maximum=2)
    assert seed == 1


def test_generated_media_paths_must_resolve_inside_controlled_root(tmp_path: Path) -> None:
    root = tmp_path / "controlled"
    media = root / "media" / "blur" / "clip.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"clip")
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(json.dumps({"id": "derived", "video": "media/blur/clip.mp4"}) + "\n")
    rows = recipe._absolute_derived_records(annotations, final_controlled_root=root)
    assert rows[0]["video"] == str(media.resolve())

    annotations.write_text(json.dumps({"id": "escape", "video": "../outside.mp4"}) + "\n")
    with pytest.raises(ValueError, match="unexpected generated media path"):
        recipe._absolute_derived_records(annotations, final_controlled_root=root)
