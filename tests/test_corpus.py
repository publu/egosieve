from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from egosieve import cli
from egosieve.corpus import (
    CORPUS_MANIFEST_NAME,
    EGO_TACTILE,
    HOLOASSIST,
    HOLOASSIST_ANNOTATION_PATH,
    PUBLISHED_REVIEW_COUNT,
    READINESS_RUBRIC_VERSION,
    CorpusAcquisitionError,
    adapt_acquired_holoassist,
    build_ego_tactile_records,
    build_holoassist_records,
    fetch_selected_files,
    verify_corpus,
    write_training_jsonl,
)
from egosieve.training.data import loads_jsonl

FIXTURES = Path(__file__).parent / "fixtures" / "ego_tactile"
HOLOASSIST_FIXTURES = Path(__file__).parent / "fixtures" / "holoassist"
REVISION = "90c4e304b8e3a9578d5bb938992206358db1a660"
HOLOASSIST_REVISION = "a57f1f2078a7b01ea87014050fdb3afe169e54f1"
HOLOASSIST_VIDEO_ID = "R999-Fixture-Task"
HOLOASSIST_VIDEO_PATH = f"HoloAssist/video/{HOLOASSIST_VIDEO_ID}.mp4"


def _fixture_json(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _fixture_frames() -> list[dict]:
    return [
        json.loads(line)
        for line in (FIXTURES / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _holoassist_annotations() -> list[dict]:
    return json.loads((HOLOASSIST_FIXTURES / "annotations.json").read_text(encoding="utf-8"))


def test_explicit_fetch_records_revision_license_attribution_and_hashes(tmp_path: Path) -> None:
    selected = ["meta/info.json", "videos/observation.images.ego/chunk-000/file-000.mp4"]
    payloads = {
        EGO_TACTILE.resolve_url(REVISION, selected[0]): b'{"codebase_version":"v3.0"}\n',
        EGO_TACTILE.resolve_url(REVISION, selected[1]): b"offline-video-fixture",
    }
    requested: list[str] = []

    def opener(request):
        requested.append(request.full_url)
        return io.BytesIO(payloads[request.full_url])

    output = tmp_path / "corpus"
    manifest = fetch_selected_files(
        EGO_TACTILE,
        revision=REVISION,
        repository_paths=selected,
        output_dir=output,
        accepted_license="CC-BY-4.0",
        opener=opener,
    )

    assert requested == [EGO_TACTILE.resolve_url(REVISION, path) for path in selected]
    assert manifest["selection_policy"] == "explicit-files-only"
    assert manifest["source"]["repository"]["revision"] == REVISION
    assert manifest["source"]["license"]["url"] == EGO_TACTILE.license_url
    assert manifest["source"]["license"]["attribution"] == EGO_TACTILE.attribution
    assert [entry["repository_path"] for entry in manifest["files"]] == selected
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    assert (output / CORPUS_MANIFEST_NAME).is_file()

    verified = verify_corpus(output)
    assert verified["file_count"] == 2
    assert verified["total_size_bytes"] == sum(len(payloads[url]) for url in requested)


@pytest.mark.parametrize(
    ("paths", "license_id", "match"),
    [
        (["../secret"], "CC-BY-4.0", "unsafe"),
        (["videos/**/*.mp4"], "CC-BY-4.0", "glob"),
        (["meta/info.json"], "cc-by-4.0", "acknowledgement"),
        (["meta/info.json", "meta/info.json"], "CC-BY-4.0", "duplicates"),
    ],
)
def test_fetch_rejects_implicit_or_unsafe_selection_before_network(
    tmp_path: Path, paths: list[str], license_id: str, match: str
) -> None:
    called = False

    def opener(_request):
        nonlocal called
        called = True
        raise AssertionError("network must not be reached")

    with pytest.raises(CorpusAcquisitionError, match=match):
        fetch_selected_files(
            EGO_TACTILE,
            revision=REVISION,
            repository_paths=paths,
            output_dir=tmp_path / "corpus",
            accepted_license=license_id,
            opener=opener,
        )
    assert not called


def test_verify_corpus_detects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "corpus"
    url = EGO_TACTILE.resolve_url(REVISION, "meta/info.json")
    fetch_selected_files(
        EGO_TACTILE,
        revision=REVISION,
        repository_paths=["meta/info.json"],
        output_dir=output,
        accepted_license="CC-BY-4.0",
        opener=lambda _request: io.BytesIO(b"original"),
    )
    (output / "files" / "meta" / "info.json").write_bytes(b"tampered")

    with pytest.raises(CorpusAcquisitionError, match="(size|SHA-256) mismatch"):
        verify_corpus(output)
    assert (
        url == json.loads((output / CORPUS_MANIFEST_NAME).read_text())["files"][0]["download_url"]
    )


def test_fetch_rejects_mutable_revision_and_verify_links_url_to_commit(tmp_path: Path) -> None:
    with pytest.raises(CorpusAcquisitionError, match="full 40-character"):
        fetch_selected_files(
            EGO_TACTILE,
            revision="main",
            repository_paths=["meta/info.json"],
            output_dir=tmp_path / "mutable",
            accepted_license="CC-BY-4.0",
            opener=lambda _request: (_ for _ in ()).throw(AssertionError("no network")),
        )

    output = tmp_path / "corpus"
    fetch_selected_files(
        EGO_TACTILE,
        revision=REVISION,
        repository_paths=["meta/info.json"],
        output_dir=output,
        accepted_license="CC-BY-4.0",
        opener=lambda _request: io.BytesIO(b"fixture"),
    )
    manifest_path = output / CORPUS_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["repository"]["revision"] = "a" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CorpusAcquisitionError, match="download URL"):
        verify_corpus(output)


def test_ego_tactile_adapter_emits_only_disclosed_proxy_boundaries(tmp_path: Path) -> None:
    info = _fixture_json("info.json")
    episodes = _fixture_json("episodes.json")
    frames = _fixture_frames()
    source = _fixture_json("manifest-source.json")

    records = build_ego_tactile_records(
        info,
        reversed(episodes),
        reversed(frames),
        episode_indexes=[1, 0],
        manifest_source=source,
    )
    repeated = build_ego_tactile_records(
        info,
        episodes,
        frames,
        episode_indexes=[0, 1],
        manifest_source=source,
    )
    assert records == repeated
    assert [record["id"] for record in records] == [
        "ego-tactile-episode-000000",
        "ego-tactile-episode-000001",
    ]
    assert records[0]["group_id"] == records[1]["group_id"]
    assert records[0]["video"] == ("files/videos/observation.images.ego/chunk-000/file-000.mp4")
    assert records[0]["license"] == "CC-BY-4.0"
    assert records[0]["label_policy"] == {
        "kind": "proxy",
        "human_reviewed": False,
        "proxy_boundary_targets": True,
        "readiness_targets": False,
        "issue_targets": False,
        "publisher_method": "physical contact and grip-force action segmentation",
    }
    assert [(window["start_s"], window["end_s"]) for window in records[0]["windows"]] == [
        (0.0, 1.0),
        (1.0, 2.0),
    ]
    assert [(window["start_s"], window["end_s"]) for window in records[1]["windows"]] == [
        (2.0, 3.5)
    ]
    for record in records:
        for window in record["windows"]:
            assert window["readiness_valid"] is False
            assert "readiness" not in window
            assert "issues" not in window
            assert window["label_source"]["kind"] == "proxy"
            assert window["label_source"]["human_reviewed"] is False

    output = write_training_jsonl(records, tmp_path / "annotations.jsonl")
    parsed = loads_jsonl(output.read_text(encoding="utf-8"))
    assert len(parsed) == 2
    assert all(not window.readiness_valid for record in parsed for window in record.windows)
    with pytest.raises(CorpusAcquisitionError, match="already exists"):
        write_training_jsonl(records, output)


def test_acquired_adapter_is_offline_and_uses_only_manifest_files(
    monkeypatch, tmp_path: Path
) -> None:
    import egosieve.corpus.ego_tactile as adapter

    info = (FIXTURES / "info.json").read_bytes()
    episodes = _fixture_json("episodes.json")
    frames = _fixture_frames()
    selected = [
        "meta/info.json",
        "meta/episodes/chunk-000/file-000.parquet",
        "data/chunk-000/file-000.parquet",
        "videos/observation.images.ego/chunk-000/file-000.mp4",
    ]
    payloads = {
        EGO_TACTILE.resolve_url(REVISION, selected[0]): info,
        EGO_TACTILE.resolve_url(REVISION, selected[1]): b"offline-episode-parquet",
        EGO_TACTILE.resolve_url(REVISION, selected[2]): b"offline-frame-parquet",
        EGO_TACTILE.resolve_url(REVISION, selected[3]): b"offline-packed-video",
    }
    corpus_dir = tmp_path / "corpus"
    fetch_selected_files(
        EGO_TACTILE,
        revision=REVISION,
        repository_paths=selected,
        output_dir=corpus_dir,
        accepted_license="CC-BY-4.0",
        opener=lambda request: io.BytesIO(payloads[request.full_url]),
    )

    def fake_read_parquet(path, _columns):
        if "meta/episodes" in path.as_posix():
            return episodes
        if "data/chunk" in path.as_posix():
            return frames
        raise AssertionError(f"unexpected local read: {path}")

    monkeypatch.setattr(adapter, "_read_parquet_rows", fake_read_parquet)
    output = tmp_path / "proxy.jsonl"
    summary = adapter.adapt_acquired_ego_tactile(
        corpus_dir,
        episode_indexes=[0],
        output_path=output,
    )

    assert summary["records"] == 1
    assert summary["windows"] == 2
    assert summary["human_reviewed"] is False
    assert summary["readiness_targets"] == 0
    parsed = loads_jsonl(output.read_text(encoding="utf-8"))
    assert parsed[0].video == f"files/{selected[3]}"


def test_corpus_fetch_cli_forwards_only_explicit_selection(monkeypatch, tmp_path, capsys) -> None:
    import egosieve.corpus as corpus

    captured: dict = {}

    def fake_fetch(source, **kwargs):
        captured.update({"source": source, **kwargs})
        return {
            "source": {
                "source_url": EGO_TACTILE.source_url,
                "repository": {"revision": REVISION},
                "license": {"id": "CC-BY-4.0"},
            },
            "files": [{"size_bytes": 5}],
        }

    monkeypatch.setattr(corpus, "fetch_selected_files", fake_fetch)
    output = tmp_path / "corpus"
    assert (
        cli.main(
            [
                "corpus",
                "fetch",
                "--source",
                "ego-tactile",
                "--revision",
                REVISION,
                "--file",
                "meta/info.json",
                "--output",
                str(output),
                "--accept-license",
                "CC-BY-4.0",
            ]
        )
        == 0
    )
    assert captured["repository_paths"] == ["meta/info.json"]
    assert captured["revision"] == REVISION
    assert captured["accepted_license"] == "CC-BY-4.0"
    summary = json.loads(capsys.readouterr().out)
    assert summary["files"] == 1
    assert summary["revision"] == REVISION


def test_holoassist_fetch_separates_official_annotations_from_pinned_mirror(
    tmp_path: Path,
) -> None:
    selected = [HOLOASSIST_ANNOTATION_PATH, HOLOASSIST_VIDEO_PATH]
    payloads = {
        HOLOASSIST.resolve_url(HOLOASSIST_REVISION, selected[0]): json.dumps(
            _holoassist_annotations()
        ).encode(),
        HOLOASSIST.resolve_url(HOLOASSIST_REVISION, selected[1]): b"offline-holoassist-video",
    }
    requested: list[str] = []

    def opener(request):
        requested.append(request.full_url)
        return io.BytesIO(payloads[request.full_url])

    output = tmp_path / "holoassist"
    manifest = fetch_selected_files(
        HOLOASSIST,
        revision=HOLOASSIST_REVISION,
        repository_paths=selected,
        output_dir=output,
        accepted_license="CDLA-Permissive-2.0",
        opener=opener,
    )

    assert requested == [HOLOASSIST.resolve_url(HOLOASSIST_REVISION, path) for path in selected]
    assert manifest["source"]["annotation_release"]["version"] == "v1_1"
    mirror = manifest["source"]["repository"]
    assert mirror["revision"] == HOLOASSIST_REVISION
    assert mirror["role"] == "media-transport-only"
    assert mirror["license_declaration_url"] is None
    assert mirror["publisher_byte_equivalence_verified"] is False
    assert [entry["origin"] for entry in manifest["files"]] == [
        "publisher_annotations",
        "media_mirror",
    ]
    assert manifest["files"][0]["local_path"] == (f"files/annotations/{HOLOASSIST_ANNOTATION_PATH}")
    assert verify_corpus(output)["file_count"] == 2


@pytest.mark.parametrize(
    ("paths", "match"),
    [
        ([HOLOASSIST_VIDEO_PATH], "requires explicit selection"),
        ([HOLOASSIST_ANNOTATION_PATH], "at least one exact"),
        (
            [HOLOASSIST_ANNOTATION_PATH, "HoloAssist/video/**/*.mp4"],
            "glob",
        ),
        (
            [HOLOASSIST_ANNOTATION_PATH, "parquet/HoloAssist/train-00000.parquet"],
            "HoloAssist selections",
        ),
    ],
)
def test_holoassist_fetch_rejects_incomplete_or_implicit_selection_before_network(
    tmp_path: Path, paths: list[str], match: str
) -> None:
    called = False

    def opener(_request):
        nonlocal called
        called = True
        raise AssertionError("network must not be reached")

    with pytest.raises(CorpusAcquisitionError, match=match):
        fetch_selected_files(
            HOLOASSIST,
            revision=HOLOASSIST_REVISION,
            repository_paths=paths,
            output_dir=tmp_path / "holoassist",
            accepted_license="CDLA-Permissive-2.0",
            opener=opener,
        )
    assert not called


def test_holoassist_fetch_rejects_unreviewed_exact_mirror_commit(tmp_path: Path) -> None:
    called = False

    def opener(_request):
        nonlocal called
        called = True
        raise AssertionError("network must not be reached")

    with pytest.raises(CorpusAcquisitionError, match="reviewed exact commit"):
        fetch_selected_files(
            HOLOASSIST,
            revision="a" * 40,
            repository_paths=[HOLOASSIST_ANNOTATION_PATH, HOLOASSIST_VIDEO_PATH],
            output_dir=tmp_path / "holoassist",
            accepted_license="CDLA-Permissive-2.0",
            opener=opener,
        )
    assert not called


def _holoassist_manifest_source(tmp_path: Path) -> tuple[Path, dict]:
    selected = [HOLOASSIST_ANNOTATION_PATH, HOLOASSIST_VIDEO_PATH]
    payloads = {
        HOLOASSIST.resolve_url(HOLOASSIST_REVISION, selected[0]): json.dumps(
            _holoassist_annotations()
        ).encode(),
        HOLOASSIST.resolve_url(HOLOASSIST_REVISION, selected[1]): b"offline-holoassist-video",
    }
    corpus_dir = tmp_path / "holoassist"
    manifest = fetch_selected_files(
        HOLOASSIST,
        revision=HOLOASSIST_REVISION,
        repository_paths=selected,
        output_dir=corpus_dir,
        accepted_license="CDLA-Permissive-2.0",
        opener=lambda request: io.BytesIO(payloads[request.full_url]),
    )
    return corpus_dir, manifest["source"]


def test_holoassist_adapter_uses_action_geometry_not_correctness_for_readiness(
    tmp_path: Path,
) -> None:
    _, source = _holoassist_manifest_source(tmp_path)
    annotations = _holoassist_annotations()
    records = build_holoassist_records(
        annotations,
        video_ids=[HOLOASSIST_VIDEO_ID],
        manifest_source=source,
        window_s=6.0,
        stride_s=3.0,
        keep_occupancy_threshold=0.5,
        max_keep_per_video=8,
        max_review_per_video=8,
        max_reject_per_video=8,
    )
    repeated = build_holoassist_records(
        annotations,
        video_ids=[HOLOASSIST_VIDEO_ID],
        manifest_source=source,
        window_s=6.0,
        stride_s=3.0,
        keep_occupancy_threshold=0.5,
        max_keep_per_video=8,
        max_review_per_video=8,
        max_reject_per_video=8,
    )
    assert records == repeated
    assert len(records) == 1
    record = records[0]
    assert record["source"] == "HoloAssist"
    assert record["group_id"] == f"holoassist:{HOLOASSIST_VIDEO_ID}"
    assert record["video"] == f"files/{HOLOASSIST_VIDEO_PATH}"
    assert record["license"] == "CDLA-Permissive-2.0"
    assert record["label_policy"]["action_correctness_used_for_readiness"] is False
    assert record["label_policy"]["issue_targets"] is False
    assert record["label_policy"]["rubric_version"] == READINESS_RUBRIC_VERSION

    by_class = {
        readiness: [window for window in record["windows"] if window["readiness"] == readiness]
        for readiness in ("KEEP", "REVIEW", "REJECT")
    }
    assert {name: len(windows) for name, windows in by_class.items()} == {
        "KEEP": 2,
        "REVIEW": 4,
        "REJECT": 3,
    }
    wrong = next(
        window
        for window in by_class["KEEP"]
        if any(
            annotation["action_correctness"].startswith("Wrong Action")
            for annotation in window["source_annotations"]
        )
    )
    assert wrong["readiness"] == "KEEP"
    assert (wrong["start_s"], wrong["end_s"]) == (12.0, 18.0)
    assert wrong["boundaries_s"] == {"start": 12.125, "end": 16.875}
    assert wrong["source_annotations"][0]["attributes"]["Incorrect Action Explanation"] == (
        "fixture explanation"
    )

    for window in record["windows"]:
        assert window["review_count"] == PUBLISHED_REVIEW_COUNT
        assert window["review_count_scope"] == "source_action_intervals"
        assert window["label_source"]["human_reviewed"] is False
        assert window["rubric_version"] == READINESS_RUBRIC_VERSION
        assert "issues" not in window
        assert "issue_valid" not in window
    assert all(0.0 < window["fine_action_occupancy"] < 0.5 for window in by_class["REVIEW"])
    assert all(
        window["derived_from"]["kind"] == "scanner_window_fine_action_union_occupancy"
        and window["fine_action_occupancy"] == 0.0
        and window["source_annotations"] == []
        for window in by_class["REJECT"]
    )
    assert all(window["end_s"] - window["start_s"] == 6.0 for window in record["windows"])

    capped = build_holoassist_records(
        annotations,
        video_ids=[HOLOASSIST_VIDEO_ID],
        manifest_source=source,
        max_keep_per_video=1,
        max_review_per_video=1,
        max_reject_per_video=1,
    )[0]
    assert {window["readiness"] for window in capped["windows"]} == {
        "KEEP",
        "REVIEW",
        "REJECT",
    }

    tail_annotations = json.loads(json.dumps(annotations))
    tail_annotations[0]["videoMetadata"]["duration"] = {
        "raw": "00:00:31.00",
        "seconds": 31,
    }
    tail_record = build_holoassist_records(
        tail_annotations,
        video_ids=[HOLOASSIST_VIDEO_ID],
        manifest_source=source,
        max_keep_per_video=None,
        max_review_per_video=None,
        max_reject_per_video=None,
    )[0]
    assert (tail_record["windows"][-1]["start_s"], tail_record["windows"][-1]["end_s"]) == (
        25.0,
        31.0,
    )
    assert tail_record["windows"][-1]["derived_from"]["include_tail"] is True


def test_acquired_holoassist_adapter_is_offline_and_manifest_bound(tmp_path: Path) -> None:
    corpus_dir, _ = _holoassist_manifest_source(tmp_path)
    output = tmp_path / "holoassist-proxies.jsonl"
    summary = adapt_acquired_holoassist(
        corpus_dir,
        video_ids=[HOLOASSIST_VIDEO_ID],
        output_path=output,
        max_keep_per_video=1,
        max_review_per_video=1,
        max_reject_per_video=1,
    )

    assert summary["records"] == 1
    assert summary["readiness_targets"] == {"KEEP": 1, "REVIEW": 1, "REJECT": 1}
    assert summary["human_reviewed"] is False
    assert summary["source_interval_review_count"] == 2
    assert summary["issue_targets"] == 0
    parsed = loads_jsonl(output.read_text(encoding="utf-8"))
    assert parsed[0].group_id == f"holoassist:{HOLOASSIST_VIDEO_ID}"
    assert {window.readiness for window in parsed[0].windows} == {"KEEP", "REVIEW", "REJECT"}


def test_holoassist_adapter_cli_forwards_caps(monkeypatch, tmp_path, capsys) -> None:
    import egosieve.corpus as corpus

    captured: dict = {}

    def fake_adapt(directory, **kwargs):
        captured.update({"directory": directory, **kwargs})
        return {"records": 1, "readiness_targets": {"KEEP": 1, "REVIEW": 1, "REJECT": 1}}

    monkeypatch.setattr(corpus, "adapt_acquired_holoassist", fake_adapt)
    output = tmp_path / "annotations.jsonl"
    assert (
        cli.main(
            [
                "corpus",
                "adapt-holoassist",
                str(tmp_path / "corpus"),
                "--video-id",
                HOLOASSIST_VIDEO_ID,
                "--output",
                str(output),
                "--window",
                "5",
                "--stride",
                "2",
                "--keep-occupancy-threshold",
                "0.6",
                "--max-keep-per-video",
                "3",
                "--max-review-per-video",
                "4",
                "--max-reject-per-video",
                "5",
            ]
        )
        == 0
    )
    assert captured["video_ids"] == [HOLOASSIST_VIDEO_ID]
    assert captured["window_s"] == 5.0
    assert captured["stride_s"] == 2.0
    assert captured["keep_occupancy_threshold"] == 0.6
    assert captured["max_keep_per_video"] == 3
    assert captured["max_review_per_video"] == 4
    assert captured["max_reject_per_video"] == 5
    assert json.loads(capsys.readouterr().out)["records"] == 1
