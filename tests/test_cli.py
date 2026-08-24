from __future__ import annotations

import json
from pathlib import Path

import pytest

from egosieve import cli
from egosieve.video import VideoMetadata


def _metadata(path: Path) -> VideoMetadata:
    return VideoMetadata(
        source_path=str(path),
        source_sha256="a" * 64,
        source_size_bytes=100,
        duration_s=10.0,
        width=1920,
        height=1080,
        display_width=1920,
        display_height=1080,
        fps=30.0,
    )


def test_inspect_writes_json(monkeypatch, tmp_path, capsys) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    monkeypatch.setattr(cli, "probe_video", lambda *args, **kwargs: _metadata(video))
    assert cli.main(["inspect", str(video)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["duration_s"] == 10.0
    assert payload["source_sha256"] == "a" * 64


def test_plan_writes_unique_sampling_contract(monkeypatch, tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    output = tmp_path / "plan.json"
    video.write_bytes(b"fixture")
    monkeypatch.setattr(cli, "probe_video", lambda *args, **kwargs: _metadata(video))
    assert (
        cli.main(
            [
                "plan",
                str(video),
                "--window",
                "4",
                "--stride",
                "2",
                "--frames",
                "4",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text())
    assert payload["schema"] == "egosieve.sampling-plan/v1"
    assert payload["plan"]["window_count"] == 4
    assert all(len(window["sample_indices"]) == 4 for window in payload["plan"]["windows"])


def test_missing_subcommand_is_usage_error() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main([])
    assert error.value.code == 2


def test_scan_parser_exposes_optional_retrieval_embeddings() -> None:
    args = cli.build_parser().parse_args(
        [
            "scan",
            "clip.mp4",
            "--model",
            "fixture/model",
            "--output",
            "manifest.jsonl",
            "--include-embeddings",
        ]
    )
    assert args.include_embeddings is True


@pytest.mark.parametrize("command", ["inspect", "plan"])
def test_metadata_commands_cannot_overwrite_source(command: str, tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"source bytes")

    with pytest.raises(SystemExit) as error:
        cli.main([command, str(video), "--output", str(video)])

    assert error.value.code == 2
    assert video.read_bytes() == b"source bytes"
