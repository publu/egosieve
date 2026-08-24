from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

from egosieve.compiler.artifacts import (
    build_clip_command,
    contact_sheet_layout,
    export_segment_clips,
)
from egosieve.compiler.segments import KEEP, REVIEW, Segment


def _segment(route: str, start: float, end: float) -> Segment:
    return Segment(start, end, route, 0.8, 0.8, 0.8, (0,))


def test_clip_command_uses_argument_array_and_optional_audio_map(tmp_path: Path) -> None:
    source = tmp_path / "unsafe; name.mp4"
    output = tmp_path / "clip.mp4"
    command = build_clip_command(source, output, start_s=1.25, end_s=3.5)

    assert isinstance(command, list)
    assert str(source.resolve()) in command
    assert str(output.resolve()) == command[-1]
    assert "0:a?" in command
    assert "1.25" in command
    assert "2.25" in command


def test_clip_export_filters_routes_and_never_invokes_shell(tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append((command, kwargs))
        return CompletedProcess(command, 0, stdout="", stderr="")

    outputs = export_segment_clips(
        tmp_path / "source.mp4",
        [_segment(KEEP, 0, 2), _segment(REVIEW, 2, 4)],
        tmp_path / "clips",
        runner=runner,
        verify_outputs=False,
    )

    assert outputs.keys() == {0}
    assert len(calls) == 1
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["timeout"] == 300.0


def test_contact_sheet_layout() -> None:
    assert contact_sheet_layout(10, columns=4) == (3, 4)
    assert contact_sheet_layout(2, columns=4) == (1, 2)
