from __future__ import annotations

from pathlib import Path

import pytest

from egosieve._paths import ensure_distinct_files


def test_output_guard_rejects_symlink_alias(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    alias = tmp_path / "alias.mp4"
    alias.symlink_to(source)

    with pytest.raises(ValueError, match="source video"):
        ensure_distinct_files(source, alias)


def test_output_guard_rejects_hardlink_alias(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    alias = tmp_path / "alias.mp4"
    alias.hardlink_to(source)

    with pytest.raises(ValueError, match="source video"):
        ensure_distinct_files(source, alias)
