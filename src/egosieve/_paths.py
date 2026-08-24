"""Small path guards shared by CLI and library entry points."""

from __future__ import annotations

import os
from pathlib import Path


def ensure_distinct_files(
    source: str | os.PathLike[str], destination: str | os.PathLike[str]
) -> tuple[Path, Path]:
    """Resolve two paths and reject aliases of the same filesystem object."""

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve(strict=False)
    same = source_path == destination_path
    if not same and destination_path.exists():
        try:
            same = source_path.samefile(destination_path)
        except OSError:
            same = False
    if same:
        raise ValueError("output path must not refer to the source video")
    return source_path, destination_path


__all__ = ["ensure_distinct_files"]
