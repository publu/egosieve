"""Stable segment compilation, manifests, and optional media artifacts."""

from .artifacts import (
    ArtifactError,
    build_clip_command,
    contact_sheet_layout,
    export_segment_clips,
    write_contact_sheet,
)
from .manifest import (
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    ManifestError,
    build_manifest_records,
    read_manifest,
    write_jsonl,
    write_jsonl_manifest,
    write_manifest,
)
from .pipeline import Compilation, VideoCompiler
from .segments import (
    DISCARD,
    KEEP,
    REVIEW,
    ROUTE_PRECEDENCE,
    ROUTES,
    Segment,
    SegmentCompilerConfig,
    WindowScore,
    compile_segments,
    hysteresis_mask,
    merge_segments,
    stabilize_segments,
)

__all__ = [
    "ArtifactError",
    "Compilation",
    "DISCARD",
    "KEEP",
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "ManifestError",
    "REVIEW",
    "ROUTE_PRECEDENCE",
    "ROUTES",
    "Segment",
    "SegmentCompilerConfig",
    "VideoCompiler",
    "WindowScore",
    "build_clip_command",
    "build_manifest_records",
    "compile_segments",
    "contact_sheet_layout",
    "export_segment_clips",
    "hysteresis_mask",
    "merge_segments",
    "read_manifest",
    "stabilize_segments",
    "write_contact_sheet",
    "write_jsonl",
    "write_jsonl_manifest",
    "write_manifest",
]
