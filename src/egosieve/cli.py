"""Command-line entry point for EgoSieve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from ._paths import ensure_distinct_files
from .release import ReleaseValidationError, validate_release
from .video import plan_frame_samples, probe_video


def _json_dump(value: Any, path: Path | None = None) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path is None:
        sys.stdout.write(payload)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


def _inspect(args: argparse.Namespace) -> int:
    if args.output is not None:
        ensure_distinct_files(args.video, args.output)
    metadata = probe_video(
        args.video,
        ffprobe_bin=args.ffprobe,
        calculate_hash=not args.no_hash,
    )
    _json_dump(metadata.to_dict(), args.output)
    return 0


def _plan(args: argparse.Namespace) -> int:
    if args.output is not None:
        ensure_distinct_files(args.video, args.output)
    metadata = probe_video(
        args.video,
        ffprobe_bin=args.ffprobe,
        calculate_hash=not args.no_hash,
    )
    plan = plan_frame_samples(
        metadata.duration_s,
        window_duration_s=args.window,
        stride_s=args.stride,
        frames_per_window=args.frames,
        start_time_s=metadata.start_time_s,
    )
    _json_dump(
        {
            "schema": "egosieve.sampling-plan/v1",
            "source": metadata.to_dict(),
            "plan": plan.to_dict(),
        },
        args.output,
    )
    return 0


def _validate_release(args: argparse.Namespace) -> int:
    result = validate_release(args.directory)
    _json_dump({"valid": True, "files": result["files"]})
    return 0


def _scan(args: argparse.Namespace) -> int:
    from .inference import ScanConfig, scan_video

    result = scan_video(
        args.video,
        model_id=args.model,
        output_path=args.output,
        revision=args.revision,
        config=ScanConfig(
            window_duration_s=args.window,
            stride_s=args.stride,
            frames_per_window=args.frames,
            batch_size=args.batch_size,
            device=args.device,
            issue_threshold=args.issue_threshold,
            include_embeddings=args.include_embeddings,
        ),
        cache_dir=args.cache_dir,
        ffmpeg_bin=args.ffmpeg,
        ffprobe_bin=args.ffprobe,
    )
    summary = {
        "output": str(result.manifest_path),
        "windows": len(result.windows),
        "segments": len(result.segments),
        "decisions": result.decision_counts,
    }
    _json_dump(summary)
    return 0


def _train(args: argparse.Namespace) -> int:
    from .training import TrainingRunConfig, train_checkpoint

    result = train_checkpoint(
        args.annotations,
        seed_checkpoint=args.seed_checkpoint,
        output_dir=args.output,
        cache_dir=args.cache_dir,
        media_root=args.media_root,
        allowed_licenses=args.allowed_license,
        config=TrainingRunConfig(
            seed=args.seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_fraction=args.warmup_fraction,
            boundary_tolerance_s=args.boundary_tolerance,
            boundary_threshold=args.boundary_threshold,
            contrastive_weight=args.contrastive_weight,
            contrastive_temperature=args.contrastive_temperature,
            patience=args.patience,
            device=args.device,
            ffmpeg_bin=args.ffmpeg,
            decode_timeout_s=args.decode_timeout,
        ),
        model_id=args.model_id,
        model_revision=args.model_revision,
        backbone=args.backbone,
        backbone_revision=args.backbone_revision,
        source_commit=args.source_commit,
        annotation_guide=args.annotation_guide,
    )
    _json_dump(result)
    return 0


def _corpus_fetch(args: argparse.Namespace) -> int:
    from .corpus import CORPUS_MANIFEST_NAME, fetch_selected_files

    manifest = fetch_selected_files(
        args.source,
        revision=args.revision,
        repository_paths=args.files,
        output_dir=args.output,
        accepted_license=args.accept_license,
    )
    _json_dump(
        {
            "manifest": str(args.output / CORPUS_MANIFEST_NAME),
            "source": manifest["source"]["source_url"],
            "revision": manifest["source"]["repository"]["revision"],
            "license": manifest["source"]["license"]["id"],
            "files": len(manifest["files"]),
            "size_bytes": sum(item["size_bytes"] for item in manifest["files"]),
        }
    )
    return 0


def _corpus_verify(args: argparse.Namespace) -> int:
    from .corpus import verify_corpus

    result = verify_corpus(args.directory)
    source = result["manifest"]["source"]
    _json_dump(
        {
            "valid": True,
            "manifest": result["manifest_path"],
            "source": source["source_url"],
            "revision": source["repository"]["revision"],
            "license": source["license"]["id"],
            "files": result["file_count"],
            "size_bytes": result["total_size_bytes"],
        }
    )
    return 0


def _corpus_adapt_ego_tactile(args: argparse.Namespace) -> int:
    from .corpus import adapt_acquired_ego_tactile

    result = adapt_acquired_ego_tactile(
        args.directory,
        episode_indexes=args.episodes,
        output_path=args.output,
    )
    _json_dump(result)
    return 0


def _corpus_adapt_holoassist(args: argparse.Namespace) -> int:
    from .corpus import adapt_acquired_holoassist

    result = adapt_acquired_holoassist(
        args.directory,
        video_ids=args.video_ids,
        output_path=args.output,
        window_s=args.window,
        stride_s=args.stride,
        keep_occupancy_threshold=args.keep_occupancy_threshold,
        max_keep_per_video=args.max_keep_per_video,
        max_review_per_video=args.max_review_per_video,
        max_reject_per_video=args.max_reject_per_video,
    )
    _json_dump(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="egosieve",
        description="Find manipulation-ready windows in first-person video.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="probe source metadata")
    inspect_parser.add_argument("video", type=Path)
    inspect_parser.add_argument("--output", type=Path)
    inspect_parser.add_argument("--ffprobe", default="ffprobe")
    inspect_parser.add_argument("--no-hash", action="store_true")
    inspect_parser.set_defaults(handler=_inspect)

    plan_parser = subparsers.add_parser("plan", help="print deterministic window sampling")
    plan_parser.add_argument("video", type=Path)
    plan_parser.add_argument("--output", type=Path)
    plan_parser.add_argument("--window", type=float, default=6.0)
    plan_parser.add_argument("--stride", type=float, default=2.0)
    plan_parser.add_argument("--frames", type=int, default=12)
    plan_parser.add_argument("--ffprobe", default="ffprobe")
    plan_parser.add_argument("--no-hash", action="store_true")
    plan_parser.set_defaults(handler=_plan)

    scan_parser = subparsers.add_parser("scan", help="score and compile a video manifest")
    scan_parser.add_argument("video", type=Path)
    scan_parser.add_argument("--model", required=True)
    scan_parser.add_argument(
        "--revision",
        help="model branch, tag, or preferably immutable Hub commit",
    )
    scan_parser.add_argument("--output", required=True, type=Path)
    scan_parser.add_argument("--window", type=float, default=6.0)
    scan_parser.add_argument("--stride", type=float, default=2.0)
    scan_parser.add_argument("--frames", type=int, default=12)
    scan_parser.add_argument("--batch-size", type=int, default=8)
    scan_parser.add_argument("--device", default="auto")
    scan_parser.add_argument(
        "--issue-threshold",
        type=float,
        help="override all checkpoint-calibrated per-issue thresholds",
    )
    scan_parser.add_argument(
        "--include-embeddings",
        action="store_true",
        help="include normalized per-window retrieval embeddings in the manifest",
    )
    scan_parser.add_argument("--cache-dir", type=Path)
    scan_parser.add_argument("--ffmpeg", default="ffmpeg")
    scan_parser.add_argument("--ffprobe", default="ffprobe")
    scan_parser.set_defaults(handler=_scan)

    train_parser = subparsers.add_parser(
        "train", help="train and evaluate a checkpoint from versioned JSONL annotations"
    )
    train_parser.add_argument("annotations", type=Path)
    train_parser.add_argument("--seed-checkpoint", required=True, type=Path)
    train_parser.add_argument("--output", required=True, type=Path)
    train_parser.add_argument("--cache-dir", required=True, type=Path)
    train_parser.add_argument("--media-root", type=Path)
    train_parser.add_argument(
        "--allowed-license",
        action="append",
        required=True,
        help="exact declared license identifier to permit; repeat for multiple licenses",
    )
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--train-fraction", type=float, default=0.8)
    train_parser.add_argument("--validation-fraction", type=float, default=0.1)
    train_parser.add_argument("--test-fraction", type=float, default=0.1)
    train_parser.add_argument("--epochs", type=int, default=12)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--warmup-fraction", type=float, default=0.1)
    train_parser.add_argument("--boundary-tolerance", type=float, default=0.3)
    train_parser.add_argument("--boundary-threshold", type=float, default=0.5)
    train_parser.add_argument("--contrastive-weight", type=float, default=0.1)
    train_parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    train_parser.add_argument("--patience", type=int, default=4)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--ffmpeg", default="ffmpeg")
    train_parser.add_argument("--decode-timeout", type=float, default=120.0)
    train_parser.add_argument("--model-id", default="itspublu/EgoSieve-S")
    train_parser.add_argument("--model-revision", default="v0.1.0")
    train_parser.add_argument("--backbone", default="facebook/dinov2-small")
    train_parser.add_argument("--backbone-revision", required=True)
    train_parser.add_argument("--source-commit", required=True)
    train_parser.add_argument("--annotation-guide", default="docs/ANNOTATION.md (v0.1)")
    train_parser.set_defaults(handler=_train)

    corpus_parser = subparsers.add_parser(
        "corpus", help="explicitly acquire and adapt licensed public corpus files"
    )
    corpus_subparsers = corpus_parser.add_subparsers(dest="corpus_command", required=True)

    corpus_fetch_parser = corpus_subparsers.add_parser(
        "fetch", help="download only explicitly named files at an immutable revision"
    )
    corpus_fetch_parser.add_argument(
        "--source", choices=("ego-tactile", "holoassist"), required=True
    )
    corpus_fetch_parser.add_argument(
        "--revision",
        required=True,
        help="full 40-character Hugging Face repository or media-mirror commit",
    )
    corpus_fetch_parser.add_argument(
        "--file",
        dest="files",
        action="append",
        required=True,
        help="exact reviewed source file path; repeat for each selected file",
    )
    corpus_fetch_parser.add_argument("--output", required=True, type=Path)
    corpus_fetch_parser.add_argument(
        "--accept-license",
        required=True,
        help="exact source license id displayed in the public-corpus guide",
    )
    corpus_fetch_parser.set_defaults(handler=_corpus_fetch)

    corpus_verify_parser = corpus_subparsers.add_parser(
        "verify", help="verify the acquisition manifest, sizes, and SHA-256 digests"
    )
    corpus_verify_parser.add_argument("directory", type=Path)
    corpus_verify_parser.set_defaults(handler=_corpus_verify)

    corpus_adapt_parser = corpus_subparsers.add_parser(
        "adapt-ego-tactile",
        help="emit proxy action-boundary records without readiness or issue labels",
    )
    corpus_adapt_parser.add_argument("directory", type=Path)
    corpus_adapt_parser.add_argument(
        "--episode",
        dest="episodes",
        action="append",
        required=True,
        type=int,
        help="explicit episode index to adapt; repeat for multiple episodes",
    )
    corpus_adapt_parser.add_argument("--output", required=True, type=Path)
    corpus_adapt_parser.set_defaults(handler=_corpus_adapt_ego_tactile)

    holoassist_adapt_parser = corpus_subparsers.add_parser(
        "adapt-holoassist",
        help="emit deterministic readiness proxies from audited fine-action intervals",
    )
    holoassist_adapt_parser.add_argument("directory", type=Path)
    holoassist_adapt_parser.add_argument(
        "--video-id",
        dest="video_ids",
        action="append",
        required=True,
        help="exact HoloAssist video_name without .mp4; repeat for multiple videos",
    )
    holoassist_adapt_parser.add_argument("--output", required=True, type=Path)
    holoassist_adapt_parser.add_argument("--window", type=float, default=6.0)
    holoassist_adapt_parser.add_argument("--stride", type=float, default=3.0)
    holoassist_adapt_parser.add_argument("--keep-occupancy-threshold", type=float, default=0.5)
    holoassist_adapt_parser.add_argument("--max-keep-per-video", type=int, default=64)
    holoassist_adapt_parser.add_argument("--max-review-per-video", type=int, default=64)
    holoassist_adapt_parser.add_argument("--max-reject-per-video", type=int, default=64)
    holoassist_adapt_parser.set_defaults(handler=_corpus_adapt_holoassist)

    release_parser = subparsers.add_parser(
        "validate-release", help="verify a model artifact before upload"
    )
    release_parser.add_argument("directory", type=Path)
    release_parser.set_defaults(handler=_validate_release)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ReleaseValidationError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"egosieve: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
