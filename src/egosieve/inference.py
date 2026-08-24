"""End-to-end video scanning and rich manifest generation."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForVideoClassification, AutoProcessor

from . import __version__
from ._paths import ensure_distinct_files
from .compiler import (
    DISCARD,
    KEEP,
    REVIEW,
    Segment,
    SegmentCompilerConfig,
    VideoCompiler,
    WindowScore,
    build_manifest_records,
    write_jsonl,
)
from .modeling import ISSUE_LABELS, READINESS_LABELS
from .video import (
    ExtractedFrame,
    PreparedVideo,
    SamplingPlan,
    VideoMetadata,
    VideoProcessingConfig,
    VideoProcessor,
)
from .video.frames import format_timestamp

ROUTE_TO_DECISION = {KEEP: "KEEP", REVIEW: "REVIEW", DISCARD: "REJECT"}


@dataclass(frozen=True)
class ScanConfig:
    window_duration_s: float = 6.0
    stride_s: float = 2.0
    frames_per_window: int = 12
    batch_size: int = 8
    device: str = "auto"
    decode_size: int | None = None
    issue_threshold: float | None = None
    issue_thresholds: Mapping[str, float] | None = None
    readiness_temperature: float | None = None
    use_checkpoint_calibration: bool = True
    include_embeddings: bool = False
    compiler: SegmentCompilerConfig = field(default_factory=SegmentCompilerConfig)

    def __post_init__(self) -> None:
        for name in ("window_duration_s", "stride_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("frames_per_window", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.decode_size is not None:
            raise ValueError(
                "decode_size is no longer supported; spatial resizing and cropping are "
                "owned by the checkpoint processor"
            )
        if self.issue_threshold is not None and not 0 <= self.issue_threshold <= 1:
            raise ValueError("issue_threshold must be between zero and one")
        if self.issue_thresholds is not None:
            if set(self.issue_thresholds) != set(ISSUE_LABELS):
                raise ValueError("issue_thresholds must contain every issue label exactly")
            if any(not 0 <= float(value) <= 1 for value in self.issue_thresholds.values()):
                raise ValueError("every per-issue threshold must be between zero and one")
        if self.readiness_temperature is not None and (
            not math.isfinite(self.readiness_temperature) or self.readiness_temperature <= 0
        ):
            raise ValueError("readiness_temperature must be finite and positive")


@dataclass(frozen=True)
class WindowPrediction:
    index: int
    start_s: float
    end_s: float
    readiness: dict[str, float]
    issues: dict[str, float]
    boundary: tuple[dict[str, float], ...]
    uncertainty: float
    embedding: tuple[float, ...] | None = None

    @property
    def selection_score(self) -> float:
        return float(self.readiness["KEEP"])

    @property
    def decision(self) -> str:
        return max(self.readiness, key=self.readiness.__getitem__)

    def as_window_score(self) -> WindowScore:
        # A confident REVIEW prediction is an uncertainty signal even when the
        # three-way distribution has low entropy.
        routed_uncertainty = max(self.uncertainty, float(self.readiness["REVIEW"]))
        return WindowScore(
            index=self.index,
            start_s=self.start_s,
            end_s=self.end_s,
            score=self.selection_score,
            uncertainty=routed_uncertainty,
        )


@dataclass(frozen=True)
class ScanResult:
    manifest_path: Path
    metadata: VideoMetadata
    plan: SamplingPlan
    windows: tuple[WindowPrediction, ...]
    segments: tuple[Segment, ...]

    @property
    def window_decision_counts(self) -> dict[str, int]:
        """Count raw model decisions over windows, including rejected videos."""

        counts = Counter(prediction.decision for prediction in self.windows)
        return {label: counts.get(label, 0) for label in READINESS_LABELS}

    @property
    def segment_decision_counts(self) -> dict[str, int]:
        """Count emitted compiler segments; omitted discard routes are absent."""

        counts = Counter(ROUTE_TO_DECISION[segment.route] for segment in self.segments)
        return {label: counts.get(label, 0) for label in READINESS_LABELS}

    @property
    def decision_counts(self) -> dict[str, int]:
        """Backward-compatible spelling for :attr:`window_decision_counts`."""

        return self.window_decision_counts


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    safe = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    return -(safe * safe.log()).sum(dim=-1) / math.log(probabilities.shape[-1])


def _issue_threshold_map(config: ScanConfig) -> dict[str, float]:
    if config.issue_threshold is not None:
        return {label: float(config.issue_threshold) for label in ISSUE_LABELS}
    if config.issue_thresholds is not None:
        return {label: float(config.issue_thresholds[label]) for label in ISSUE_LABELS}
    return {label: 0.35 for label in ISSUE_LABELS}


def _checkpoint_calibrated_config(config: ScanConfig, model: Any) -> ScanConfig:
    if not config.use_checkpoint_calibration:
        return config
    model_config = getattr(model, "config", None)
    updates: dict[str, Any] = {}
    if config.readiness_temperature is None:
        updates["readiness_temperature"] = float(
            getattr(model_config, "readiness_temperature", 1.0)
        )
    if config.issue_threshold is None and config.issue_thresholds is None:
        checkpoint_issues = getattr(model_config, "issue_thresholds", None)
        if checkpoint_issues is not None:
            updates["issue_thresholds"] = dict(checkpoint_issues)
    checkpoint_compiler = getattr(model_config, "compiler_thresholds", None)
    if checkpoint_compiler is not None and config.compiler == SegmentCompilerConfig():
        updates["compiler"] = SegmentCompilerConfig(**dict(checkpoint_compiler))
    return replace(config, **updates) if updates else config


def _component_num_frames(component: Any, *, is_model: bool) -> int | None:
    owner = getattr(component, "config", None) if is_model else component
    value = getattr(owner, "num_frames", None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        source = "model.config.num_frames" if is_model else "processor.num_frames"
        raise ValueError(f"{source} must be a positive integer; received {value!r}")
    return value


def _validate_frame_contract(
    *,
    config: ScanConfig,
    model: Any,
    processor: Any,
    plan: SamplingPlan | None = None,
) -> None:
    """Fail before decoding or model execution when temporal contracts differ."""

    expected = config.frames_per_window
    declared = {
        "processor.num_frames": _component_num_frames(processor, is_model=False),
        "model.config.num_frames": _component_num_frames(model, is_model=True),
    }
    if plan is not None:
        declared["sampling plan frames_per_window"] = plan.frames_per_window
    missing = [name for name, value in declared.items() if value is None]
    if missing:
        raise ValueError("frame-count contract is incomplete; missing " + ", ".join(missing))
    mismatches = [f"{name}={value}" for name, value in declared.items() if value != expected]
    if mismatches:
        details = ", ".join(mismatches)
        raise ValueError(
            f"frame-count contract mismatch: ScanConfig.frames_per_window={expected}, {details}"
        )

    model_config = getattr(model, "config", None)
    max_frames = getattr(model_config, "max_frames", None)
    if max_frames is not None and expected > max_frames:
        raise ValueError(
            f"frames_per_window={expected} exceeds model.config.max_frames={max_frames}"
        )


def _validate_issue_contract(model: Any) -> None:
    """Reject checkpoints whose issue heads cannot be interpreted safely."""

    model_config = getattr(model, "config", None)
    declared_labels = getattr(model_config, "issue_labels", None)
    if declared_labels is None:
        raise ValueError("issue-label contract is incomplete; missing model.config.issue_labels")
    if tuple(declared_labels) != ISSUE_LABELS:
        raise ValueError(
            "issue-label contract mismatch: model.config.issue_labels must exactly match "
            f"{list(ISSUE_LABELS)!r}"
        )
    declared_width = getattr(model_config, "num_issue_labels", None)
    if (
        isinstance(declared_width, bool)
        or not isinstance(declared_width, int)
        or declared_width != len(ISSUE_LABELS)
    ):
        raise ValueError(
            "issue-label contract mismatch: model.config.num_issue_labels must equal "
            f"the integer {len(ISSUE_LABELS)}"
        )


def predict_prepared_video(
    prepared: PreparedVideo,
    *,
    model: Any,
    processor: Any,
    config: ScanConfig,
) -> tuple[WindowPrediction, ...]:
    """Score an already sampled video; exposed for testing and custom decoders."""

    _validate_frame_contract(
        config=config,
        model=model,
        processor=processor,
        plan=prepared.plan,
    )
    _validate_issue_contract(model)
    device = _device(config.device)
    model = model.to(device).eval()
    frame_windows = prepared.frame_paths_by_window()
    predictions: list[WindowPrediction] = []

    unique_embeddings: torch.Tensor | None = None
    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None and prepared.frames:
        encoded_chunks: list[torch.Tensor] = []
        chunk_size = config.frames_per_window
        with torch.inference_mode():
            for offset in range(0, len(prepared.frames), chunk_size):
                chunk = prepared.frames[offset : offset + chunk_size]
                inputs = processor(
                    videos=[frame.path for frame in chunk],
                    return_tensors="pt",
                )
                pixels = inputs["pixel_values"].to(device)
                batch, frames_count, channels, height, width = pixels.shape
                vision_output = vision_model(
                    pixel_values=pixels.reshape(
                        batch * frames_count,
                        channels,
                        height,
                        width,
                    ),
                    return_dict=True,
                )
                encoded_chunks.append(
                    vision_output.last_hidden_state[: len(chunk), 0].detach().cpu()
                )
        unique_embeddings = torch.cat(encoded_chunks, dim=0)

    with torch.inference_mode():
        for offset in range(0, len(frame_windows), config.batch_size):
            paths_batch = frame_windows[offset : offset + config.batch_size]
            if unique_embeddings is None:
                inputs = processor(videos=paths_batch, return_tensors="pt")
                pixel_values = inputs.get("pixel_values")
                frame_mask = inputs.get("frame_mask")
                if pixel_values is None or pixel_values.ndim != 5:
                    raise ValueError(
                        "processor must return pixel_values with shape "
                        "[batch, frames, channels, height, width]"
                    )
                if pixel_values.shape[1] != config.frames_per_window:
                    raise ValueError(
                        "processor output frame dimension does not match frames_per_window; "
                        f"expected {config.frames_per_window}, received {pixel_values.shape[1]}"
                    )
                if frame_mask is None or tuple(frame_mask.shape) != tuple(pixel_values.shape[:2]):
                    raise ValueError(
                        "processor must return frame_mask matching pixel_values [batch, frames]"
                    )
                model_inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                    if key in {"pixel_values", "frame_mask"}
                }
            else:
                windows_batch = prepared.plan.windows[offset : offset + len(paths_batch)]
                gathered = torch.stack(
                    [unique_embeddings[list(window.sample_indices)] for window in windows_batch]
                )
                model_inputs = {
                    "frame_embeddings": gathered.to(device),
                    "frame_mask": torch.ones(
                        gathered.shape[:2],
                        dtype=torch.bool,
                        device=device,
                    ),
                }
            output = model(**model_inputs)
            logits = getattr(output, "logits", None)
            if logits is None:
                logits = getattr(output, "readiness_logits", None)
            if logits is None:
                raise ValueError("model output must provide standard video-classification logits")
            temperature = config.readiness_temperature
            if temperature is None:
                temperature = (
                    float(getattr(model.config, "readiness_temperature", 1.0))
                    if config.use_checkpoint_calibration
                    else 1.0
                )
            readiness = torch.softmax(logits.float() / temperature, dim=-1).cpu()
            issue_logits = getattr(output, "issue_logits", None)
            if issue_logits is None:
                raise ValueError("model output must provide issue_logits")
            expected_issue_shape = (readiness.shape[0], len(ISSUE_LABELS))
            if tuple(issue_logits.shape) != expected_issue_shape:
                raise ValueError(
                    "model output issue_logits shape mismatch: expected "
                    f"{expected_issue_shape}, received {tuple(issue_logits.shape)}"
                )
            issues = torch.sigmoid(issue_logits.float()).cpu()
            boundaries = torch.sigmoid(output.boundary_logits.float()).cpu()
            embeddings = output.clip_embedding.float().cpu()
            uncertainties = _entropy(readiness)

            for batch_index in range(readiness.shape[0]):
                window_index = offset + batch_index
                window = prepared.plan.windows[window_index]
                samples = prepared.plan.samples_for_window(window)
                readiness_row = {
                    label: float(readiness[batch_index, label_index])
                    for label_index, label in enumerate(READINESS_LABELS)
                }
                issue_row = {
                    label: float(issues[batch_index, label_index])
                    for label_index, label in enumerate(ISSUE_LABELS)
                }
                boundary_row = tuple(
                    {
                        "timestamp_s": float(sample.timestamp_s),
                        "start": float(boundaries[batch_index, sample_index, 0]),
                        "end": float(boundaries[batch_index, sample_index, 1]),
                    }
                    for sample_index, sample in enumerate(samples)
                )
                embedding = (
                    tuple(float(value) for value in embeddings[batch_index])
                    if config.include_embeddings
                    else None
                )
                predictions.append(
                    WindowPrediction(
                        index=window.index,
                        start_s=window.start_s,
                        end_s=window.end_s,
                        readiness=readiness_row,
                        issues=issue_row,
                        boundary=boundary_row,
                        uncertainty=float(uncertainties[batch_index]),
                        embedding=embedding,
                    )
                )
    return tuple(predictions)


def _source_cache_key(prepared_hash: str, config: ScanConfig) -> str:
    payload = json.dumps(
        {
            "source": prepared_hash,
            "window": config.window_duration_s,
            "stride": config.stride_s,
            "frames": config.frames_per_window,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _local_artifact_revision(model_id: str) -> str | None:
    root = Path(model_id).expanduser()
    if not root.is_dir():
        return None
    candidates = [
        path
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name in {"config.json", "preprocessor_config.json"}
            or path.suffix == ".safetensors"
            or path.name.endswith(".safetensors.index.json")
        )
    ]
    if not candidates:
        return None
    digest = hashlib.sha256()
    for path in sorted(candidates, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_components(
    model_id: str,
    *,
    revision: str | None,
    model: Any | None,
    processor: Any | None,
) -> tuple[Any, Any, str | None]:
    """Load model and processor from one resolved revision."""

    local_revision = _local_artifact_revision(model_id)
    if model is None:
        model = AutoModelForVideoClassification.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
        )
    resolved_revision = (
        getattr(getattr(model, "config", None), "_commit_hash", None) or local_revision or revision
    )
    if processor is None:
        processor = AutoProcessor.from_pretrained(
            model_id,
            revision=None if local_revision else resolved_revision,
            trust_remote_code=True,
        )
    return model, processor, resolved_revision


def _expected_cached_frames(
    prepared_dir: Path,
    plan: Any,
    *,
    image_format: str,
) -> tuple[ExtractedFrame, ...]:
    frames = []
    suffix = image_format.lower().lstrip(".")
    for sample in plan.samples:
        stamp = format_timestamp(sample.timestamp_s).replace(".", "_")
        path = prepared_dir / f"frame_{sample.index:06d}_{stamp}.{suffix}"
        frames.append(ExtractedFrame(sample=sample, path=str(path)))
    return tuple(frames)


def _prepare_video(
    source: Path,
    *,
    config: ScanConfig,
    work_root: Path,
    ffmpeg_bin: str,
    ffprobe_bin: str,
) -> PreparedVideo:
    video_processor = VideoProcessor(
        VideoProcessingConfig(
            window_duration_s=config.window_duration_s,
            stride_s=config.stride_s,
            frames_per_window=config.frames_per_window,
            # Preserve decoded source geometry. The checkpoint processor owns
            # the sole resize/crop/normalization transform in training and
            # inference.
            output_size=None,
            image_format="png",
        ),
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
    )
    metadata = video_processor.probe(source, calculate_hash=True)
    plan = video_processor.plan(metadata)
    key = _source_cache_key(metadata.source_sha256 or "no-hash", config)
    frame_dir = work_root / key
    expected = _expected_cached_frames(
        frame_dir,
        plan,
        image_format=video_processor.config.image_format,
    )
    if expected and all(Path(frame.path).is_file() for frame in expected):
        frames = expected
    else:
        frames = video_processor.extract(source, plan, frame_dir, overwrite=True)
    return PreparedVideo(metadata=metadata, plan=plan, frames=frames)


def _segment_id(source_hash: str, model_id: str, segment: Segment) -> str:
    payload = (
        f"{source_hash}\0{model_id}\0{segment.start_s:.9f}\0{segment.end_s:.9f}\0{segment.route}"
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def build_scan_records(
    prepared: PreparedVideo,
    predictions: Sequence[WindowPrediction],
    segments: Sequence[Segment],
    *,
    model_id: str,
    model_revision: str | None = None,
    requested_revision: str | None = None,
    config: ScanConfig,
) -> tuple[dict[str, Any], ...]:
    """Build the compiler schema plus model probabilities and evidence."""

    base = list(
        build_manifest_records(
            prepared.metadata,
            plan=prepared.plan,
            scores=[prediction.as_window_score() for prediction in predictions],
            segments=segments,
            generator=f"egosieve {__version__}",
        )
    )
    prediction_by_index = {prediction.index: prediction for prediction in predictions}
    header = base[0]
    header["model"] = {
        "id": model_id,
        "revision": model_revision,
        "requested_revision": requested_revision,
        "processor_revision": model_revision,
        "readiness_labels": list(READINESS_LABELS),
    }
    header["policy"] = asdict(config.compiler)
    header["issue_labels"] = list(ISSUE_LABELS)
    issue_thresholds = _issue_threshold_map(config)
    header["issue_reporting_thresholds"] = issue_thresholds
    window_decisions = Counter(prediction.decision for prediction in predictions)
    segment_decisions = Counter(ROUTE_TO_DECISION[segment.route] for segment in segments)
    header["counts"]["window_decisions"] = {
        label: window_decisions.get(label, 0) for label in READINESS_LABELS
    }
    header["counts"]["segment_decisions"] = {
        label: segment_decisions.get(label, 0) for label in READINESS_LABELS
    }

    source_hash = prepared.metadata.source_sha256 or "unhashed"
    segment_by_index = {index: segment for index, segment in enumerate(segments)}
    for record in base[1:]:
        if record["record_type"] == "window":
            prediction = prediction_by_index[record["window_index"]]
            record["decision"] = prediction.decision
            record["uncertainty"] = prediction.uncertainty
            record["routing_uncertainty"] = prediction.as_window_score().uncertainty
            record["readiness"] = prediction.readiness
            record["issues"] = prediction.issues
            record["reported_issues"] = [
                label
                for label, probability in prediction.issues.items()
                if probability >= issue_thresholds[label]
            ]
            record["boundary"] = list(prediction.boundary)
            if prediction.embedding is not None:
                record["embedding"] = list(prediction.embedding)
        elif record["record_type"] == "segment":
            segment = segment_by_index[record["segment_index"]]
            member_predictions = [
                prediction_by_index[index]
                for index in segment.window_indices
                if index in prediction_by_index
            ]
            model_identity = f"{model_id}@{model_revision or 'unresolved'}"
            record["segment_id"] = _segment_id(source_hash, model_identity, segment)
            record["decision"] = ROUTE_TO_DECISION[segment.route]
            if member_predictions:
                record["readiness"] = {
                    label: sum(prediction.readiness[label] for prediction in member_predictions)
                    / len(member_predictions)
                    for label in READINESS_LABELS
                }
                record["issues"] = {
                    label: max(prediction.issues[label] for prediction in member_predictions)
                    for label in ISSUE_LABELS
                }
                record["reported_issues"] = [
                    label
                    for label, probability in record["issues"].items()
                    if probability >= issue_thresholds[label]
                ]
    return tuple(base)


def scan_video(
    source_path: str | Path,
    *,
    model_id: str,
    output_path: str | Path,
    revision: str | None = None,
    config: ScanConfig | None = None,
    cache_dir: str | Path | None = None,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    model: Any | None = None,
    processor: Any | None = None,
) -> ScanResult:
    """Probe, timestamp-sample, score, compile, and write one video."""

    config = config or ScanConfig()
    source, destination = ensure_distinct_files(source_path, output_path)
    if not source.is_file():
        raise FileNotFoundError(source)

    model, processor, resolved_revision = _load_components(
        model_id,
        revision=revision,
        model=model,
        processor=processor,
    )
    # Reject semantically incompatible checkpoints before calibration metadata
    # is read or any media is probed and decoded.
    _validate_issue_contract(model)
    config = _checkpoint_calibrated_config(config, model)

    _validate_frame_contract(config=config, model=model, processor=processor)

    def execute(work_root: Path) -> ScanResult:
        prepared = _prepare_video(
            source,
            config=config,
            work_root=work_root,
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
        )
        predictions = predict_prepared_video(
            prepared,
            model=model,
            processor=processor,
            config=config,
        )
        compiler = VideoCompiler(config.compiler)
        compilation = compiler.compile(
            prepared.metadata,
            [prediction.as_window_score() for prediction in predictions],
            plan=prepared.plan,
        )
        records = build_scan_records(
            prepared,
            predictions,
            compilation.segments,
            model_id=model_id,
            model_revision=resolved_revision,
            requested_revision=revision,
            config=config,
        )
        manifest_path = write_jsonl(destination, records)
        return ScanResult(
            manifest_path=manifest_path,
            metadata=prepared.metadata,
            plan=prepared.plan,
            windows=predictions,
            segments=compilation.segments,
        )

    if cache_dir is not None:
        root = Path(cache_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return execute(root)
    with tempfile.TemporaryDirectory(prefix="egosieve-") as temporary:
        return execute(Path(temporary))


__all__ = [
    "ScanConfig",
    "ScanResult",
    "WindowPrediction",
    "build_scan_records",
    "predict_prepared_video",
    "scan_video",
]
