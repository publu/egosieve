"""Video metadata, deterministic sampling, extraction, and tensor transforms."""

from .frames import (
    FrameExtractionError,
    batch_frames_to_tensor,
    build_frame_extract_command,
    extract_plan_frames,
    frame_to_tensor,
    frames_to_tensor,
    to_tensor,
)
from .models import ExtractedFrame, FrameSample, SamplingPlan, SamplingWindow, VideoMetadata
from .probe import (
    VideoProbeError,
    build_ffprobe_command,
    parse_ffprobe_json,
    parse_rate,
    probe_video,
    sha256_file,
)
from .processor import PreparedVideo, VideoProcessingConfig, VideoProcessor
from .sampling import (
    gather_window_values,
    materialize_plan,
    plan_frame_samples,
    plan_sampling,
    sample_once,
    sample_timestamps,
    sliding_windows,
)

__all__ = [
    "ExtractedFrame",
    "FrameExtractionError",
    "FrameSample",
    "SamplingPlan",
    "SamplingWindow",
    "PreparedVideo",
    "VideoMetadata",
    "VideoProcessingConfig",
    "VideoProcessor",
    "VideoProbeError",
    "batch_frames_to_tensor",
    "build_ffprobe_command",
    "build_frame_extract_command",
    "extract_plan_frames",
    "frame_to_tensor",
    "frames_to_tensor",
    "gather_window_values",
    "materialize_plan",
    "parse_ffprobe_json",
    "parse_rate",
    "plan_frame_samples",
    "plan_sampling",
    "probe_video",
    "sample_once",
    "sample_timestamps",
    "sha256_file",
    "sliding_windows",
    "to_tensor",
]
