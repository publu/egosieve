from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("av")

from PIL import Image
from transformers import AutoModelForVideoClassification, AutoProcessor, pipeline

from egosieve import EgoSieveConfig, EgoSieveModel, EgoSieveProcessor
from egosieve.compiler import SegmentCompilerConfig
from egosieve.inference import ScanConfig, scan_video
from egosieve.initialization import save_training_seed


def _tiny_model() -> EgoSieveModel:
    config = EgoSieveConfig(
        vision_config={
            "model_type": "dinov2",
            "image_size": 16,
            "patch_size": 8,
            "hidden_size": 24,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 48,
        },
        temporal_hidden_size=16,
        temporal_num_layers=1,
        temporal_intermediate_size=24,
        num_frames=2,
        max_frames=4,
        projection_dim=8,
        dropout=0,
    )
    return EgoSieveModel(config).eval()


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is not installed")
    source = tmp_path / "non-square.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=orange:size=64x48:rate=12",
            "-t",
            "1",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            str(source),
        ],
        check=True,
        shell=False,
    )
    return source


def test_transformers_video_classification_pipeline_contract(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    model = _tiny_model()
    processor = EgoSieveProcessor(size=16, num_frames=2)
    checkpoint = tmp_path / "checkpoint"
    save_training_seed(model, checkpoint, processor=processor)
    loaded_model = AutoModelForVideoClassification.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
    )
    loaded_processor = AutoProcessor.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
    )
    classifier = pipeline(
        "video-classification",
        model=loaded_model,
        image_processor=loaded_processor,
        device="cpu",
    )

    predictions = classifier(str(synthetic_video), top_k=3)

    assert {prediction["label"] for prediction in predictions} == {
        "KEEP",
        "REVIEW",
        "REJECT",
    }
    assert sum(prediction["score"] for prediction in predictions) == pytest.approx(1.0)


def test_scan_preserves_source_geometry_until_checkpoint_processor(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    model = _tiny_model()
    processor = EgoSieveProcessor(size=16, num_frames=2)
    cache = tmp_path / "cache"
    manifest = tmp_path / "manifest.jsonl"

    result = scan_video(
        synthetic_video,
        model_id="local/tiny-egosieve",
        output_path=manifest,
        config=ScanConfig(
            window_duration_s=0.5,
            stride_s=0.5,
            frames_per_window=2,
            batch_size=1,
            device="cpu",
            compiler=SegmentCompilerConfig(merge_gap_s=0, min_duration_s=0),
        ),
        cache_dir=cache,
        model=model,
        processor=processor,
    )

    extracted = sorted(cache.rglob("*.png"))
    assert result.manifest_path == manifest.resolve()
    assert extracted
    with Image.open(extracted[0]) as image:
        assert image.size == (64, 48)
    inputs = processor(videos=[extracted[:2]], return_tensors="pt")
    assert inputs["pixel_values"].shape == (1, 2, 3, 16, 16)
