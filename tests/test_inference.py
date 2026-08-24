from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import egosieve.inference as inference_module
from egosieve.compiler import SegmentCompilerConfig, VideoCompiler
from egosieve.inference import (
    ScanConfig,
    ScanResult,
    WindowPrediction,
    build_scan_records,
    predict_prepared_video,
    scan_video,
)
from egosieve.video import ExtractedFrame, PreparedVideo, VideoMetadata, plan_frame_samples


class FakeProcessor:
    num_frames = 2

    def __call__(self, *, videos, return_tensors):
        batch = len(videos)
        frames = len(videos[0])
        return {
            "pixel_values": torch.zeros(batch, frames, 3, 4, 4),
            "frame_mask": torch.ones(batch, frames, dtype=torch.long),
        }


class FakeOutput:
    def __init__(self, batch: int, frames: int):
        self.logits = torch.tensor([[4.0, 0.0, -2.0]]).repeat(batch, 1)
        self.issue_logits = torch.tensor([[1.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0]]).repeat(
            batch, 1
        )
        self.boundary_logits = torch.zeros(batch, frames, 2)
        self.clip_embedding = torch.ones(batch, 4) / 2


class FakeModel:
    config = SimpleNamespace(
        num_frames=2,
        max_frames=8,
        issue_labels=[
            "acting_hand_not_visible",
            "low_hand_activity",
            "camera_instability",
            "blur",
            "exposure",
            "scene_cut",
            "duplicate_frames",
        ],
        num_issue_labels=7,
        _commit_hash=None,
    )

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, pixel_values, frame_mask):
        return FakeOutput(pixel_values.shape[0], pixel_values.shape[1])


def _prepared(tmp_path) -> PreparedVideo:
    metadata = VideoMetadata(
        source_path="video.mp4",
        source_sha256="b" * 64,
        duration_s=8.0,
        width=100,
        height=100,
        display_width=100,
        display_height=100,
    )
    plan = plan_frame_samples(
        metadata.duration_s,
        window_duration_s=4.0,
        stride_s=4.0,
        frames_per_window=2,
    )
    frames = []
    for sample in plan.samples:
        path = tmp_path / f"{sample.index}.jpg"
        path.write_bytes(b"fixture")
        frames.append(ExtractedFrame(sample, str(path)))
    return PreparedVideo(metadata, plan, tuple(frames))


def test_predictions_and_rich_manifest(tmp_path) -> None:
    prepared = _prepared(tmp_path)
    config = ScanConfig(
        frames_per_window=2,
        batch_size=1,
        device="cpu",
        include_embeddings=True,
        compiler=SegmentCompilerConfig(merge_gap_s=0, min_duration_s=0),
    )
    predictions = predict_prepared_video(
        prepared,
        model=FakeModel(),
        processor=FakeProcessor(),
        config=config,
    )
    assert len(predictions) == 2
    assert predictions[0].decision == "KEEP"
    assert predictions[0].issues["acting_hand_not_visible"] > 0.5
    assert len(predictions[0].embedding) == 4

    compilation = VideoCompiler(config.compiler).compile(
        prepared.metadata,
        [prediction.as_window_score() for prediction in predictions],
        plan=prepared.plan,
    )
    records = build_scan_records(
        prepared,
        predictions,
        compilation.segments,
        model_id="local/test",
        config=config,
    )
    json.dumps(records, allow_nan=False)
    assert records[0]["model"]["id"] == "local/test"
    assert records[0]["counts"]["window_decisions"] == {
        "KEEP": 2,
        "REVIEW": 0,
        "REJECT": 0,
    }
    window = next(record for record in records if record["record_type"] == "window")
    assert window["decision"] == "KEEP"
    assert window["reported_issues"] == ["acting_hand_not_visible"]
    segment = next(record for record in records if record["record_type"] == "segment")
    assert segment["segment_id"].startswith("sha256:")


def test_scan_result_counts_raw_rejects_separately_from_emitted_segments(tmp_path) -> None:
    prepared = _prepared(tmp_path)
    prediction = WindowPrediction(
        index=0,
        start_s=0,
        end_s=4,
        readiness={"KEEP": 0.01, "REVIEW": 0.04, "REJECT": 0.95},
        issues={},
        boundary=(),
        uncertainty=0.1,
    )
    result = ScanResult(
        manifest_path=tmp_path / "manifest.jsonl",
        metadata=prepared.metadata,
        plan=prepared.plan,
        windows=(prediction,),
        segments=(),
    )

    assert result.decision_counts == {"KEEP": 0, "REVIEW": 0, "REJECT": 1}
    assert result.window_decision_counts == result.decision_counts
    assert result.segment_decision_counts == {"KEEP": 0, "REVIEW": 0, "REJECT": 0}


def test_scan_rejects_frame_contract_mismatch_before_probe(tmp_path) -> None:
    source = tmp_path / "not-a-real-video.mp4"
    source.write_bytes(b"the mismatch must be detected before ffprobe")

    with pytest.raises(ValueError, match="frame-count contract mismatch"):
        scan_video(
            source,
            model_id="local/test",
            output_path=tmp_path / "manifest.jsonl",
            config=ScanConfig(frames_per_window=3, device="cpu"),
            ffprobe_bin="this-command-must-not-run",
            model=FakeModel(),
            processor=FakeProcessor(),
        )


def test_prepared_inference_requires_declared_frame_contract(tmp_path) -> None:
    with pytest.raises(ValueError, match="contract is incomplete"):
        predict_prepared_video(
            _prepared(tmp_path),
            model=FakeModel(),
            processor=object(),
            config=ScanConfig(frames_per_window=2, device="cpu"),
        )


def test_prepared_inference_rejects_legacy_issue_label_contract(tmp_path) -> None:
    model = FakeModel()
    model.config = SimpleNamespace(
        num_frames=2,
        max_frames=8,
        issue_labels=[
            "no_hands",
            "low_hand_activity",
            "hand_occlusion",
            "camera_instability",
            "blur",
            "exposure",
            "scene_cut",
            "duplicate_frames",
        ],
        num_issue_labels=8,
    )

    with pytest.raises(ValueError, match="issue-label contract mismatch"):
        predict_prepared_video(
            _prepared(tmp_path),
            model=model,
            processor=FakeProcessor(),
            config=ScanConfig(
                frames_per_window=2,
                device="cpu",
                use_checkpoint_calibration=False,
            ),
        )


def test_prepared_inference_rejects_wrong_issue_output_width(tmp_path) -> None:
    class WideIssueModel(FakeModel):
        def __call__(self, pixel_values, frame_mask):
            output = super().__call__(pixel_values, frame_mask)
            output.issue_logits = torch.zeros(pixel_values.shape[0], 8)
            return output

    with pytest.raises(ValueError, match="issue_logits shape mismatch"):
        predict_prepared_video(
            _prepared(tmp_path),
            model=WideIssueModel(),
            processor=FakeProcessor(),
            config=ScanConfig(frames_per_window=2, device="cpu"),
        )


def test_scan_cannot_overwrite_its_source(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source bytes")

    with pytest.raises(ValueError, match="source video"):
        scan_video(
            source,
            model_id="local/test",
            output_path=source,
            model=FakeModel(),
            processor=FakeProcessor(),
            config=ScanConfig(frames_per_window=2, device="cpu"),
        )

    assert source.read_bytes() == b"source bytes"


def test_hub_processor_load_uses_model_resolved_commit(monkeypatch) -> None:
    calls = []
    model = FakeModel()
    model.config = SimpleNamespace(num_frames=2, max_frames=8, _commit_hash="a" * 40)
    processor = FakeProcessor()

    class ModelLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("model", model_id, kwargs))
            return model

    class ProcessorLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("processor", model_id, kwargs))
            return processor

    monkeypatch.setattr(inference_module, "AutoModelForVideoClassification", ModelLoader)
    monkeypatch.setattr(inference_module, "AutoProcessor", ProcessorLoader)

    loaded_model, loaded_processor, revision = inference_module._load_components(
        "org/model",
        revision="main",
        model=None,
        processor=None,
    )

    assert loaded_model is model
    assert loaded_processor is processor
    assert revision == "a" * 40
    assert calls[0][2]["revision"] == "main"
    assert calls[1][2]["revision"] == "a" * 40
