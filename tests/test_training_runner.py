from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from transformers import Dinov2Config, Dinov2Model

from egosieve.initialization import initialize_from_backbone, save_training_seed
from egosieve.modeling import ISSUE_LABELS, EgoSieveModel
from egosieve.processing_egosieve import EgoSieveProcessor
from egosieve.training import TrainingRunConfig, train_checkpoint
from egosieve.training.data import TrainingWindow
from egosieve.training.features import WindowExample, feature_cache_key
from egosieve.training.runner import _contrastive_loss, _test_prediction_document


def _video(path: Path, hue: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=32x24:rate=12, hue=h={hue}",
            "-t",
            "1.2",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        check=True,
    )


def _seed(path: Path) -> EgoSieveModel:
    backbone = Dinov2Model(
        Dinov2Config(
            image_size=16,
            patch_size=8,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    )
    model = initialize_from_backbone(
        backbone,
        num_frames=2,
        max_frames=4,
        temporal_hidden_size=12,
        temporal_num_layers=1,
        temporal_intermediate_size=20,
        projection_dim=8,
        dropout=0.2,
    )
    save_training_seed(
        model,
        path,
        processor=EgoSieveProcessor(
            size=16,
            resize_shortest_edge=16,
            num_frames=2,
        ),
    )
    return model


def test_feature_cache_identity_depends_on_media_and_timestamps_not_labels(tmp_path: Path) -> None:
    video = tmp_path / "same.mp4"
    video.write_bytes(b"stable fixture identity")
    first = WindowExample(
        key="record-a:0",
        record_id="record-a",
        group_id="capture",
        source="fixture",
        license="CC0-1.0",
        video_path=video,
        window_index=0,
        window=TrainingWindow(0.0, 1.0, readiness="KEEP", readiness_valid=True),
        timestamps_s=(0.25, 0.75),
    )
    second = WindowExample(
        key="record-b:7",
        record_id="record-b",
        group_id="capture",
        source="fixture",
        license="CC0-1.0",
        video_path=video,
        window_index=7,
        window=TrainingWindow(0.0, 1.0, issues={"blur": False}),
        timestamps_s=(0.25, 0.75),
    )
    assert feature_cache_key(first, "artifact") == feature_cache_key(second, "artifact")


def test_contrastive_loss_masks_different_examples_from_the_same_group() -> None:
    first = torch.eye(3, requires_grad=True)
    second = torch.eye(3, requires_grad=True)
    groups = ("capture-a", "capture-a", "capture-b")

    actual = _contrastive_loss(first, second, 1.0, groups)
    logits = first @ second.T
    valid = torch.tensor(
        [
            [True, False, True],
            [False, True, True],
            [True, True, True],
        ]
    )
    masked = logits.masked_fill(~valid, -torch.inf)
    labels = torch.arange(3)
    expected = 0.5 * (
        torch.nn.functional.cross_entropy(masked, labels)
        + torch.nn.functional.cross_entropy(masked.T, labels)
    )
    ordinary_instance_loss = torch.nn.functional.cross_entropy(logits, labels)

    torch.testing.assert_close(actual, expected)
    assert actual < ordinary_instance_loss


def test_contrastive_loss_preserves_distinct_group_instance_discrimination() -> None:
    first = torch.tensor([[1.0, 0.0], [0.2, 0.8], [-0.5, 0.5]])
    second = torch.tensor([[0.9, 0.1], [0.0, 1.0], [-0.4, 0.6]])
    labels = torch.arange(3)
    logits = first @ second.T / 0.25
    expected = 0.5 * (
        torch.nn.functional.cross_entropy(logits, labels)
        + torch.nn.functional.cross_entropy(logits.T, labels)
    )

    actual = _contrastive_loss(first, second, 0.25, ("a", "b", "c"))

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_contrastive_loss_is_safe_without_any_eligible_negatives(batch_size: int) -> None:
    first = torch.randn(batch_size, 4, requires_grad=True)
    second = torch.randn(batch_size, 4, requires_grad=True)

    loss = _contrastive_loss(first, second, 0.1, ("one-capture",) * batch_size)

    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    assert first.grad is not None
    assert second.grad is not None
    torch.testing.assert_close(first.grad, torch.zeros_like(first))
    torch.testing.assert_close(second.grad, torch.zeros_like(second))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_tiny_feature_cached_training_builds_evidence(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    initial = _seed(seed_dir)
    rows = []
    labels = ("KEEP", "REVIEW", "REJECT")
    for group_index in range(3):
        video = tmp_path / f"group-{group_index}.mp4"
        _video(video, group_index * 45)
        windows = []
        for window_index, label in enumerate(labels):
            start = window_index * 0.3
            end = start + 0.3
            issues = {
                name: bool((window_index + issue_index) % 2)
                for issue_index, name in enumerate(ISSUE_LABELS)
            }
            windows.append(
                {
                    "start_s": start,
                    "end_s": end,
                    "readiness": label,
                    "readiness_valid": True,
                    "issues": issues,
                    "issue_valid": {name: True for name in ISSUE_LABELS},
                    "boundaries_s": {"start": start + 0.075, "end": end - 0.075},
                    "boundary_valid": True,
                    "annotator": "human-fixture",
                    "review_count": 2,
                    "rubric_version": "fixture-v1",
                    "label_provenance": {
                        "readiness": {"kind": "human-derived" if window_index == 1 else "human"},
                        "issues": {"kind": "human"},
                        "boundaries": {"kind": "human-derived"},
                    },
                }
            )
        windows.append(
            {
                "start_s": 0.0,
                "end_s": 0.3,
                "readiness_valid": False,
                "issues": {"blur": True},
                "issue_valid": {"blur": True},
                "boundary_valid": False,
                "annotator": "programmatic-fixture",
                "label_provenance": {
                    "readiness": {"kind": "unlabeled"},
                    "issues": {"kind": "programmatic-controlled-corruption"},
                    "boundaries": {"kind": "unlabeled"},
                },
            }
        )
        rows.append(
            {
                "schema": "egosieve.training/v1",
                "id": f"episode-{group_index}",
                "group_id": f"capture-{group_index}",
                "video": video.name,
                "license": "CC-BY-4.0",
                "source": "fixture",
                "windows": windows,
            }
        )
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "release"
    result = train_checkpoint(
        annotations,
        seed_checkpoint=seed_dir,
        output_dir=output,
        cache_dir=tmp_path / "features",
        allowed_licenses=["CC-BY-4.0"],
        config=TrainingRunConfig(
            seed=3,
            train_fraction=1 / 3,
            validation_fraction=1 / 3,
            test_fraction=1 / 3,
            epochs=2,
            patience=1,
            batch_size=3,
            device="cpu",
            boundary_tolerance_s=0.16,
        ),
        backbone_revision="fixture-sha",
        source_commit="fixture-source-commit",
    )

    assert result["test_examples"] == 4
    for name in (
        "model.safetensors",
        "metrics.json",
        "test_predictions.json",
        "splits.json",
        "training_report.json",
        "evidence.json",
    ):
        assert (output / name).is_file()
    report = json.loads((output / "training_report.json").read_text())
    assert report["calibration"]["readiness_temperature"] > 0
    assert report["backbone"]["revision"] == "fixture-sha"
    predictions = json.loads((output / "test_predictions.json").read_text())
    assert len(predictions["examples"]) == 4
    corruption = next(row for row in predictions["examples"] if not row["readiness_valid"])
    assert corruption["readiness_target"] is None
    assert corruption["label_provenance"] == {
        "readiness": {"kind": "unlabeled"},
        "issues": {"kind": "programmatic-controlled-corruption"},
        "boundaries": {"kind": "unlabeled"},
    }
    assert "review_count" not in corruption
    assert "rubric_version" not in corruption
    metrics = json.loads((output / "metrics.json").read_text())
    evaluation = metrics["evaluation"]
    readiness_macro_f1 = evaluation["per_source"]["fixture"]["readiness_macro_f1"]
    assert "human_reviewed" not in evaluation
    assert evaluation == {
        "annotation_guide": "docs/ANNOTATION.md (v0.1)",
        "boundary_examples": 3,
        "grouped_split": True,
        "issue_examples": 4,
        "issue_examples_by_provenance": {
            "human": 3,
            "human-derived": 0,
            "programmatic-controlled-corruption": 1,
        },
        "issues_controlled_corruptions": True,
        "per_source": {
            "fixture": {
                "readiness_examples": 3,
                "readiness_macro_f1": readiness_macro_f1,
            }
        },
        "readiness_examples": 3,
        "readiness_examples_by_provenance": {
            "human": 2,
            "human-derived": 1,
        },
        "readiness_human_grounded": True,
        "test_examples": 4,
    }
    card = (output / "README.md").read_text()
    assert "task-level provenance" in card
    assert "Human-derived rows are not direct, independent EgoSieve rubric judgments" in card
    assert "dataset-specific proxy details" in card
    assert "human-reviewed" not in card
    trained = EgoSieveModel.from_pretrained(output, local_files_only=True)
    assert not torch.equal(
        initial.clip_projection.weight,
        trained.clip_projection.weight,
    )


def test_release_prediction_export_rejects_flat_provenance(tmp_path: Path) -> None:
    example = WindowExample(
        key="record-a:0",
        record_id="record-a",
        group_id="capture-a",
        source="fixture",
        license="CC0-1.0",
        video_path=tmp_path / "unused.mp4",
        window_index=0,
        window=TrainingWindow(
            0.0,
            1.0,
            readiness="KEEP",
            readiness_valid=True,
            issues={"blur": True},
            issue_valid={"blur": True},
            boundaries_s={"start": 0.1, "end": 0.9},
            boundary_valid=True,
            extra={
                "review_count": 2,
                "rubric_version": "fixture-v1",
                "label_provenance": {"kind": "human"},
            },
        ),
        timestamps_s=(0.25, 0.75),
    )
    predictions = {
        "readiness_valid": [True],
        "readiness_labels": [0],
        "readiness_probabilities": [[0.8, 0.1, 0.1]],
        "issue_valid": [[label == "blur" for label in ISSUE_LABELS]],
        "issue_labels": [[1.0 if label == "blur" else 0.0 for label in ISSUE_LABELS]],
        "issue_probabilities": [[0.9 if label == "blur" else 0.1 for label in ISSUE_LABELS]],
        "boundary_reference_s": [[0.1, 0.9]],
        "boundary_reference_valid": [[True, True]],
        "boundary_prediction_s": [[0.1, 0.9]],
        "boundary_prediction_valid": [[True, True]],
    }
    with pytest.raises(ValueError, match="flat provenance kinds are not release evidence"):
        _test_prediction_document(
            predictions,
            [example],
            checkpoint_sha256="0" * 64,
            splits_sha256="1" * 64,
        )
