from __future__ import annotations

import json

import pytest
import torch

from egosieve import (
    ISSUE_LABELS,
    READINESS_LABELS,
    EgoSieveConfig,
    EgoSieveModel,
)


def tiny_config(**overrides) -> EgoSieveConfig:
    values = {
        "vision_config": {
            "model_type": "dinov2",
            "image_size": 16,
            "patch_size": 8,
            "num_channels": 3,
            "hidden_size": 24,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 48,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "drop_path_rate": 0.0,
        },
        "temporal_hidden_size": 16,
        "temporal_num_layers": 2,
        "temporal_intermediate_size": 24,
        "temporal_kernel_size": 3,
        "num_frames": 4,
        "max_frames": 8,
        "projection_dim": 12,
        "dropout": 0.0,
    }
    values.update(overrides)
    return EgoSieveConfig(**values)


def test_pixel_forward_shapes_and_label_contract() -> None:
    torch.manual_seed(1)
    config = tiny_config()
    model = EgoSieveModel(config).eval()
    pixels = torch.randn(2, 4, 3, 16, 16)
    frame_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)

    with torch.no_grad():
        output = model(pixel_values=pixels, frame_mask=frame_mask)

    assert output.loss is None
    assert output.logits.shape == (2, 3)
    assert output.readiness_logits is output.logits
    assert output.readiness_logits.shape == (2, 3)
    assert output.issue_logits.shape == (2, 7)
    assert output.boundary_logits.shape == (2, 4, 2)
    assert output.clip_embedding.shape == (2, 12)
    torch.testing.assert_close(output.clip_embedding.norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(output.boundary_logits[1, 2:], torch.zeros(2, 2))
    assert tuple(config.readiness_labels) == READINESS_LABELS
    assert tuple(config.issue_labels) == ISSUE_LABELS
    assert ISSUE_LABELS == (
        "acting_hand_not_visible",
        "low_hand_activity",
        "camera_instability",
        "blur",
        "exposure",
        "scene_cut",
        "duplicate_frames",
    )


@pytest.mark.parametrize(
    "legacy_metadata",
    [
        {
            "issue_labels": [
                "no_hands",
                "low_hand_activity",
                "hand_occlusion",
                "camera_instability",
                "blur",
                "exposure",
                "scene_cut",
                "duplicate_frames",
            ]
        },
        {"num_issue_labels": 8},
    ],
)
def test_legacy_eight_issue_checkpoint_metadata_is_rejected(
    legacy_metadata: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="regenerate the seed checkpoint and retrain"):
        tiny_config(**legacy_metadata)


def test_masked_pixels_cannot_change_any_output() -> None:
    torch.manual_seed(2)
    model = EgoSieveModel(tiny_config(num_frames=5)).eval()
    pixels = torch.randn(1, 5, 3, 16, 16)
    changed = pixels.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 10_000
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)

    with torch.no_grad():
        first = model(pixel_values=pixels, frame_mask=mask)
        second = model(pixel_values=changed, frame_mask=mask)

    for field in (
        "readiness_logits",
        "issue_logits",
        "boundary_logits",
        "clip_embedding",
    ):
        torch.testing.assert_close(getattr(first, field), getattr(second, field), rtol=0, atol=0)


def test_multitask_loss_supports_ignore_values_and_masks() -> None:
    torch.manual_seed(3)
    config = tiny_config(
        readiness_loss_weight=0.5,
        issue_loss_weight=1.5,
        boundary_loss_weight=2.0,
    )
    model = EgoSieveModel(config)
    embeddings = torch.randn(2, 4, config.vision_config.hidden_size)
    frame_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    readiness_labels = torch.tensor([2, -100])
    issue_labels = torch.randint(0, 2, (2, len(ISSUE_LABELS)), dtype=torch.float32)
    issue_labels[0, 1] = float("nan")
    issue_labels[1, 3] = -100
    issue_label_mask = torch.tensor([1, 0], dtype=torch.bool)
    boundary_labels = torch.randint(0, 2, (2, 4, 2), dtype=torch.float32)
    # These values are ignored by frame_mask, even though one is non-finite.
    boundary_labels[1, 2, 0] = float("nan")
    boundary_labels[1, 3] = -100

    output = model(
        frame_embeddings=embeddings,
        frame_mask=frame_mask,
        readiness_labels=readiness_labels,
        issue_labels=issue_labels,
        boundary_labels=boundary_labels,
        issue_label_mask=issue_label_mask,
    )

    assert output.loss is not None
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.readiness_classifier.weight.grad is not None
    assert model.issue_classifier.weight.grad is not None
    assert model.boundary_classifier.weight.grad is not None

    # All-ignored targets return a differentiable zero instead of NaN.
    ignored = model(
        frame_embeddings=embeddings,
        readiness_labels=torch.full((2,), -100),
        issue_labels=torch.full((2, len(ISSUE_LABELS)), -100.0),
        boundary_labels=torch.full((2, 4, 2), -100.0),
    )
    assert ignored.loss is not None
    torch.testing.assert_close(ignored.loss, torch.zeros_like(ignored.loss))


def test_local_save_and_load_round_trip(tmp_path) -> None:
    torch.manual_seed(4)
    model = EgoSieveModel(tiny_config()).eval()
    embeddings = torch.randn(2, 3, model.config.vision_config.hidden_size)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    with torch.no_grad():
        expected = model(frame_embeddings=embeddings, frame_mask=mask)

    model.save_pretrained(tmp_path, safe_serialization=False)
    restored = EgoSieveModel.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.no_grad():
        actual = restored(frame_embeddings=embeddings, frame_mask=mask)

    for field in (
        "readiness_logits",
        "issue_logits",
        "boundary_logits",
        "clip_embedding",
    ):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field))
    assert isinstance(restored.config, EgoSieveConfig)
    assert isinstance(restored.config.vision_config.hidden_size, int)

    saved_config = json.loads((tmp_path / "config.json").read_text())
    assert saved_config["model_type"] == "egosieve"
    assert saved_config["vision_config"]["model_type"] == "dinov2"
    assert saved_config["num_frames"] == 4
    assert saved_config["issue_labels"] == list(ISSUE_LABELS)


def test_input_and_label_validation() -> None:
    model = EgoSieveModel(tiny_config())
    embeddings = torch.randn(1, 2, model.config.vision_config.hidden_size)

    with pytest.raises(ValueError, match="exactly one"):
        model()
    with pytest.raises(ValueError, match="exactly one"):
        model(
            pixel_values=torch.randn(1, 2, 3, 16, 16),
            frame_embeddings=embeddings,
        )
    with pytest.raises(ValueError, match="readiness_labels"):
        model(frame_embeddings=embeddings, readiness_labels=torch.zeros(1, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="checkpoint num_frames"):
        model(pixel_values=torch.randn(1, 3, 3, 16, 16))
