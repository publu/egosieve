from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import (
    AutoConfig,
    AutoModelForVideoClassification,
    AutoProcessor,
    Dinov2Config,
    Dinov2Model,
)

from egosieve.initialization import UNTRAINED_MARKER, initialize_from_backbone, save_training_seed
from egosieve.processing_egosieve import EgoSieveProcessor


def _tiny_backbone() -> Dinov2Model:
    return Dinov2Model(
        Dinov2Config(
            image_size=16,
            patch_size=8,
            hidden_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=48,
        )
    )


def test_initialization_copies_backbone_exactly() -> None:
    torch.manual_seed(11)
    backbone = _tiny_backbone()
    model = initialize_from_backbone(
        backbone,
        temporal_hidden_size=16,
        temporal_num_layers=1,
        temporal_intermediate_size=24,
        projection_dim=8,
        num_frames=4,
        max_frames=4,
    )
    for expected, actual in zip(
        backbone.parameters(), model.vision_model.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual)


def test_training_seed_is_fail_closed(tmp_path) -> None:
    model = initialize_from_backbone(
        _tiny_backbone(),
        temporal_hidden_size=16,
        temporal_num_layers=1,
        temporal_intermediate_size=24,
        projection_dim=8,
        num_frames=4,
        max_frames=4,
    )
    save_training_seed(model, tmp_path, processor=EgoSieveProcessor(size=16, num_frames=4))
    assert (tmp_path / "model.safetensors").is_file()
    assert (tmp_path / UNTRAINED_MARKER).is_file()
    assert (tmp_path / "preprocessor_config.json").is_file()

    auto_config = AutoConfig.from_pretrained(
        tmp_path, trust_remote_code=True, local_files_only=True
    )
    auto_processor = AutoProcessor.from_pretrained(
        tmp_path, trust_remote_code=True, local_files_only=True
    )
    auto_model = AutoModelForVideoClassification.from_pretrained(
        tmp_path, trust_remote_code=True, local_files_only=True
    )
    assert auto_config.model_type == "egosieve"
    assert auto_processor.num_frames == 4
    assert auto_model.config.num_frames == 4
    assert auto_model.config.projection_dim == 8


def test_training_seed_rejects_processor_frame_mismatch(tmp_path) -> None:
    model = initialize_from_backbone(
        _tiny_backbone(),
        temporal_hidden_size=16,
        temporal_num_layers=1,
        temporal_intermediate_size=24,
        projection_dim=8,
        num_frames=4,
        max_frames=4,
    )
    with pytest.raises(ValueError, match="processor.num_frames"):
        save_training_seed(model, tmp_path, processor=EgoSieveProcessor(size=16, num_frames=3))

    assert not (tmp_path / "model.safetensors").exists()
