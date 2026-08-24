"""Create a trainable EgoSieve checkpoint from a licensed vision backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import Dinov2Model

from .modeling import EgoSieveConfig, EgoSieveModel
from .processing_egosieve import EgoSieveProcessor

UNTRAINED_MARKER = "UNTRAINED_HEADS"


def initialize_from_backbone(
    backbone: Dinov2Model,
    **config_overrides: Any,
) -> EgoSieveModel:
    """Build a model and copy a caller-provided DINOv2 backbone exactly.

    The function never resolves a model id or accesses the network. Callers
    remain responsible for choosing and documenting an allowed backbone.
    Task-specific heads are newly initialized and must be trained before use.
    """

    config = EgoSieveConfig(
        vision_config=backbone.config.to_dict(),
        **config_overrides,
    )
    model = EgoSieveModel(config)
    incompatible = model.vision_model.load_state_dict(backbone.state_dict(), strict=True)
    if (
        incompatible.missing_keys or incompatible.unexpected_keys
    ):  # pragma: no cover - strict guards
        raise RuntimeError(f"vision-backbone copy failed: {incompatible}")
    return model


def register_for_hub() -> None:
    """Attach custom AutoClass metadata before ``save_pretrained``."""

    EgoSieveConfig.register_for_auto_class("AutoConfig")
    EgoSieveModel.register_for_auto_class("AutoModelForVideoClassification")
    EgoSieveProcessor.register_for_auto_class("AutoProcessor")


def save_training_seed(
    model: EgoSieveModel,
    output_dir: str | Path,
    *,
    processor: EgoSieveProcessor | None = None,
) -> Path:
    """Save initialized backbone weights with an explicit non-release marker."""

    if processor is None:
        processor = EgoSieveProcessor(
            size=model.config.vision_config.image_size,
            num_frames=model.config.num_frames,
        )
    elif processor.num_frames != model.config.num_frames:
        raise ValueError(
            "processor.num_frames must match model.config.num_frames before saving; "
            f"received {processor.num_frames} and {model.config.num_frames}."
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    register_for_hub()
    model.save_pretrained(root, safe_serialization=True)
    processor.save_pretrained(root)
    (root / UNTRAINED_MARKER).write_text(
        "The task heads in this directory are initialized, not trained.\n",
        encoding="utf-8",
    )
    return root


__all__ = [
    "UNTRAINED_MARKER",
    "initialize_from_backbone",
    "register_for_hub",
    "save_training_seed",
]
