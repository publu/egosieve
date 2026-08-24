"""Configuration for the EgoSieve video-readiness model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import isfinite
from typing import Any

from transformers import Dinov2Config, PretrainedConfig

# The order is part of the public model contract.  In particular, changing it
# would silently reinterpret existing classifier weights.
READINESS_LABELS = ("KEEP", "REVIEW", "REJECT")
ISSUE_LABELS = (
    "no_hands",
    "low_hand_activity",
    "hand_occlusion",
    "camera_instability",
    "blur",
    "exposure",
    "scene_cut",
    "duplicate_frames",
)
BOUNDARY_LABELS = ("start", "end")


def _default_vision_config() -> dict[str, Any]:
    """Return a DINOv2-S/14-shaped config without resolving remote assets."""

    return {
        "model_type": "dinov2",
        "image_size": 224,
        "patch_size": 14,
        "num_channels": 3,
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "drop_path_rate": 0.0,
    }


class EgoSieveConfig(PretrainedConfig):
    """Configuration for :class:`~egosieve.EgoSieveModel`.

    ``vision_config`` is either a :class:`transformers.Dinov2Config` or its
    dictionary representation.  Supplying a dictionary makes it easy to build
    very small, fully offline backbones for tests.  The default describes a
    DINOv2-S/14-sized encoder, but does not download weights or configuration.
    """

    model_type = "egosieve"
    is_composition = True

    def __init__(
        self,
        vision_config: Dinov2Config | Mapping[str, Any] | None = None,
        temporal_hidden_size: int = 256,
        temporal_num_layers: int = 3,
        temporal_intermediate_size: int = 512,
        temporal_kernel_size: int = 3,
        num_frames: int = 12,
        max_frames: int = 512,
        projection_dim: int = 128,
        dropout: float = 0.1,
        initializer_range: float = 0.02,
        readiness_loss_weight: float = 1.0,
        issue_loss_weight: float = 1.0,
        boundary_loss_weight: float = 1.0,
        issue_pos_weight: Sequence[float] | None = None,
        readiness_temperature: float = 1.0,
        issue_thresholds: Mapping[str, float] | None = None,
        compiler_thresholds: Mapping[str, Any] | None = None,
        calibration_source: str | None = None,
        ignore_index: int = -100,
        **kwargs: Any,
    ) -> None:
        id2label = {index: label for index, label in enumerate(READINESS_LABELS)}
        label2id = {label: index for index, label in enumerate(READINESS_LABELS)}
        # These mappings are fixed because they describe the three readiness
        # logits.  Keeping them stable also makes Trainer prediction metadata
        # unambiguous.
        kwargs.pop("num_labels", None)
        kwargs["id2label"] = id2label
        kwargs["label2id"] = label2id
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = _default_vision_config()
        if isinstance(vision_config, Dinov2Config):
            self.vision_config = vision_config
        elif isinstance(vision_config, Mapping):
            vision_dict = deepcopy(dict(vision_config))
            model_type = vision_dict.pop("model_type", "dinov2")
            if model_type != "dinov2":
                raise ValueError(
                    "EgoSieveConfig requires a DINOv2 vision_config; "
                    f"received model_type={model_type!r}."
                )
            self.vision_config = Dinov2Config(**vision_dict)
        else:
            raise TypeError(
                "vision_config must be a Dinov2Config, a mapping, or None; "
                f"received {type(vision_config).__name__}."
            )

        positive_ints = {
            "temporal_hidden_size": temporal_hidden_size,
            "temporal_num_layers": temporal_num_layers,
            "temporal_intermediate_size": temporal_intermediate_size,
            "temporal_kernel_size": temporal_kernel_size,
            "num_frames": num_frames,
            "max_frames": max_frames,
            "projection_dim": projection_dim,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; received {value!r}.")
        if temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be odd so sequence length is preserved.")
        if num_frames > max_frames:
            raise ValueError(f"num_frames ({num_frames}) cannot exceed max_frames ({max_frames}).")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); received {dropout!r}.")
        if initializer_range <= 0:
            raise ValueError("initializer_range must be positive.")

        loss_weights = {
            "readiness_loss_weight": readiness_loss_weight,
            "issue_loss_weight": issue_loss_weight,
            "boundary_loss_weight": boundary_loss_weight,
        }
        for name, value in loss_weights.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative; received {value!r}.")

        if issue_pos_weight is not None:
            issue_pos_weight = tuple(float(value) for value in issue_pos_weight)
            if len(issue_pos_weight) != len(ISSUE_LABELS):
                raise ValueError(
                    "issue_pos_weight must contain one value for each issue "
                    f"({len(ISSUE_LABELS)} values)."
                )
            if any(value <= 0 for value in issue_pos_weight):
                raise ValueError("Every issue_pos_weight value must be positive.")

        readiness_temperature = float(readiness_temperature)
        if not isfinite(readiness_temperature) or readiness_temperature <= 0:
            raise ValueError("readiness_temperature must be finite and positive")
        if issue_thresholds is None:
            issue_thresholds = {name: 0.35 for name in ISSUE_LABELS}
        else:
            issue_thresholds = dict(issue_thresholds)
        if set(issue_thresholds) != set(ISSUE_LABELS):
            raise ValueError("issue_thresholds must contain every configured issue label exactly")
        normalized_issue_thresholds = {}
        for name in ISSUE_LABELS:
            value = float(issue_thresholds[name])
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"issue threshold for {name!r} must lie in [0, 1]")
            normalized_issue_thresholds[name] = value

        default_compiler = {
            "enter_threshold": 0.7,
            "exit_threshold": 0.5,
            "merge_gap_s": 0.5,
            "min_duration_s": 1.0,
            "uncertainty_threshold": 0.5,
            "uncertainty_route": "review",
            "short_segment_route": "discard",
            "include_discard": False,
        }
        if compiler_thresholds is not None:
            unknown = set(compiler_thresholds) - set(default_compiler)
            if unknown:
                raise ValueError(f"unknown compiler_thresholds keys: {sorted(unknown)}")
            default_compiler.update(dict(compiler_thresholds))

        self.temporal_hidden_size = temporal_hidden_size
        # ``hidden_size`` is useful to generic Transformers tooling.
        self.hidden_size = temporal_hidden_size
        self.temporal_num_layers = temporal_num_layers
        self.temporal_intermediate_size = temporal_intermediate_size
        self.temporal_kernel_size = temporal_kernel_size
        # Hugging Face's video-classification pipeline reads this value to
        # determine how many frames to decode for a checkpoint.
        self.num_frames = num_frames
        self.max_frames = max_frames
        self.projection_dim = projection_dim
        self.dropout = float(dropout)
        self.initializer_range = float(initializer_range)
        self.readiness_loss_weight = float(readiness_loss_weight)
        self.issue_loss_weight = float(issue_loss_weight)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.issue_pos_weight = issue_pos_weight
        self.readiness_temperature = readiness_temperature
        self.issue_thresholds = normalized_issue_thresholds
        self.compiler_thresholds = default_compiler
        self.calibration_source = calibration_source
        self.ignore_index = int(ignore_index)

        # JSON-friendly copies make the output ordering discoverable without
        # importing Python constants.
        self.readiness_labels = list(READINESS_LABELS)
        self.issue_labels = list(ISSUE_LABELS)
        self.boundary_labels = list(BOUNDARY_LABELS)
        self.num_readiness_labels = len(READINESS_LABELS)
        self.num_issue_labels = len(ISSUE_LABELS)
        self.num_boundary_labels = len(BOUNDARY_LABELS)


__all__ = [
    "BOUNDARY_LABELS",
    "EgoSieveConfig",
    "ISSUE_LABELS",
    "READINESS_LABELS",
]
