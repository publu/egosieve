from __future__ import annotations

import json

import numpy as np
import torch

from egosieve.modeling import EgoSieveConfig, EgoSieveModel
from egosieve.training import SCHEMA_VERSION, TrainingCollator, loads_jsonl


def _tiny_config() -> EgoSieveConfig:
    return EgoSieveConfig(
        vision_config={
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
        temporal_hidden_size=16,
        temporal_num_layers=1,
        temporal_intermediate_size=24,
        temporal_kernel_size=3,
        num_frames=4,
        max_frames=4,
        projection_dim=12,
        dropout=0.0,
    )


def test_parsed_record_to_sampled_collator_to_tiny_model_loss() -> None:
    raw = {
        "schema": SCHEMA_VERSION,
        "id": "episode-1",
        "group_id": "session-1",
        "video": "videos/episode-1.mp4",
        "license": "dataset-owner-declared",
        "windows": [
            {
                "start_s": 0.0,
                "end_s": 4.0,
                "readiness": "KEEP",
                "readiness_valid": True,
                "issues": {"blur": True},
                "issue_valid": {"blur": True},
                "boundaries_s": {"start": 0.9, "end": 2.8},
                "boundary_valid": True,
                "annotator": "human",
            }
        ],
    }
    record = loads_jsonl(json.dumps(raw))[0]
    config = _tiny_config()
    embeddings = np.arange(4 * 24, dtype=np.float32).reshape(4, 24) / 100
    batch = TrainingCollator(boundary_tolerance_s=0.25)(
        [
            {
                "window": record.windows[0],
                "sampled_timestamps_s": [0.0, 1.0, 2.0, 3.0],
                "frame_embeddings": embeddings,
            }
        ]
    )

    assert batch["boundary_labels"].shape == (1, 4, 2)
    assert batch["boundary_label_mask"].shape == (1, 4, 2)
    assert batch["boundary_labels"][0, :, 0].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert batch["boundary_labels"][0, :, 1].tolist() == [0.0, 0.0, 0.0, 1.0]

    model = EgoSieveModel(config)
    tensor_batch = {key: torch.from_numpy(value) for key, value in batch.items()}
    output = model(**tensor_batch)

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.readiness_classifier.weight.grad is not None
    assert model.issue_classifier.weight.grad is not None
    assert model.boundary_classifier.weight.grad is not None
