from __future__ import annotations

import hashlib
import json

import pytest
import torch
from torch.nn import functional as F
from transformers import Dinov2Config, Dinov2Model

from egosieve.initialization import initialize_from_backbone, save_training_seed
from egosieve.modeling import ISSUE_LABELS, READINESS_LABELS
from egosieve.processing_egosieve import EgoSieveProcessor
from egosieve.release import (
    HASHED_RELEASE_FILES,
    RELEASE_EVIDENCE_SCHEMA,
    ReleaseValidationError,
    validate_release,
)


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_trained_model():
    torch.manual_seed(7)
    backbone = Dinov2Model(
        Dinov2Config(
            image_size=16,
            patch_size=8,
            hidden_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=48,
        )
    )
    model = initialize_from_backbone(
        backbone,
        temporal_hidden_size=16,
        temporal_num_layers=1,
        temporal_intermediate_size=24,
        projection_dim=8,
        max_frames=4,
        num_frames=4,
        dropout=0,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    frame_embeddings = torch.randn(6, 4, 24)
    readiness_labels = torch.tensor([0, 1, 2, 0, 1, 2])
    issue_labels = torch.tensor(
        [[float((row + column) % 2) for column in range(len(ISSUE_LABELS))] for row in range(6)]
    )
    boundary_labels = torch.zeros(6, 4, 2)
    boundary_labels[:, 1, 0] = 1
    boundary_labels[:, 2, 1] = 1
    output = model(
        frame_embeddings=frame_embeddings,
        readiness_labels=readiness_labels,
        issue_labels=issue_labels,
        boundary_labels=boundary_labels,
    )
    retrieval_targets = F.normalize(torch.randn_like(output.clip_embedding), dim=-1)
    retrieval_loss = F.mse_loss(output.clip_embedding, retrieval_targets)
    assert output.loss is not None
    (output.loss + retrieval_loss).backward()
    assert model.clip_projection.weight.grad is not None
    optimizer.step()
    return model


def _split_document() -> dict:
    examples = [
        {"id": "train-0", "group_id": "train-group", "source": "train-source", "split": "train"},
        {"id": "train-1", "group_id": "train-group", "source": "train-source", "split": "train"},
        {
            "id": "validation-0",
            "group_id": "validation-group",
            "source": "validation-source",
            "split": "validation",
        },
        {
            "id": "validation-1",
            "group_id": "validation-group",
            "source": "validation-source",
            "split": "validation",
        },
    ]
    examples.extend(
        {
            "id": f"test-{index}",
            "group_id": (f"test-group-{index // 3}" if index < 6 else "test-group-corruptions"),
            "source": f"source-{index // 3}" if index < 6 else "controlled-corruptions",
            "split": "test",
        }
        for index in range(8)
    )
    return {"schema": "egosieve.splits/v1", "examples": examples}


def _prediction_document(checkpoint_hash: str, splits_hash: str) -> dict:
    examples = []
    for index in range(6):
        readiness_index = index % 3
        readiness = [0.05, 0.05, 0.05]
        readiness[readiness_index] = 0.9
        issue_kind = "human" if index == 0 else "human-derived" if index == 1 else "unlabeled"
        issue_mask = [index < 2] * 3 + [False] * (len(ISSUE_LABELS) - 3)
        issue_targets = [index if index < 2 else None] * 3 + [None] * (len(ISSUE_LABELS) - 3)
        examples.append(
            {
                "id": f"test-{index}",
                "group_id": f"test-group-{index // 3}",
                "source": f"source-{index // 3}",
                "label_provenance": {
                    "readiness": {"kind": "human-derived" if index == 1 else "human"},
                    "issues": {"kind": issue_kind},
                    "boundaries": {"kind": "human-derived"},
                },
                "review_count": 2,
                "rubric_version": "egosieve-annotation/v1",
                "readiness_valid": True,
                "readiness_target": READINESS_LABELS[readiness_index],
                "readiness_probabilities": readiness,
                "issue_targets": issue_targets,
                "issue_valid": issue_mask,
                "issue_probabilities": [0.9 if index == 1 else 0.1] * 3
                + [0.1] * (len(ISSUE_LABELS) - 3),
                "boundary_reference_s": [index * 10 + 1.0, index * 10 + 4.0],
                "boundary_prediction_s": [index * 10 + 1.0, index * 10 + 4.0],
                "boundary_reference_valid": [True, True],
                "boundary_prediction_valid": [True, True],
            }
        )
    for index, issue_target in ((6, 0), (7, 1)):
        examples.append(
            {
                "id": f"test-{index}",
                "group_id": "test-group-corruptions",
                "source": "controlled-corruptions",
                "label_provenance": {
                    "readiness": {"kind": "unlabeled"},
                    "issues": {"kind": "programmatic-controlled-corruption"},
                    "boundaries": {"kind": "unlabeled"},
                },
                "readiness_valid": False,
                "readiness_target": None,
                "readiness_probabilities": [0.05, 0.05, 0.9],
                "issue_targets": [issue_target] * len(ISSUE_LABELS),
                "issue_valid": [True] * len(ISSUE_LABELS),
                "issue_probabilities": [0.9 if issue_target else 0.1] * len(ISSUE_LABELS),
                "boundary_reference_s": [None, None],
                "boundary_prediction_s": [None, None],
                "boundary_reference_valid": [False, False],
                "boundary_prediction_valid": [False, False],
            }
        )
    return {
        "schema": "egosieve.test-predictions/v1",
        "checkpoint_sha256": checkpoint_hash,
        "splits_sha256": splits_hash,
        "readiness_labels": list(READINESS_LABELS),
        "issue_labels": list(ISSUE_LABELS),
        "boundary_labels": ["start", "end"],
        "examples": examples,
    }


def _metrics_document(checkpoint_hash: str, splits_hash: str, predictions_hash: str) -> dict:
    return {
        "schema": "egosieve.release-metrics/v1",
        "readiness": {
            "macro_f1": 1.0,
            "per_class": {
                label: {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2}
                for label in READINESS_LABELS
            },
            "confusion_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        },
        "issues": {
            "macro_auroc": 1.0,
            "macro_average_precision": 1.0,
            "per_issue": {
                label: {
                    "auroc": 1.0,
                    "average_precision": 1.0,
                    "positives": 2 if index < 3 else 1,
                    "negatives": 2 if index < 3 else 1,
                }
                for index, label in enumerate(ISSUE_LABELS)
            },
        },
        "boundaries": {
            "f1": 1.0,
            "tolerance_s": 0.25,
            "per_boundary": {
                label: {
                    "f1": 1.0,
                    "true_positives": 6,
                    "false_positives": 0,
                    "false_negatives": 0,
                }
                for label in ("start", "end")
            },
        },
        "calibration": {
            "ece": 0.1,
            "n_bins": 10,
            "selective_risk": [{"coverage": 1.0, "risk": 0.0, "threshold": 0.9}],
        },
        "throughput": {
            "cpu_windows_per_second": 1.2,
            "gpu_windows_per_second": 20.0,
        },
        "evaluation": {
            "readiness_human_grounded": True,
            "issues_controlled_corruptions": True,
            "grouped_split": True,
            "test_examples": 8,
            "readiness_examples": 6,
            "readiness_examples_by_provenance": {
                "human": 5,
                "human-derived": 1,
            },
            "issue_examples": 4,
            "boundary_examples": 6,
            "issue_examples_by_provenance": {
                "human": 1,
                "human-derived": 1,
                "programmatic-controlled-corruption": 2,
            },
            "annotation_guide": "docs/ANNOTATION.md",
            "per_source": {
                source: {"readiness_examples": 3, "readiness_macro_f1": 1.0}
                for source in ("source-0", "source-1")
            },
        },
        "data": {"datasets": [{"name": "fixture", "license": "apache-2.0"}]},
        "evidence": {
            "checkpoint_sha256": checkpoint_hash,
            "splits_sha256": splits_hash,
            "test_predictions_sha256": predictions_hash,
        },
    }


def _refresh_evidence(root) -> None:
    evidence = {
        "schema": RELEASE_EVIDENCE_SCHEMA,
        "artifacts": {name: _sha256(root / name) for name in HASHED_RELEASE_FILES},
    }
    _write_json(root / "evidence.json", evidence)


def _refresh_linkages(root) -> None:
    checkpoint_hash = _sha256(root / "model.safetensors")
    splits_hash = _sha256(root / "splits.json")

    report = json.loads((root / "training_report.json").read_text())
    report["checkpoint_sha256"] = checkpoint_hash
    report["splits_sha256"] = splits_hash
    _write_json(root / "training_report.json", report)

    predictions = json.loads((root / "test_predictions.json").read_text())
    predictions["checkpoint_sha256"] = checkpoint_hash
    predictions["splits_sha256"] = splits_hash
    _write_json(root / "test_predictions.json", predictions)

    metrics = json.loads((root / "metrics.json").read_text())
    metrics["evidence"] = {
        "checkpoint_sha256": checkpoint_hash,
        "splits_sha256": splits_hash,
        "test_predictions_sha256": _sha256(root / "test_predictions.json"),
    }
    _write_json(root / "metrics.json", metrics)

    _refresh_evidence(root)


def _artifact(tmp_path):
    model = _tiny_trained_model()
    model.config.backbone_model_id = "facebook/dinov2-small"
    model.config.backbone_revision = "a" * 40
    model.config.vision_config._name_or_path = "facebook/dinov2-small"
    save_training_seed(model, tmp_path, processor=EgoSieveProcessor(size=16, num_frames=4))
    (tmp_path / "UNTRAINED_HEADS").unlink()
    (tmp_path / "README.md").write_text(
        "---\nlicense: apache-2.0\nlibrary_name: transformers\n"
        "pipeline_tag: video-classification\n---\n# model\n",
        encoding="utf-8",
    )

    _write_json(tmp_path / "splits.json", _split_document())
    checkpoint_hash = _sha256(tmp_path / "model.safetensors")
    splits_hash = _sha256(tmp_path / "splits.json")
    _write_json(
        tmp_path / "training_report.json",
        {
            "schema": "egosieve.training-report/v1",
            "completed": True,
            "run_id": "fixture-run",
            "source_commit": "b" * 40,
            "backbone": {
                "model_id": "facebook/dinov2-small",
                "revision": "a" * 40,
            },
            "optimizer_steps": 1,
            "split_counts": {"train": 2, "validation": 2, "test": 8},
            "checkpoint_sha256": checkpoint_hash,
            "splits_sha256": splits_hash,
            "retrieval_objective": {
                "name": "fixture-projection-alignment",
                "weight": 1.0,
                "positive_pairs": 3,
            },
        },
    )
    _write_json(
        tmp_path / "test_predictions.json",
        _prediction_document(checkpoint_hash, splits_hash),
    )
    _write_json(
        tmp_path / "metrics.json",
        _metrics_document(
            checkpoint_hash,
            splits_hash,
            _sha256(tmp_path / "test_predictions.json"),
        ),
    )
    _refresh_linkages(tmp_path)
    return tmp_path


def _declared_only_artifact(tmp_path):
    backbone = Dinov2Model(
        Dinov2Config(
            image_size=16,
            patch_size=8,
            hidden_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=48,
        )
    )
    model = initialize_from_backbone(backbone, num_frames=4, max_frames=4)
    save_training_seed(model, tmp_path, processor=EgoSieveProcessor(size=16, num_frames=4))
    (tmp_path / "UNTRAINED_HEADS").unlink()
    (tmp_path / "README.md").write_text(
        "---\nlicense: apache-2.0\nlibrary_name: transformers\n"
        "pipeline_tag: video-classification\n---\n# declared only\n",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "metrics.json",
        {"evaluation": {"readiness_human_grounded": True}},
    )
    return tmp_path


def test_valid_release_requires_self_consistent_training_and_evaluation_evidence(tmp_path) -> None:
    result = validate_release(_artifact(tmp_path))
    assert result["config"]["model_type"] == "egosieve"
    assert result["training_report"]["optimizer_steps"] == 1
    assert result["evidence"]["schema"] == RELEASE_EVIDENCE_SCHEMA


def test_random_model_and_declared_metrics_are_not_release_evidence(tmp_path) -> None:
    with pytest.raises(ReleaseValidationError, match="missing release files.*training_report"):
        validate_release(_declared_only_artifact(tmp_path))


def test_release_rejects_placeholder(tmp_path) -> None:
    root = _artifact(tmp_path)
    with (root / "README.md").open("a") as handle:
        handle.write("{{EVALUATION}}")
    with pytest.raises(ReleaseValidationError, match="template"):
        validate_release(root)


def test_release_rejects_blanket_human_review_provenance(tmp_path) -> None:
    root = _artifact(tmp_path)
    metrics = json.loads((root / "metrics.json").read_text())
    metrics["evaluation"]["human_reviewed"] = True
    _write_json(root / "metrics.json", metrics)
    _refresh_linkages(root)
    with pytest.raises(ReleaseValidationError, match="human_reviewed is too broad"):
        validate_release(root)


def test_release_allows_programmatic_issue_rows_without_review_fields(tmp_path) -> None:
    root = _artifact(tmp_path)
    result = validate_release(root)
    evaluation = result["metrics"]["evaluation"]
    assert evaluation["test_examples"] == 8
    assert evaluation["readiness_examples"] == 6
    assert evaluation["issue_examples"] == 4
    assert evaluation["issue_examples_by_provenance"] == {
        "human": 1,
        "human-derived": 1,
        "programmatic-controlled-corruption": 2,
    }
    assert set(evaluation["per_source"]) == {"source-0", "source-1"}


def test_release_rejects_target_when_readiness_is_invalid(tmp_path) -> None:
    root = _artifact(tmp_path)
    predictions = json.loads((root / "test_predictions.json").read_text())
    predictions["examples"][6]["readiness_target"] = "KEEP"
    _write_json(root / "test_predictions.json", predictions)
    _refresh_linkages(root)
    with pytest.raises(ReleaseValidationError, match="must be null when readiness_valid is false"):
        validate_release(root)


def test_release_requires_two_reviews_for_human_readiness_only(tmp_path) -> None:
    root = _artifact(tmp_path)
    predictions = json.loads((root / "test_predictions.json").read_text())
    predictions["examples"][0]["review_count"] = 1
    _write_json(root / "test_predictions.json", predictions)
    _refresh_linkages(root)
    with pytest.raises(ReleaseValidationError, match="review_count must be at least 2"):
        validate_release(root)


def test_release_requires_every_readiness_class_among_valid_rows(tmp_path) -> None:
    root = _artifact(tmp_path)
    predictions = json.loads((root / "test_predictions.json").read_text())
    for row in predictions["examples"]:
        if row["readiness_target"] == "REJECT":
            row["readiness_valid"] = False
            row["readiness_target"] = None
            row["label_provenance"]["readiness"]["kind"] = "unlabeled"
    _write_json(root / "test_predictions.json", predictions)
    _refresh_linkages(root)
    with pytest.raises(ReleaseValidationError, match="no valid `REJECT` readiness target"):
        validate_release(root)


def test_release_ignores_unlabeled_rows_for_readiness_metrics(tmp_path) -> None:
    root = _artifact(tmp_path)
    predictions = json.loads((root / "test_predictions.json").read_text())
    predictions["examples"][6]["readiness_probabilities"] = [1.0, 0.0, 0.0]
    predictions["examples"][7]["readiness_probabilities"] = [0.0, 1.0, 0.0]
    _write_json(root / "test_predictions.json", predictions)
    _refresh_linkages(root)
    validate_release(root)


def test_release_requires_some_controlled_corruption_issue_provenance(tmp_path) -> None:
    root = _artifact(tmp_path)
    predictions = json.loads((root / "test_predictions.json").read_text())
    predictions["examples"][6]["label_provenance"]["issues"]["kind"] = "human"
    predictions["examples"][7]["label_provenance"]["issues"]["kind"] = "human-derived"
    _write_json(root / "test_predictions.json", predictions)
    _refresh_linkages(root)
    with pytest.raises(ReleaseValidationError, match="at least one issue-valid row"):
        validate_release(root)


def test_release_recomputes_readiness_provenance_counts(tmp_path) -> None:
    root = _artifact(tmp_path)
    metrics = json.loads((root / "metrics.json").read_text())
    metrics["evaluation"]["readiness_examples_by_provenance"] = {
        "human": 4,
        "human-derived": 2,
    }
    _write_json(root / "metrics.json", metrics)
    _refresh_linkages(root)
    with pytest.raises(
        ReleaseValidationError,
        match=r"readiness_examples_by_provenance\.human.*does not match",
    ):
        validate_release(root)


def test_release_rejects_untrained_marker(tmp_path) -> None:
    root = _artifact(tmp_path)
    (root / "UNTRAINED_HEADS").write_text("not trained")
    with pytest.raises(ReleaseValidationError, match="non-release"):
        validate_release(root)


def test_release_rejects_empty_weights_before_loading(tmp_path) -> None:
    root = _artifact(tmp_path)
    (root / "model.safetensors").write_bytes(b"")
    with pytest.raises(ReleaseValidationError, match="cannot be empty"):
        validate_release(root)


def test_release_requires_custom_code(tmp_path) -> None:
    root = _artifact(tmp_path)
    (root / "modeling_egosieve.py").unlink()
    with pytest.raises(ReleaseValidationError, match="custom model code"):
        validate_release(root)


def test_release_rejects_artifact_hash_tampering(tmp_path) -> None:
    root = _artifact(tmp_path)
    with (root / "metrics.json").open("a") as handle:
        handle.write(" \n")
    with pytest.raises(ReleaseValidationError, match="hash mismatch.*metrics.json"):
        validate_release(root)


def test_release_rejects_broken_cross_hash_with_valid_outer_manifest(tmp_path) -> None:
    root = _artifact(tmp_path)
    report = json.loads((root / "training_report.json").read_text())
    report["splits_sha256"] = "0" * 64
    _write_json(root / "training_report.json", report)
    _refresh_evidence(root)
    with pytest.raises(ReleaseValidationError, match="training_report.splits_sha256"):
        validate_release(root)


def test_release_rejects_unknown_versioned_schema(tmp_path) -> None:
    root = _artifact(tmp_path)
    splits = json.loads((root / "splits.json").read_text())
    splits["schema"] = "egosieve.splits/v999"
    _write_json(root / "splits.json", splits)
    _refresh_linkages(root)
    with pytest.raises(ReleaseValidationError, match="splits.json schema"):
        validate_release(root)


def test_release_rejects_group_leakage_even_with_refreshed_hashes(tmp_path) -> None:
    root = _artifact(tmp_path)
    splits = json.loads((root / "splits.json").read_text())
    test_row = next(row for row in splits["examples"] if row["split"] == "test")
    test_row["group_id"] = "train-group"
    _write_json(root / "splits.json", splits)
    _refresh_linkages(root)
    with pytest.raises(ReleaseValidationError, match="leaks group.*train-group"):
        validate_release(root)


def test_release_recomputes_metrics_from_raw_predictions(tmp_path) -> None:
    root = _artifact(tmp_path)
    predictions = json.loads((root / "test_predictions.json").read_text())
    predictions["examples"][0]["readiness_probabilities"] = [0.05, 0.9, 0.05]
    _write_json(root / "test_predictions.json", predictions)
    _refresh_linkages(root)
    with pytest.raises(ReleaseValidationError, match="does not match test_predictions"):
        validate_release(root)


def test_release_requires_retrieval_training_evidence(tmp_path) -> None:
    root = _artifact(tmp_path)
    report = json.loads((root / "training_report.json").read_text())
    report["retrieval_objective"]["weight"] = 0
    _write_json(root / "training_report.json", report)
    _refresh_linkages(root)
    with pytest.raises(ReleaseValidationError, match="retrieval_objective.weight must be positive"):
        validate_release(root)


@pytest.mark.parametrize(
    ("path", "match"),
    [
        (("source_commit",), "source_commit"),
        (("backbone", "revision"), "backbone.revision"),
    ],
)
def test_release_requires_source_and_backbone_revisions(tmp_path, path, match) -> None:
    root = _artifact(tmp_path)
    report = json.loads((root / "training_report.json").read_text())
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = ""
    _write_json(root / "training_report.json", report)
    _refresh_linkages(root)
    with pytest.raises(ReleaseValidationError, match=match):
        validate_release(root)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("backbone_model_id", "other/model", "backbone_model_id does not match"),
        ("backbone_revision", "c" * 40, "backbone_revision does not match"),
    ],
)
def test_release_links_config_to_training_backbone(tmp_path, field, value, match) -> None:
    root = _artifact(tmp_path)
    config = json.loads((root / "config.json").read_text())
    config[field] = value
    _write_json(root / "config.json", config)
    _refresh_evidence(root)
    with pytest.raises(ReleaseValidationError, match=match):
        validate_release(root)


def test_release_rejects_machine_local_paths(tmp_path) -> None:
    root = _artifact(tmp_path)
    config = json.loads((root / "config.json").read_text())
    config["vision_config"]["_name_or_path"] = "/private/model-cache/snapshot"
    _write_json(root / "config.json", config)
    _refresh_evidence(root)
    with pytest.raises(ReleaseValidationError, match="machine-local absolute path"):
        validate_release(root)
