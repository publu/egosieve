"""Reproducible feature-cached training and held-out evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ..initialization import register_for_hub
from ..modeling import EgoSieveModel
from ..processing_egosieve import EgoSieveProcessor
from .calibration import (
    fit_issue_thresholds,
    fit_readiness_temperature,
    fit_routing_thresholds,
)
from .data import BOUNDARY_LABELS, ISSUE_LABELS, READINESS_LABELS, load_jsonl
from .features import FeatureCache, WindowExample, artifact_fingerprint, expand_training_windows
from .metrics import (
    expected_calibration_error,
    issue_metrics,
    readiness_metrics,
    selective_risk_curve,
    temporal_boundary_metrics,
)
from .splits import group_assignments
from .targets import TrainingCollator

RELEASE_PROVENANCE_KINDS = (
    "human",
    "human-derived",
    "programmatic-controlled-corruption",
    "unlabeled",
)
HUMAN_GROUNDED_PROVENANCE = frozenset({"human", "human-derived"})
ISSUE_EVIDENCE_PROVENANCE = (
    "human",
    "human-derived",
    "programmatic-controlled-corruption",
)


@dataclass(frozen=True)
class TrainingRunConfig:
    """Configuration persisted verbatim with every training run."""

    seed: int = 17
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    epochs: int = 12
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    max_grad_norm: float = 1.0
    boundary_tolerance_s: float = 0.3
    boundary_threshold: float = 0.5
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.1
    patience: int = 4
    device: str = "auto"
    ffmpeg_bin: str = "ffmpeg"
    decode_timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.patience <= 0:
            raise ValueError("epochs, batch_size, and patience must be positive")
        for name in ("learning_rate", "max_grad_norm", "boundary_tolerance_s"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.weight_decay < 0 or not 0 <= self.warmup_fraction < 1:
            raise ValueError("weight_decay must be non-negative and warmup_fraction in [0, 1)")
        if not 0 < self.boundary_threshold < 1:
            raise ValueError("boundary_threshold must lie in (0, 1)")
        if self.contrastive_weight < 0 or self.contrastive_temperature <= 0:
            raise ValueError("contrastive_weight must be non-negative and temperature positive")
        fractions = self.train_fraction + self.validation_fraction + self.test_fraction
        if not math.isclose(fractions, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("train/validation/test fractions must sum to 1")


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _collate(
    examples: Sequence[WindowExample],
    feature_cache: FeatureCache,
    collator: TrainingCollator,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    rows = [
        {
            "window": example.window,
            "sampled_timestamps_s": example.timestamps_s,
            "frame_embeddings": feature_cache.load(example),
        }
        for example in examples
    ]
    arrays = collator(rows)
    result: dict[str, torch.Tensor] = {}
    for name, array in arrays.items():
        tensor = torch.from_numpy(np.asarray(array))
        result[name] = tensor.to(device)
    return result


def _contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    temperature: float,
    group_ids: Sequence[str] | None = None,
) -> torch.Tensor:
    """Symmetric instance discrimination without same-group false negatives.

    The diagonal remains the positive pair. Off-diagonal examples sharing a
    source group are removed from both softmax denominators because nearby
    windows or derived variants from one capture are not defensible negatives.
    When no group ids are supplied, every off-diagonal item remains a negative
    for backward compatibility.
    """

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("contrastive views must have matching [batch, embedding] shapes")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("contrastive temperature must be finite and positive")
    groups: tuple[str, ...] | None = None
    if group_ids is not None:
        if isinstance(group_ids, (str, bytes)):
            raise ValueError("group_ids must be a sequence with one id per batch item")
        groups = tuple(group_ids)
        if len(groups) != first.shape[0]:
            raise ValueError("group_ids must contain one id per batch item")
        if any(not isinstance(group_id, str) or not group_id for group_id in groups):
            raise ValueError("group_ids must contain non-empty strings")
    if first.shape[0] < 2:
        return (first.sum() + second.sum()) * 0.0
    logits = first @ second.transpose(0, 1) / temperature
    if groups is not None:
        valid = torch.tensor(
            [
                [
                    row == column or row_group != column_group
                    for column, column_group in enumerate(groups)
                ]
                for row, row_group in enumerate(groups)
            ],
            dtype=torch.bool,
            device=logits.device,
        )
        logits = logits.masked_fill(~valid, -torch.inf)
    labels = torch.arange(first.shape[0], device=first.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _linear_warmup_decay(step: int, total_steps: int, warmup_fraction: float) -> float:
    warmup_steps = max(1, round(total_steps * warmup_fraction))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    remaining = max(total_steps - warmup_steps, 1)
    return max(0.0, (total_steps - step) / remaining)


def _readiness_score(report: Mapping[str, Any]) -> float:
    value = report.get("readiness", {}).get("macro_f1")
    return -1.0 if value is None else float(value)


def _boundary_reference(examples: Sequence[WindowExample]) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((len(examples), len(BOUNDARY_LABELS)), np.nan, dtype=np.float64)
    valid = np.zeros_like(values, dtype=bool)
    for row, example in enumerate(examples):
        window = example.window
        for column, name in enumerate(BOUNDARY_LABELS):
            timestamp = window.boundaries_s.get(name)
            declared = (
                window.boundary_valid
                if isinstance(window.boundary_valid, bool)
                else window.boundary_valid.get(name, False)
            )
            if declared and timestamp is not None:
                values[row, column] = timestamp
                valid[row, column] = True
    return values, valid


def _source_readiness(
    examples: Sequence[WindowExample],
    labels: np.ndarray,
    probabilities: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    sources: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        if bool(valid[index]):
            sources[example.source].append(index)
    result: dict[str, Any] = {}
    for source, indices in sorted(sources.items()):
        selected = np.asarray(indices, dtype=np.int64)
        metrics = readiness_metrics(
            labels[selected],
            probabilities[selected],
        )
        result[source] = {
            "readiness_examples": len(indices),
            "readiness_macro_f1": metrics["macro_f1"],
        }
    return result


def evaluate_examples(
    model: EgoSieveModel,
    examples: Sequence[WindowExample],
    *,
    feature_cache: FeatureCache,
    config: TrainingRunConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate every task head on a fixed ordered example sequence."""

    if not examples:
        raise ValueError("evaluation split is empty")
    collator = TrainingCollator(boundary_tolerance_s=config.boundary_tolerance_s)
    model.eval()
    readiness_logits: list[np.ndarray] = []
    issue_logits: list[np.ndarray] = []
    boundary_logits: list[np.ndarray] = []
    readiness_labels: list[np.ndarray] = []
    readiness_masks: list[np.ndarray] = []
    issue_labels: list[np.ndarray] = []
    issue_masks: list[np.ndarray] = []
    losses: list[float] = []
    with torch.inference_mode():
        for offset in range(0, len(examples), config.batch_size):
            batch_examples = examples[offset : offset + config.batch_size]
            batch = _collate(batch_examples, feature_cache, collator, device)
            output = model(**batch)
            losses.append(float(output.loss.detach().cpu()))
            readiness_logits.append(output.logits.detach().cpu().float().numpy())
            issue_logits.append(output.issue_logits.detach().cpu().float().numpy())
            boundary_logits.append(output.boundary_logits.detach().cpu().float().numpy())
            readiness_labels.append(batch["readiness_labels"].cpu().numpy())
            readiness_masks.append(batch["readiness_label_mask"].cpu().numpy())
            issue_labels.append(batch["issue_labels"].cpu().numpy())
            issue_masks.append(batch["issue_label_mask"].cpu().numpy())

    ready_raw = np.concatenate(readiness_logits)
    issue_raw = np.concatenate(issue_logits)
    boundary_raw = np.concatenate(boundary_logits)
    ready_true = np.concatenate(readiness_labels)
    ready_valid = np.concatenate(readiness_masks).astype(bool)
    issue_true = np.concatenate(issue_labels)
    issue_valid = np.concatenate(issue_masks).astype(bool)
    temperature = float(getattr(model.config, "readiness_temperature", 1.0))
    ready_prob = torch.softmax(torch.from_numpy(ready_raw) / temperature, dim=-1).numpy()
    issue_prob = torch.sigmoid(torch.from_numpy(issue_raw)).numpy()

    reference_times, reference_valid = _boundary_reference(examples)
    predicted_times = np.full_like(reference_times, np.nan)
    predicted_valid = np.zeros_like(reference_valid)
    boundary_prob = torch.sigmoid(torch.from_numpy(boundary_raw)).numpy()
    for row, example in enumerate(examples):
        timestamps = np.asarray(example.timestamps_s)
        for column in range(len(BOUNDARY_LABELS)):
            best = int(np.argmax(boundary_prob[row, :, column]))
            if boundary_prob[row, best, column] >= config.boundary_threshold:
                predicted_times[row, column] = timestamps[best]
                predicted_valid[row, column] = reference_valid[row, column]

    readiness = readiness_metrics(ready_true, ready_prob, valid_mask=ready_valid)
    issues = issue_metrics(
        issue_true,
        issue_prob,
        valid_mask=issue_valid,
        issue_names=ISSUE_LABELS,
    )
    boundaries = temporal_boundary_metrics(
        reference_times,
        predicted_times,
        tolerance_s=config.boundary_tolerance_s,
        reference_mask=reference_valid,
        prediction_mask=predicted_valid,
    )
    calibration = {
        "ece": expected_calibration_error(ready_true, ready_prob, valid_mask=ready_valid),
        "selective_risk": selective_risk_curve(
            ready_true,
            ready_prob,
            valid_mask=ready_valid,
        ),
    }
    return _jsonable(
        {
            "loss": float(np.mean(losses)),
            "readiness": readiness,
            "issues": issues,
            "boundaries": boundaries,
            "calibration": calibration,
            "per_source": _source_readiness(
                examples,
                ready_true,
                ready_prob,
                ready_valid,
            ),
            "predictions": {
                "example_keys": [example.key for example in examples],
                "readiness_logits": ready_raw,
                "readiness_labels": ready_true,
                "readiness_valid": ready_valid,
                "readiness_probabilities": ready_prob,
                "issue_labels": issue_true,
                "issue_logits": issue_raw,
                "issue_valid": issue_valid,
                "issue_probabilities": issue_prob,
                "boundary_reference_s": reference_times,
                "boundary_reference_valid": reference_valid,
                "boundary_prediction_s": predicted_times,
                "boundary_prediction_valid": predicted_valid,
                "boundary_probabilities": boundary_prob,
            },
        }
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _benchmark_once(
    model: EgoSieveModel,
    device: torch.device,
    *,
    iterations: int = 3,
) -> float:
    model.to(device).eval()
    size = int(model.config.vision_config.image_size)
    frames = int(model.config.num_frames)
    pixels = torch.zeros((1, frames, 3, size, size), dtype=torch.float32, device=device)
    mask = torch.ones((1, frames), dtype=torch.bool, device=device)
    with torch.inference_mode():
        model(pixel_values=pixels, frame_mask=mask)
        _sync(device)
        started = time.perf_counter()
        for _ in range(iterations):
            model(pixel_values=pixels, frame_mask=mask)
        _sync(device)
    elapsed = time.perf_counter() - started
    return iterations / elapsed


def benchmark_model(model: EgoSieveModel, preferred_device: torch.device) -> dict[str, Any]:
    """Measure preprocessed-window forward throughput on CPU and one accelerator."""

    cpu = _benchmark_once(model, torch.device("cpu"))
    if preferred_device.type == "cpu":
        accelerator = None
        accelerator_rate = 0.0
    else:
        accelerator = preferred_device.type
        accelerator_rate = _benchmark_once(model, preferred_device)
    model.to(preferred_device)
    return {
        "scope": "model-forward-on-preprocessed-windows",
        "batch_size": 1,
        "iterations": 3,
        "cpu_windows_per_second": cpu,
        "gpu_windows_per_second": accelerator_rate,
        "accelerator": accelerator,
    }


def _metric_or_zero(value: Any) -> float:
    return 0.0 if value is None else float(value)


def _release_label_provenance(
    example: WindowExample,
    *,
    readiness_valid: bool,
    issue_valid: Sequence[bool],
    boundary_valid: Sequence[bool],
) -> dict[str, dict[str, str]]:
    """Normalize an explicitly task-scoped held-out provenance declaration."""

    raw = example.window.extra.get("label_provenance")
    location = f"test example {example.key} label_provenance"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{location} must be a task-level object")

    result: dict[str, dict[str, str]] = {}
    for task in ("readiness", "issues", "boundaries"):
        task_value = raw.get(task)
        if not isinstance(task_value, Mapping):
            raise ValueError(
                f"{location}.{task} must be an object; flat provenance kinds are not "
                "release evidence"
            )
        kind = task_value.get("kind")
        if not isinstance(kind, str) or kind not in RELEASE_PROVENANCE_KINDS:
            raise ValueError(f"{location}.{task}.kind must be one of {RELEASE_PROVENANCE_KINDS}")
        result[task] = {"kind": kind}

    readiness_kind = result["readiness"]["kind"]
    if readiness_valid and readiness_kind not in HUMAN_GROUNDED_PROVENANCE:
        raise ValueError(f"{location}.readiness must be human or human-derived for a valid target")
    if not readiness_valid and readiness_kind == "programmatic-controlled-corruption":
        raise ValueError(
            f"{location}.readiness cannot claim a controlled corruption without a target"
        )

    has_issue_target = any(issue_valid)
    issue_kind = result["issues"]["kind"]
    if has_issue_target and issue_kind == "unlabeled":
        raise ValueError(f"{location}.issues cannot be unlabeled when issue targets are valid")
    if not has_issue_target and issue_kind != "unlabeled":
        raise ValueError(f"{location}.issues must be unlabeled when no issue target is valid")

    has_boundary_target = any(boundary_valid)
    boundary_kind = result["boundaries"]["kind"]
    if has_boundary_target and boundary_kind not in HUMAN_GROUNDED_PROVENANCE:
        raise ValueError(f"{location}.boundaries must be human or human-derived for valid targets")
    if not has_boundary_target and boundary_kind != "unlabeled":
        raise ValueError(
            f"{location}.boundaries must be unlabeled when no boundary target is valid"
        )
    return result


def _release_evaluation_provenance(
    predictions: Mapping[str, Any],
    examples: Sequence[WindowExample],
) -> dict[str, Any]:
    readiness_count = 0
    issue_count = 0
    boundary_count = 0
    issue_counts = {kind: 0 for kind in ISSUE_EVIDENCE_PROVENANCE}
    readiness_human_grounded = True

    for index, example in enumerate(examples):
        readiness_valid = bool(predictions["readiness_valid"][index])
        issue_valid = [bool(value) for value in predictions["issue_valid"][index]]
        boundary_valid = [bool(value) for value in predictions["boundary_reference_valid"][index]]
        provenance = _release_label_provenance(
            example,
            readiness_valid=readiness_valid,
            issue_valid=issue_valid,
            boundary_valid=boundary_valid,
        )
        if readiness_valid:
            readiness_count += 1
            readiness_human_grounded &= provenance["readiness"]["kind"] in HUMAN_GROUNDED_PROVENANCE
        if any(issue_valid):
            issue_count += 1
            issue_counts[provenance["issues"]["kind"]] += 1
        if any(boundary_valid):
            boundary_count += 1

    controlled_count = issue_counts["programmatic-controlled-corruption"]
    if readiness_count == 0 or not readiness_human_grounded:
        raise ValueError("held-out readiness evidence must contain human-grounded targets")
    if controlled_count == 0:
        raise ValueError(
            "held-out issue evidence needs at least one programmatic controlled-corruption row"
        )
    return {
        "readiness_human_grounded": True,
        "issues_controlled_corruptions": True,
        "test_examples": len(examples),
        "readiness_examples": readiness_count,
        "issue_examples": issue_count,
        "boundary_examples": boundary_count,
        "issue_examples_by_provenance": issue_counts,
    }


def _release_metrics(
    report: Mapping[str, Any],
    predictions: Mapping[str, Any],
    test_examples: Sequence[WindowExample],
    all_examples: Sequence[WindowExample],
    throughput: Mapping[str, Any],
    *,
    annotation_guide: str,
) -> dict[str, Any]:
    ready = report["readiness"]
    issue = report["issues"]
    boundary = report["boundaries"]
    calibration = report["calibration"]
    per_class = {}
    for label in READINESS_LABELS:
        row = ready["per_class"][label]
        per_class[label] = {
            "precision": _metric_or_zero(row["precision"]),
            "recall": _metric_or_zero(row["recall"]),
            "f1": _metric_or_zero(row["f1"]),
            "support": int(row["support"]),
        }
    curve = calibration["selective_risk"]
    selective = [
        {
            "coverage": float(coverage),
            "risk": float(risk),
            "threshold": float(threshold),
        }
        for coverage, risk, threshold in zip(
            curve["coverage"], curve["risk"], curve["threshold"], strict=True
        )
    ]
    evaluation_provenance = _release_evaluation_provenance(predictions, test_examples)
    datasets: dict[tuple[str, str], int] = defaultdict(int)
    for example in all_examples:
        datasets[(example.source, example.license)] += 1
    return {
        "schema": "egosieve.release-metrics/v1",
        "readiness": {
            "macro_f1": _metric_or_zero(ready["macro_f1"]),
            "accuracy": _metric_or_zero(ready["accuracy"]),
            "per_class": per_class,
            "confusion_matrix": ready["confusion_matrix"],
        },
        "issues": {
            "macro_auroc": _metric_or_zero(issue["macro_auroc"]),
            "macro_average_precision": _metric_or_zero(issue["macro_average_precision"]),
            "per_issue": issue["per_issue"],
        },
        "boundaries": {
            "f1": _metric_or_zero(boundary["micro_f1"]),
            "macro_f1": _metric_or_zero(boundary["macro_f1"]),
            "tolerance_s": float(boundary["tolerance_s"]),
            "per_boundary": boundary["per_boundary"],
        },
        "calibration": {
            "ece": _metric_or_zero(calibration["ece"]),
            "n_bins": 15,
            "selective_risk": selective,
        },
        "throughput": dict(throughput),
        "evaluation": {
            **evaluation_provenance,
            "grouped_split": True,
            "annotation_guide": annotation_guide,
            "per_source": report["per_source"],
        },
        "data": {
            "datasets": [
                {"name": source, "license": license_name, "training_windows": count}
                for (source, license_name), count in sorted(datasets.items())
            ]
        },
    }


def _split_document(
    split_examples: Mapping[str, Sequence[WindowExample]],
    *,
    seed: int,
) -> dict[str, Any]:
    rows = []
    for split in ("train", "validation", "test"):
        rows.extend(
            {
                "id": example.key,
                "group_id": example.group_id,
                "source": example.source,
                "split": split,
            }
            for example in split_examples[split]
        )
    return {"schema": "egosieve.splits/v1", "seed": seed, "examples": rows}


def _test_prediction_document(
    predictions: Mapping[str, Any],
    examples: Sequence[WindowExample],
    *,
    checkpoint_sha256: str,
    splits_sha256: str,
) -> dict[str, Any]:
    rows = []
    for index, example in enumerate(examples):
        readiness_valid = bool(predictions["readiness_valid"][index])
        readiness_id = int(predictions["readiness_labels"][index])
        if readiness_valid and not 0 <= readiness_id < len(READINESS_LABELS):
            raise ValueError(f"test example {example.key} has an invalid readiness target")
        issue_valid = [bool(value) for value in predictions["issue_valid"][index]]
        issue_targets = [
            int(predictions["issue_labels"][index][column]) if valid else None
            for column, valid in enumerate(issue_valid)
        ]
        reference_valid = [bool(value) for value in predictions["boundary_reference_valid"][index]]
        prediction_valid = [
            bool(value) for value in predictions["boundary_prediction_valid"][index]
        ]
        provenance = _release_label_provenance(
            example,
            readiness_valid=readiness_valid,
            issue_valid=issue_valid,
            boundary_valid=reference_valid,
        )
        row = {
            "id": example.key,
            "group_id": example.group_id,
            "source": example.source,
            "label_provenance": provenance,
            "readiness_valid": readiness_valid,
            "readiness_target": READINESS_LABELS[readiness_id] if readiness_valid else None,
            "readiness_probabilities": predictions["readiness_probabilities"][index],
            "issue_targets": issue_targets,
            "issue_valid": issue_valid,
            "issue_probabilities": predictions["issue_probabilities"][index],
            "boundary_reference_s": predictions["boundary_reference_s"][index],
            "boundary_reference_valid": reference_valid,
            "boundary_prediction_s": predictions["boundary_prediction_s"][index],
            "boundary_prediction_valid": prediction_valid,
        }
        if provenance["readiness"]["kind"] in HUMAN_GROUNDED_PROVENANCE:
            review_count = example.window.extra.get("review_count")
            rubric_version = example.window.extra.get("rubric_version")
            if (
                isinstance(review_count, bool)
                or not isinstance(review_count, int)
                or review_count < 2
            ):
                raise ValueError(
                    f"test example {example.key} human-grounded readiness requires "
                    "review_count >= 2"
                )
            if not isinstance(rubric_version, str) or not rubric_version.strip():
                raise ValueError(
                    f"test example {example.key} human-grounded readiness requires a rubric_version"
                )
            row["review_count"] = review_count
            row["rubric_version"] = rubric_version.strip()
        rows.append(row)
    return {
        "schema": "egosieve.test-predictions/v1",
        "checkpoint_sha256": checkpoint_sha256,
        "splits_sha256": splits_sha256,
        "readiness_labels": list(READINESS_LABELS),
        "issue_labels": list(ISSUE_LABELS),
        "boundary_labels": list(BOUNDARY_LABELS),
        "examples": rows,
    }


def _render_model_card(
    metrics: Mapping[str, Any],
    *,
    model_id: str,
    model_revision: str,
    backbone: str,
) -> str:
    datasets = ", ".join(f"{row['name']} ({row['license']})" for row in metrics["data"]["datasets"])
    evaluation = metrics["evaluation"]
    issue_provenance = evaluation["issue_examples_by_provenance"]
    return f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: video-classification
base_model: {backbone}
tags:
  - robotics
  - egocentric-video
  - video-quality
  - dataset-curation
  - physical-ai
---

# EgoSieve-S

EgoSieve-S ranks manipulation-ready spans in first-person video. It produces
three readiness logits (`KEEP`, `REVIEW`, `REJECT`), eight observable issue
scores, diagnostic start/end boundary proposals, and a normalized retrieval
embedding. It is a dataset-curation model, not a robot policy.

## Usage

```python
from transformers import AutoModelForVideoClassification, AutoProcessor

processor = AutoProcessor.from_pretrained(
    "{model_id}", revision="{model_revision}", trust_remote_code=True
)
model = AutoModelForVideoClassification.from_pretrained(
    "{model_id}", revision="{model_revision}", trust_remote_code=True
).eval()
outputs = model(**processor(frames, return_tensors="pt"))
```

The timestamp-aware scanner and JSONL compiler are provided by the `egosieve`
package. The checkpoint expects {int(metrics.get("input_frames", 12))} center-sampled
RGB frames per window; use its bundled processor.

## Training and evaluation

Data represented in the held-out evaluation: {datasets}. Splits are grouped by
original capture unit. Readiness, calibration, and boundary results use
{evaluation["readiness_examples"]} human-grounded readiness rows and
{evaluation["boundary_examples"]} human-grounded boundary rows. Issue results
use {evaluation["issue_examples"]} labeled rows: {issue_provenance["human"]}
human, {issue_provenance["human-derived"]} human-derived, and
{issue_provenance["programmatic-controlled-corruption"]} programmatic
controlled corruptions. Unlabeled task targets are masked. The release bundle
includes raw held-out predictions with task-level provenance, split
assignments, exact run configuration, and metric provenance.

- Readiness macro F1: {metrics["readiness"]["macro_f1"]:.4f}
- Issue macro AUROC: {metrics["issues"]["macro_auroc"]:.4f}
- Issue macro average precision: {metrics["issues"]["macro_average_precision"]:.4f}
- Boundary micro F1 at {metrics["boundaries"]["tolerance_s"]:.2f}s: {metrics["boundaries"]["f1"]:.4f}
- Readiness ECE: {metrics["calibration"]["ece"]:.4f}

## Intended use and limitations

Use the model to rank raw egocentric windows, route uncertain spans for review,
and create embeddings for near-duplicate search. Readiness remains dependent on
the published rubric and capture domain. RGB cannot establish force, physical
success, consent, safety, metric depth, or legal publishability. Boundary
scores are proposals and are diagnostic-only in the v0.1 compiler.

First-person recordings can contain faces, screens, homes, and bystanders.
Apply a separate privacy and consent review before sharing any media.
"""


def _ensure_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory must be absent or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def train_checkpoint(
    annotations: str | Path,
    *,
    seed_checkpoint: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    media_root: str | Path | None = None,
    allowed_licenses: Iterable[str] | None = None,
    config: TrainingRunConfig | None = None,
    model_id: str = "itspublu/EgoSieve-S",
    model_revision: str = "v0.1.0",
    backbone: str = "facebook/dinov2-small",
    backbone_revision: str | None = None,
    source_commit: str | None = None,
    annotation_guide: str = "docs/ANNOTATION.md (v0.1)",
) -> dict[str, Any]:
    """Train temporal heads on cached frozen-backbone features and build a candidate."""

    run = config or TrainingRunConfig()
    if not isinstance(backbone_revision, str) or not backbone_revision.strip():
        raise ValueError("an immutable backbone_revision is required")
    if not isinstance(source_commit, str) or not source_commit.strip():
        raise ValueError("the source_commit used for training is required")
    _seed_everything(run.seed)
    device = _device(run.device)
    annotation_path = Path(annotations).expanduser().resolve(strict=True)
    seed_root = Path(seed_checkpoint).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    _ensure_output_directory(destination)
    records = load_jsonl(annotation_path)
    if len({record.group_id for record in records}) < 3:
        raise ValueError("at least three independent groups are required for train/validation/test")

    model = EgoSieveModel.from_pretrained(seed_root, local_files_only=True)
    processor = EgoSieveProcessor.from_pretrained(seed_root, local_files_only=True)
    if processor.num_frames != model.config.num_frames:
        raise ValueError("seed model and processor disagree about num_frames")
    fingerprint = artifact_fingerprint(seed_root)
    root = annotation_path.parent if media_root is None else Path(media_root)
    examples = expand_training_windows(
        records,
        media_root=root,
        num_frames=model.config.num_frames,
        allowed_licenses=allowed_licenses,
    )
    assignments = group_assignments(
        records,
        train_fraction=run.train_fraction,
        validation_fraction=run.validation_fraction,
        test_fraction=run.test_fraction,
        seed=run.seed,
    )
    split_examples: dict[str, list[WindowExample]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for example in examples:
        split_examples[assignments[example.group_id]].append(example)
    if any(not values for values in split_examples.values()):
        raise ValueError("every split must contain at least one labeled window")
    if run.contrastive_weight > 0 and len(split_examples["train"]) < 2:
        raise ValueError("contrastive retrieval training requires at least two train windows")

    feature_cache = FeatureCache(
        cache_dir,
        artifact_sha256=fingerprint,
        processor=processor,
        vision_model=model.vision_model,
        device=device,
        ffmpeg_bin=run.ffmpeg_bin,
        timeout_s=run.decode_timeout_s,
    )
    cache_report = feature_cache.prepare(examples)

    for parameter in model.vision_model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=run.learning_rate,
        weight_decay=run.weight_decay,
    )
    steps_per_epoch = math.ceil(len(split_examples["train"]) / run.batch_size)
    total_steps = steps_per_epoch * run.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _linear_warmup_decay(step, total_steps, run.warmup_fraction),
    )
    collator = TrainingCollator(boundary_tolerance_s=run.boundary_tolerance_s)
    rng = np.random.default_rng(run.seed)
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    global_step = 0
    contrastive_positive_pairs = 0

    for epoch in range(1, run.epochs + 1):
        model.train()
        model.vision_model.eval()
        order = rng.permutation(len(split_examples["train"]))
        epoch_loss = 0.0
        epoch_supervised = 0.0
        epoch_contrastive = 0.0
        batches = 0
        for offset in range(0, len(order), run.batch_size):
            indices = order[offset : offset + run.batch_size]
            selected = [split_examples["train"][int(index)] for index in indices]
            if run.contrastive_weight > 0 and len(selected) == 1:
                partner = int(order[0])
                if partner == int(indices[0]):
                    partner = int(order[1])
                selected.append(split_examples["train"][partner])
            batch = _collate(selected, feature_cache, collator, device)
            optimizer.zero_grad(set_to_none=True)
            first = model(**batch)
            if first.loss is None:
                raise RuntimeError("the training batch contains no valid supervised targets")
            if run.contrastive_weight > 0 and len(selected) > 1:
                second = model(
                    frame_embeddings=batch["frame_embeddings"],
                    frame_mask=batch["frame_mask"],
                )
                contrastive = _contrastive_loss(
                    first.clip_embedding,
                    second.clip_embedding,
                    run.contrastive_temperature,
                    [example.group_id for example in selected],
                )
                contrastive_positive_pairs += len(selected)
            else:
                contrastive = first.loss * 0.0
            loss = first.loss + run.contrastive_weight * contrastive
            loss.backward()
            if run.contrastive_weight > 0 and model.clip_projection.weight.grad is None:
                raise RuntimeError("retrieval projection received no gradient")
            torch.nn.utils.clip_grad_norm_(trainable, run.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            batches += 1
            epoch_loss += float(loss.detach().cpu())
            epoch_supervised += float(first.loss.detach().cpu())
            epoch_contrastive += float(contrastive.detach().cpu())

        validation = evaluate_examples(
            model,
            split_examples["validation"],
            feature_cache=feature_cache,
            config=run,
            device=device,
        )
        score = _readiness_score(validation)
        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": epoch_loss / batches,
                "supervised_loss": epoch_supervised / batches,
                "contrastive_loss": epoch_contrastive / batches,
                "learning_rate": scheduler.get_last_lr()[0],
                "validation": {
                    key: value for key, value in validation.items() if key != "predictions"
                },
            }
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
                if not name.startswith("vision_model.")
            }
        else:
            stale_epochs += 1
            if stale_epochs >= run.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a selectable checkpoint")
    model.load_state_dict(best_state, strict=False)
    model.to(device).eval()
    calibration_report = evaluate_examples(
        model,
        split_examples["validation"],
        feature_cache=feature_cache,
        config=run,
        device=device,
    )
    calibration_predictions = calibration_report["predictions"]
    temperature = fit_readiness_temperature(
        calibration_predictions["readiness_logits"],
        calibration_predictions["readiness_labels"],
        calibration_predictions["readiness_valid"],
    )
    calibrated_probabilities = torch.softmax(
        torch.tensor(calibration_predictions["readiness_logits"]) / temperature,
        dim=-1,
    ).numpy()
    issue_thresholds = fit_issue_thresholds(
        calibration_predictions["issue_labels"],
        calibration_predictions["issue_probabilities"],
        calibration_predictions["issue_valid"],
    )
    routing_thresholds = fit_routing_thresholds(
        calibration_predictions["readiness_labels"],
        calibrated_probabilities,
        calibration_predictions["readiness_valid"],
    )
    model.config.readiness_temperature = temperature
    model.config.issue_thresholds = issue_thresholds
    model.config.compiler_thresholds = {
        **model.config.compiler_thresholds,
        **routing_thresholds,
    }
    model.config.calibration_source = "grouped-validation-split"
    test_report = evaluate_examples(
        model,
        split_examples["test"],
        feature_cache=feature_cache,
        config=run,
        device=device,
    )
    predictions = test_report.pop("predictions")
    throughput = benchmark_model(model, device)
    metrics = _release_metrics(
        test_report,
        predictions,
        split_examples["test"],
        examples,
        throughput,
        annotation_guide=annotation_guide,
    )
    metrics["input_frames"] = model.config.num_frames

    register_for_hub()
    model.save_pretrained(destination, safe_serialization=True)
    processor.save_pretrained(destination)
    (destination / "README.md").write_text(
        _render_model_card(
            metrics,
            model_id=model_id,
            model_revision=model_revision,
            backbone=backbone,
        ),
        encoding="utf-8",
    )
    split_manifest = _split_document(split_examples, seed=run.seed)
    _write_json(destination / "splits.json", split_manifest)
    checkpoint_sha256 = _file_sha256(destination / "model.safetensors")
    splits_sha256 = _file_sha256(destination / "splits.json")
    prediction_document = _test_prediction_document(
        predictions,
        split_examples["test"],
        checkpoint_sha256=checkpoint_sha256,
        splits_sha256=splits_sha256,
    )
    _write_json(destination / "test_predictions.json", prediction_document)
    predictions_sha256 = _file_sha256(destination / "test_predictions.json")
    metrics["evidence"] = {
        "checkpoint_sha256": checkpoint_sha256,
        "splits_sha256": splits_sha256,
        "test_predictions_sha256": predictions_sha256,
    }
    _write_json(destination / "metrics.json", metrics)
    annotation_sha256 = _file_sha256(annotation_path)
    run_id_payload = json.dumps(
        {
            "annotations": annotation_sha256,
            "seed_artifact": fingerprint,
            "source_commit": source_commit,
            "config": asdict(run),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    run_report = {
        "schema": "egosieve.training-report/v1",
        "run_id": "run-" + hashlib.sha256(run_id_payload).hexdigest()[:16],
        "source_commit": source_commit,
        "completed": True,
        "optimizer_steps": global_step,
        "split_counts": {name: len(values) for name, values in split_examples.items()},
        "checkpoint_sha256": checkpoint_sha256,
        "splits_sha256": splits_sha256,
        "retrieval_objective": {
            "name": "symmetric-instance-contrastive",
            "weight": run.contrastive_weight,
            "temperature": run.contrastive_temperature,
            "same_group_negatives_masked": True,
            "positive_pairs": contrastive_positive_pairs,
        },
        "config": asdict(run),
        "annotations": {
            "filename": annotation_path.name,
            "sha256": annotation_sha256,
            "records": len(records),
            "windows": len(examples),
        },
        "seed_artifact_sha256": fingerprint,
        "backbone": {"model_id": backbone, "revision": backbone_revision},
        "device": str(device),
        "cache": cache_report,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "calibration": {
            "readiness_temperature": temperature,
            "issue_thresholds": issue_thresholds,
            "compiler_thresholds": model.config.compiler_thresholds,
            "split": "validation",
        },
        "history": history,
    }
    _write_json(destination / "training_report.json", run_report)
    evidence_files = (
        "README.md",
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "metrics.json",
        "training_report.json",
        "splits.json",
        "test_predictions.json",
        "configuration_egosieve.py",
        "modeling_egosieve.py",
        "processing_egosieve.py",
    )
    evidence = {
        "schema": "egosieve.release-evidence/v1",
        "artifacts": {name: _file_sha256(destination / name) for name in evidence_files},
    }
    _write_json(destination / "evidence.json", evidence)
    return {
        "output_dir": str(destination),
        "best_epoch": best_epoch,
        "test_macro_f1": metrics["readiness"]["macro_f1"],
        "test_examples": len(split_examples["test"]),
        "device": str(device),
    }


__all__ = [
    "TrainingRunConfig",
    "benchmark_model",
    "evaluate_examples",
    "train_checkpoint",
]
