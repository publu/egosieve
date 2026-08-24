"""Fit deterministic post-training probabilities and routing thresholds."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .data import ISSUE_LABELS
from .metrics import readiness_metrics


def fit_readiness_temperature(
    logits: Any,
    labels: Any,
    valid_mask: Any,
    *,
    max_steps: int = 64,
) -> float:
    """Fit one positive temperature by minimizing held-out cross entropy."""

    raw = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels)
    valid = np.asarray(valid_mask, dtype=bool)
    if raw.ndim != 2 or raw.shape[1] != 3 or truth.shape != (len(raw),):
        raise ValueError("readiness calibration expects logits [N,3] and labels [N]")
    valid = valid & np.isfinite(truth) & (truth >= 0)
    if not np.any(valid):
        raise ValueError("temperature fitting requires at least one valid label")
    selected_logits = torch.tensor(raw[valid], dtype=torch.float64)
    selected_labels = torch.tensor(truth[valid], dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max_steps,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = torch.nn.functional.cross_entropy(selected_logits / temperature, selected_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def _binary_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    true_positive = int(np.count_nonzero(labels & predictions))
    false_positive = int(np.count_nonzero(~labels & predictions))
    false_negative = int(np.count_nonzero(labels & ~predictions))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def fit_issue_thresholds(
    labels: Any,
    probabilities: Any,
    valid_mask: Any,
    *,
    issue_names: Sequence[str] = ISSUE_LABELS,
) -> dict[str, float]:
    """Choose per-label F1 thresholds on validation data with stable tie-breaking."""

    truth = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(probabilities, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    names = tuple(issue_names)
    if truth.shape != scores.shape or truth.shape != valid.shape or truth.ndim != 2:
        raise ValueError("issue calibration arrays must share [N, issues] shape")
    if truth.shape[1] != len(names):
        raise ValueError("issue_names does not match the calibration arrays")
    result: dict[str, float] = {}
    for column, name in enumerate(names):
        selected = valid[:, column] & np.isfinite(truth[:, column])
        y_true = truth[selected, column].astype(bool)
        y_score = scores[selected, column]
        if len(y_true) == 0 or not np.any(y_true) or np.all(y_true):
            result[name] = 0.5
            continue
        candidates = np.unique(np.r_[0.05, np.linspace(0.1, 0.9, 17), y_score, 0.95])
        scored = [
            (_binary_f1(y_true, y_score >= threshold), float(threshold)) for threshold in candidates
        ]
        # Prefer the threshold closest to 0.5, then the larger threshold, when
        # validation F1 ties. This is deterministic and mildly conservative.
        _, threshold = max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5), item[1]))
        result[name] = threshold
    return result


def fit_routing_thresholds(
    readiness_labels: Any,
    probabilities: Any,
    valid_mask: Any,
) -> dict[str, float]:
    """Fit the exact uncertainty/KEEP decision used before segment merging."""

    labels = np.asarray(readiness_labels)
    probs = np.asarray(probabilities, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if probs.ndim != 2 or probs.shape[1] != 3 or labels.shape != (len(probs),):
        raise ValueError("routing calibration expects probabilities [N,3] and labels [N]")
    valid = valid & np.isfinite(labels) & (labels >= 0)
    if not np.any(valid):
        raise ValueError("routing calibration requires valid readiness labels")
    selected_probs = probs[valid]
    selected_labels = labels[valid]
    safe = np.clip(selected_probs, np.finfo(np.float64).tiny, 1.0)
    entropy = -(safe * np.log(safe)).sum(axis=1) / math.log(3)
    routing_uncertainty = np.maximum(entropy, selected_probs[:, 1])
    best: tuple[float, float, float] | None = None
    for uncertainty_threshold in np.linspace(0.35, 0.85, 11):
        for enter_threshold in np.linspace(0.35, 0.9, 12):
            predicted = np.where(
                routing_uncertainty >= uncertainty_threshold,
                1,
                np.where(selected_probs[:, 0] >= enter_threshold, 0, 2),
            )
            score = readiness_metrics(selected_labels, predicted)["macro_f1"]
            numeric = 0.0 if score is None else float(score)
            candidate = (numeric, float(uncertainty_threshold), float(enter_threshold))
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    _, uncertainty_threshold, enter_threshold = best
    return {
        "enter_threshold": enter_threshold,
        "exit_threshold": max(0.0, enter_threshold - 0.15),
        "uncertainty_threshold": uncertainty_threshold,
    }


__all__ = [
    "fit_issue_thresholds",
    "fit_readiness_temperature",
    "fit_routing_thresholds",
]
