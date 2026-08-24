"""NumPy-only evaluation metrics for EgoSieve's masked prediction heads."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from .data import BOUNDARY_LABELS, ISSUE_LABELS, READINESS_LABELS


def _mask_1d(mask: Any, length: int, name: str) -> np.ndarray:
    if mask is None:
        return np.ones(length, dtype=bool)
    array = np.asarray(mask)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape {(length,)}, received {array.shape}")
    return array.astype(bool, copy=False)


def _true_class_labels(
    y_true: Any,
    *,
    num_classes: int,
    valid_mask: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(y_true)
    if raw.ndim != 1:
        raise ValueError(f"y_true must be one-dimensional, received shape {raw.shape}")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("y_true must contain numeric class ids") from error
    declared = _mask_1d(valid_mask, len(raw), "valid_mask")
    # Negative values and non-finite values are conventional unknown-label
    # sentinels.  They remain harmless even when no explicit mask is supplied.
    valid = declared & np.isfinite(numeric) & (numeric >= 0)
    selected = numeric[valid]
    if np.any(selected != np.floor(selected)):
        raise ValueError("valid y_true values must be integer class ids")
    if np.any(selected >= num_classes):
        raise ValueError(f"valid y_true values must be in [0, {num_classes - 1}]")
    labels = np.zeros(len(raw), dtype=np.int64)
    labels[valid] = selected.astype(np.int64)
    return labels, valid


def _predicted_class_labels(
    y_pred: Any, *, length: int, num_classes: int, valid: np.ndarray
) -> np.ndarray:
    raw = np.asarray(y_pred)
    if raw.ndim == 2:
        if raw.shape != (length, num_classes):
            raise ValueError(
                f"class scores must have shape {(length, num_classes)}, received {raw.shape}"
            )
        try:
            scores = raw.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("class scores must be numeric") from error
        if np.any(~np.isfinite(scores[valid])):
            raise ValueError("class scores must be finite for valid examples")
        # Replace values on ignored rows so argmax is well-defined there too.
        safe = np.where(valid[:, None], scores, 0.0)
        return np.argmax(safe, axis=1).astype(np.int64)
    if raw.shape != (length,):
        raise ValueError(f"y_pred must have shape {(length,)} or {(length, num_classes)}")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("y_pred must contain numeric class ids") from error
    selected = numeric[valid]
    if np.any(~np.isfinite(selected)) or np.any(selected != np.floor(selected)):
        raise ValueError("predictions for valid examples must be finite integer class ids")
    if np.any((selected < 0) | (selected >= num_classes)):
        raise ValueError(f"predicted class ids must be in [0, {num_classes - 1}]")
    labels = np.zeros(length, dtype=np.int64)
    labels[valid] = selected.astype(np.int64)
    return labels


def confusion_matrix(
    y_true: Any,
    y_pred: Any,
    *,
    num_classes: int = 3,
    valid_mask: Any = None,
) -> np.ndarray:
    """Return a fixed-size confusion matrix with rows=true and columns=predicted."""

    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    true, valid = _true_class_labels(y_true, num_classes=num_classes, valid_mask=valid_mask)
    predicted = _predicted_class_labels(
        y_pred,
        length=len(true),
        num_classes=num_classes,
        valid=valid,
    )
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (true[valid], predicted[valid]), 1)
    return matrix


readiness_confusion_matrix = confusion_matrix


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _defined_mean(values: Iterable[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return None if not defined else float(np.mean(defined))


def readiness_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    valid_mask: Any = None,
    label_names: Sequence[str] = READINESS_LABELS,
) -> dict[str, Any]:
    """Compute fixed three-class confusion, per-class PRF, and macro F1.

    Undefined per-class quantities are represented by ``None`` in
    ``per_class`` and by ``NaN`` in the vector views.  Macro values average
    only defined classes and are ``None`` when there are no valid classes.
    """

    names = tuple(label_names)
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("label_names must contain three unique names")
    matrix = confusion_matrix(y_true, y_pred, num_classes=3, valid_mask=valid_mask)
    precision_values: list[float | None] = []
    recall_values: list[float | None] = []
    f1_values: list[float | None] = []
    per_class: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        support = int(matrix[index, :].sum())
        predicted = int(matrix[:, index].sum())
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        undefined = []
        if precision is None:
            undefined.append("precision")
        if recall is None:
            undefined.append("recall")
        if f1 is None:
            undefined.append("f1")
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
            "undefined": tuple(undefined),
        }

    def vector(values: list[float | None]) -> np.ndarray:
        return np.asarray(
            [np.nan if value is None else value for value in values], dtype=np.float64
        )

    total = int(matrix.sum())
    return {
        "confusion_matrix": matrix,
        "per_class": per_class,
        "precision": vector(precision_values),
        "recall": vector(recall_values),
        "f1": vector(f1_values),
        "support": matrix.sum(axis=1),
        "macro_precision": _defined_mean(precision_values),
        "macro_recall": _defined_mean(recall_values),
        "macro_f1": _defined_mean(f1_values),
        "accuracy": _safe_ratio(int(np.trace(matrix)), total),
        "n_valid": total,
    }


classification_metrics = readiness_metrics


def _binary_values(
    y_true: Any, y_score: Any, valid_mask: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    truth_raw = np.asarray(y_true)
    score_raw = np.asarray(y_score)
    if truth_raw.shape != score_raw.shape:
        raise ValueError(
            f"y_true and y_score must have matching shapes; received {truth_raw.shape} and {score_raw.shape}"
        )
    if truth_raw.ndim != 1:
        raise ValueError("binary metric inputs must be one-dimensional")
    try:
        truth = truth_raw.astype(np.float64)
        scores = score_raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("binary metric inputs must be numeric") from error
    declared = _mask_1d(valid_mask, len(truth), "valid_mask")
    valid = declared & np.isfinite(truth) & (truth >= 0)
    if np.any(~np.isin(truth[valid], (0.0, 1.0))):
        raise ValueError("valid binary labels must be 0 or 1")
    if np.any(~np.isfinite(scores[valid])):
        raise ValueError("scores must be finite for valid labels")
    return truth[valid].astype(np.int8), scores[valid]


def binary_auroc(y_true: Any, y_score: Any, *, valid_mask: Any = None) -> float | None:
    """Compute tie-aware binary AUROC, or ``None`` when either class is absent."""

    truth, scores = _binary_values(y_true, y_score, valid_mask)
    positive_count = int(np.count_nonzero(truth == 1))
    negative_count = int(np.count_nonzero(truth == 0))
    if positive_count == 0 or negative_count == 0:
        return None
    # Mann-Whitney U on average ranks gives tied positive/negative pairs half
    # credit without allocating a potentially enormous pairwise matrix.
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    group_starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    group_ends = np.r_[group_starts[1:], len(scores)]
    for start, end in zip(group_starts, group_ends, strict=True):
        # Ranks are one-based; tied observations receive their average rank.
        ranks[order[start:end]] = (start + 1 + end) / 2.0
    positive_rank_sum = float(ranks[truth == 1].sum())
    u_statistic = positive_rank_sum - positive_count * (positive_count + 1) / 2
    return float(u_statistic / (positive_count * negative_count))


def binary_average_precision(
    y_true: Any,
    y_score: Any,
    *,
    valid_mask: Any = None,
) -> float | None:
    """Compute non-interpolated average precision with tied scores grouped."""

    truth, scores = _binary_values(y_true, y_score, valid_mask)
    positive_count = int(np.count_nonzero(truth == 1))
    if positive_count == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    cumulative_true = np.cumsum(sorted_truth)
    cumulative_count = np.arange(1, len(truth) + 1)
    # A threshold includes every sample tied at that score.  Evaluating only
    # group endpoints avoids order-dependent AP within ties.
    endpoints = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    true_at_threshold = cumulative_true[endpoints]
    count_at_threshold = cumulative_count[endpoints]
    recall = true_at_threshold / positive_count
    precision = true_at_threshold / count_at_threshold
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def _binary_matrix(
    y_true: Any,
    y_score: Any,
    valid_mask: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth_raw = np.asarray(y_true)
    score_raw = np.asarray(y_score)
    if truth_raw.ndim == 1:
        truth_raw = truth_raw[:, None]
    if score_raw.ndim == 1:
        score_raw = score_raw[:, None]
    if truth_raw.ndim != 2 or truth_raw.shape != score_raw.shape:
        raise ValueError(
            "issue labels and scores must have matching [examples, issues] shapes; "
            f"received {truth_raw.shape} and {score_raw.shape}"
        )
    try:
        truth = truth_raw.astype(np.float64)
        scores = score_raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("issue labels and scores must be numeric") from error
    if valid_mask is None:
        declared = np.ones(truth.shape, dtype=bool)
    else:
        mask = np.asarray(valid_mask)
        if mask.shape == (truth.shape[0],):
            declared = np.broadcast_to(mask.astype(bool)[:, None], truth.shape)
        elif mask.shape == truth.shape:
            declared = mask.astype(bool, copy=False)
        else:
            raise ValueError(
                f"valid_mask must have shape {truth.shape} or {(truth.shape[0],)}, received {mask.shape}"
            )
    valid = declared & np.isfinite(truth) & (truth >= 0)
    if np.any(~np.isin(truth[valid], (0.0, 1.0))):
        raise ValueError("valid issue labels must be 0 or 1")
    if np.any(~np.isfinite(scores[valid])):
        raise ValueError("issue scores must be finite for valid labels")
    return truth, scores, valid


def issue_metrics(
    y_true: Any,
    y_score: Any,
    *,
    valid_mask: Any = None,
    issue_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute per-issue and macro AUROC/AP without scikit-learn.

    AUROC is undefined when a valid column lacks positives or negatives.  AP
    is undefined when it lacks positives.  These cases are returned as
    ``None`` and excluded from the corresponding macro average.
    """

    truth, scores, valid = _binary_matrix(y_true, y_score, valid_mask)
    issue_count = truth.shape[1]
    if issue_names is None:
        names = (
            ISSUE_LABELS
            if issue_count == len(ISSUE_LABELS)
            else tuple(f"issue_{index}" for index in range(issue_count))
        )
    else:
        names = tuple(issue_names)
    if len(names) != issue_count or len(set(names)) != len(names):
        raise ValueError(f"issue_names must contain {issue_count} unique names")

    aurocs: list[float | None] = []
    average_precisions: list[float | None] = []
    per_issue: dict[str, dict[str, Any]] = {}
    for column, name in enumerate(names):
        column_mask = valid[:, column]
        labels = truth[:, column]
        column_scores = scores[:, column]
        positives = int(np.count_nonzero(column_mask & (labels == 1)))
        negatives = int(np.count_nonzero(column_mask & (labels == 0)))
        auroc = binary_auroc(labels, column_scores, valid_mask=column_mask)
        average_precision = binary_average_precision(labels, column_scores, valid_mask=column_mask)
        aurocs.append(auroc)
        average_precisions.append(average_precision)
        undefined = []
        if auroc is None:
            undefined.append("auroc")
        if average_precision is None:
            undefined.append("average_precision")
        per_issue[name] = {
            "auroc": auroc,
            "ap": average_precision,
            "average_precision": average_precision,
            "n_valid": positives + negatives,
            "positives": positives,
            "negatives": negatives,
            "undefined": tuple(undefined),
        }

    auroc_vector = np.asarray([np.nan if value is None else value for value in aurocs])
    ap_vector = np.asarray([np.nan if value is None else value for value in average_precisions])
    macro_ap = _defined_mean(average_precisions)
    return {
        "per_issue": per_issue,
        "auroc": auroc_vector,
        "ap": ap_vector,
        "average_precision": ap_vector,
        "macro_auroc": _defined_mean(aurocs),
        "macro_ap": macro_ap,
        "macro_average_precision": macro_ap,
        "undefined_auroc": tuple(
            name for name, value in zip(names, aurocs, strict=True) if value is None
        ),
        "undefined_average_precision": tuple(
            name for name, value in zip(names, average_precisions, strict=True) if value is None
        ),
    }


issue_classification_metrics = issue_metrics


def _probability_inputs(
    y_true: Any,
    probabilities: Any,
    valid_mask: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs_raw = np.asarray(probabilities)
    if probs_raw.ndim != 2 or probs_raw.shape[1] < 2:
        raise ValueError(
            "probabilities must have shape [examples, classes] with at least two classes"
        )
    try:
        probs = probs_raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("probabilities must be numeric") from error
    true, valid = _true_class_labels(
        y_true,
        num_classes=probs.shape[1],
        valid_mask=valid_mask,
    )
    if len(true) != len(probs):
        raise ValueError("y_true and probabilities must contain the same number of examples")
    selected = probs[valid]
    if np.any(~np.isfinite(selected)):
        raise ValueError("probabilities must be finite for valid examples")
    if np.any((selected < -1e-12) | (selected > 1.0 + 1e-12)):
        raise ValueError("probabilities must lie in [0, 1]")
    if len(selected) and not np.allclose(selected.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("each valid probability row must sum to 1")
    probs = np.clip(probs, 0.0, 1.0)
    return true, probs, valid


def calibration_bins(
    y_true: Any,
    probabilities: Any,
    *,
    n_bins: int = 15,
    valid_mask: Any = None,
) -> dict[str, np.ndarray]:
    """Return equal-width top-label calibration-bin statistics."""

    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    true, probs, valid = _probability_inputs(y_true, probabilities, valid_mask)
    selected = probs[valid]
    selected_true = true[valid]
    counts = np.zeros(n_bins, dtype=np.int64)
    accuracy = np.full(n_bins, np.nan, dtype=np.float64)
    confidence = np.full(n_bins, np.nan, dtype=np.float64)
    if len(selected):
        predicted = np.argmax(selected, axis=1)
        confidences = np.max(selected, axis=1)
        correct = predicted == selected_true
        indices = np.minimum((confidences * n_bins).astype(np.int64), n_bins - 1)
        for index in range(n_bins):
            members = indices == index
            counts[index] = int(np.count_nonzero(members))
            if counts[index]:
                accuracy[index] = float(np.mean(correct[members]))
                confidence[index] = float(np.mean(confidences[members]))
    return {
        "lower": np.arange(n_bins, dtype=np.float64) / n_bins,
        "upper": np.arange(1, n_bins + 1, dtype=np.float64) / n_bins,
        "count": counts,
        "accuracy": accuracy,
        "confidence": confidence,
    }


def expected_calibration_error(
    y_true: Any,
    probabilities: Any,
    *,
    n_bins: int = 15,
    valid_mask: Any = None,
) -> float | None:
    """Compute equal-width multiclass top-label expected calibration error."""

    bins = calibration_bins(y_true, probabilities, n_bins=n_bins, valid_mask=valid_mask)
    total = int(bins["count"].sum())
    if total == 0:
        return None
    occupied = bins["count"] > 0
    gaps = np.abs(bins["accuracy"][occupied] - bins["confidence"][occupied])
    return float(np.sum(bins["count"][occupied] / total * gaps))


ece = expected_calibration_error


def _event_sequences(values: Any, name: str) -> list[np.ndarray]:
    if isinstance(values, np.ndarray) and values.ndim > 2:
        raise ValueError(f"{name} must be a one- or two-dimensional collection")
    if isinstance(values, np.ndarray) and values.ndim == 2:
        raw_sequences = [row for row in values]
    else:
        try:
            outer = list(values)
        except TypeError as error:
            raise ValueError(f"{name} must be an iterable of timestamps") from error
        nested = bool(outer) and any(isinstance(item, (list, tuple, np.ndarray)) for item in outer)
        raw_sequences = outer if nested else [outer]

    result: list[np.ndarray] = []
    for index, sequence in enumerate(raw_sequences):
        try:
            array = np.asarray(sequence, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name}[{index}] must contain numeric timestamps") from error
        if array.ndim == 0:
            array = array.reshape(1)
        if array.ndim != 1:
            raise ValueError(f"{name}[{index}] must be one-dimensional")
        result.append(array)
    return result


def _event_masks(mask: Any, sequences: list[np.ndarray], name: str) -> list[np.ndarray]:
    if mask is None:
        return [np.isfinite(sequence) for sequence in sequences]
    masks = _event_sequences(mask, name)
    if len(masks) != len(sequences):
        raise ValueError(f"{name} must contain one mask sequence per timestamp sequence")
    result = []
    for index, (item, sequence) in enumerate(zip(masks, sequences, strict=True)):
        if item.shape != sequence.shape:
            raise ValueError(
                f"{name}[{index}] must have shape {sequence.shape}, received {item.shape}"
            )
        result.append(item.astype(bool) & np.isfinite(sequence))
    return result


def _match_sorted(reference: np.ndarray, predicted: np.ndarray, tolerance_s: float) -> int:
    reference = np.sort(reference)
    predicted = np.sort(predicted)
    true_index = 0
    pred_index = 0
    matched = 0
    while true_index < len(reference) and pred_index < len(predicted):
        delta = predicted[pred_index] - reference[true_index]
        within_tolerance = abs(delta) <= tolerance_s or math.isclose(
            abs(delta), tolerance_s, rel_tol=1e-12, abs_tol=1e-12
        )
        if within_tolerance:
            matched += 1
            true_index += 1
            pred_index += 1
        elif delta < 0:
            pred_index += 1
        else:
            true_index += 1
    return matched


def boundary_matching_metrics(
    reference_times_s: Any,
    predicted_times_s: Any,
    *,
    tolerance_s: float,
    reference_mask: Any = None,
    prediction_mask: Any = None,
) -> dict[str, Any]:
    """One-to-one timestamp matching metrics, isolated per input sequence."""

    if isinstance(tolerance_s, bool) or not isinstance(tolerance_s, (int, float)):
        raise TypeError("tolerance_s must be numeric")
    tolerance = float(tolerance_s)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance_s must be finite and non-negative")
    references = _event_sequences(reference_times_s, "reference_times_s")
    predictions = _event_sequences(predicted_times_s, "predicted_times_s")
    if len(references) != len(predictions):
        raise ValueError(
            "reference and prediction inputs must contain the same number of sequences"
        )
    reference_masks = _event_masks(reference_mask, references, "reference_mask")
    prediction_masks = _event_masks(prediction_mask, predictions, "prediction_mask")
    true_positive = 0
    reference_count = 0
    prediction_count = 0
    for reference, predicted, true_valid, pred_valid in zip(
        references,
        predictions,
        reference_masks,
        prediction_masks,
        strict=True,
    ):
        clean_reference = reference[true_valid]
        clean_prediction = predicted[pred_valid]
        true_positive += _match_sorted(clean_reference, clean_prediction, tolerance)
        reference_count += len(clean_reference)
        prediction_count += len(clean_prediction)
    false_positive = prediction_count - true_positive
    false_negative = reference_count - true_positive
    precision = _safe_ratio(true_positive, prediction_count)
    recall = _safe_ratio(true_positive, reference_count)
    f1 = _safe_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "n_reference": reference_count,
        "n_predicted": prediction_count,
        "tolerance_s": tolerance,
    }


def boundary_matching_f1(
    reference_times_s: Any,
    predicted_times_s: Any,
    *,
    tolerance_s: float,
    reference_mask: Any = None,
    prediction_mask: Any = None,
) -> float | None:
    """Return only the F1 from :func:`boundary_matching_metrics`."""

    return boundary_matching_metrics(
        reference_times_s,
        predicted_times_s,
        tolerance_s=tolerance_s,
        reference_mask=reference_mask,
        prediction_mask=prediction_mask,
    )["f1"]


boundary_f1 = boundary_matching_f1


def temporal_boundary_metrics(
    reference_times_s: Any,
    predicted_times_s: Any,
    *,
    tolerance_s: float,
    reference_mask: Any = None,
    prediction_mask: Any = None,
    boundary_names: Sequence[str] = BOUNDARY_LABELS,
) -> dict[str, Any]:
    """Evaluate an ``[examples, boundary types]`` timestamp matrix.

    Matching is kept separate by example and boundary type, preventing a
    prediction for one video (or an ``end`` prediction) from satisfying a
    different video's ``start`` target.
    """

    reference = np.asarray(reference_times_s, dtype=np.float64)
    predicted = np.asarray(predicted_times_s, dtype=np.float64)
    if reference.ndim != 2 or predicted.shape != reference.shape:
        raise ValueError("boundary timestamps must have matching [examples, boundary types] shapes")
    names = tuple(boundary_names)
    if len(names) != reference.shape[1] or len(set(names)) != len(names):
        raise ValueError(f"boundary_names must contain {reference.shape[1]} unique names")

    def matrix_mask(value: Any, name: str, finite: np.ndarray) -> np.ndarray:
        if value is None:
            return finite
        array = np.asarray(value)
        if array.shape != reference.shape:
            raise ValueError(f"{name} must have shape {reference.shape}, received {array.shape}")
        return array.astype(bool) & finite

    true_valid = matrix_mask(reference_mask, "reference_mask", np.isfinite(reference))
    if prediction_mask is None and reference_mask is not None:
        # An unknown target cell is outside the evaluable region; a prediction
        # there must not become a false positive merely because annotation is
        # unavailable.
        pred_valid = true_valid & np.isfinite(predicted)
    else:
        pred_valid = matrix_mask(prediction_mask, "prediction_mask", np.isfinite(predicted))
    per_boundary: dict[str, dict[str, Any]] = {}
    f1_values: list[float | None] = []
    totals = {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
    for column, name in enumerate(names):
        metrics = boundary_matching_metrics(
            reference[:, column, None],
            predicted[:, column, None],
            tolerance_s=tolerance_s,
            reference_mask=true_valid[:, column, None],
            prediction_mask=pred_valid[:, column, None],
        )
        per_boundary[name] = metrics
        f1_values.append(metrics["f1"])
        for key in totals:
            totals[key] += metrics[key]
    micro_denominator = (
        2 * totals["true_positives"] + totals["false_positives"] + totals["false_negatives"]
    )
    return {
        "per_boundary": per_boundary,
        "macro_f1": _defined_mean(f1_values),
        "micro_f1": _safe_ratio(2 * totals["true_positives"], micro_denominator),
        **totals,
        "tolerance_s": float(tolerance_s),
    }


def selective_risk_curve(
    y_true: Any,
    probabilities: Any,
    *,
    valid_mask: Any = None,
) -> dict[str, Any]:
    """Return error risk as progressively lower-confidence examples are kept.

    There is one operating point per unique confidence threshold, so tied
    examples enter together and the curve cannot depend on their input order.
    """

    true, probs, valid = _probability_inputs(y_true, probabilities, valid_mask)
    selected = probs[valid]
    selected_true = true[valid]
    if len(selected) == 0:
        empty = np.asarray([], dtype=np.float64)
        return {"coverage": empty, "risk": empty, "threshold": empty, "aurc": None, "n_valid": 0}
    predicted = np.argmax(selected, axis=1)
    confidence = np.max(selected, axis=1)
    errors = (predicted != selected_true).astype(np.int64)
    order = np.argsort(-confidence, kind="stable")
    confidence = confidence[order]
    errors = errors[order]
    endpoints = np.flatnonzero(np.r_[confidence[1:] != confidence[:-1], True])
    retained = endpoints + 1
    coverage = retained / len(selected)
    risk = np.cumsum(errors)[endpoints] / retained
    threshold = confidence[endpoints]
    coverage_steps = np.diff(np.r_[0.0, coverage])
    aurc = float(np.sum(coverage_steps * risk))
    return {
        "coverage": coverage.astype(np.float64),
        "risk": risk.astype(np.float64),
        "threshold": threshold.astype(np.float64),
        "aurc": aurc,
        "n_valid": len(selected),
    }


risk_coverage_curve = selective_risk_curve


__all__ = [
    "binary_auroc",
    "binary_average_precision",
    "boundary_f1",
    "boundary_matching_f1",
    "boundary_matching_metrics",
    "calibration_bins",
    "classification_metrics",
    "confusion_matrix",
    "ece",
    "expected_calibration_error",
    "issue_classification_metrics",
    "issue_metrics",
    "readiness_confusion_matrix",
    "readiness_metrics",
    "risk_coverage_curve",
    "selective_risk_curve",
    "temporal_boundary_metrics",
]
