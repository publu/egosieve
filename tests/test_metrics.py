from __future__ import annotations

import numpy as np
import pytest

from egosieve.training.metrics import (
    boundary_matching_metrics,
    expected_calibration_error,
    issue_metrics,
    readiness_metrics,
    selective_risk_curve,
    temporal_boundary_metrics,
)


def test_readiness_metrics_for_perfect_and_entirely_wrong_predictions() -> None:
    truth = np.array([0, 1, 2, 0, 1, 2])
    perfect = readiness_metrics(truth, truth)
    np.testing.assert_array_equal(perfect["confusion_matrix"], np.diag([2, 2, 2]))
    assert perfect["macro_f1"] == pytest.approx(1.0)

    wrong = readiness_metrics(truth, (truth + 1) % 3)
    assert wrong["macro_f1"] == pytest.approx(0.0)
    assert wrong["accuracy"] == pytest.approx(0.0)


def test_readiness_missing_labels_and_undefined_classes_are_explicit() -> None:
    result = readiness_metrics(
        np.array([0, -100, 0]),
        np.array([0, 2, 0]),
        valid_mask=np.array([True, True, True]),
    )
    assert result["n_valid"] == 2
    assert result["per_class"]["KEEP"]["f1"] == pytest.approx(1.0)
    assert result["per_class"]["REVIEW"]["f1"] is None
    assert result["per_class"]["REJECT"]["recall"] is None


def test_issue_metrics_report_undefined_columns_without_warnings() -> None:
    truth = np.array(
        [
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
            [1, 0, 1],
        ]
    )
    scores = np.array(
        [
            [0.1, 0.2, 0.8],
            [0.9, 0.1, 0.7],
            [0.2, 0.3, 0.9],
            [0.8, 0.2, 0.6],
        ]
    )
    result = issue_metrics(truth, scores, issue_names=("mixed", "negative", "positive"))

    assert result["per_issue"]["mixed"]["auroc"] == pytest.approx(1.0)
    assert result["per_issue"]["negative"]["auroc"] is None
    assert result["per_issue"]["negative"]["average_precision"] is None
    assert result["per_issue"]["positive"]["auroc"] is None
    assert result["per_issue"]["positive"]["average_precision"] == pytest.approx(1.0)
    assert result["macro_auroc"] == pytest.approx(1.0)
    assert result["macro_average_precision"] == pytest.approx(1.0)


def test_calibration_and_selective_risk_perfect_predictions() -> None:
    truth = np.array([0, 1, 2])
    probabilities = np.eye(3)
    assert expected_calibration_error(truth, probabilities, n_bins=5) == pytest.approx(0.0)
    curve = selective_risk_curve(truth, probabilities)
    np.testing.assert_allclose(curve["coverage"], [1.0])
    np.testing.assert_allclose(curve["risk"], [0.0])
    assert curve["aurc"] == pytest.approx(0.0)


def test_boundary_matching_is_one_to_one_and_sequence_isolated() -> None:
    duplicate = boundary_matching_metrics([1.0], [0.95, 1.02], tolerance_s=0.1)
    assert duplicate["true_positives"] == 1
    assert duplicate["false_positives"] == 1
    assert duplicate["f1"] == pytest.approx(2 / 3)

    isolated = boundary_matching_metrics([[1.0], [10.0]], [[10.0], [1.0]], tolerance_s=0.0)
    assert isolated["true_positives"] == 0
    assert isolated["f1"] == pytest.approx(0.0)


def test_temporal_boundary_metrics_respect_validity_masks() -> None:
    reference = np.array([[1.0, 4.0], [10.0, 14.0]])
    predicted = np.array([[1.05, 8.0], [20.0, 14.05]])
    valid = np.array([[True, False], [False, True]])
    result = temporal_boundary_metrics(
        reference,
        predicted,
        tolerance_s=0.1,
        reference_mask=valid,
        prediction_mask=valid,
    )
    assert result["true_positives"] == 2
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["micro_f1"] == pytest.approx(1.0)
