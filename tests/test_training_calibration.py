from __future__ import annotations

import numpy as np

from egosieve.training import (
    fit_issue_thresholds,
    fit_readiness_temperature,
    fit_routing_thresholds,
)


def test_temperature_is_positive_and_improves_overconfident_logits() -> None:
    logits = np.asarray(
        [
            [8.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [0.0, 0.0, 8.0],
        ]
    )
    labels = np.asarray([0, 1, 1, 2])
    temperature = fit_readiness_temperature(logits, labels, np.ones(4, dtype=bool))
    assert temperature > 1.0


def test_issue_thresholds_use_only_declared_targets() -> None:
    labels = np.tile(np.asarray([[0.0], [1.0], [1.0], [0.0]]), (1, 8))
    scores = np.tile(np.asarray([[0.1], [0.4], [0.8], [0.2]]), (1, 8))
    valid = np.ones_like(labels, dtype=bool)
    valid[:, 3] = False
    thresholds = fit_issue_thresholds(labels, scores, valid)
    assert len(thresholds) == 8
    assert thresholds["camera_instability"] == 0.5
    assert 0.2 < thresholds["blur"] <= 0.4


def test_routing_thresholds_are_compiler_compatible() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.05, 0.9],
            [0.75, 0.15, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.1, 0.8],
        ]
    )
    fitted = fit_routing_thresholds(
        np.asarray([0, 1, 2, 0, 1, 2]),
        probabilities,
        np.ones(6, dtype=bool),
    )
    assert 0 <= fitted["exit_threshold"] <= fitted["enter_threshold"] <= 1
    assert 0 <= fitted["uncertainty_threshold"] <= 1
